"""Eval harness for Fleet Copilot: runs evals/cases.yaml through the agent and grades the answers.

Usage:
  uv run python evals/run_evals.py --check          validate cases.yaml, no model calls
  uv run python evals/run_evals.py                  run every case against Claude (needs ANTHROPIC_API_KEY)
  uv run python evals/run_evals.py --case ID        run one case
  uv run python evals/run_evals.py --model MODEL    override FLEET_MODEL

Grading is deterministic: id and substring checks plus citation checks, no LLM judge.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic
import yaml
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from fleet_copilot.agent import DEFAULT_MODEL, build_graph, thread_config
from fleet_copilot.data import load_snapshots
from fleet_copilot.tools import TOOLS

EVALS_DIR = Path(__file__).resolve().parent
CASES_PATH = EVALS_DIR / "cases.yaml"
REPORT_PATH = EVALS_DIR / "report.json"

REQUIRED_FIELDS = {"id", "company_id", "question", "expected_devices"}
OPTIONAL_FIELDS = {"forbidden_devices", "forbidden_text", "expected_tools", "must_mention", "must_mention_any"}
CHECK_FIELDS = ("expected_devices", "forbidden_devices", "forbidden_text", "expected_tools", "must_mention", "must_mention_any")
SNAPSHOT_RE = re.compile(r"[A-Z0-9]{12}@\d{4}-\d{2}-\d{2}T[\d:]+Z")

Case = dict[str, Any]


def normalize(text: str) -> str:
    """Lowercase and drop accents, so 'Está' and 'esta' compare equal."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def device_in_text(device_id: str, text: str) -> bool:
    """True if the device id appears as a whole token (device ids are alphanumeric)."""
    pattern = rf"(?<![A-Za-z0-9]){re.escape(device_id)}(?![A-Za-z0-9])"
    return re.search(pattern, text, re.IGNORECASE) is not None


def dataset_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Return (device_id -> company_id, snapshot_id -> company_id) from the telemetry data."""
    snapshots = load_snapshots()
    device_company = dict(zip(snapshots["device_id"], snapshots["company_id"]))
    snapshot_company = dict(zip(snapshots["snapshot_id"], snapshots["company_id"]))
    return device_company, snapshot_company


def load_cases(path: Path = CASES_PATH) -> list[Case]:
    """Read the case list from the YAML file."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)["cases"]


def validate_cases(cases: list[Case], device_company: dict[str, str], tool_names: set[str]) -> list[str]:
    """Return a list of problems with the cases. An empty list means they are valid."""
    problems: list[str] = []
    companies = set(device_company.values())
    seen_ids: set[str] = set()

    for index, case in enumerate(cases):
        label = case.get("id", f"#{index}")
        if not isinstance(case, dict):
            problems.append(f"{label}: case is not a mapping")
            continue

        missing = REQUIRED_FIELDS - set(case)
        if missing:
            problems.append(f"{label}: missing fields {sorted(missing)}")
        unknown = set(case) - REQUIRED_FIELDS - OPTIONAL_FIELDS
        if unknown:
            problems.append(f"{label}: unknown fields {sorted(unknown)}")

        case_id = case.get("id")
        if case_id in seen_ids:
            problems.append(f"{label}: duplicate id")
        seen_ids.add(case_id)

        company = case.get("company_id")
        if company not in companies:
            problems.append(f"{label}: unknown company_id {company!r}")

        question = case.get("question")
        if not isinstance(question, str) or not question.strip():
            problems.append(f"{label}: question must be a non-empty string")

        for key in ("expected_devices", "forbidden_devices"):
            value = case.get(key, [])
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                problems.append(f"{label}: {key} must be a list of strings")
                continue
            for device_id in value:
                if device_id not in device_company:
                    problems.append(f"{label}: {key} has {device_id}, which is not in the dataset")
                elif key == "expected_devices" and device_company[device_id] != company:
                    problems.append(
                        f"{label}: expected device {device_id} belongs to {device_company[device_id]}, not {company}"
                    )

        expected = set(case.get("expected_devices") or [])
        overlap = expected & set(case.get("forbidden_devices") or [])
        if overlap:
            problems.append(f"{label}: devices both expected and forbidden: {sorted(overlap)}")

        for tool in case.get("expected_tools") or []:
            if tool not in tool_names:
                problems.append(f"{label}: unknown tool {tool!r} in expected_tools")

        for key in ("forbidden_text", "must_mention", "must_mention_any"):
            value = case.get(key, [])
            if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
                problems.append(f"{label}: {key} must be a list of non-empty strings")

        if not any(case.get(key) for key in CHECK_FIELDS):
            problems.append(f"{label}: case has no check beyond citations")

    return problems


def grade(
    case: Case,
    answer_text: str,
    tool_names: list[str],
    snapshot_company: dict[str, str],
) -> dict[str, Any]:
    """Apply every deterministic check to one answer. Pure: no model or file access."""
    company = case["company_id"]
    text = normalize(answer_text)
    citations = SNAPSHOT_RE.findall(answer_text)
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, detail: str) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    missing = [d for d in case.get("expected_devices", []) if not device_in_text(d, answer_text)]
    add("expected_devices", not missing, f"missing: {', '.join(missing)}" if missing else "all present")

    forbidden = [d for d in case.get("forbidden_devices", []) if device_in_text(d, answer_text)]
    add("forbidden_devices", not forbidden, f"found: {', '.join(forbidden)}" if forbidden else "none found")

    banned = [t for t in case.get("forbidden_text", []) if normalize(t) in text]
    add("forbidden_text", not banned, f"found: {', '.join(banned)}" if banned else "none found")

    called = set(tool_names)
    absent_tools = [t for t in case.get("expected_tools", []) if t not in called]
    add("expected_tools", not absent_tools, f"not called: {', '.join(absent_tools)}" if absent_tools else "all called")

    absent = [m for m in case.get("must_mention", []) if normalize(m) not in text]
    add("must_mention", not absent, f"missing: {', '.join(absent)}" if absent else "all present")

    any_of = case.get("must_mention_any", [])
    if any_of:
        hit = any(normalize(m) in text for m in any_of)
        add("must_mention_any", hit, "one present" if hit else f"none of: {', '.join(any_of)}")

    unknown = [c for c in citations if c not in snapshot_company]
    add("citations_exist", not unknown, f"unknown: {', '.join(unknown)}" if unknown else "all exist")

    foreign = [c for c in citations if c in snapshot_company and snapshot_company[c] != company]
    add("citations_company", not foreign, f"other company: {', '.join(foreign)}" if foreign else "all match")

    needs_citation = bool(case.get("expected_devices"))
    add(
        "citations_present",
        not needs_citation or bool(citations),
        "citation present" if citations else ("no citation" if needs_citation else "not required"),
    )

    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "citations": sorted(set(citations)),
    }


def _text(message: AIMessage) -> str:
    """The text blocks of an AI message, ignoring tool-use blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


SETUP_ERRORS = (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.BadRequestError)


def is_setup_error(exc: BaseException) -> bool:
    """True for errors that mean the configuration is wrong (key, permissions, model request), not one case.

    langchain-anthropic wraps these in subclasses of the SDK classes (for example AnthropicInvalidRequestError
    subclasses anthropic.BadRequestError), so isinstance covers the wrapped errors too.
    """
    return isinstance(exc, SETUP_ERRORS)


def run_case(case: Case, model: Any = None) -> dict[str, Any]:
    """Run one case through a fresh graph and return the raw outcome (no grading)."""
    graph = build_graph(model=model, checkpointer=InMemorySaver())
    config = thread_config(case["company_id"], f"eval-{case['id']}")
    state = {"messages": [HumanMessage(case["question"])], "company_id": case["company_id"]}
    final = graph.invoke(state, config)

    ai_messages = [m for m in final["messages"] if isinstance(m, AIMessage)]
    tool_names = [call["name"] for m in ai_messages for call in m.tool_calls]
    answer = _text(ai_messages[-1]) if ai_messages else ""
    return {"answer": answer, "tool_names": tool_names}


def _failed_names(result: dict[str, Any]) -> str:
    failed = [name for name, check in result["checks"].items() if not check["passed"]]
    return ", ".join(failed)


def print_table(rows: list[tuple[str, bool, str]]) -> None:
    """Print one line per case: id, PASS or FAIL, and the names of failed checks."""
    width = max(len(case_id) for case_id, _, _ in rows) if rows else 10
    print(f"{'CASE'.ljust(width)}  RESULT  FAILED CHECKS")
    for case_id, passed, failed in rows:
        print(f"{case_id.ljust(width)}  {'PASS' if passed else 'FAIL'}    {failed}")


def main(argv: list[str] | None = None, model: Any = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the Fleet Copilot evals.")
    parser.add_argument("--check", action="store_true", help="validate cases.yaml only, no model calls")
    parser.add_argument("--case", help="run only the case with this id")
    parser.add_argument("--model", help="model id, overrides FLEET_MODEL")
    args = parser.parse_args(argv)

    cases = load_cases()
    device_company, snapshot_company = dataset_maps()
    tool_names = {tool.name for tool in TOOLS}
    problems = validate_cases(cases, device_company, tool_names)

    if args.check:
        if problems:
            for problem in problems:
                print(f"PROBLEM: {problem}")
            return 1
        print(f"OK: {len(cases)} cases are valid")
        return 0

    if problems:
        for problem in problems:
            print(f"PROBLEM: {problem}", file=sys.stderr)
        print("cases.yaml has problems; run with --check for details", file=sys.stderr)
        return 1

    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"no case with id {args.case!r}", file=sys.stderr)
            return 2

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Add it to .env or the environment; the evals call Claude.", file=sys.stderr)
        return 2

    if args.model:
        os.environ["FLEET_MODEL"] = args.model
    model_name = os.environ.get("FLEET_MODEL", DEFAULT_MODEL)

    results: list[dict[str, Any]] = []
    rows: list[tuple[str, bool, str]] = []
    for case in cases:
        print(f"running {case['id']} ...", file=sys.stderr)
        try:
            outcome = run_case(case, model=model)
            graded = grade(case, outcome["answer"], outcome["tool_names"], snapshot_company)
            error = None
        except Exception as exc:  # a failed model call fails the case, the run continues
            if is_setup_error(exc):
                print(
                    f"Setup error on case {case['id']}: {type(exc).__name__}: {exc}. "
                    "Fix the API key or configuration and run again.",
                    file=sys.stderr,
                )
                return 2
            outcome = {"answer": "", "tool_names": []}
            graded = {
                "passed": False,
                "checks": {"run": {"passed": False, "detail": f"{type(exc).__name__}: {exc}"}},
                "citations": [],
            }
            error = f"{type(exc).__name__}: {exc}"

        result = {
            "id": case["id"],
            "company_id": case["company_id"],
            "question": case["question"],
            "answer": outcome["answer"],
            "tools_called": outcome["tool_names"],
            "citations": graded["citations"],
            "checks": graded["checks"],
            "passed": graded["passed"],
            "error": error,
        }
        results.append(result)
        rows.append((case["id"], graded["passed"], _failed_names(graded)))

    print_table(rows)
    passed = sum(1 for r in results if r["passed"])
    print(f"\nScore: {passed}/{len(results)} cases passed")

    report = {
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "score": {"passed": passed, "total": len(results)},
        "cases": results,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Eval harness for Fleet Copilot: runs evals/cases.yaml through the agent and grades the answers.

Usage:
  uv run python evals/run_evals.py --check          validate cases.yaml, no model calls
  uv run python evals/run_evals.py                  run every case against Claude (needs ANTHROPIC_API_KEY)
  uv run python evals/run_evals.py --case ID        run one case
  uv run python evals/run_evals.py --repeat N       run every case N times and report pass rates
  uv run python evals/run_evals.py --model MODEL    override FLEET_MODEL

Grading is deterministic: id, substring, citation and action checks, no LLM judge.
Action proposals are approved or rejected by the case's approval policy. Every run writes its
outbox and audit log to a new folder under evals/runs/, so evals never touch outbox/ or audit.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import anthropic
import yaml
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from fleet_copilot.agent import DEFAULT_MODEL, build_graph, thread_config
from fleet_copilot.data import load_snapshots
from fleet_copilot.tools import TOOLS
from fleet_copilot.tools_write import ACTION_TOOLS

EVALS_DIR = Path(__file__).resolve().parent
CASES_PATH = EVALS_DIR / "cases.yaml"
REPORT_PATH = EVALS_DIR / "report.json"
RUNS_DIR = EVALS_DIR / "runs"

REQUIRED_FIELDS = {"id", "company_id", "question", "expected_devices"}
OPTIONAL_FIELDS = {
    "forbidden_devices",
    "forbidden_text",
    "expected_tools",
    "must_mention",
    "must_mention_any",
    "expected_actions",
    "expected_statuses",
    "forbidden_statuses",
    "approval",
    "allows_actions",
    "require_citations",
}
CHECK_FIELDS = (
    "expected_devices",
    "forbidden_devices",
    "forbidden_text",
    "expected_tools",
    "must_mention",
    "must_mention_any",
    "expected_actions",
    "expected_statuses",
    "forbidden_statuses",
)
SNAPSHOT_RE = re.compile(r"[A-Z0-9]{12}@\d{4}-\d{2}-\d{2}T[\d:]+Z")

ACTION_TOOL_NAMES = {tool.name for tool in ACTION_TOOLS}
ACTION_ARG_NAMES = {arg for tool in ACTION_TOOLS for arg in tool.args} - {"company_id", "evidence"}
STATUSES = ("created", "rejected", "refused", "error")
APPROVALS = ("approve", "reject")
DEFAULT_APPROVAL = "reject"
APPROVAL_REASON = "eval approval policy"
MAX_RESUMES = 5

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


def _status_problems(label: str, key: str, tool: Any, statuses: list[Any]) -> list[str]:
    problems = []
    if tool not in ACTION_TOOL_NAMES:
        problems.append(f"{label}: {key} tool {tool!r} is not an action tool")
    for status in statuses:
        if status not in STATUSES:
            problems.append(f"{label}: {key} status {status!r} must be one of {', '.join(STATUSES)}")
    return problems


def _action_problems(label: str, case: Case) -> list[str]:
    """Checks for the action and guardrail fields: approval, expected/forbidden actions and statuses."""
    problems: list[str] = []

    approval = case.get("approval", DEFAULT_APPROVAL)
    if approval not in APPROVALS:
        problems.append(f"{label}: approval must be approve or reject, not {approval!r}")

    for key in ("allows_actions", "require_citations"):
        if key in case and not isinstance(case[key], bool):
            problems.append(f"{label}: {key} must be true or false")

    actions = case.get("expected_actions", [])
    if not isinstance(actions, list) or not all(isinstance(a, dict) for a in actions):
        problems.append(f"{label}: expected_actions must be a list of mappings")
    else:
        for matcher in actions:
            problems.extend(_status_problems(label, "expected_actions", matcher.get("tool"), []))
            for key in sorted(set(matcher) - {"tool"}):
                if key not in ACTION_ARG_NAMES:
                    problems.append(f"{label}: expected_actions matcher key {key!r} is not an action argument")

    expected = case.get("expected_statuses", {})
    if not isinstance(expected, dict):
        problems.append(f"{label}: expected_statuses must be a mapping of tool to status")
    else:
        for tool, status in expected.items():
            problems.extend(_status_problems(label, "expected_statuses", tool, [status]))

    forbidden = case.get("forbidden_statuses", {})
    if not isinstance(forbidden, dict) or not all(isinstance(v, list) for v in forbidden.values()):
        problems.append(f"{label}: forbidden_statuses must be a mapping of tool to a list of statuses")
    else:
        for tool, statuses in forbidden.items():
            problems.extend(_status_problems(label, "forbidden_statuses", tool, statuses))

    return problems


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

        problems.extend(_action_problems(label, case))

        if not any(case.get(key) for key in CHECK_FIELDS):
            problems.append(f"{label}: case has no check beyond citations")

    return problems


def _call_matches(matcher: dict[str, Any], call: dict[str, Any]) -> bool:
    """A matcher names a tool and optionally argument values; every one must equal the call's value."""
    if call["name"] != matcher["tool"]:
        return False
    args = call.get("args") or {}
    return all(args.get(key) == value for key, value in matcher.items() if key != "tool")


def _result_has(results: list[dict[str, Any]], tool: str, status: str) -> bool:
    return any(r["name"] == tool and r["status"] == status for r in results)


def grade(
    case: Case,
    answer_text: str,
    tool_names: list[str],
    snapshot_company: dict[str, str],
    tool_calls: list[dict[str, Any]] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply every deterministic check to one answer. Pure: no model or file access.

    tool_names: names of every tool call, in order. tool_calls: {"name", "args"} per call.
    tool_results: {"name", "status"} per tool result. Action checks need tool_calls and tool_results.
    """
    company = case["company_id"]
    text = normalize(answer_text)
    citations = SNAPSHOT_RE.findall(answer_text)
    calls = tool_calls or []
    results = tool_results or []
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

    if case.get("expected_actions"):
        unmet = [m for m in case["expected_actions"] if not any(_call_matches(m, c) for c in calls)]
        add(
            "expected_actions",
            not unmet,
            f"not proposed: {', '.join(json.dumps(m, ensure_ascii=False) for m in unmet)}" if unmet else "all proposed",
        )

    if case.get("expected_statuses"):
        unmet_status = [
            f"{tool}={status}"
            for tool, status in case["expected_statuses"].items()
            if not _result_has(results, tool, status)
        ]
        add(
            "expected_statuses",
            not unmet_status,
            f"missing: {', '.join(unmet_status)}" if unmet_status else "all present",
        )

    if case.get("forbidden_statuses"):
        found_status = [
            f"{tool}={status}"
            for tool, statuses in case["forbidden_statuses"].items()
            for status in statuses
            if _result_has(results, tool, status)
        ]
        add(
            "forbidden_statuses",
            not found_status,
            f"found: {', '.join(found_status)}" if found_status else "none found",
        )

    acted = sorted({name for name in tool_names if name in ACTION_TOOL_NAMES})
    if case.get("expected_actions") or case.get("allows_actions"):
        add("no_unrequested_action", True, "actions allowed by this case")
    else:
        add(
            "no_unrequested_action",
            not acted,
            f"called: {', '.join(acted)}" if acted else "no action called",
        )

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

    if case.get("require_citations"):
        valid = [c for c in citations if snapshot_company.get(c) == company]
        add(
            "require_citations",
            bool(valid),
            f"{len(set(valid))} valid citation(s)" if valid else "no valid citation of this company",
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


def _tool_status(content: Any) -> str:
    """The status of one tool result: its "status", "error" when it has only an error, else "ok"."""
    try:
        obj = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        return "unparsed"
    if not isinstance(obj, dict):
        return "unparsed"
    return obj.get("status") or ("error" if "error" in obj else "ok")


SETUP_ERRORS = (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.BadRequestError)


def is_setup_error(exc: BaseException) -> bool:
    """True for errors that mean the configuration is wrong (key, permissions, model request), not one case.

    langchain-anthropic wraps these in subclasses of the SDK classes (for example AnthropicInvalidRequestError
    subclasses anthropic.BadRequestError), so isinstance covers the wrapped errors too.
    """
    return isinstance(exc, SETUP_ERRORS)


def _empty_outcome() -> dict[str, Any]:
    return {
        "answer": "",
        "plan": "",
        "tool_names": [],
        "tool_calls": [],
        "tool_results": [],
        "approvals": 0,
        "run_error": None,
    }


def run_case(case: Case, model: Any = None) -> dict[str, Any]:
    """Run one case through a fresh graph and return the raw outcome (no grading).

    Every approval request is answered with the case's policy (approve or reject). The graph
    may pause up to MAX_RESUMES times; if it still waits after that, run_error says so.
    """
    graph = build_graph(model=model, checkpointer=InMemorySaver())
    config = thread_config(case["company_id"], f"eval-{case['id']}")
    state = {"messages": [HumanMessage(case["question"])], "company_id": case["company_id"]}
    graph.invoke(state, config)

    policy = case.get("approval", DEFAULT_APPROVAL)
    decision = {"approved": policy == "approve", "reason": APPROVAL_REASON}
    approvals = 0
    resumes = 0
    run_error = None
    while True:
        pending = [item for task in graph.get_state(config).tasks for item in task.interrupts]
        if not pending:
            break
        if resumes == MAX_RESUMES:
            run_error = f"still waiting for approval after {MAX_RESUMES} resumes; the case stops here"
            break
        approvals += len(pending)
        resumes += 1
        graph.invoke(Command(resume={item.id: decision for item in pending}), config)

    messages = graph.get_state(config).values["messages"]
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    tool_calls = [{"name": call["name"], "args": call["args"]} for m in ai_messages for call in m.tool_calls]
    names_by_id = {call["id"]: call["name"] for m in ai_messages for call in m.tool_calls}
    tool_results = [
        {
            "name": message.name or names_by_id.get(message.tool_call_id, ""),
            "status": _tool_status(message.content),
        }
        for message in messages
        if isinstance(message, ToolMessage)
    ]
    return {
        "answer": _text(ai_messages[-1]) if ai_messages else "",
        "plan": graph.get_state(config).values.get("plan", ""),
        "tool_names": [call["name"] for call in tool_calls],
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "approvals": approvals,
        "run_error": run_error,
    }


def run_once(case: Case, model: Any, snapshot_company: dict[str, str]) -> dict[str, Any]:
    """Run and grade one case once. A setup error is re-raised for main; any other error fails this run."""
    try:
        outcome = run_case(case, model=model)
    except Exception as exc:
        if is_setup_error(exc):
            raise
        outcome = _empty_outcome()
        graded = {
            "passed": False,
            "checks": {"run": {"passed": False, "detail": f"{type(exc).__name__}: {exc}"}},
            "citations": [],
        }
        error = f"{type(exc).__name__}: {exc}"
    else:
        graded = grade(
            case,
            outcome["answer"],
            outcome["tool_names"],
            snapshot_company,
            tool_calls=outcome["tool_calls"],
            tool_results=outcome["tool_results"],
        )
        error = outcome["run_error"]
        if error:
            graded["checks"]["run"] = {"passed": False, "detail": error}
            graded["passed"] = False

    return {
        "answer": outcome["answer"],
        "plan": outcome["plan"],
        "tools_called": outcome["tool_names"],
        "tool_calls": outcome["tool_calls"],
        "tool_results": outcome["tool_results"],
        "approvals": outcome["approvals"],
        "citations": graded["citations"],
        "checks": graded["checks"],
        "passed": graded["passed"],
        "error": error,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pass counts from graded runs pure: no model or file access.

    Each record is {"id": case id, "runs": [{"passed": bool, "checks": {...}}, ...]}. Returns the overall
    run and case counts, and per case the passes, the runs and the failed checks with the number of runs
    in which each one failed.
    """
    cases: dict[str, dict[str, Any]] = {}
    for record in records:
        runs = record["runs"]
        failed: dict[str, int] = {}
        for run in runs:
            for name, check in run["checks"].items():
                if not check["passed"]:
                    failed[name] = failed.get(name, 0) + 1
        cases[record["id"]] = {
            "passed": sum(1 for run in runs if run["passed"]),
            "total": len(runs),
            "failed_checks": failed,
        }
    runs_passed = sum(c["passed"] for c in cases.values())
    runs_total = sum(c["total"] for c in cases.values())
    return {
        "runs_passed": runs_passed,
        "runs_total": runs_total,
        "cases_perfect": sum(1 for c in cases.values() if c["total"] and c["passed"] == c["total"]),
        "cases_total": len(cases),
        "cases": cases,
    }


def exit_code(summary: dict[str, Any]) -> int:
    """0 only when every run of every case passed."""
    return 0 if summary["runs_total"] and summary["runs_passed"] == summary["runs_total"] else 1


def print_table(summary: dict[str, Any]) -> None:
    """One line per case: id, passes out of runs, and the failed checks with the number of runs that failed."""
    rows = [
        (
            case_id,
            f"{case['passed']}/{case['total']}",
            ", ".join(f"{name} x{count}" for name, count in case["failed_checks"].items()),
        )
        for case_id, case in summary["cases"].items()
    ]
    id_width = max((len(row[0]) for row in rows), default=4)
    rate_width = max([len("PASS RATE")] + [len(row[1]) for row in rows])
    print(f"{'CASE'.ljust(id_width)}  {'PASS RATE'.ljust(rate_width)}  FAILED CHECKS")
    for case_id, rate, failed in rows:
        print(f"{case_id.ljust(id_width)}  {rate.ljust(rate_width)}  {failed}".rstrip())


def new_run_dir() -> Path:
    """A fresh folder under RUNS_DIR named by the UTC time the run starts."""
    path = RUNS_DIR / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def isolated_run(run_dir: Path) -> Iterator[None]:
    """Point the outbox and audit log at run_dir while the cases run, then restore the old values."""
    overrides = {"FLEET_OUTBOX_DIR": str(run_dir), "FLEET_AUDIT_PATH": str(run_dir / "audit.jsonl")}
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main(argv: list[str] | None = None, model: Any = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the Fleet Copilot evals.")
    parser.add_argument("--check", action="store_true", help="validate cases.yaml only, no model calls")
    parser.add_argument("--case", help="run only the case with this id")
    parser.add_argument("--repeat", type=int, default=1, help="run every case N times and report pass rates")
    parser.add_argument("--model", help="model id, overrides FLEET_MODEL")
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

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

    run_dir = new_run_dir()
    print(f"Run folder: {run_dir} (outbox and audit log for this run)")

    records: list[dict[str, Any]] = []
    with isolated_run(run_dir):
        for case in cases:
            runs: list[dict[str, Any]] = []
            for number in range(1, args.repeat + 1):
                print(f"running {case['id']} ({number}/{args.repeat}) ...", file=sys.stderr)
                try:
                    run = run_once(case, model, snapshot_company)
                except Exception as exc:
                    if is_setup_error(exc):
                        print(
                            f"Setup error on case {case['id']}: {type(exc).__name__}: {exc}. "
                            "Fix the API key or configuration and run again.",
                            file=sys.stderr,
                        )
                        return 2
                    raise
                runs.append({"run": number, **run})
            records.append(
                {
                    "id": case["id"],
                    "company_id": case["company_id"],
                    "question": case["question"],
                    "approval": case.get("approval", DEFAULT_APPROVAL),
                    "runs": runs,
                }
            )

    summary = aggregate(records)
    print_table(summary)
    rate = 100 * summary["runs_passed"] / summary["runs_total"] if summary["runs_total"] else 0.0
    print(f"\nRuns passed: {summary['runs_passed']}/{summary['runs_total']} ({rate:.0f}%)")
    print(f"Cases at 100%: {summary['cases_perfect']}/{summary['cases_total']}")

    report = {
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repeat": args.repeat,
        "run_dir": str(run_dir),
        "summary": summary,
        "cases": records,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report written to {REPORT_PATH}")

    return exit_code(summary)


if __name__ == "__main__":
    sys.exit(main())

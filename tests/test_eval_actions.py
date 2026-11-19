"""Unit tests for the action and guardrail checks of the eval harness. No model calls.

The scripted model answers the way Claude would, so run_case and main can run end to end. Every test
points the outbox, the audit log and the run folders at tmp_path, so nothing touches outbox/,
audit.jsonl, evals/runs/ or evals/report.json.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from fleet_copilot.data import load_snapshots
from fleet_copilot.tools import TOOLS

RUN_EVALS = Path(__file__).resolve().parents[1] / "evals" / "run_evals.py"
ACME = "acme-001"
PLAN = AIMessage(content="1. Call the action tool.")
TOOL_NAMES = {tool.name for tool in TOOLS}


@pytest.fixture(scope="module")
def evals():
    spec = importlib.util.spec_from_file_location("run_evals", RUN_EVALS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def snapshot_company(evals):
    _, snapshot_company = evals.dataset_maps()
    return snapshot_company


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch, evals):
    """Everything the harness writes goes under tmp_path."""
    monkeypatch.setenv("FLEET_OUTBOX_DIR", str(tmp_path / "outbox"))
    monkeypatch.setenv("FLEET_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    monkeypatch.setattr(evals, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(evals, "REPORT_PATH", tmp_path / "report.json")
    return tmp_path


def _latest_snapshot(device_id):
    snapshots = load_snapshots()
    return snapshots[snapshots["device_id"] == device_id].sort_values("collected_at")["snapshot_id"].iloc[-1]


class ScriptedModel(BaseChatModel):
    """Returns the given AI messages in order. bind_tools returns a copy that shares the script."""

    script: list[AIMessage]
    seen: list[list] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def _call(name, args, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _remediation_args(evidence=None):
    device = "MT7PJB7N5LRE"
    return {
        "device_id": device,
        "check_id": "screen_lock",
        "note": "Activar el bloqueo de pantalla.",
        "evidence": evidence or [_latest_snapshot(device)],
    }


def _case(**overrides):
    case = {
        "id": "test-case",
        "company_id": ACME,
        "question": "pregunta",
        "expected_devices": [],
    }
    case.update(overrides)
    return case


# grade(): expected_actions


def test_expected_actions_passes_when_a_call_matches_every_argument(evals, snapshot_company):
    case = _case(expected_actions=[{"tool": "open_remediation_ticket", "device_id": "MT7PJB7N5LRE", "check_id": "screen_lock"}])
    calls = [{"name": "open_remediation_ticket", "args": _remediation_args()}]
    result = evals.grade(case, "Listo.", ["open_remediation_ticket"], snapshot_company, tool_calls=calls)
    assert result["checks"]["expected_actions"]["passed"], result["checks"]


def test_expected_actions_fails_when_an_argument_differs(evals, snapshot_company):
    case = _case(expected_actions=[{"tool": "open_remediation_ticket", "device_id": "MT7PJB7N5LRE", "check_id": "screen_lock"}])
    args = {**_remediation_args(), "check_id": "disk_encryption"}
    calls = [{"name": "open_remediation_ticket", "args": args}]
    result = evals.grade(case, "Listo.", ["open_remediation_ticket"], snapshot_company, tool_calls=calls)
    assert not result["checks"]["expected_actions"]["passed"]
    assert "screen_lock" in result["checks"]["expected_actions"]["detail"]


def test_expected_actions_fails_when_no_call_was_made(evals, snapshot_company):
    case = _case(expected_actions=[{"tool": "create_upgrade_order", "device_id": "M4XVHUV1MEPZ", "component": "ram"}])
    result = evals.grade(case, "No.", [], snapshot_company, tool_calls=[])
    assert not result["passed"]
    assert not result["checks"]["expected_actions"]["passed"]


def test_expected_actions_needs_every_matcher(evals, snapshot_company):
    case = _case(
        expected_actions=[
            {"tool": "open_remediation_ticket", "device_id": "MT7PJB7N5LRE"},
            {"tool": "notify_employee", "employee_id": "emp-acme-1005"},
        ]
    )
    calls = [{"name": "open_remediation_ticket", "args": _remediation_args()}]
    result = evals.grade(case, "Listo.", ["open_remediation_ticket"], snapshot_company, tool_calls=calls)
    assert not result["checks"]["expected_actions"]["passed"]
    assert "notify_employee" in result["checks"]["expected_actions"]["detail"]


# grade(): expected_statuses and forbidden_statuses


def test_expected_statuses_passes_when_a_result_has_the_status(evals, snapshot_company):
    case = _case(expected_statuses={"open_remediation_ticket": "created"})
    results = [{"name": "open_remediation_ticket", "status": "created"}]
    result = evals.grade(case, "", [], snapshot_company, tool_results=results)
    assert result["checks"]["expected_statuses"]["passed"]


def test_expected_statuses_fails_for_another_status(evals, snapshot_company):
    case = _case(expected_statuses={"open_remediation_ticket": "created"})
    results = [{"name": "open_remediation_ticket", "status": "rejected"}]
    result = evals.grade(case, "", [], snapshot_company, tool_results=results)
    assert not result["checks"]["expected_statuses"]["passed"]
    assert "open_remediation_ticket=created" in result["checks"]["expected_statuses"]["detail"]


def test_forbidden_statuses_fails_when_the_status_occurs(evals, snapshot_company):
    case = _case(allows_actions=True, forbidden_statuses={"create_upgrade_order": ["created"]})
    results = [{"name": "create_upgrade_order", "status": "created"}]
    result = evals.grade(case, "", ["create_upgrade_order"], snapshot_company, tool_results=results)
    assert not result["checks"]["forbidden_statuses"]["passed"]


def test_forbidden_statuses_passes_when_only_refused(evals, snapshot_company):
    case = _case(allows_actions=True, forbidden_statuses={"create_upgrade_order": ["created"]})
    results = [{"name": "create_upgrade_order", "status": "refused"}]
    result = evals.grade(case, "", ["create_upgrade_order"], snapshot_company, tool_results=results)
    assert result["checks"]["forbidden_statuses"]["passed"]


# grade(): no_unrequested_action


def test_action_call_in_a_question_only_case_fails_the_guardrail(evals, snapshot_company):
    case = _case()
    result = evals.grade(case, "Listo.", ["notify_employee"], snapshot_company, tool_calls=[])
    assert not result["checks"]["no_unrequested_action"]["passed"]
    assert "notify_employee" in result["checks"]["no_unrequested_action"]["detail"]
    assert not result["passed"]


def test_no_action_call_passes_the_guardrail(evals, snapshot_company):
    result = evals.grade(_case(), "Nada que hacer.", ["fleet_insights"], snapshot_company)
    assert result["checks"]["no_unrequested_action"]["passed"]


def test_action_call_is_allowed_when_the_case_says_so(evals, snapshot_company):
    case = _case(allows_actions=True)
    result = evals.grade(case, "", ["flag_device_for_replacement"], snapshot_company)
    assert result["checks"]["no_unrequested_action"]["passed"]


def test_action_call_is_allowed_when_the_case_expects_actions(evals, snapshot_company):
    case = _case(expected_actions=[{"tool": "notify_employee", "employee_id": "emp-acme-1005"}])
    calls = [{"name": "notify_employee", "args": {"employee_id": "emp-acme-1005"}}]
    result = evals.grade(case, "", ["notify_employee"], snapshot_company, tool_calls=calls)
    assert result["checks"]["no_unrequested_action"]["passed"]


# grade(): require_citations


def test_require_citations_passes_with_a_valid_citation_and_no_devices(evals, snapshot_company):
    case = _case(require_citations=True)
    answer = f"Ninguno. [{_latest_snapshot('MT7PJB7N5LRE')}]"
    result = evals.grade(case, answer, [], snapshot_company)
    assert result["checks"]["require_citations"]["passed"]


def test_require_citations_fails_without_a_citation(evals, snapshot_company):
    result = evals.grade(_case(require_citations=True), "Ninguno.", [], snapshot_company)
    assert not result["checks"]["require_citations"]["passed"]


def test_require_citations_fails_when_the_only_citation_is_another_company(evals, snapshot_company):
    foreign = _latest_snapshot("I3JLKILKQTAC")  # initech-003
    result = evals.grade(_case(require_citations=True), f"Ninguno. [{foreign}]", [], snapshot_company)
    assert not result["checks"]["require_citations"]["passed"]


# run_case(): approvals decide what happens to a proposal


def test_run_case_approve_creates_the_record_in_the_temp_outbox(evals, isolated):
    case = _case(
        id="action-approve",
        expected_devices=["MT7PJB7N5LRE"],
        approval="approve",
        expected_actions=[{"tool": "open_remediation_ticket", "device_id": "MT7PJB7N5LRE"}],
        expected_statuses={"open_remediation_ticket": "created"},
    )
    model = ScriptedModel(
        script=[
            PLAN,
            _call("open_remediation_ticket", _remediation_args()),
            AIMessage(content=f"Ticket abierto para MT7PJB7N5LRE [{_latest_snapshot('MT7PJB7N5LRE')}]."),
        ],
        seen=[],
    )
    outcome = evals.run_case(case, model=model)

    assert outcome["run_error"] is None
    assert outcome["approvals"] == 1
    assert outcome["tool_results"] == [{"name": "open_remediation_ticket", "status": "created"}]
    assert outcome["tool_calls"][0]["name"] == "open_remediation_ticket"
    assert outcome["tool_calls"][0]["args"]["device_id"] == "MT7PJB7N5LRE"

    record_file = isolated / "outbox" / "open_remediation_ticket.jsonl"
    assert record_file.exists()
    record = json.loads(record_file.read_text().splitlines()[0])
    assert record["device_id"] == "MT7PJB7N5LRE"
    assert record["check_id"] == "screen_lock"

    grade = evals.grade(
        case,
        outcome["answer"],
        outcome["tool_names"],
        evals.dataset_maps()[1],
        tool_calls=outcome["tool_calls"],
        tool_results=outcome["tool_results"],
    )
    assert grade["passed"], grade["checks"]


def test_run_case_reject_writes_nothing_and_reports_rejected(evals, isolated):
    case = _case(id="action-reject", approval="reject")
    model = ScriptedModel(
        script=[PLAN, _call("open_remediation_ticket", _remediation_args()), AIMessage(content="No se abrió el ticket.")],
        seen=[],
    )
    outcome = evals.run_case(case, model=model)

    assert outcome["approvals"] == 1
    assert outcome["tool_results"] == [{"name": "open_remediation_ticket", "status": "rejected"}]
    assert not (isolated / "outbox" / "open_remediation_ticket.jsonl").exists()


def test_run_case_defaults_to_reject_when_no_approval_is_given(evals, isolated):
    model = ScriptedModel(
        script=[PLAN, _call("open_remediation_ticket", _remediation_args()), AIMessage(content="No.")],
        seen=[],
    )
    outcome = evals.run_case(_case(), model=model)
    assert outcome["tool_results"] == [{"name": "open_remediation_ticket", "status": "rejected"}]


def test_run_case_answers_several_pending_approvals_together(evals, isolated):
    two_calls = AIMessage(
        content="",
        tool_calls=[
            {"name": "open_remediation_ticket", "args": _remediation_args(), "id": "call_a"},
            {"name": "open_remediation_ticket", "args": _remediation_args(), "id": "call_b"},
        ],
    )
    model = ScriptedModel(script=[PLAN, two_calls, AIMessage(content="Dos tickets.")], seen=[])
    outcome = evals.run_case(_case(approval="approve"), model=model)

    assert outcome["approvals"] == 2
    assert outcome["run_error"] is None
    assert sorted(r["status"] for r in outcome["tool_results"]) == ["created", "created"]


def test_two_approved_proposals_in_one_step_write_two_records(evals, isolated):
    two_calls = AIMessage(
        content="",
        tool_calls=[
            {"name": "open_remediation_ticket", "args": _remediation_args(), "id": "call_a"},
            {"name": "open_remediation_ticket", "args": _remediation_args(), "id": "call_b"},
        ],
    )
    model = ScriptedModel(script=[PLAN, two_calls, AIMessage(content="Dos tickets.")], seen=[])
    evals.run_case(_case(approval="approve"), model=model)
    assert len((isolated / "outbox" / "open_remediation_ticket.jsonl").read_text().splitlines()) == 2


def test_run_case_stops_after_five_resumes_and_says_so(evals, isolated):
    calls = [_call("open_remediation_ticket", _remediation_args(), call_id=f"call_{i}") for i in range(6)]
    model = ScriptedModel(script=[PLAN, *calls, AIMessage(content="Listo.")], seen=[])
    outcome = evals.run_case(_case(approval="approve"), model=model)

    assert outcome["approvals"] == 5
    assert outcome["run_error"] is not None
    assert "5 resumes" in outcome["run_error"]


def test_action_in_a_question_only_case_is_graded_as_a_failure(evals, isolated, snapshot_company):
    case = _case(id="qa-only", expected_devices=[])
    model = ScriptedModel(
        script=[PLAN, _call("open_remediation_ticket", _remediation_args()), AIMessage(content="Hecho.")],
        seen=[],
    )
    outcome = evals.run_case(case, model=model)
    graded = evals.grade(
        case,
        outcome["answer"],
        outcome["tool_names"],
        snapshot_company,
        tool_calls=outcome["tool_calls"],
        tool_results=outcome["tool_results"],
    )
    assert not graded["checks"]["no_unrequested_action"]["passed"]
    assert not graded["passed"]


def test_tool_status_reads_status_error_and_ok(evals):
    assert evals._tool_status('{"status": "created", "action_id": "x"}') == "created"
    assert evals._tool_status('{"error": "Not available."}') == "error"
    assert evals._tool_status('{"rows": []}') == "ok"
    assert evals._tool_status("not json") == "unparsed"


# main(): run folders and repeats


def _main_model(answer_text, device="MT7PJB7N5LRE"):
    """A scripted model that proposes the remediation ticket and then answers with a citation."""
    snap = _latest_snapshot(device)
    return ScriptedModel(
        script=[
            PLAN,
            _call("open_remediation_ticket", _remediation_args()),
            AIMessage(content=f"{answer_text} [{snap}]"),
        ],
        seen=[],
    )


@pytest.fixture
def approved_case_only(evals, monkeypatch):
    cases = [c for c in evals.load_cases() if c["id"] == "acme-action-remediation-approved"]
    monkeypatch.setattr(evals, "load_cases", lambda *a, **k: cases)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return cases


def test_main_writes_actions_to_a_run_folder_and_restores_the_environment(evals, isolated, monkeypatch, approved_case_only, capsys):
    monkeypatch.setenv("FLEET_OUTBOX_DIR", "/previous/outbox")
    monkeypatch.delenv("FLEET_AUDIT_PATH", raising=False)
    model = _main_model("Ticket abierto para MT7PJB7N5LRE.")

    assert evals.main([], model=model) == 0

    run_dirs = list((isolated / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "open_remediation_ticket.jsonl").exists()
    assert (run_dir / "audit.jsonl").exists()
    assert str(run_dir) in capsys.readouterr().out

    report = json.loads((isolated / "report.json").read_text())
    assert report["repeat"] == 1
    assert report["run_dir"] == str(run_dir)
    assert report["summary"]["runs_passed"] == 1

    assert os.environ["FLEET_OUTBOX_DIR"] == "/previous/outbox"
    assert "FLEET_AUDIT_PATH" not in os.environ


def test_main_repeat_runs_each_case_n_times_and_keeps_every_run(evals, isolated, monkeypatch, approved_case_only):
    model = ScriptedModel(
        script=[
            PLAN,
            _call("open_remediation_ticket", _remediation_args(), call_id="c1"),
            AIMessage(content=f"Ticket abierto para MT7PJB7N5LRE [{_latest_snapshot('MT7PJB7N5LRE')}]."),
            PLAN,
            AIMessage(content="No lo sé."),  # second run: no action, no citation: fails
            PLAN,
            _call("open_remediation_ticket", _remediation_args(), call_id="c3"),
            AIMessage(content=f"Ticket abierto para MT7PJB7N5LRE [{_latest_snapshot('MT7PJB7N5LRE')}]."),
        ],
        seen=[],
    )
    assert evals.main(["--repeat", "3"], model=model) == 1

    report = json.loads((isolated / "report.json").read_text())
    assert report["repeat"] == 3
    runs = report["cases"][0]["runs"]
    assert [r["passed"] for r in runs] == [True, False, True]
    assert report["summary"]["cases"]["acme-action-remediation-approved"]["passed"] == 2
    assert report["summary"]["cases"]["acme-action-remediation-approved"]["total"] == 3
    assert "expected_actions" in report["summary"]["cases"]["acme-action-remediation-approved"]["failed_checks"]


def test_main_rejects_repeat_below_one(evals, isolated, monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with pytest.raises(SystemExit) as exc:
        evals.main(["--repeat", "0"], model=ScriptedModel(script=[], seen=[]))
    assert exc.value.code == 2


# aggregate() and exit_code()


def _run(passed, **failed):
    checks = {name: {"passed": True, "detail": ""} for name in ["expected_devices"]}
    for name in failed:
        checks[name] = {"passed": False, "detail": ""}
    return {"passed": passed, "checks": checks}


def test_aggregate_computes_per_case_rates_and_failed_check_counts(evals):
    records = [
        {"id": "a", "runs": [_run(True), _run(True), _run(True)]},
        {"id": "b", "runs": [_run(False, expected_actions=1), _run(True), _run(False, expected_actions=1, expected_statuses=1)]},
        {"id": "c", "runs": [_run(False, forbidden_statuses=1)]},
    ]
    summary = evals.aggregate(records)

    assert summary["cases"]["a"] == {"passed": 3, "total": 3, "failed_checks": {}}
    assert summary["cases"]["b"]["passed"] == 1
    assert summary["cases"]["b"]["total"] == 3
    assert summary["cases"]["b"]["failed_checks"] == {"expected_actions": 2, "expected_statuses": 1}
    assert summary["cases"]["c"]["failed_checks"] == {"forbidden_statuses": 1}
    assert summary["runs_passed"] == 4  # 3 from a, 1 from b, 0 from c
    assert summary["runs_total"] == 7
    assert summary["cases_perfect"] == 1
    assert summary["cases_total"] == 3


def test_exit_code_is_zero_only_when_every_run_passed(evals):
    all_pass = evals.aggregate([{"id": "a", "runs": [_run(True), _run(True)]}])
    one_fail = evals.aggregate([{"id": "a", "runs": [_run(True), _run(False, expected_devices=1)]}])
    assert evals.exit_code(all_pass) == 0
    assert evals.exit_code(one_fail) == 1


def test_print_table_shows_rate_and_failed_check_counts(evals, capsys):
    summary = evals.aggregate(
        [{"id": "acme-x", "runs": [_run(True), _run(False, expected_actions=1), _run(False, expected_actions=1)]}]
    )
    evals.print_table(summary)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split()[:2] == ["CASE", "PASS"]
    assert lines[1].split()[:2] == ["acme-x", "1/3"]
    assert "expected_actions x2" in lines[1]


# validate_cases(): the new fields


def _valid_action_case(**overrides):
    case = _case(
        id="action-valid",
        expected_devices=["MT7PJB7N5LRE"],
        expected_actions=[{"tool": "open_remediation_ticket", "device_id": "MT7PJB7N5LRE"}],
        expected_statuses={"open_remediation_ticket": "created"},
        approval="approve",
    )
    case.update(overrides)
    return case


def test_validate_accepts_a_well_formed_action_case(evals):
    device_company, _ = evals.dataset_maps()
    assert evals.validate_cases([_valid_action_case()], device_company, TOOL_NAMES) == []


def test_validate_rejects_an_unknown_action_tool(evals):
    device_company, _ = evals.dataset_maps()
    problems = evals.validate_cases(
        [_valid_action_case(expected_actions=[{"tool": "create_ticket", "device_id": "MT7PJB7N5LRE"}])],
        device_company,
        TOOL_NAMES,
    )
    assert any("create_ticket" in p and "action tool" in p for p in problems), problems


def test_validate_rejects_a_read_tool_as_an_action(evals):
    device_company, _ = evals.dataset_maps()
    problems = evals.validate_cases(
        [_valid_action_case(expected_actions=[{"tool": "fleet_insights"}])],
        device_company,
        TOOL_NAMES,
    )
    assert any("fleet_insights" in p and "action tool" in p for p in problems), problems


def test_validate_rejects_an_unknown_tool_in_expected_statuses(evals):
    device_company, _ = evals.dataset_maps()
    problems = evals.validate_cases(
        [_valid_action_case(expected_statuses={"get_device": "created"})],
        device_company,
        TOOL_NAMES,
    )
    assert any("get_device" in p and "action tool" in p for p in problems), problems


def test_validate_rejects_a_bad_approval_value(evals):
    device_company, _ = evals.dataset_maps()
    problems = evals.validate_cases([_valid_action_case(approval="maybe")], device_company, TOOL_NAMES)
    assert any("approval" in p and "maybe" in p for p in problems), problems


def test_validate_rejects_a_bad_status(evals):
    device_company, _ = evals.dataset_maps()
    problems = evals.validate_cases(
        [_valid_action_case(expected_statuses={"open_remediation_ticket": "done"})],
        device_company,
        TOOL_NAMES,
    )
    assert any("'done'" in p and "created" in p for p in problems), problems


def test_validate_rejects_a_bad_status_in_forbidden_statuses(evals):
    device_company, _ = evals.dataset_maps()
    problems = evals.validate_cases(
        [_valid_action_case(forbidden_statuses={"create_upgrade_order": ["approved"]})],
        device_company,
        TOOL_NAMES,
    )
    assert any("'approved'" in p for p in problems), problems


def test_validate_rejects_an_unknown_matcher_argument(evals):
    device_company, _ = evals.dataset_maps()
    problems = evals.validate_cases(
        [_valid_action_case(expected_actions=[{"tool": "open_remediation_ticket", "device": "MT7PJB7N5LRE"}])],
        device_company,
        TOOL_NAMES,
    )
    assert any("'device'" in p for p in problems), problems


@pytest.mark.parametrize("approval, outcome", [("approve", "approved"), ("reject", "rejected")])
def test_two_proposals_in_one_step_are_each_audited_once(evals, isolated, approval, outcome):
    two_calls = AIMessage(
        content="",
        tool_calls=[
            {"name": "open_remediation_ticket", "args": _remediation_args(), "id": "call_a"},
            {"name": "open_remediation_ticket", "args": _remediation_args(), "id": "call_b"},
        ],
    )
    model = ScriptedModel(script=[PLAN, two_calls, AIMessage(content="Listo.")], seen=[])
    evals.run_case(_case(approval=approval), model=model)
    events = [json.loads(line) for line in (isolated / "audit.jsonl").read_text().splitlines()]
    final = [e["tool_call_id"] for e in events if e.get("outcome") == outcome]
    assert sorted(final) == ["call_a", "call_b"]

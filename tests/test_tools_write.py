import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from fleet_copilot import insights, tools_write
from fleet_copilot.agent import TOOLS, build_graph, thread_config
from fleet_copilot.data import latest_per_device, load_snapshots
from fleet_copilot.tools_read import NOT_AVAILABLE
from fleet_copilot.tools_write import ACTION_TOOLS

ACME = "acme-001"
INITECH_DEVICE = "I3JLKILKQTAC"  # belongs to initech-003
PLAN = AIMessage(content="1. Call fleet_insights.")
NOT_FOUND_EMPLOYEE = "Employee not found for this company."
UNKNOWN_SNAPSHOT = "NOPE00000000@2026-01-01T00:00:00Z"


class ScriptedModel(BaseChatModel):
    """Returns the given AI messages in order and records what it was sent."""

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


# Ground truth, read from the dataset and the insight detectors. Nothing is hardcoded
# except the device ids the spec names.


def _history(device_id):
    snapshots = load_snapshots()
    return snapshots[snapshots["device_id"] == device_id].sort_values("collected_at")


def _latest_id(device_id):
    return _history(device_id)["snapshot_id"].iloc[-1]


def _employee_of(device_id):
    return _history(device_id)["employee_id"].iloc[-1]


def _company_devices(company):
    snapshots = load_snapshots()
    return latest_per_device(snapshots[snapshots["company_id"] == company])


def _device_ids(company):
    return list(_company_devices(company)["device_id"])


def _flagged(detector, company):
    return {row["device_id"] for row in detector(company)["rows"]}


def _hardware_flagged(company):
    return (
        _flagged(insights.detect_ram_pressure, company)
        | _flagged(insights.detect_low_storage, company)
        | _flagged(insights.detect_battery_end_of_life, company)
    )


def _failing_on_latest(company, check_id):
    return [
        row["device_id"]
        for _, row in _company_devices(company).iterrows()
        if any(c["check_id"] == check_id and c["status"] == "fail" for c in row["compliance"])
    ]


def _passing_check(company):
    """A (device_id, check_id) where the check passes on the device's latest snapshot."""
    for _, row in _company_devices(company).iterrows():
        for check in row["compliance"]:
            if check["status"] == "pass":
                return row["device_id"], check["check_id"]
    raise AssertionError("no passing check in the data")


def _other_device(company, device_id):
    return next(d for d in _device_ids(company) if d != device_id)


# Fixtures and runners


@pytest.fixture(autouse=True)
def paths(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit.jsonl"
    outbox = tmp_path / "outbox"
    monkeypatch.setenv("FLEET_AUDIT_PATH", str(audit_path))
    monkeypatch.setenv("FLEET_OUTBOX_DIR", str(outbox))
    return audit_path, outbox


def _lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _outbox_has_records(outbox):
    return outbox.exists() and any(outbox.iterdir())


def _start(script, company=ACME, question="propose an action"):
    """Run until the graph finishes or pauses for approval. Returns (graph, config, result)."""
    model = ScriptedModel(script=[PLAN, *script], seen=[])  # the first reply is the planner's
    graph = build_graph(model=model, checkpointer=InMemorySaver())
    config = thread_config(company, "test")
    result = graph.invoke({"messages": [HumanMessage(question)], "company_id": company}, config)
    return graph, config, result


def _pending(graph, config):
    return [i for task in graph.get_state(config).tasks for i in task.interrupts]


def _tool_content(result):
    return json.loads(next(m for m in result["messages"] if isinstance(m, ToolMessage)).content)


def _direct(paths, action, args, company=ACME):
    """Run one action that should not pause. Returns (tool content, audit events)."""
    audit_path, outbox = paths
    before = len(_lines(audit_path))
    graph, config, final = _start([_call(action, args), AIMessage(content="ok")], company=company)
    assert _pending(graph, config) == []
    assert not _outbox_has_records(outbox)
    return _tool_content(final), _lines(audit_path)[before:]  # only the events of this call


def _approve(paths, action, args, company=ACME):
    """Propose one action, approve it, and return (the one outbox record, the proposal shown to the human, tool content)."""
    _, outbox = paths
    graph, config, _ = _start([_call(action, args), AIMessage(content="Done.")], company=company)
    pending = _pending(graph, config)
    assert [i.value["action"] for i in pending] == [action]
    proposal = pending[0].value["proposal"]
    assert "action_id" not in proposal and "created_at" not in proposal  # created only after approval
    assert not _outbox_has_records(outbox)

    final = graph.invoke(Command(resume={"approved": True}), config)

    records = _lines(outbox / f"{action}.jsonl")
    assert len(records) == 1
    content = _tool_content(final)
    assert content == {"status": "created", "action_id": records[0]["action_id"]}
    return records[0], proposal, content


# Schema and wiring


def test_create_ticket_is_gone():
    assert not hasattr(tools_write, "create_ticket")
    assert "create_ticket" not in {tool.name for tool in TOOLS}


def test_agent_exposes_read_tools_and_the_four_action_tools():
    assert {tool.name for tool in ACTION_TOOLS} == {
        "open_remediation_ticket",
        "create_upgrade_order",
        "flag_device_for_replacement",
        "notify_employee",
    }
    assert {tool.name for tool in ACTION_TOOLS} <= {tool.name for tool in TOOLS}


def test_company_id_is_hidden_from_every_tool_schema():
    for tool in TOOLS:
        schema = tool.tool_call_schema.model_json_schema()
        assert "company_id" not in schema.get("properties", {}), tool.name


# Approved actions: one line in the outbox and one approved audit event


def test_remediation_ticket_approved_writes_one_line(paths):
    audit_path, _ = paths
    device = _failing_on_latest(ACME, "screen_lock")[0]
    evidence = [_latest_id(device)]
    record, proposal, _ = _approve(
        paths,
        "open_remediation_ticket",
        {"device_id": device, "check_id": "screen_lock", "note": "Enable screen lock.", "evidence": evidence},
    )

    assert proposal["company_id"] == ACME and proposal["device_id"] == device
    assert proposal["evidence"] == evidence
    assert record["action"] == "open_remediation_ticket"
    assert record["company_id"] == ACME
    assert record["device_id"] == device
    assert record["employee_id"] == _employee_of(device)
    assert record["check_id"] == "screen_lock"
    assert record["evidence"] == evidence
    assert len(record["action_id"]) == 32 and record["created_at"]

    events = _lines(audit_path)
    assert [e["outcome"] for e in events] == ["approved"]
    assert events[0]["company_id"] == ACME
    assert events[0]["action_id"] == record["action_id"]


def test_upgrade_order_approved_writes_one_line(paths):
    audit_path, _ = paths
    device = _flagged(insights.detect_ram_pressure, ACME).pop()
    assert device == "M4XVHUV1MEPZ"  # ground truth for acme ram pressure
    evidence = [_latest_id(device)]
    record, _, _ = _approve(
        paths,
        "create_upgrade_order",
        {"device_id": device, "component": "ram", "spec": "32 GB", "evidence": evidence},
    )

    assert record["action"] == "create_upgrade_order"
    assert record["company_id"] == ACME
    assert record["device_id"] == device
    assert record["employee_id"] == _employee_of(device)
    assert record["component"] == "ram" and record["spec"] == "32 GB"
    events = _lines(audit_path)
    assert [e["outcome"] for e in events] == ["approved"]
    assert events[0]["action_id"] == record["action_id"]


def test_replacement_flag_approved_writes_one_line(paths):
    audit_path, _ = paths
    device = sorted(_hardware_flagged(ACME))[0]
    record, _, _ = _approve(
        paths,
        "flag_device_for_replacement",
        {"device_id": device, "reason": "Memory saturated.", "evidence": [_latest_id(device)]},
    )

    assert record["action"] == "flag_device_for_replacement"
    assert record["company_id"] == ACME
    assert record["device_id"] == device
    assert record["employee_id"] == _employee_of(device)
    assert [e["outcome"] for e in _lines(audit_path)] == ["approved"]


def test_notify_employee_approved_writes_one_line(paths):
    audit_path, _ = paths
    device = "8NM23J95R5I6"
    employee = _employee_of(device)
    assert employee == "emp-acme-1005"  # ground truth for this device's owner
    record, _, _ = _approve(
        paths,
        "notify_employee",
        {"employee_id": employee, "message": "Free up space on your disk.", "evidence": [_latest_id(device)]},
    )

    assert record["action"] == "notify_employee"
    assert record["company_id"] == ACME
    assert record["employee_id"] == employee
    assert "device_id" not in record
    assert record["message"] == "Free up space on your disk."
    assert [e["outcome"] for e in _lines(audit_path)] == ["approved"]


def test_rejected_action_writes_nothing_and_is_audited(paths):
    audit_path, outbox = paths
    device = _failing_on_latest(ACME, "screen_lock")[0]
    args = {
        "device_id": device,
        "check_id": "screen_lock",
        "note": "Screen lock.",
        "evidence": [_latest_id(device)],
    }
    graph, config, _ = _start([_call("open_remediation_ticket", args), AIMessage(content="Not opened.")])
    assert _pending(graph, config)

    final = graph.invoke(Command(resume={"approved": False, "reason": "wait until Monday"}), config)

    assert not _outbox_has_records(outbox)
    assert _tool_content(final) == {"status": "rejected", "reason": "wait until Monday"}
    events = _lines(audit_path)
    assert [e["outcome"] for e in events] == ["rejected"]
    assert events[0]["reason"] == "wait until Monday"


# Refusals: no interrupt, nothing written, specific audit outcome


def test_empty_evidence_is_refused_without_interrupt(paths):
    device = _failing_on_latest(ACME, "screen_lock")[0]
    content, events = _direct(
        paths,
        "open_remediation_ticket",
        {"device_id": device, "check_id": "screen_lock", "note": "x", "evidence": []},
    )
    assert content["status"] == "refused"
    assert "evidence" in content["reason"].lower()
    assert [e["outcome"] for e in events] == ["refused_insufficient_evidence"]
    assert events[0]["reason"] == content["reason"]


def test_evidence_from_another_device_of_the_same_company_is_refused(paths):
    device = _failing_on_latest(ACME, "screen_lock")[0]
    other = _other_device(ACME, device)
    other_id = _latest_id(other)
    content, events = _direct(
        paths,
        "open_remediation_ticket",
        {"device_id": device, "check_id": "screen_lock", "note": "x", "evidence": [other_id]},
    )
    assert content["status"] == "refused"
    assert other_id in content["reason"]
    assert [e["outcome"] for e in events] == ["refused_insufficient_evidence"]


def test_evidence_from_another_company_is_refused_without_revealing_it(paths):
    device = _failing_on_latest(ACME, "screen_lock")[0]
    foreign_id = _latest_id(INITECH_DEVICE)
    content, events = _direct(
        paths,
        "open_remediation_ticket",
        {"device_id": device, "check_id": "screen_lock", "note": "x", "evidence": [foreign_id]},
    )
    assert content == {"status": "refused", "reason": f"Snapshot {foreign_id} is not available for this company."}
    text = json.dumps(content).lower()
    assert "initech" not in text and "globex" not in text
    assert [e["outcome"] for e in events] == ["refused_insufficient_evidence"]


def test_foreign_and_unknown_snapshots_get_the_same_refusal(paths):
    device = _failing_on_latest(ACME, "screen_lock")[0]
    foreign_id = _latest_id(INITECH_DEVICE)

    def refusal(snap_id):
        content, _ = _direct(
            paths,
            "open_remediation_ticket",
            {"device_id": device, "check_id": "screen_lock", "note": "x", "evidence": [snap_id]},
        )
        return content["reason"]

    foreign_reason = refusal(foreign_id)
    unknown_reason = refusal(UNKNOWN_SNAPSHOT)
    assert foreign_reason.replace(foreign_id, "SNAP") == unknown_reason.replace(UNKNOWN_SNAPSHOT, "SNAP")
    assert unknown_reason == f"Snapshot {UNKNOWN_SNAPSHOT} is not available for this company."


def test_remediation_for_a_check_that_passes_is_refused(paths):
    device, check_id = _passing_check(ACME)
    content, events = _direct(
        paths,
        "open_remediation_ticket",
        {"device_id": device, "check_id": check_id, "note": "x", "evidence": [_latest_id(device)]},
    )
    assert content == {
        "status": "refused",
        "reason": f"{check_id} is passing on the latest snapshot of {device}.",
    }
    assert [e["outcome"] for e in events] == ["refused_insufficient_evidence"]
    assert events[0]["reason"] == content["reason"]


def test_ram_upgrade_for_a_device_without_ram_pressure_is_refused(paths):
    device = next(d for d in _device_ids(ACME) if d not in _flagged(insights.detect_ram_pressure, ACME))
    content, events = _direct(
        paths,
        "create_upgrade_order",
        {"device_id": device, "component": "ram", "spec": "32 GB", "evidence": [_latest_id(device)]},
    )
    assert content == {"status": "refused", "reason": f"{device} has no ram_pressure finding."}
    assert [e["outcome"] for e in events] == ["refused_insufficient_evidence"]


def test_disk_upgrade_for_a_device_without_low_storage_is_refused(paths):
    device = next(d for d in _device_ids(ACME) if d not in _flagged(insights.detect_low_storage, ACME))
    content, _ = _direct(
        paths,
        "create_upgrade_order",
        {"device_id": device, "component": "disk", "spec": "1 TB", "evidence": [_latest_id(device)]},
    )
    assert content == {"status": "refused", "reason": f"{device} has no low_storage finding."}


def test_replacement_for_a_device_without_hardware_finding_is_refused(paths):
    device = next(d for d in _device_ids(ACME) if d not in _hardware_flagged(ACME))
    content, events = _direct(
        paths,
        "flag_device_for_replacement",
        {"device_id": device, "reason": "x", "evidence": [_latest_id(device)]},
    )
    assert content == {
        "status": "refused",
        "reason": f"{device} has no hardware finding (low_storage, ram_pressure or battery_end_of_life).",
    }
    assert [e["outcome"] for e in events] == ["refused_insufficient_evidence"]


def test_replacement_is_not_allowed_on_compliance_drift_alone(paths):
    # Acme has screen_lock drift on devices with no hardware finding; that must not be enough.
    device = next(d for d in _failing_on_latest(ACME, "screen_lock") if d not in _hardware_flagged(ACME))
    content, _ = _direct(
        paths,
        "flag_device_for_replacement",
        {"device_id": device, "reason": "x", "evidence": [_latest_id(device)]},
    )
    assert content["status"] == "refused"


def test_notify_employee_of_another_company_looks_like_an_unknown_employee(paths):
    foreign_employee = _employee_of(INITECH_DEVICE)
    evidence = [_latest_id(_device_ids(ACME)[0])]

    foreign, foreign_events = _direct(
        paths,
        "notify_employee",
        {"employee_id": foreign_employee, "message": "hola", "evidence": evidence},
    )
    unknown, unknown_events = _direct(
        paths,
        "notify_employee",
        {"employee_id": "emp-does-not-exist", "message": "hola", "evidence": evidence},
    )

    assert foreign == unknown == {"error": NOT_FOUND_EMPLOYEE}
    assert "initech" not in json.dumps(foreign).lower()
    assert [e["outcome"] for e in foreign_events] == ["not_found"]
    assert [e["outcome"] for e in unknown_events] == ["not_found"]


def test_notify_employee_evidence_must_be_one_of_that_employees_devices(paths):
    employee = _employee_of("8NM23J95R5I6")
    other_device = next(d for d in _device_ids(ACME) if _employee_of(d) != employee)
    other_id = _latest_id(other_device)
    content, events = _direct(
        paths,
        "notify_employee",
        {"employee_id": employee, "message": "hola", "evidence": [other_id]},
    )
    assert content["status"] == "refused"
    assert other_id in content["reason"]
    assert [e["outcome"] for e in events] == ["refused_insufficient_evidence"]


# Cross-tenant device ids


@pytest.mark.parametrize(
    "action, args",
    [
        ("open_remediation_ticket", {"check_id": "screen_lock", "note": "x"}),
        ("create_upgrade_order", {"component": "ram", "spec": "32 GB"}),
        ("flag_device_for_replacement", {"reason": "x"}),
    ],
)
def test_cross_tenant_device_is_refused_without_interrupt(paths, action, args):
    evidence = [_latest_id(INITECH_DEVICE)]
    content, events = _direct(
        paths,
        action,
        {"device_id": INITECH_DEVICE, "evidence": evidence, **args},
    )
    assert content == {"error": NOT_AVAILABLE}
    assert [e["outcome"] for e in events] == ["cross_tenant_denied"]
    assert events[0]["args"]["device_id"] == INITECH_DEVICE


def test_unknown_device_is_refused_without_interrupt(paths):
    content, events = _direct(
        paths,
        "flag_device_for_replacement",
        {"device_id": "NOPE", "reason": "x", "evidence": [UNKNOWN_SNAPSHOT]},
    )
    assert content == {"error": NOT_AVAILABLE}
    assert [e["outcome"] for e in events] == ["not_found"]

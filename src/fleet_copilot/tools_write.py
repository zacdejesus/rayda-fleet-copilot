"""Action tools: each one proposes a change that a human must approve before anything is written.

Every action runs the same flow, in code, before a human is asked:
  a. target check: the device or employee must belong to this company;
  b. evidence check: the snapshot_ids must exist, belong to this company and to the target;
  c. precondition check: the data must support the action (computed here, never trusted from the model);
  d. approval: the graph pauses with interrupt() and waits for the human;
  e. on approval, one JSON line is appended to outbox/<action>.jsonl.

A refused proposal never reaches the human. LangGraph re-runs the tool from the top
on resume, so steps a-c only read data, and ids and timestamps are created after approval.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal

from langchain_core.tools import tool
from langgraph.types import interrupt

from fleet_copilot import audit, insights, tools_read
from fleet_copilot.data import load_snapshots
from fleet_copilot.tools import CompanyId
from fleet_copilot.tools_read import NOT_AVAILABLE, AccessDenied, NotFound

DEFAULT_OUTBOX_DIR = Path(__file__).resolve().parents[2] / "outbox"
NOT_FOUND_EMPLOYEE = "Employee not found for this company."

# Precondition: (company_id, device history oldest first) -> refusal reason, or None when the action is supported.
Precondition = Callable[[str, list[dict[str, Any]]], str | None]

HARDWARE_INSIGHTS: dict[str, Callable[[str], dict[str, Any]]] = {
    "ram_pressure": insights.detect_ram_pressure,
    "low_storage": insights.detect_low_storage,
    "battery_end_of_life": insights.detect_battery_end_of_life,
}
UPGRADE_INSIGHT = {"ram": "ram_pressure", "disk": "low_storage", "battery": "battery_end_of_life"}


def outbox_dir() -> Path:
    """Where approved actions go. FLEET_OUTBOX_DIR overrides the default, which tests use."""
    return Path(os.environ.get("FLEET_OUTBOX_DIR", DEFAULT_OUTBOX_DIR))


def _append_record(action: str, record: dict[str, Any]) -> None:
    """Append one approved action as a JSON line to outbox/<action>.jsonl. Creates the directory if needed."""
    directory = outbox_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / f"{action}.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _refused_json(reason: str) -> str:
    return json.dumps({"status": "refused", "reason": reason}, ensure_ascii=False)


def _hardware_findings(company_id: str, device_id: str) -> set[str]:
    """Names of the hardware detectors whose rows include this device."""
    return {
        name
        for name, detector in HARDWARE_INSIGHTS.items()
        if any(row["device_id"] == device_id for row in detector(company_id)["rows"])
    }


def _check_failing(check_id: str, company_id: str, history: list[dict[str, Any]]) -> str | None:
    """open_remediation_ticket: the check must fail on the device's latest snapshot."""
    device_id = history[-1]["device_id"]
    statuses = {check["check_id"]: check["status"] for check in history[-1]["compliance"]}
    status = statuses.get(check_id)
    if status == "fail":
        return None
    if status is None:
        return f"{device_id} has no {check_id} result on its latest snapshot."
    if status == "pass":
        return f"{check_id} is passing on the latest snapshot of {device_id}."
    return f"{check_id} is {status}, not failing, on the latest snapshot of {device_id}."


def _upgrade_supported(component: str, company_id: str, history: list[dict[str, Any]]) -> str | None:
    """create_upgrade_order: the device must appear in the matching hardware detector."""
    device_id = history[-1]["device_id"]
    insight = UPGRADE_INSIGHT[component]
    if insight in _hardware_findings(company_id, device_id):
        return None
    return f"{device_id} has no {insight} finding."


def _replacement_supported(company_id: str, history: list[dict[str, Any]]) -> str | None:
    """flag_device_for_replacement: at least one hardware finding. compliance_drift does not count."""
    device_id = history[-1]["device_id"]
    if _hardware_findings(company_id, device_id):
        return None
    return f"{device_id} has no hardware finding (low_storage, ram_pressure or battery_end_of_life)."


def _evidence_problem(evidence: list[str], company_snapshot_ids: set[str], allowed: set[str], target: str) -> str | None:
    """Return the refusal reason for bad evidence, or None when every id is valid."""
    if not evidence:
        return "No evidence given. Cite the snapshot_ids from tool results that support this action."
    for snap_id in evidence:
        # Unknown and foreign ids get the same message on purpose, so other tenants are not revealed.
        if snap_id not in company_snapshot_ids:
            return f"Snapshot {snap_id} is not available for this company."
        if snap_id not in allowed:
            return f"Snapshot {snap_id} is not a snapshot of {target}."
    return None


def _act(
    action: str,
    company_id: str,
    *,
    fields: dict[str, Any],
    evidence: list[str],
    device_id: str | None = None,
    employee_id: str | None = None,
    precondition: Precondition | None = None,
) -> str:
    """The shared flow for every action. Pass exactly one of device_id or employee_id."""
    subject = {"device_id": device_id} if device_id is not None else {"employee_id": employee_id}
    event: dict[str, Any] = {
        "company_id": company_id,
        "tool": action,
        "args": {**subject, **fields, "evidence": list(evidence)},
    }
    snapshots = load_snapshots()
    company_snapshot_ids = set(snapshots[snapshots["company_id"] == company_id]["snapshot_id"])

    # a. Target check. Device actions use the device's history; employee actions use the company's devices.
    if device_id is not None:
        try:
            history = tools_read._device_history(company_id, device_id)
        except AccessDenied:
            audit.record({**event, "outcome": "cross_tenant_denied"})
            return json.dumps({"error": NOT_AVAILABLE})
        except NotFound:
            audit.record({**event, "outcome": "not_found"})
            return json.dumps({"error": NOT_AVAILABLE})
        owner = history[-1]["employee_id"]
        allowed = {snap["snapshot_id"] for snap in history}
        target = f"device {device_id}"
        identity = {"device_id": device_id, "employee_id": owner}
    else:
        owned = snapshots[(snapshots["company_id"] == company_id) & (snapshots["employee_id"] == employee_id)]
        if owned.empty:
            # Same answer for an unknown employee and one from another company.
            audit.record({**event, "outcome": "not_found"})
            return json.dumps({"error": NOT_FOUND_EMPLOYEE})
        history = []
        allowed = set(owned["snapshot_id"])
        target = f"a device of employee {employee_id}"
        identity = {"employee_id": employee_id}

    # b. Evidence check.
    problem = _evidence_problem(list(evidence), company_snapshot_ids, allowed, target)
    if problem is None and precondition is not None:
        # c. Precondition check, computed from the data.
        problem = precondition(company_id, history)
    if problem is not None:
        audit.record({**event, "outcome": "refused_insufficient_evidence", "reason": problem})
        return _refused_json(problem)

    # d. Approval. Nothing before this point creates ids or timestamps, and nothing is written.
    proposal = {"company_id": company_id, **identity, **fields, "evidence": list(evidence)}
    decision = interrupt({"action": action, "proposal": proposal})

    # e. Approved or rejected.
    if not decision.get("approved"):
        reason = decision.get("reason") or ""
        audit.record({**event, "outcome": "rejected", "reason": reason})
        return json.dumps({"status": "rejected", "reason": reason}, ensure_ascii=False)

    record = {
        "action_id": uuid.uuid4().hex,
        "action": action,
        **proposal,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _append_record(action, record)
    except OSError as error:
        audit.record({**event, "outcome": "write_failed", "detail": str(error)})
        return json.dumps({"status": "error", "error": "The action could not be saved."})

    audit.record({**event, "outcome": "approved", "action_id": record["action_id"]})
    return json.dumps({"status": "created", "action_id": record["action_id"]}, ensure_ascii=False)


@tool
def open_remediation_ticket(
    company_id: CompanyId,
    device_id: str,
    check_id: Literal["disk_encryption", "screen_lock", "os_up_to_date"],
    note: str,
    evidence: list[str],
) -> str:
    """Propose an IT ticket to fix one failing compliance check on one device. A human must approve it.

    Precondition: check_id must be failing on the device's latest snapshot. Evidence must be
    snapshot_ids of this device, taken from tool results that show the failure (for example
    compliance_failures or get_device). Status "created" means approved; "refused" means
    nothing was proposed, so explain the reason to the user.
    """
    return _act(
        "open_remediation_ticket",
        company_id,
        device_id=device_id,
        fields={"check_id": check_id, "note": note},
        evidence=evidence,
        precondition=partial(_check_failing, check_id),
    )


@tool
def create_upgrade_order(
    company_id: CompanyId,
    device_id: str,
    component: Literal["ram", "disk", "battery"],
    spec: str,
    evidence: list[str],
) -> str:
    """Propose a hardware upgrade order (ram, disk or battery) for one device. A human must approve it.

    Precondition: the device must have the matching finding: ram needs ram_pressure, disk needs
    low_storage, battery needs battery_end_of_life (use fleet_insights). Evidence must be snapshot_ids
    of this device from those results. Status "created" means approved; "refused" means nothing was proposed.
    """
    return _act(
        "create_upgrade_order",
        company_id,
        device_id=device_id,
        fields={"component": component, "spec": spec},
        evidence=evidence,
        precondition=partial(_upgrade_supported, component),
    )


@tool
def flag_device_for_replacement(
    company_id: CompanyId,
    device_id: str,
    reason: str,
    evidence: list[str],
) -> str:
    """Propose flagging one device for replacement. A human must approve it.

    Precondition: the device must have at least one hardware finding (low_storage, ram_pressure or
    battery_end_of_life). A compliance finding alone is not enough. Evidence must be snapshot_ids of
    this device from those results. Status "created" means approved; "refused" means nothing was proposed.
    """
    return _act(
        "flag_device_for_replacement",
        company_id,
        device_id=device_id,
        fields={"reason": reason},
        evidence=evidence,
        precondition=_replacement_supported,
    )


@tool
def notify_employee(
    company_id: CompanyId,
    employee_id: str,
    message: str,
    evidence: list[str],
) -> str:
    """Propose a message to an employee about their device. A human must approve it before it is sent.

    The employee must have at least one device in this company. Evidence must be snapshot_ids of one
    of that employee's devices, taken from tool results that support the message. Status "created"
    means approved; "refused" means nothing was proposed.
    """
    return _act(
        "notify_employee",
        company_id,
        employee_id=employee_id,
        fields={"message": message},
        evidence=evidence,
    )


ACTION_TOOLS = [open_remediation_ticket, create_upgrade_order, flag_device_for_replacement, notify_employee]

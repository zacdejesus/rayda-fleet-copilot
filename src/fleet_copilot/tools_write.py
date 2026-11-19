"""Write tools. Right now there is one: opening an IT ticket, which needs human approval.

The tool pauses the graph with interrupt(). Nothing is written until the human
resumes the graph with {"approved": True}. LangGraph re-runs the code before the
interrupt on resume, so ids and timestamps are created only after approval.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import tool
from langgraph.types import interrupt

from fleet_copilot import audit, tools_read
from fleet_copilot.tools import CompanyId
from fleet_copilot.tools_read import NOT_AVAILABLE, AccessDenied, NotFound

DEFAULT_OUTBOX_DIR = Path(__file__).resolve().parents[2] / "outbox"


def outbox_dir() -> Path:
    """Where approved tickets go. FLEET_OUTBOX_DIR overrides the default, which tests use."""
    return Path(os.environ.get("FLEET_OUTBOX_DIR", DEFAULT_OUTBOX_DIR))


def _append_ticket(ticket: dict[str, Any]) -> None:
    """Append one ticket as a JSON line. Creates the outbox directory if needed."""
    directory = outbox_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "tickets.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(ticket, ensure_ascii=False) + "\n")


@tool
def create_ticket(
    company_id: CompanyId,
    device_id: str,
    title: str,
    details: str,
    priority: Literal["low", "medium", "high"],
) -> str:
    """Open an IT ticket for one device. A human must approve it before it is created.

    Put the supporting snapshot_ids in details. The result has status "created"
    only if the human approved; "rejected" means nothing was created.
    """
    event: dict[str, Any] = {
        "company_id": company_id,
        "tool": "create_ticket",
        "args": {"device_id": device_id, "title": title, "priority": priority},
    }
    try:
        history = tools_read._device_history(company_id, device_id)
    except AccessDenied:
        audit.record({**event, "outcome": "cross_tenant_denied"})
        return json.dumps({"error": NOT_AVAILABLE})
    except NotFound:
        audit.record({**event, "outcome": "not_found"})
        return json.dumps({"error": NOT_AVAILABLE})

    preview = {
        "company_id": company_id,
        "device_id": device_id,
        "employee_id": history[-1]["employee_id"],
        "title": title,
        "details": details,
        "priority": priority,
    }
    decision = interrupt({"action": "create_ticket", "ticket": preview})

    if not decision.get("approved"):
        reason = decision.get("reason") or ""
        audit.record({**event, "outcome": "rejected", "reason": reason})
        return json.dumps({"status": "rejected", "reason": reason}, ensure_ascii=False)

    ticket = {
        "ticket_id": uuid.uuid4().hex,
        **preview,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _append_ticket(ticket)
    except OSError as error:
        audit.record({**event, "outcome": "write_failed", "detail": str(error)})
        return json.dumps({"status": "error", "error": "The ticket could not be saved."})

    audit.record({**event, "outcome": "approved", "ticket_id": ticket["ticket_id"]})
    return json.dumps({"status": "created", "ticket_id": ticket["ticket_id"]}, ensure_ascii=False)

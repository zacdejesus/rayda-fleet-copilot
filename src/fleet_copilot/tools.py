"""LangChain tool wrappers around the read tools and the insight detectors.

company_id is injected from the graph state with InjectedState, so it never
appears in the schema the model sees and the model cannot choose or change it.
Every call is written to the audit log, including refused ones.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Callable, Literal

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from fleet_copilot import audit, insights, tools_read
from fleet_copilot.tools_read import NOT_AVAILABLE, AccessDenied, NotFound

CompanyId = Annotated[str, InjectedState("company_id")]

INSIGHTS = {
    "low_storage": insights.detect_low_storage,
    "ram_pressure": insights.detect_ram_pressure,
    "battery_end_of_life": insights.detect_battery_end_of_life,
    "compliance_drift": insights.detect_compliance_drift,
}


def _run(name: str, fn: Callable[..., dict[str, Any]], company_id: str, **kwargs: Any) -> str:
    """Call one tool function, audit the outcome, and return JSON for the model."""
    event: dict[str, Any] = {"company_id": company_id, "tool": name, "args": kwargs}
    try:
        result = fn(company_id, **kwargs)
    except AccessDenied:
        audit.record({**event, "outcome": "cross_tenant_denied"})
        return json.dumps({"error": NOT_AVAILABLE})
    except NotFound:
        audit.record({**event, "outcome": "not_found"})
        return json.dumps({"error": NOT_AVAILABLE})
    except ValueError as error:
        audit.record({**event, "outcome": "invalid_args", "detail": str(error)})
        return json.dumps({"error": str(error)})

    audit.record({**event, "outcome": "ok", "rows": len(result["rows"]), "evidence": len(result["evidence"])})
    return json.dumps(result, ensure_ascii=False)


@tool
def list_devices(
    company_id: CompanyId,
    form_factor: Literal["laptop", "desktop"] | None = None,
    os_name: Literal["macOS", "Windows"] | None = None,
) -> str:
    """List the company's devices, one row per device from its latest snapshot.

    Use this to see the fleet, count devices, or find a device_id. Optional
    filters: form_factor (laptop or desktop) and os_name (macOS or Windows).
    """
    return _run("list_devices", tools_read.list_devices, company_id, form_factor=form_factor, os_name=os_name)


@tool
def get_device(company_id: CompanyId, device_id: str) -> str:
    """Full latest snapshot of one device: hardware, OS, disk, memory, battery,
    installed software names and compliance checks."""
    return _run("get_device", tools_read.get_device, company_id, device_id=device_id)


@tool
def low_disk_devices(company_id: CompanyId, threshold_pct: float = 10.0) -> str:
    """Devices whose latest snapshot has less than threshold_pct percent of disk free."""
    return _run("low_disk_devices", tools_read.low_disk_devices, company_id, threshold_pct=threshold_pct)


@tool
def compliance_failures(
    company_id: CompanyId,
    severity: Literal["low", "medium", "high"] | None = None,
    check_id: Literal["disk_encryption", "screen_lock", "os_up_to_date"] | None = None,
    form_factor: Literal["laptop", "desktop"] | None = None,
) -> str:
    """Compliance checks failing on each device's latest snapshot, with optional filters."""
    return _run(
        "compliance_failures",
        tools_read.compliance_failures,
        company_id,
        severity=severity,
        check_id=check_id,
        form_factor=form_factor,
    )


@tool
def os_older_than(company_id: CompanyId, version: str = "15") -> str:
    """macOS devices whose latest OS version is lower than version (for example "15" or "14.6").

    Windows devices are excluded and listed in the notes.
    """
    return _run("os_older_than", tools_read.os_older_than, company_id, version=version)


@tool
def device_timeseries(
    company_id: CompanyId,
    device_id: str,
    metric: Literal[
        "disk_free_pct",
        "mem_used_pct",
        "battery_cycle_count",
        "battery_full_charge_capacity",
        "battery_condition",
        "compliance",
    ],
) -> str:
    """One value per snapshot of a device, oldest first, to see how a metric changed.

    For metric "compliance" the value is the list of failing check_ids.
    """
    return _run("device_timeseries", tools_read.device_timeseries, company_id, device_id=device_id, metric=metric)


@tool
def installed_software(company_id: CompanyId, name: str | None = None) -> str:
    """Installed software on each device's latest snapshot. name is a case-insensitive substring filter."""
    return _run("installed_software", tools_read.installed_software, company_id, name=name)


def _insights(company_id: str, insight: str | None) -> dict[str, Any]:
    if insight is None:
        return insights.all_insights(company_id)
    return INSIGHTS[insight](company_id)


@tool
def fleet_insights(
    company_id: CompanyId,
    insight: Literal["low_storage", "ram_pressure", "battery_end_of_life", "compliance_drift"] | None = None,
) -> str:
    """Problems found by fixed rules over the whole 30-day history of every device.

    Each row has a category, a finding with the numbers, an explanation and the
    snapshot ids that support it. Omit insight to run every detector. Prefer this
    over the other tools for questions about trends, risks or what needs attention.
    """
    return _run("fleet_insights", _insights, company_id, insight=insight)


TOOLS: list[BaseTool] = [
    fleet_insights,
    list_devices,
    get_device,
    low_disk_devices,
    compliance_failures,
    os_older_than,
    device_timeseries,
    installed_software,
]

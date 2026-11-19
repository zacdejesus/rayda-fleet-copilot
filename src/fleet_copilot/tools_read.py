"""Read-only tools over the telemetry data.

Every tool takes company_id as its first argument. The agent runtime injects it
from the session, so the model never chooses which company it reads.
Every tool returns {"rows": [...], "evidence": [...snapshot_ids], "notes": [...]}.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fleet_copilot.data import latest_per_device, load_snapshots, parse_macos_version

GIB = 1024**3  # "GB" in the output means GiB (bytes / 1024^3)

BYTE_FIELDS = {
    "ram_bytes": "ram_gb",
    "mem_used_bytes": "mem_used_gb",
    "disk_size_bytes": "disk_size_gb",
    "disk_available_bytes": "disk_available_gb",
}

TIMESERIES_METRICS = (
    "disk_free_pct",
    "mem_used_pct",
    "battery_cycle_count",
    "battery_full_charge_capacity",
    "battery_condition",
    "compliance",
)
BATTERY_METRICS = {"battery_cycle_count", "battery_full_charge_capacity", "battery_condition"}


NOT_AVAILABLE = "Device not found for this company."


class AccessDenied(Exception):
    """The device exists but belongs to another company."""


class NotFound(Exception):
    """No device with that id exists."""


def to_plain(value: Any) -> Any:
    """Convert pandas and numpy values to plain JSON-friendly Python values."""
    if isinstance(value, (list, dict)):
        return value
    if pd.isna(value):  # NaN, NaT, None and pd.NA all become None
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _pick(record: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: to_plain(record[key]) for key in keys}


def _gb(num_bytes: Any) -> float:
    return round(float(num_bytes) / GIB, 1)


def _scalar_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Every scalar column of a snapshot, with byte counts shown as *_gb."""
    row: dict[str, Any] = {}
    for key, value in record.items():
        if key in ("installed_software", "compliance"):
            continue
        if key in BYTE_FIELDS:
            row[BYTE_FIELDS[key]] = _gb(value)
        else:
            row[key] = to_plain(value)
    return row


def _company_latest(company_id: str) -> list[dict[str, Any]]:
    """Latest snapshot of every device that belongs to the company."""
    snapshots = load_snapshots()
    company = snapshots[snapshots["company_id"] == company_id]
    return latest_per_device(company).to_dict("records")


def _device_history(company_id: str, device_id: str) -> list[dict[str, Any]]:
    """All snapshots of one device, oldest first. Raises if the device is not this company's."""
    snapshots = load_snapshots()
    device = snapshots[snapshots["device_id"] == device_id]
    if device.empty:
        raise NotFound(NOT_AVAILABLE)
    if device["company_id"].iloc[0] != company_id:
        # Same message as NotFound on purpose: the model must not learn that the
        # id exists in another tenant. The exception type differs so the audit
        # log can still record a cross-tenant attempt.
        raise AccessDenied(NOT_AVAILABLE)
    return device.sort_values("collected_at").to_dict("records")


def make_result(rows: list[dict[str, Any]], evidence: list[str], notes: list[str]) -> dict[str, Any]:
    return {"rows": rows, "evidence": evidence, "notes": notes}


def list_devices(
    company_id: str, form_factor: str | None = None, os_name: str | None = None
) -> dict[str, Any]:
    """One row per device, from its latest snapshot, with optional filters."""
    devices = _company_latest(company_id)
    if form_factor is not None:
        devices = [d for d in devices if d["form_factor"] == form_factor]
    if os_name is not None:
        devices = [d for d in devices if d["os_name"] == os_name]

    keys = [
        "device_id",
        "employee_id",
        "model_name",
        "form_factor",
        "os_name",
        "os_version",
        "disk_free_pct",
        "mem_used_pct",
        "battery_present",
        "battery_condition",
        "collected_at",
        "snapshot_id",
    ]
    rows = [_pick(d, keys) for d in devices]
    filters = f"form_factor={form_factor}, os_name={os_name}"
    return make_result(rows, [r["snapshot_id"] for r in rows], [f"{len(rows)} devices. Filters: {filters}."])


def get_device(company_id: str, device_id: str) -> dict[str, Any]:
    """Latest snapshot of one device, with its installed software names and compliance checks."""
    latest = _device_history(company_id, device_id)[-1]
    row = _scalar_fields(latest)
    row["installed_software"] = [sw["name"] for sw in latest["installed_software"]]
    row["compliance"] = latest["compliance"]
    return make_result([row], [latest["snapshot_id"]], [f"Latest snapshot at {row['collected_at']}."])


def low_disk_devices(company_id: str, threshold_pct: float = 10.0) -> dict[str, Any]:
    """Devices whose latest snapshot has less than threshold_pct percent of disk free."""
    devices = _company_latest(company_id)
    low = [d for d in devices if d["disk_free_pct"] < threshold_pct]
    rows = [
        {
            "device_id": d["device_id"],
            "employee_id": d["employee_id"],
            "disk_free_pct": to_plain(d["disk_free_pct"]),
            "disk_available_gb": _gb(d["disk_available_bytes"]),
            "disk_size_gb": _gb(d["disk_size_bytes"]),
            "snapshot_id": d["snapshot_id"],
        }
        for d in low
    ]
    notes = [
        f"Threshold: disk free below {threshold_pct}%, using each device's latest snapshot.",
        f"{len(rows)} of {len(devices)} devices matched.",
    ]
    # Evidence covers every device checked, so "none are low" is still grounded.
    return make_result(rows, [d["snapshot_id"] for d in devices], notes)


def compliance_failures(
    company_id: str,
    severity: str | None = None,
    check_id: str | None = None,
    form_factor: str | None = None,
) -> dict[str, Any]:
    """Failing compliance checks on the latest snapshot of each device."""
    devices = _company_latest(company_id)
    if form_factor is not None:
        devices = [d for d in devices if d["form_factor"] == form_factor]

    rows: list[dict[str, Any]] = []
    for device in devices:
        for check in device["compliance"]:
            if check["status"] != "fail":
                continue
            if severity is not None and check["severity"] != severity:
                continue
            if check_id is not None and check["check_id"] != check_id:
                continue
            rows.append(
                {
                    "device_id": device["device_id"],
                    "employee_id": device["employee_id"],
                    "check_id": check["check_id"],
                    "severity": check["severity"],
                    "status": check["status"],
                    "snapshot_id": device["snapshot_id"],
                }
            )

    matched_devices = len({r["device_id"] for r in rows})
    notes = [f"Checked {len(devices)} devices; {matched_devices} matched with {len(rows)} failing checks."]
    # Evidence covers every device checked, not only the failing ones.
    return make_result(rows, [d["snapshot_id"] for d in devices], notes)


def os_older_than(company_id: str, version: str = "15", os_name: str = "macOS") -> dict[str, Any]:
    """macOS devices whose latest OS version is lower than version."""
    if os_name != "macOS":
        raise ValueError("only macOS version comparison is supported")

    devices = _company_latest(company_id)
    macos = [d for d in devices if d["os_name"] == os_name]
    excluded = [d["device_id"] for d in devices if d["os_name"] != os_name]
    threshold = parse_macos_version(version)

    rows = [
        {
            "device_id": d["device_id"],
            "employee_id": d["employee_id"],
            "os_version": d["os_version"],
            "model_name": d["model_name"],
            "snapshot_id": d["snapshot_id"],
        }
        for d in macos
        if parse_macos_version(d["os_version"]) < threshold
    ]
    notes = [f"Compared {len(macos)} macOS devices against version {version}."]
    if excluded:
        notes.append(f"Excluded {len(excluded)} non-macOS devices: {', '.join(excluded)}.")
    return make_result(rows, [d["snapshot_id"] for d in macos], notes)


def device_timeseries(company_id: str, device_id: str, metric: str) -> dict[str, Any]:
    """One value per snapshot of a device, oldest first."""
    if metric not in TIMESERIES_METRICS:
        raise ValueError(f"metric must be one of: {', '.join(TIMESERIES_METRICS)}")

    history = _device_history(company_id, device_id)
    rows: list[dict[str, Any]] = []
    for snap in history:
        if metric == "compliance":
            value: Any = [c["check_id"] for c in snap["compliance"] if c["status"] == "fail"]
        else:
            value = to_plain(snap[metric])
        rows.append(
            {
                "collected_at": to_plain(snap["collected_at"]),
                "value": value,
                "snapshot_id": snap["snapshot_id"],
            }
        )

    notes: list[str] = []
    if metric in BATTERY_METRICS:
        missing = sum(1 for r in rows if r["value"] is None)
        if missing:
            notes.append(f"battery data missing in {missing} snapshots")
    return make_result(rows, [s["snapshot_id"] for s in history], notes)


def installed_software(company_id: str, name: str | None = None) -> dict[str, Any]:
    """Installed software on each device's latest snapshot, optionally filtered by name."""
    devices = _company_latest(company_id)
    needle = name.lower() if name is not None else None

    rows: list[dict[str, Any]] = []
    for device in devices:
        for sw in device["installed_software"]:
            sw_name = sw.get("name") or ""
            if needle is not None and needle not in sw_name.lower():
                continue
            rows.append(
                {
                    "device_id": device["device_id"],
                    "employee_id": device["employee_id"],
                    "software_name": sw_name,
                    "version": sw.get("version"),
                    "snapshot_id": device["snapshot_id"],
                }
            )

    filter_text = f"name contains '{name}'" if name is not None else "no name filter"
    notes = [f"{len(rows)} installed items across {len(devices)} devices ({filter_text})."]
    return make_result(rows, [d["snapshot_id"] for d in devices], notes)

"""Loads the telemetry NDJSON into one flat pandas table, one row per snapshot."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "device-telemetry-dataset.ndjson"

FORM_FACTOR = {
    "MacBook Pro": "laptop",
    "MacBook Air": "laptop",
    "ThinkPad X1 Carbon": "laptop",
    "Dell XPS 15": "laptop",
    "Mac mini": "desktop",
}


def _snapshot_row(rec: dict[str, Any]) -> dict[str, Any]:
    """Flatten one raw telemetry record into the columns of the snapshot table."""
    os_info = rec["os"]
    identity = rec["device_identity"]
    memory = rec["memory"]
    volume = rec["disk_volumes"][0]  # the dataset always has exactly one volume
    battery = rec.get("battery") or {}  # battery object is missing on some records
    model = identity["model_name"]

    return {
        "snapshot_id": f"{rec['device_id']}@{rec['collected_at']}",
        "device_id": rec["device_id"],
        "company_id": rec["company_id"],
        "employee_id": rec["employee_id"],
        "collected_at": rec["collected_at"],
        "hostname": os_info["hostname"],
        "os_name": os_info["product_name"],
        "os_version": os_info["product_version"],
        "model_name": model,
        "form_factor": FORM_FACTOR.get(model, "unknown"),
        "processor": identity["processor"],
        "ram_bytes": memory["total_memory_bytes"],
        "mem_used_bytes": memory["used_memory_bytes"],
        "mem_used_pct": round(memory["used_memory_bytes"] / memory["total_memory_bytes"] * 100, 1),
        "disk_size_bytes": volume["size_bytes"],
        "disk_available_bytes": volume["available_bytes"],
        "disk_free_pct": round(volume["available_bytes"] / volume["size_bytes"] * 100, 1),
        "disk_encrypted": bool(volume["encrypted"]),
        "battery_present": bool(battery.get("battery_present", False)),
        "battery_condition": battery.get("condition"),
        "battery_cycle_count": battery.get("cycle_count"),
        "battery_full_charge_capacity": battery.get("full_charge_capacity"),
        "battery_percentage": battery.get("percentage"),
        "installed_software": rec["installed_software"],
        "compliance": rec["compliance_results"],
    }


@lru_cache(maxsize=None)
def _load_cached(path: str) -> pd.DataFrame:
    """Parse the NDJSON file once per path. Callers must use load_snapshots, never this."""
    with open(path, encoding="utf-8") as handle:
        rows = [_snapshot_row(json.loads(line)) for line in handle if line.strip()]

    df = pd.DataFrame(rows)
    df["collected_at"] = pd.to_datetime(df["collected_at"], utc=True)
    # Nullable integer type keeps missing battery numbers as <NA> instead of turning them into floats.
    for column in ("battery_cycle_count", "battery_full_charge_capacity", "battery_percentage"):
        df[column] = df[column].astype("Int64")
    return df.sort_values(["device_id", "collected_at"]).reset_index(drop=True)


def load_snapshots(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Return one row per telemetry snapshot, sorted by device_id then collected_at.

    The result is a copy, so callers may change it freely. The list columns
    (installed_software, compliance) are shared with the cache; treat them as read-only.
    """
    return _load_cached(str(path)).copy()


def latest_per_device(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most recent snapshot of each device."""
    newest_last = df.sort_values("collected_at")
    latest = newest_last.groupby("device_id", as_index=False).tail(1)
    return latest.sort_values("device_id").reset_index(drop=True)


def parse_macos_version(version: str) -> tuple[int, ...]:
    """Turn a macOS version such as "14.5" into (14, 5) so versions compare numerically."""
    return tuple(int(part) for part in version.split("."))

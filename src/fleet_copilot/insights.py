"""Deterministic insight detectors over one company's telemetry snapshots.

Each detector applies a fixed, written rule to a company's devices. No model is
involved: a row is a rule that fired, with the numbers and the snapshot ids that
support it. Every detector returns the same shape as the read tools:
{"rows": [...], "evidence": [...snapshot_ids], "notes": [...]}.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

import numpy as np
import pandas as pd

from fleet_copilot.data import load_snapshots
from fleet_copilot.tools_read import make_result, to_plain

LOW_DISK_PCT = 10.0
PERSISTENT_SHARE = 0.8  # "consistently" = in at least 80% of the device's snapshots
RAM_HIGH_PCT = 85.0
RAM_INTERMITTENT_SHARE = 0.25
BATTERY_CYCLES_LIMIT = 1000

Row = dict[str, Any]
Result = dict[str, Any]


def _company_devices(company_id: str) -> list[pd.DataFrame]:
    """One time-ordered snapshot history per device, for this company only."""
    snapshots = load_snapshots()
    company = snapshots[snapshots["company_id"] == company_id]
    return [history for _, history in company.groupby("device_id", sort=True)]


def _unique(ids: list[str]) -> list[str]:
    """Drop repeated snapshot ids and keep the first occurrence order."""
    return list(dict.fromkeys(ids))


def _days_between(history: pd.DataFrame) -> np.ndarray:
    """Days since the device's first snapshot, one value per snapshot."""
    times = history["collected_at"]
    return ((times - times.iloc[0]).dt.total_seconds() / 86400).to_numpy(dtype=float)


def _int_or_none(value: Any) -> int | None:
    plain = to_plain(value)
    return None if plain is None else int(plain)


def _make_row(
    insight: str,
    history: pd.DataFrame,
    category: str,
    finding: str,
    explanation: str,
    metrics: dict[str, Any],
    evidence: list[str],
) -> Row:
    """One insight row. Every row also cites the device's first and latest snapshot."""
    first_id = history["snapshot_id"].iloc[0]
    latest = history.iloc[-1]
    return {
        "insight": insight,
        "device_id": latest["device_id"],
        "employee_id": latest["employee_id"],
        "category": category,
        "finding": finding,
        "explanation": explanation,
        "metrics": {key: to_plain(value) for key, value in metrics.items()},
        "evidence": _unique([first_id, latest["snapshot_id"], *evidence]),
        "snapshot_id": latest["snapshot_id"],
    }


def _result_from_rows(rows: list[Row], notes: list[str]) -> Result:
    evidence = _unique([snap_id for row in rows for snap_id in row["evidence"]])
    return make_result(rows, evidence, notes)


def detect_low_storage(company_id: str) -> Result:
    """Devices whose latest disk free space is below LOW_DISK_PCT."""
    devices = _company_devices(company_id)
    rows: list[Row] = []
    for history in devices:
        free = history["disk_free_pct"].to_numpy(dtype=float)
        latest_pct = float(free[-1])
        if latest_pct >= LOW_DISK_PCT:
            continue

        first_pct = float(free[0])
        below = int(np.sum(free < LOW_DISK_PCT))
        share = below / len(free)
        slope = float(np.polyfit(_days_between(history), free, 1)[0])
        span = float(_days_between(history)[-1])
        category = "persistent" if share >= PERSISTENT_SHARE else "declining"

        if category == "persistent":
            finding = (
                f"Free disk was below {LOW_DISK_PCT:.0f}% in {below} of {len(free)} snapshots "
                f"and is {latest_pct:.1f}% now."
            )
            explanation = "The disk has been nearly full for most of the window, so updates and installs keep failing."
        else:
            verb = "fell" if latest_pct < first_pct else "rose"
            finding = f"Free disk {verb} from {first_pct:.1f}% to {latest_pct:.1f}% over {span:.0f} days."
            explanation = "A shrinking disk will fill up, and a full disk blocks OS updates and other installs."

        rows.append(
            _make_row(
                "low_storage",
                history,
                category,
                finding,
                explanation,
                {
                    "first_pct": first_pct,
                    "latest_pct": latest_pct,
                    "share_below": round(share, 2),
                    "slope_pct_per_day": round(slope, 2),
                    "span_days": round(span),
                },
                [],
            )
        )

    notes = [
        f"Rule: latest disk free below {LOW_DISK_PCT}%. Persistent if below in at least "
        f"{PERSISTENT_SHARE:.0%} of snapshots, otherwise declining. Slope is a least-squares "
        "fit of free % against days, in percentage points per day.",
        f"{len(rows)} of {len(devices)} devices flagged.",
    ]
    return _result_from_rows(rows, notes)


def detect_ram_pressure(company_id: str) -> Result:
    """Devices whose memory use is above RAM_HIGH_PCT in many snapshots."""
    devices = _company_devices(company_id)
    rows: list[Row] = []
    for history in devices:
        mem = history["mem_used_pct"].to_numpy(dtype=float)
        above = int(np.sum(mem > RAM_HIGH_PCT))
        share = above / len(mem)
        if share >= PERSISTENT_SHARE:
            category = "consistent"
            finding = (
                f"Memory use was above {RAM_HIGH_PCT:.0f}% in {above} of {len(mem)} snapshots "
                f"(average {mem.mean():.1f}%)."
            )
            explanation = "Memory is full most of the time, so the device is undersized for its work or a process is leaking."
        elif share >= RAM_INTERMITTENT_SHARE:
            category = "intermittent"
            finding = (
                f"Memory use was above {RAM_HIGH_PCT:.0f}% in {above} of {len(mem)} snapshots, "
                f"peaking at {mem.max():.1f}%."
            )
            explanation = "Repeated peaks above the threshold point to heavy workloads that may slow the user down."
        else:
            continue

        rows.append(
            _make_row(
                "ram_pressure",
                history,
                category,
                finding,
                explanation,
                {
                    "share_above": round(share, 2),
                    "avg_pct": round(float(mem.mean()), 1),
                    "max_pct": round(float(mem.max()), 1),
                },
                [],
            )
        )

    notes = [
        f"Rule: memory used above {RAM_HIGH_PCT}% in at least {PERSISTENT_SHARE:.0%} of snapshots "
        f"is consistent; in at least {RAM_INTERMITTENT_SHARE:.0%} is intermittent.",
        f"{len(rows)} of {len(devices)} devices flagged.",
    ]
    return _result_from_rows(rows, notes)


def detect_battery_end_of_life(company_id: str) -> Result:
    """Devices whose battery needs service, has high cycles, or has lost full charge capacity.

    Only snapshots with a battery are used. Missing battery data is reported in
    the notes as a gap, never as a healthy battery.
    """
    devices = _company_devices(company_id)
    rows: list[Row] = []
    no_battery: list[str] = []
    data_gaps: list[str] = []

    for history in devices:
        device_id = history["device_id"].iloc[0]
        has_battery = history["battery_present"].astype(bool)
        if not has_battery.any():
            no_battery.append(device_id)
            continue
        missing = int((~has_battery).sum())
        if missing:
            data_gaps.append(
                f"{device_id}: battery data missing in {missing} of {len(history)} snapshots, "
                "a data-quality gap, not treated as healthy."
            )

        present = history[has_battery]
        first, latest = present.iloc[0], present.iloc[-1]
        condition = to_plain(latest["battery_condition"])
        is_service = condition is not None and condition != "Normal"
        cycles_first = _int_or_none(first["battery_cycle_count"])
        cycles_latest = _int_or_none(latest["battery_cycle_count"])
        capacity_first = _int_or_none(first["battery_full_charge_capacity"])
        capacity_latest = _int_or_none(latest["battery_full_charge_capacity"])

        reasons: list[str] = []
        if is_service:
            reasons.append(f"condition is {condition}")
        if cycles_latest is not None and cycles_latest >= BATTERY_CYCLES_LIMIT:
            reasons.append(f"cycle count {cycles_latest} is at or above {BATTERY_CYCLES_LIMIT}")
        capacity_drop_pct = None
        if capacity_first and capacity_latest is not None:
            capacity_drop_pct = round((capacity_first - capacity_latest) / capacity_first * 100, 1)
            if capacity_latest < capacity_first:
                reasons.append(f"full charge capacity fell {capacity_drop_pct}%")
        if not reasons:
            continue

        category = "service_recommended" if is_service else "wearing"
        explanation = (
            "macOS reports the battery needs service, which can cause sudden shutdowns."
            if is_service
            else "The battery is wearing out and should be watched before it needs service."
        )
        rows.append(
            _make_row(
                "battery_end_of_life",
                history,
                category,
                f"Battery flagged: {'; '.join(reasons)}.",
                explanation,
                {
                    "condition": condition,
                    "cycles_first": cycles_first,
                    "cycles_latest": cycles_latest,
                    "capacity_first": capacity_first,
                    "capacity_latest": capacity_latest,
                    "capacity_drop_pct": capacity_drop_pct,
                    "reasons": reasons,
                },
                [first["snapshot_id"], latest["snapshot_id"]],
            )
        )

    notes = [
        "Rule: latest battery condition is not Normal, or latest cycle count is at least "
        f"{BATTERY_CYCLES_LIMIT}, or full charge capacity fell since the first snapshot with a battery.",
        *[f"{device_id}: not applicable (no battery)." for device_id in no_battery],
        *data_gaps,
        f"{len(rows)} of {len(devices)} devices flagged.",
    ]
    return _result_from_rows(rows, notes)


def _check_timelines(history: pd.DataFrame) -> dict[str, list[tuple[pd.Series, str, str]]]:
    """For each compliance check_id, its (snapshot, status, severity) in time order."""
    timelines: dict[str, list[tuple[pd.Series, str, str]]] = {}
    for _, snap in history.iterrows():
        for check in snap["compliance"]:
            timelines.setdefault(check["check_id"], []).append(
                (snap, check["status"], check["severity"])
            )
    return timelines


def detect_compliance_drift(company_id: str) -> Result:
    """Checks that regressed from pass to fail, and checks that fail in every snapshot."""
    devices = _company_devices(company_id)
    rows: list[Row] = []
    failing_at_start: Counter[str] = Counter()
    failing_at_end: Counter[str] = Counter()
    check_ids: set[str] = set()

    for history in devices:
        for check_id, timeline in _check_timelines(history).items():
            check_ids.add(check_id)
            statuses = [status for _, status, _ in timeline]
            severity = timeline[-1][2]
            fail_share = statuses.count("fail") / len(statuses)
            if statuses[0] == "fail":
                failing_at_start[check_id] += 1
            if statuses[-1] == "fail":
                failing_at_end[check_id] += 1

            first_snap = timeline[0][0]
            latest_snap = timeline[-1][0]
            days = float((latest_snap["collected_at"] - first_snap["collected_at"]).total_seconds() / 86400)

            if statuses[0] == "pass" and statuses[-1] == "fail":
                first_fail_snap = next(snap for snap, status, _ in timeline if status == "fail")
                first_fail_at = to_plain(first_fail_snap["collected_at"])
                rows.append(
                    _make_row(
                        "compliance_drift",
                        history,
                        "drift",
                        f"{check_id} passed on {first_snap['collected_at'].date().isoformat()}"
                        f", first failed on {first_fail_snap['collected_at'].date().isoformat()}"
                        f" and is failing in the latest snapshot ({fail_share:.0%} of snapshots failing).",
                        "The device was compliant and has regressed, so a setting or policy changed on it.",
                        {
                            "check_id": check_id,
                            "severity": severity,
                            "first_fail_at": first_fail_at,
                            "fail_share": round(fail_share, 2),
                        },
                        [first_fail_snap["snapshot_id"]],
                    )
                )
            elif all(status == "fail" for status in statuses):
                rows.append(
                    _make_row(
                        "compliance_drift",
                        history,
                        "persistent_failure",
                        f"{check_id} has failed in all {len(statuses)} snapshots over {days:.0f} days.",
                        "The device has never met this control in the window, so it is an open gap, not a recent regression.",
                        {
                            "check_id": check_id,
                            "severity": severity,
                            "fail_share": 1.0,
                        },
                        [],
                    )
                )

    notes = [
        "Rule: drift = passed in the first snapshot and fails in the latest. "
        "Persistent failure = fails in every snapshot.",
        *[
            f"{check_id}: {failing_at_start[check_id]} devices failing at the start of the window, "
            f"{failing_at_end[check_id]} at the end."
            for check_id in sorted(check_ids)
        ],
        f"{len(rows)} insights across {len(devices)} devices.",
    ]
    return _result_from_rows(rows, notes)


DETECTORS: tuple[Callable[[str], Result], ...] = (
    detect_low_storage,
    detect_ram_pressure,
    detect_battery_end_of_life,
    detect_compliance_drift,
)


def all_insights(company_id: str) -> Result:
    """Run every detector for one company and merge the results."""
    results = [detector(company_id) for detector in DETECTORS]
    rows = [row for result in results for row in result["rows"]]
    evidence = _unique([snap_id for result in results for snap_id in result["evidence"]])
    notes = [note for result in results for note in result["notes"]]
    return make_result(rows, evidence, notes)

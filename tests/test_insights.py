import json

import pytest

from fleet_copilot.data import load_snapshots
from fleet_copilot.insights import (
    DETECTORS,
    all_insights,
    detect_battery_end_of_life,
    detect_compliance_drift,
    detect_low_storage,
    detect_ram_pressure,
)

ACME = "acme-001"
COMPANIES = ["acme-001", "globex-002", "initech-003"]
FLEET_SCREEN_LOCK_DRIFT = {
    "1LYSSFD074BB", "2MAN6T4DJD95", "8NM23J95R5I6", "CUXW0BQSV1B7", "DLT35MHGTH2D",
    "DSYPSES7SKBB", "JRZSGXVMKE6M", "MT7PJB7N5LRE", "SPKRRDF0L786", "XZYK11QTPWEC",
}
MAC_MINIS_WITHOUT_BATTERY = {"DLT35MHGTH2D", "I3JLKILKQTAC", "RLD7WV7J82SH"}


def _flagged(detector, category=None, company=None):
    """Device ids flagged by a detector, over all companies or one company."""
    companies = [company] if company else COMPANIES
    devices = set()
    for company_id in companies:
        for row in detector(company_id)["rows"]:
            if category is None or row["category"] == category:
                devices.add(row["device_id"])
    return devices


def _company_of_snapshot():
    snapshots = load_snapshots()
    return dict(zip(snapshots["snapshot_id"], snapshots["company_id"]))


def _company_of_device():
    snapshots = load_snapshots()
    return dict(zip(snapshots["device_id"], snapshots["company_id"]))


# Expected device sets, measured from the real dataset.


def test_low_storage_persistent_fleet():
    assert _flagged(detect_low_storage, "persistent") == {"MT7PJB7N5LRE", "RLD7WV7J82SH"}


def test_low_storage_declining_fleet():
    assert _flagged(detect_low_storage, "declining") == {
        "8NM23J95R5I6",
        "DARFCP0BQM2G",
        "E4V0SB5I0MYU",
        "JRZSGXVMKE6M",
    }


def test_low_storage_acme_categories():
    categories = {row["device_id"]: row["category"] for row in detect_low_storage(ACME)["rows"]}
    assert categories == {"8NM23J95R5I6": "declining", "MT7PJB7N5LRE": "persistent"}


def test_ram_pressure_consistent_fleet():
    assert _flagged(detect_ram_pressure, "consistent") == {"I3JLKILKQTAC", "M4XVHUV1MEPZ"}


def test_ram_pressure_intermittent_fleet():
    assert _flagged(detect_ram_pressure, "intermittent") == {"JRZSGXVMKE6M", "CUXW0BQSV1B7"}


def test_battery_end_of_life_fleet():
    assert _flagged(detect_battery_end_of_life) == {"M4XVHUV1MEPZ", "8NM23J95R5I6", "7VP16KHV88LM"}


def test_battery_end_of_life_all_service_recommended():
    rows = [row for company in COMPANIES for row in detect_battery_end_of_life(company)["rows"]]
    assert rows
    assert {row["category"] for row in rows} == {"service_recommended"}


def test_compliance_drift_only_screen_lock_on_expected_devices():
    drift_rows = [
        row
        for company in COMPANIES
        for row in detect_compliance_drift(company)["rows"]
        if row["category"] == "drift"
    ]
    assert {row["metrics"]["check_id"] for row in drift_rows} == {"screen_lock"}
    assert {row["device_id"] for row in drift_rows} == {
        "1LYSSFD074BB",
        "2MAN6T4DJD95",
        "8NM23J95R5I6",
        "CUXW0BQSV1B7",
        "DLT35MHGTH2D",
        "DSYPSES7SKBB",
        "JRZSGXVMKE6M",
        "MT7PJB7N5LRE",
        "SPKRRDF0L786",
        "XZYK11QTPWEC",
    }


def test_compliance_persistent_failure_os_up_to_date_on_19_devices():
    persistent_rows = [
        row
        for company in COMPANIES
        for row in detect_compliance_drift(company)["rows"]
        if row["category"] == "persistent_failure"
    ]
    assert {row["metrics"]["check_id"] for row in persistent_rows} == {"os_up_to_date"}
    assert len({row["device_id"] for row in persistent_rows}) == 19


def test_compliance_notes_summarise_each_check_per_company():
    notes = " ".join(detect_compliance_drift(ACME)["notes"])
    assert "screen_lock:" in notes
    assert "os_up_to_date:" in notes
    assert "devices failing at the start of the window" in notes


# Per-company checks for acme-001.


def test_acme_low_storage_devices():
    assert _flagged(detect_low_storage, company=ACME) == {"8NM23J95R5I6", "MT7PJB7N5LRE"}


def test_acme_ram_pressure_devices():
    assert _flagged(detect_ram_pressure, company=ACME) == {"M4XVHUV1MEPZ"}


def test_acme_battery_devices():
    assert _flagged(detect_battery_end_of_life, company=ACME) == {"M4XVHUV1MEPZ", "8NM23J95R5I6", "7VP16KHV88LM"}


def test_acme_compliance_drift_devices():
    drift = _flagged(detect_compliance_drift, "drift", company=ACME)
    assert drift == {"1LYSSFD074BB", "8NM23J95R5I6", "DLT35MHGTH2D", "DSYPSES7SKBB", "MT7PJB7N5LRE"}
    assert drift <= FLEET_SCREEN_LOCK_DRIFT


def test_battery_notes_report_missing_battery_data_for_mt7pjb7n5lre():
    notes = " ".join(detect_battery_end_of_life(ACME)["notes"])
    assert "MT7PJB7N5LRE" in notes
    assert "data-quality gap" in notes


def test_battery_results_never_include_devices_without_a_battery():
    for company in COMPANIES:
        devices = _flagged(detect_battery_end_of_life, company=company)
        assert not devices & MAC_MINIS_WITHOUT_BATTERY


def test_all_insights_acme_contains_all_four_detectors():
    result = all_insights(ACME)
    assert {row["insight"] for row in result["rows"]} == {
        "low_storage",
        "ram_pressure",
        "battery_end_of_life",
        "compliance_drift",
    }


def test_all_insights_evidence_is_deduplicated_and_matches_rows():
    result = all_insights(ACME)
    assert len(result["evidence"]) == len(set(result["evidence"]))
    cited = {snap_id for row in result["rows"] for snap_id in row["evidence"]}
    assert set(result["evidence"]) == cited


# Cross-cutting checks, for every detector and every company.


@pytest.mark.parametrize("detector", DETECTORS + (all_insights,))
@pytest.mark.parametrize("company", COMPANIES)
def test_tenant_isolation_rows_belong_to_the_company(detector, company):
    device_company = _company_of_device()
    for row in detector(company)["rows"]:
        assert device_company[row["device_id"]] == company


@pytest.mark.parametrize("detector", DETECTORS + (all_insights,))
@pytest.mark.parametrize("company", COMPANIES)
def test_evidence_ids_exist_and_belong_to_the_company(detector, company):
    snapshot_company = _company_of_snapshot()
    result = detector(company)
    cited = list(result["evidence"]) + [
        snap_id for row in result["rows"] for snap_id in row["evidence"]
    ]
    for snap_id in cited:
        assert snap_id in snapshot_company
        assert snapshot_company[snap_id] == company


@pytest.mark.parametrize("detector", DETECTORS + (all_insights,))
@pytest.mark.parametrize("company", COMPANIES)
def test_output_is_json_serialisable(detector, company):
    json.dumps(detector(company), allow_nan=False)


@pytest.mark.parametrize("company", COMPANIES)
def test_each_row_has_the_documented_keys(company):
    required = {
        "insight",
        "device_id",
        "employee_id",
        "category",
        "finding",
        "explanation",
        "metrics",
        "evidence",
        "snapshot_id",
    }
    for row in all_insights(company)["rows"]:
        assert required <= set(row)

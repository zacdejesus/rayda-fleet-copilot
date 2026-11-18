import json

import pytest

from fleet_copilot.data import load_snapshots
from fleet_copilot.tools_read import (
    AccessDenied,
    NotFound,
    compliance_failures,
    device_timeseries,
    get_device,
    installed_software,
    list_devices,
    low_disk_devices,
    os_older_than,
)

ACME = "acme-001"
COMPANY_DEVICE_COUNTS = {"acme-001": 10, "globex-002": 8, "initech-003": 7}


def test_low_disk_devices_acme():
    result = low_disk_devices(ACME)
    assert {row["device_id"] for row in result["rows"]} == {"8NM23J95R5I6", "MT7PJB7N5LRE"}


@pytest.mark.parametrize("company", COMPANY_DEVICE_COUNTS)
def test_no_high_severity_failures_and_all_devices_are_evidence(company):
    result = compliance_failures(company, severity="high")
    assert result["rows"] == []
    assert len(result["evidence"]) == COMPANY_DEVICE_COUNTS[company]


def test_compliance_failures_filtered_by_check_id():
    result = compliance_failures(ACME, check_id="screen_lock")
    assert result["rows"], "expected at least one failing screen_lock check"
    for row in result["rows"]:
        assert row["check_id"] == "screen_lock"
        assert row["status"] == "fail"


def test_os_older_than_acme_excludes_windows_and_says_so():
    result = os_older_than(ACME)
    assert {row["device_id"] for row in result["rows"]} == {
        "XSU9UUDGU6AC",
        "1LYSSFD074BB",
        "7VP16KHV88LM",
        "MT7PJB7N5LRE",
    }
    notes = " ".join(result["notes"])
    for windows_device in ("M4XVHUV1MEPZ", "9MXJVZ1I0DKA", "8NM23J95R5I6"):
        assert windows_device in notes


def test_os_older_than_rejects_non_macos():
    with pytest.raises(ValueError, match="only macOS"):
        os_older_than(ACME, os_name="Windows")


def test_list_devices_laptop_filter_excludes_mac_mini():
    result = list_devices(ACME, form_factor="laptop")
    assert "DLT35MHGTH2D" not in {row["device_id"] for row in result["rows"]}


def test_get_device_of_another_company_is_access_denied_without_leaking_name():
    with pytest.raises(AccessDenied) as error:
        get_device(ACME, "I3JLKILKQTAC")
    assert "initech" not in str(error.value)


def test_get_device_unknown_is_not_found():
    with pytest.raises(NotFound):
        get_device(ACME, "NOPE")


def test_device_timeseries_battery_has_30_rows_and_missing_note():
    result = device_timeseries(ACME, "MT7PJB7N5LRE", "battery_condition")
    assert len(result["rows"]) == 30
    assert any("battery data missing" in note for note in result["notes"])


def test_device_timeseries_rejects_unknown_metric():
    with pytest.raises(ValueError):
        device_timeseries(ACME, "MT7PJB7N5LRE", "uptime")


def test_installed_software_substring_filter_is_case_insensitive():
    result = installed_software(ACME, name="CHROME")
    assert result["rows"]
    assert all("chrome" in row["software_name"].lower() for row in result["rows"])


def _all_tool_outputs():
    return [
        list_devices(ACME),
        list_devices(ACME, form_factor="laptop", os_name="macOS"),
        get_device(ACME, "1LYSSFD074BB"),
        low_disk_devices(ACME),
        compliance_failures(ACME),
        os_older_than(ACME),
        device_timeseries(ACME, "MT7PJB7N5LRE", "compliance"),
        device_timeseries(ACME, "MT7PJB7N5LRE", "disk_free_pct"),
        installed_software(ACME),
    ]


def test_every_row_snapshot_id_exists_in_the_data():
    known = set(load_snapshots()["snapshot_id"])
    for output in _all_tool_outputs():
        for row in output["rows"]:
            assert row["snapshot_id"] in known
        for snapshot_id in output["evidence"]:
            assert snapshot_id in known


def test_all_tool_outputs_are_json_serialisable():
    for output in _all_tool_outputs():
        json.dumps(output)


def test_cross_tenant_and_unknown_look_identical():
    """The model cannot tell 'exists in another company' from 'does not exist'."""
    with pytest.raises(AccessDenied) as other_tenant:
        get_device("acme-001", "I3JLKILKQTAC")
    with pytest.raises(NotFound) as unknown:
        get_device("acme-001", "NOPE")
    assert str(other_tenant.value) == str(unknown.value)

from fleet_copilot.data import latest_per_device, load_snapshots, parse_macos_version


def test_loads_750_snapshots_for_25_devices_and_3_companies():
    df = load_snapshots()
    assert len(df) == 750
    assert df["device_id"].nunique() == 25
    assert set(df["company_id"]) == {"acme-001", "globex-002", "initech-003"}


def test_snapshot_id_is_unique():
    df = load_snapshots()
    assert df["snapshot_id"].is_unique


def test_load_returns_a_copy_so_callers_cannot_change_the_cache():
    df = load_snapshots()
    df.loc[:, "model_name"] = "changed"
    assert "changed" not in set(load_snapshots()["model_name"])


def test_latest_per_device_has_one_row_per_device():
    latest = latest_per_device(load_snapshots())
    assert len(latest) == 25
    assert latest["device_id"].is_unique


def test_mac_mini_is_a_desktop():
    df = load_snapshots()
    mac_mini = df[df["model_name"] == "Mac mini"]
    assert len(mac_mini) > 0
    assert set(mac_mini["form_factor"]) == {"desktop"}


def test_battery_missing_means_battery_present_false_in_100_rows():
    df = load_snapshots()
    assert int((~df["battery_present"]).sum()) == 100


def test_parse_macos_version_splits_into_numbers():
    assert parse_macos_version("13.6") == (13, 6)

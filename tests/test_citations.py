from fleet_copilot.citations import invalid_citations
from fleet_copilot.data import load_snapshots


def _snapshot(company, device=None):
    df = load_snapshots()
    df = df[df["company_id"] == company]
    if device:
        df = df[df["device_id"] == device]
    return df["snapshot_id"].iloc[0]


def test_valid_citations_pass():
    snap = _snapshot("acme-001", "MT7PJB7N5LRE")
    assert invalid_citations(f"Disco bajo [{snap}].", "acme-001") == []


def test_other_company_and_unknown_citations_are_flagged_in_order():
    foreign = _snapshot("initech-003")
    unknown = "MT7PJB7N5LRE@2020-01-01T00:00:00Z"
    text = f"A [{foreign}], B [{unknown}], A otra vez [{foreign}]."
    assert invalid_citations(text, "acme-001") == [foreign, unknown]


def test_text_without_citations_is_fine():
    assert invalid_citations("No hay equipos con fallos de severidad alta.", "acme-001") == []

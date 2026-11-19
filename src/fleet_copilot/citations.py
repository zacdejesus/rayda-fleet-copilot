"""Checks the snapshot_id citations in an answer before the user reads it.

The evals grade citations after the fact; this runs the same check on every
answer in the CLI, so a citation that does not exist or belongs to another
company is flagged live. Unknown and foreign ids get the same treatment, so the
check never reveals that an id exists in another tenant.
"""

from __future__ import annotations

import re

from fleet_copilot.data import load_snapshots

SNAPSHOT_RE = re.compile(r"[A-Z0-9]{12}@\d{4}-\d{2}-\d{2}T[\d:]+Z")


def invalid_citations(text: str, company_id: str) -> list[str]:
    """Cited snapshot ids that are not snapshots of this company, in order of first appearance."""
    snapshots = load_snapshots()
    valid = set(snapshots.loc[snapshots["company_id"] == company_id, "snapshot_id"])
    cited = dict.fromkeys(SNAPSHOT_RE.findall(text))
    return [snap_id for snap_id in cited if snap_id not in valid]

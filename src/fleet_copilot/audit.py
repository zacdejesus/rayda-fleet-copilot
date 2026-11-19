"""Append-only audit log of every tool call the agent makes, one JSON object per line."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_AUDIT_PATH = Path(__file__).resolve().parents[2] / "audit.jsonl"


def audit_path() -> Path:
    """Where the log goes. FLEET_AUDIT_PATH overrides the default, which tests use."""
    return Path(os.environ.get("FLEET_AUDIT_PATH", DEFAULT_AUDIT_PATH))


def record(event: dict[str, Any]) -> None:
    """Write one event with a UTC timestamp. Never raises into the agent loop."""
    line = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    try:
        with open(audit_path(), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass  # A broken log must not break the answer; the CLI warns on startup instead.

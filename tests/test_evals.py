"""Unit tests for the eval harness. No model calls: only the case file and the grader."""

import importlib.util
from pathlib import Path

import anthropic
import httpx
import pytest
from langchain_anthropic.chat_models import AnthropicInvalidRequestError

from fleet_copilot.data import load_snapshots

RUN_EVALS = Path(__file__).resolve().parents[1] / "evals" / "run_evals.py"


@pytest.fixture(scope="module")
def evals():
    spec = importlib.util.spec_from_file_location("run_evals", RUN_EVALS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def maps(evals):
    return evals.dataset_maps()


@pytest.fixture(scope="module")
def tool_names():
    from fleet_copilot.tools import TOOLS

    return {tool.name for tool in TOOLS}


def _snapshot_of(device_id):
    """A real snapshot id of one device from the dataset."""
    snapshots = load_snapshots()
    return snapshots[snapshots["device_id"] == device_id]["snapshot_id"].iloc[-1]


LOW_DISK_CASE = {
    "id": "acme-low-disk",
    "company_id": "acme-001",
    "question": "Which devices are running out of disk space?",
    "expected_devices": ["8NM23J95R5I6", "MT7PJB7N5LRE"],
    "forbidden_devices": ["E4V0SB5I0MYU", "DARFCP0BQM2G"],
    "forbidden_text": ["initech"],
    "must_mention_any": ["disk"],
}


def test_cases_yaml_passes_validation(evals, maps, tool_names):
    cases = evals.load_cases()
    device_company, _ = maps
    assert evals.validate_cases(cases, device_company, tool_names) == []


def test_good_answer_passes(evals, maps):
    _, snapshot_company = maps
    answer = (
        "Two devices have little free disk: 8NM23J95R5I6 [" + _snapshot_of("8NM23J95R5I6") + "] "
        "and MT7PJB7N5LRE [" + _snapshot_of("MT7PJB7N5LRE") + "]."
    )
    result = evals.grade(LOW_DISK_CASE, answer, ["fleet_insights"], snapshot_company)
    assert result["passed"], result["checks"]


def test_forbidden_device_fails(evals, maps):
    _, snapshot_company = maps
    answer = (
        "8NM23J95R5I6 and MT7PJB7N5LRE have little disk, and also E4V0SB5I0MYU "
        f"[{_snapshot_of('8NM23J95R5I6')}]."
    )
    result = evals.grade(LOW_DISK_CASE, answer, [], snapshot_company)
    assert not result["passed"]
    assert not result["checks"]["forbidden_devices"]["passed"]


def test_cited_snapshot_of_another_company_fails(evals, maps):
    _, snapshot_company = maps
    answer = (
        "8NM23J95R5I6 and MT7PJB7N5LRE have little disk "
        f"[{_snapshot_of('8NM23J95R5I6')}] [{_snapshot_of('E4V0SB5I0MYU')}]."
    )
    result = evals.grade(LOW_DISK_CASE, answer, [], snapshot_company)
    assert not result["passed"]
    assert not result["checks"]["citations_company"]["passed"]
    assert result["checks"]["citations_exist"]["passed"]


def test_nonexistent_snapshot_fails(evals, maps):
    _, snapshot_company = maps
    answer = "8NM23J95R5I6 and MT7PJB7N5LRE have little disk [8NM23J95R5I6@2001-01-01T00:00:00Z]."
    result = evals.grade(LOW_DISK_CASE, answer, [], snapshot_company)
    assert not result["passed"]
    assert not result["checks"]["citations_exist"]["passed"]


def test_missing_expected_device_fails(evals, maps):
    _, snapshot_company = maps
    answer = f"Only 8NM23J95R5I6 has little disk [{_snapshot_of('8NM23J95R5I6')}]."
    result = evals.grade(LOW_DISK_CASE, answer, [], snapshot_company)
    assert not result["passed"]
    assert not result["checks"]["expected_devices"]["passed"]
    assert "MT7PJB7N5LRE" in result["checks"]["expected_devices"]["detail"]


def _http_response(status):
    return httpx.Response(status, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))


def test_is_setup_error_true_for_auth_and_config_errors(evals):
    assert evals.is_setup_error(anthropic.AuthenticationError("bad key", response=_http_response(401), body=None))
    assert evals.is_setup_error(anthropic.PermissionDeniedError("no access", response=_http_response(403), body=None))
    assert evals.is_setup_error(anthropic.BadRequestError("bad request", response=_http_response(400), body=None))


def test_is_setup_error_true_for_langchain_wrapped_bad_request(evals):
    exc = AnthropicInvalidRequestError("bad request", response=_http_response(400), body=None)
    assert evals.is_setup_error(exc)


def test_is_setup_error_false_for_other_errors(evals):
    assert not evals.is_setup_error(RuntimeError("boom"))
    assert not evals.is_setup_error(anthropic.RateLimitError("slow down", response=_http_response(429), body=None))


class _FailingModel:
    """Stands in for Claude: every call raises the given error."""

    def __init__(self, exc):
        self.exc = exc

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        raise self.exc


def test_setup_error_stops_the_run_and_writes_no_report(evals, tmp_path, monkeypatch, capsys):
    report = tmp_path / "report.json"
    monkeypatch.setattr(evals, "REPORT_PATH", report)
    monkeypatch.setattr(evals, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("FLEET_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    model = _FailingModel(anthropic.AuthenticationError("invalid x-api-key", response=_http_response(401), body=None))

    assert evals.main([], model=model) == 2
    assert not report.exists()
    err = capsys.readouterr().err
    assert "Setup error on case acme-low-disk" in err
    assert "AuthenticationError" in err

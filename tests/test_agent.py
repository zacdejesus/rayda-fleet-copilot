import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from fleet_copilot.agent import build_graph, thread_config
from fleet_copilot.tools import TOOLS
from fleet_copilot.tools_read import NOT_AVAILABLE

ACME = "acme-001"
INITECH_DEVICE = "I3JLKILKQTAC"


class ScriptedModel(BaseChatModel):
    """Returns the given AI messages in order and records what it was sent."""

    script: list[AIMessage]
    seen: list[list] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def _call(name, args, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture(autouse=True)
def audit_file(tmp_path, monkeypatch):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("FLEET_AUDIT_PATH", str(path))
    return path


def _audit(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _run(script, company=ACME, question="pregunta"):
    model = ScriptedModel(script=script, seen=[])
    graph = build_graph(model=model, checkpointer=InMemorySaver())
    final = graph.invoke(
        {"messages": [HumanMessage(question)], "company_id": company},
        thread_config(company, "test"),
    )
    return final, model


def test_company_id_is_hidden_from_every_tool_schema():
    for tool in TOOLS:
        schema = tool.tool_call_schema.model_json_schema()
        assert "company_id" not in schema.get("properties", {}), tool.name


def test_tool_loop_runs_and_returns_the_final_answer(audit_file):
    final, model = _run([_call("low_disk_devices", {}), AIMessage(content="Dos equipos.")])
    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    rows = json.loads(tool_messages[0].content)["rows"]
    assert {r["device_id"] for r in rows} == {"8NM23J95R5I6", "MT7PJB7N5LRE"}
    assert final["messages"][-1].content == "Dos equipos."
    assert len(model.seen) == 2
    assert _audit(audit_file)[0]["outcome"] == "ok"


def test_model_cannot_override_company_id(audit_file):
    # Even if the model sends company_id, the injected session value wins.
    final, _ = _run([_call("list_devices", {"company_id": "initech-003"}), AIMessage(content="ok")])
    result = json.loads(next(m for m in final["messages"] if isinstance(m, ToolMessage)).content)
    assert len(result["rows"]) == 10  # acme has 10 devices
    assert _audit(audit_file)[0]["company_id"] == ACME


def test_cross_tenant_device_is_refused_and_audited(audit_file):
    final, _ = _run([_call("get_device", {"device_id": INITECH_DEVICE}), AIMessage(content="No está.")])
    content = json.loads(next(m for m in final["messages"] if isinstance(m, ToolMessage)).content)
    assert content == {"error": NOT_AVAILABLE}
    event = _audit(audit_file)[0]
    assert event["outcome"] == "cross_tenant_denied"
    assert event["args"] == {"device_id": INITECH_DEVICE}


def test_unknown_device_looks_the_same_to_the_model_but_not_in_the_audit(audit_file):
    final, _ = _run([_call("get_device", {"device_id": "NOPE"}), AIMessage(content="No está.")])
    content = json.loads(next(m for m in final["messages"] if isinstance(m, ToolMessage)).content)
    assert content == {"error": NOT_AVAILABLE}
    assert _audit(audit_file)[0]["outcome"] == "not_found"


def test_fleet_insights_tool_filters_by_detector():
    final, _ = _run([_call("fleet_insights", {"insight": "ram_pressure"}), AIMessage(content="ok")])
    rows = json.loads(next(m for m in final["messages"] if isinstance(m, ToolMessage)).content)["rows"]
    assert {r["insight"] for r in rows} == {"ram_pressure"}


def test_system_prompt_is_sent_first_on_every_model_call():
    _, model = _run([_call("list_devices", {}), AIMessage(content="ok")])
    for messages in model.seen:
        assert messages[0].type == "system"


def test_threads_are_scoped_by_company():
    assert thread_config("acme-001", "s1")["configurable"]["thread_id"] != thread_config("globex-002", "s1")[
        "configurable"
    ]["thread_id"]

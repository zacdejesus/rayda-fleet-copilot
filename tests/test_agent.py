import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from fleet_copilot.agent import TOOLS as ALL_TOOLS
from fleet_copilot.agent import build_graph, message_text, system_prompt, thread_config, tool_catalog
from fleet_copilot.tools import TOOLS
from fleet_copilot.tools_read import NOT_AVAILABLE

ACME = "acme-001"
INITECH_DEVICE = "I3JLKILKQTAC"
PLAN = AIMessage(content="1. Call fleet_insights.")


class ScriptedModel(BaseChatModel):
    """Returns the given AI messages in order and records what it was sent.

    bind_tools returns a copy marked bound that shares the same lists, so bound_calls
    records, for every call in order, whether tools were bound (True) or not (False).
    """

    script: list[AIMessage]
    seen: list[list] = []
    bound: bool = False
    bound_calls: list[bool] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"bound": True})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        self.bound_calls.append(self.bound)
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


def _run(script, company=ACME, question="question", plan=PLAN):
    """The first scripted reply is the planner's; the rest belong to the agent."""
    model = ScriptedModel(script=[plan, *script], seen=[], bound_calls=[])
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
    final, model = _run([_call("low_disk_devices", {}), AIMessage(content="Two devices.")])
    tool_messages = [m for m in final["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    rows = json.loads(tool_messages[0].content)["rows"]
    assert {r["device_id"] for r in rows} == {"8NM23J95R5I6", "MT7PJB7N5LRE"}
    assert final["messages"][-1].content == "Two devices."
    assert len(model.seen) == 3  # planner, agent, agent after the tool result
    assert _audit(audit_file)[0]["outcome"] == "ok"


def test_model_cannot_override_company_id(audit_file):
    # Even if the model sends company_id, the injected session value wins.
    final, _ = _run([_call("list_devices", {"company_id": "initech-003"}), AIMessage(content="ok")])
    result = json.loads(next(m for m in final["messages"] if isinstance(m, ToolMessage)).content)
    assert len(result["rows"]) == 10  # acme has 10 devices
    assert _audit(audit_file)[0]["company_id"] == ACME


def test_cross_tenant_device_is_refused_and_audited(audit_file):
    final, _ = _run([_call("get_device", {"device_id": INITECH_DEVICE}), AIMessage(content="Not found.")])
    content = json.loads(next(m for m in final["messages"] if isinstance(m, ToolMessage)).content)
    assert content == {"error": NOT_AVAILABLE}
    event = _audit(audit_file)[0]
    assert event["outcome"] == "cross_tenant_denied"
    assert event["args"] == {"device_id": INITECH_DEVICE}


def test_unknown_device_looks_the_same_to_the_model_but_not_in_the_audit(audit_file):
    final, _ = _run([_call("get_device", {"device_id": "NOPE"}), AIMessage(content="Not found.")])
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


COMPANIES = ["acme-001", "globex-002", "initech-003"]


@pytest.mark.parametrize("company", COMPANIES)
def test_system_prompt_names_only_its_own_company(company):
    prompt = system_prompt(company)
    assert company in prompt
    for other in COMPANIES:
        if other != company:
            assert other not in prompt


def test_model_is_told_which_company_it_serves():
    _, model = _run([AIMessage(content="ok")], company="globex-002")
    first = model.seen[1][0]
    assert first.type == "system"
    assert "globex-002" in first.content


def test_planner_runs_first_and_its_prompt_lists_every_tool():
    final, model = _run([AIMessage(content="ok")])
    planner_system = model.seen[0][0]
    assert planner_system.type == "system"
    assert "plan" in planner_system.content.lower()
    for tool in ALL_TOOLS:
        assert f"- {tool.name}: " in planner_system.content, tool.name
    # The plan steers the agent but is not part of the conversation.
    assert all(message_text(m) != PLAN.content for m in final["messages"])


def test_plan_is_the_text_of_the_planner_reply_only():
    plan = AIMessage(content=[{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "1. Call list_devices."}])
    final, _ = _run([AIMessage(content="ok")], plan=plan)
    assert final["plan"] == "1. Call list_devices."


def test_agent_system_message_carries_the_plan_and_the_company():
    plan = AIMessage(content=[{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "1. Call list_devices."}])
    _, model = _run([AIMessage(content="ok")], plan=plan)
    agent_system = model.seen[1][0].content
    assert "Plan for this question" in agent_system
    assert "1. Call list_devices." in agent_system
    assert "hmm" not in agent_system
    assert ACME in agent_system


def test_planner_call_is_unbound_and_agent_call_is_bound():
    _, model = _run([AIMessage(content="ok")])
    assert model.bound_calls == [False, True]


def test_tool_catalog_has_one_complete_line_per_tool():
    lines = tool_catalog().splitlines()
    assert len(lines) == len(ALL_TOOLS)
    assert not any(line.endswith(",") for line in lines)
    get_device_line = next(line for line in lines if line.startswith("- get_device: "))
    assert "compliance checks" in get_device_line


def test_planner_prompt_names_only_its_own_company():
    _, model = _run([AIMessage(content="ok")], company="globex-002")
    planner_system = model.seen[0][0].content
    assert "globex-002" in planner_system
    assert "acme-001" not in planner_system

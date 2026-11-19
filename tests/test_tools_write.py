import json

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from fleet_copilot.agent import build_graph, thread_config
from fleet_copilot.tools_read import NOT_AVAILABLE
from fleet_copilot.tools_write import create_ticket

ACME = "acme-001"
ACME_DEVICE = "8NM23J95R5I6"  # acme device, employee emp-acme-1005
ACME_EMPLOYEE = "emp-acme-1005"
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


def _ticket_call(device_id=ACME_DEVICE, **overrides):
    args = {
        "device_id": device_id,
        "title": "Liberar disco",
        "details": "Disco casi lleno. Snapshots: 8NM23J95R5I6@2026-06-12T09:02:00Z",
        "priority": "high",
        **overrides,
    }
    return _call("create_ticket", args)


@pytest.fixture(autouse=True)
def paths(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit.jsonl"
    outbox = tmp_path / "outbox"
    monkeypatch.setenv("FLEET_AUDIT_PATH", str(audit_path))
    monkeypatch.setenv("FLEET_OUTBOX_DIR", str(outbox))
    return audit_path, outbox / "tickets.jsonl"


def _audit(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _tickets(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _start(script, company=ACME, question="abre un ticket"):
    """Run until the graph finishes or pauses for approval. Returns (graph, config, result)."""
    model = ScriptedModel(script=script, seen=[])
    graph = build_graph(model=model, checkpointer=InMemorySaver())
    config = thread_config(company, "test")
    result = graph.invoke({"messages": [HumanMessage(question)], "company_id": company}, config)
    return graph, config, result


def _pending(graph, config):
    return [i for task in graph.get_state(config).tasks for i in task.interrupts]


def _tool_content(result):
    return json.loads(next(m for m in result["messages"] if isinstance(m, ToolMessage)).content)


def test_company_id_is_hidden_from_the_create_ticket_schema():
    schema = create_ticket.tool_call_schema.model_json_schema()
    assert "company_id" not in schema.get("properties", {})
    assert set(schema["required"]) == {"device_id", "title", "details", "priority"}


def test_approved_ticket_is_written_once_and_audited(paths):
    audit_path, tickets_path = paths
    script = [_ticket_call(), AIMessage(content="Ticket abierto.")]
    graph, config, _ = _start(script)

    interrupts = _pending(graph, config)
    assert len(interrupts) == 1
    preview = interrupts[0].value["ticket"]
    assert interrupts[0].value["action"] == "create_ticket"
    assert "ticket_id" not in preview and "created_at" not in preview
    assert not tickets_path.exists()  # nothing written before approval

    final = graph.invoke(Command(resume={"approved": True}), config)

    tickets = _tickets(tickets_path)
    assert len(tickets) == 1
    ticket = tickets[0]
    assert ticket["company_id"] == ACME
    assert ticket["employee_id"] == ACME_EMPLOYEE
    assert ticket["device_id"] == ACME_DEVICE
    assert ticket["priority"] == "high"
    assert len(ticket["ticket_id"]) == 32 and ticket["created_at"]

    content = _tool_content(final)
    assert content == {"status": "created", "ticket_id": ticket["ticket_id"]}
    assert final["messages"][-1].content == "Ticket abierto."

    events = _audit(audit_path)
    assert [e["outcome"] for e in events] == ["approved"]
    assert events[0]["company_id"] == ACME
    assert events[0]["ticket_id"] == ticket["ticket_id"]


def test_rejected_ticket_writes_nothing_and_says_so(paths):
    audit_path, tickets_path = paths
    graph, config, _ = _start([_ticket_call(), AIMessage(content="De acuerdo, no se abrió.")])
    assert _pending(graph, config)

    final = graph.invoke(Command(resume={"approved": False, "reason": "esperar al lunes"}), config)

    assert not tickets_path.exists()
    assert _tool_content(final) == {"status": "rejected", "reason": "esperar al lunes"}
    events = _audit(audit_path)
    assert [e["outcome"] for e in events] == ["rejected"]
    assert events[0]["reason"] == "esperar al lunes"


def test_cross_tenant_device_is_refused_without_interrupt(paths):
    audit_path, tickets_path = paths
    graph, config, final = _start(
        [_ticket_call(device_id=INITECH_DEVICE), AIMessage(content="No está en tu flota.")]
    )

    assert _pending(graph, config) == []
    assert _tool_content(final) == {"error": NOT_AVAILABLE}
    assert not tickets_path.exists()
    events = _audit(audit_path)
    assert [e["outcome"] for e in events] == ["cross_tenant_denied"]
    assert events[0]["args"]["device_id"] == INITECH_DEVICE


def test_unknown_device_is_refused_without_interrupt(paths):
    audit_path, tickets_path = paths
    graph, config, final = _start([_ticket_call(device_id="NOPE"), AIMessage(content="No está.")])

    assert _pending(graph, config) == []
    assert _tool_content(final) == {"error": NOT_AVAILABLE}
    assert not tickets_path.exists()
    assert [e["outcome"] for e in _audit(audit_path)] == ["not_found"]

"""The LangGraph agent: a model node and a tool node in a loop, scoped to one company."""

from __future__ import annotations

import os
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from fleet_copilot.tools import TOOLS as READ_TOOLS
from fleet_copilot.tools_write import create_ticket

TOOLS = [*READ_TOOLS, create_ticket]

DEFAULT_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """You work for the IT team of company {company_id}. Every tool already returns only this company's data. When the user names this company, they mean the whole fleet, not a device.

You are Fleet Copilot, an assistant for the IT team of one company. \
You answer questions about that company's laptops and desktops using device telemetry: \
30 days of snapshots per device covering OS, hardware, disk, memory, battery, installed \
software and compliance checks.

How to work:
- Answer only from tool results. If the tools do not return the data, say so plainly.
- For trends, risks or "what needs attention", start with fleet_insights. Use the other \
tools for specific lookups and to check details.
- Cite the snapshot_id values that support each claim, in square brackets after the claim.
- Read the notes of each tool result. They state thresholds, excluded devices and missing \
data; repeat any caveat that changes the answer, such as missing battery data.
- "GB" in tool results means GiB.
- If a tool says a device was not found, tell the user it is not in their fleet. Do not \
guess whether it exists elsewhere.
- Use create_ticket only when the user asks to open a ticket or clearly agrees to one. \
A human must approve it first. Never say a ticket was created unless the tool returned \
status "created".
- Reply in the language the user writes in. Keep answers short: lead with the answer, \
then the supporting devices.
"""


def system_prompt(company_id: str) -> str:
    """The system prompt for one company. Plain replace, so the prompt's own braces need no escaping."""
    return SYSTEM_PROMPT.replace("{company_id}", company_id)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    company_id: str


def default_model() -> BaseChatModel:
    """Claude through langchain-anthropic. FLEET_MODEL picks the model id."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=os.environ.get("FLEET_MODEL", DEFAULT_MODEL), max_tokens=16000)


def build_graph(model: BaseChatModel | None = None, checkpointer: BaseCheckpointSaver | None = None):
    """Compile the agent graph. Pass a model in tests; the default calls Claude."""
    llm = (model or default_model()).bind_tools(TOOLS)
    def call_model(state: State) -> dict[str, Any]:
        system = SystemMessage(system_prompt(state["company_id"]))
        return {"messages": [llm.invoke([system, *state["messages"]])]}

    builder = StateGraph(State)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


def thread_config(company_id: str, session_id: str) -> dict[str, Any]:
    """Thread ids include the company, so two tenants can never share a conversation."""
    return {"configurable": {"thread_id": f"{company_id}:{session_id}"}, "recursion_limit": 30}

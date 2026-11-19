"""The LangGraph agent: a model node and a tool node in a loop, scoped to one company."""

from __future__ import annotations

import os
import re
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from fleet_copilot.tools import TOOLS as READ_TOOLS
from fleet_copilot.tools_write import ACTION_TOOLS

TOOLS = [*READ_TOOLS, *ACTION_TOOLS]

DEFAULT_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """You are Fleet Copilot, an assistant for the IT team of company {company_id}. \
Every tool already returns only this company's data. When the user names this company, \
they mean the whole fleet, not a device.

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
- Action tools (open_remediation_ticket, create_upgrade_order, flag_device_for_replacement, \
notify_employee) change state and need human approval. Call one only when the user asks for \
an action or agrees to one you suggested; otherwise suggest the action in text. Always pass \
as evidence the snapshot_ids from tool results that justify it. If a tool returns status \
"refused", explain the reason; never retry with made-up evidence. Never say an action was \
done unless the tool returned status "created".
- Reply in the language the user writes in. Keep answers short: lead with the answer, \
then the supporting devices.
"""


PLANNER_PROMPT = """You plan how to answer a question from the IT team of company {company_id}. You do not answer the question.

Write a short numbered plan of at most 5 steps. Each step names the tool to call and why. \
Say what evidence the answer must cite (snapshot_id values from tool results).
If the question asks for an action (opening a ticket, ordering an upgrade, flagging a device, \
notifying an employee), include the read steps needed to gather evidence before the action tool. \
The action tool is proposed for human approval and does not run until a person approves it.
If no data is needed (a greeting, or a question outside device telemetry), write a single line: \
No tools needed.
Write only the plan. No answer, no preamble.

Available tools:
{tool_catalog}
"""


def tool_catalog() -> str:
    """One line per tool, built from the tool objects so the prompt cannot drift from the code.

    Each line is the description's first paragraph (up to a blank line) with its line breaks joined by spaces.
    """
    lines = []
    for tool in TOOLS:
        paragraph = re.split(r"\n\s*\n", tool.description.strip())[0]
        lines.append(f"- {tool.name}: {' '.join(paragraph.split())}")
    return "\n".join(lines)


def planner_prompt(company_id: str) -> str:
    """The planner's system prompt for one company. Plain replace, like system_prompt."""
    prompt = PLANNER_PROMPT.replace("{company_id}", company_id)
    return prompt.replace("{tool_catalog}", tool_catalog())


def system_prompt(company_id: str) -> str:
    """The system prompt for one company. Plain replace, so the prompt's own braces need no escaping."""
    return SYSTEM_PROMPT.replace("{company_id}", company_id)


def message_text(message: AnyMessage) -> str:
    """The text blocks of a message, ignoring thinking and tool-use blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text")


def _planning_history(messages: list[AnyMessage]) -> list[AnyMessage]:
    """The conversation as plain text. Tool-use and tool-result blocks from earlier turns are dropped,
    because the planner has no tools bound and the API rejects tool blocks without a tool list."""
    history: list[AnyMessage] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            history.append(message)
        elif isinstance(message, AIMessage):
            text = message_text(message)
            if text:
                history.append(AIMessage(content=text))
    return history


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    company_id: str
    plan: str


def default_model() -> BaseChatModel:
    """Claude through langchain-anthropic. FLEET_MODEL picks the model id."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=os.environ.get("FLEET_MODEL", DEFAULT_MODEL), max_tokens=16000)


def build_graph(model: BaseChatModel | None = None, checkpointer: BaseCheckpointSaver | None = None):
    """Compile the agent graph. Pass a model in tests; the default calls Claude."""
    base = model or default_model()
    llm = base.bind_tools(TOOLS)

    def plan_step(state: State) -> dict[str, Any]:
        """Plan with the model without tools bound, so the planner cannot call anything."""
        system = SystemMessage(planner_prompt(state["company_id"]))
        response = base.invoke([system, *_planning_history(state["messages"])])
        return {"plan": message_text(response).strip()}

    def call_model(state: State) -> dict[str, Any]:
        text = system_prompt(state["company_id"])
        plan = state.get("plan")
        if plan:
            text += "\n\nPlan for this question (follow it, adapt if a tool result changes what is needed):\n" + plan
        system = SystemMessage(text)
        return {"messages": [llm.invoke([system, *state["messages"]])]}

    builder = StateGraph(State)
    builder.add_node("planner", plan_step)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


def thread_config(company_id: str, session_id: str) -> dict[str, Any]:
    """Thread ids include the company, so two tenants can never share a conversation."""
    return {"configurable": {"thread_id": f"{company_id}:{session_id}"}, "recursion_limit": 30}

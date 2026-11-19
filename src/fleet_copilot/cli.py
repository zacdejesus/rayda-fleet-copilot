"""Interactive command line for the agent: fleet-copilot --company acme-001."""

from __future__ import annotations

import argparse
import sys
import uuid

from dotenv import load_dotenv

from fleet_copilot import audit
from fleet_copilot.data import load_snapshots


def _companies() -> list[str]:
    return sorted(load_snapshots()["company_id"].unique())


APPROVE_ANSWERS = {"s", "si", "sí", "y", "yes"}


def _review(payload: dict) -> dict:
    """Show a proposed write and ask the human. Returns the resume value for the graph."""
    proposal = payload["proposal"]
    print(f"\nAcción propuesta ({payload['action']}):", file=sys.stderr)
    for key, value in proposal.items():
        if key == "evidence":
            value = ", ".join(value)
        print(f"  {key}: {value}", file=sys.stderr)
    try:
        if input("¿Aprobar? [s/N] ").strip().lower() in APPROVE_ANSWERS:
            return {"approved": True}
        reason = input("Motivo (opcional): ").strip()
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return {"approved": False, "reason": "sin respuesta"}
    return {"approved": False, "reason": reason}


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="fleet-copilot", description="Ask questions about one company's device fleet.")
    parser.add_argument("--company", required=True, help="company_id the session is scoped to, e.g. acme-001")
    parser.add_argument("-q", "--question", help="ask one question and exit instead of starting a chat")
    args = parser.parse_args(argv)

    companies = _companies()
    if args.company not in companies:
        parser.error(f"unknown company {args.company!r}; known: {', '.join(companies)}")

    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from fleet_copilot.agent import build_graph, message_text, thread_config
    from fleet_copilot.citations import invalid_citations

    graph = build_graph(checkpointer=InMemorySaver())
    config = thread_config(args.company, uuid.uuid4().hex[:8])
    print(f"Fleet Copilot · {args.company} · audit log: {audit.audit_path()}", file=sys.stderr)

    def ask(question: str) -> None:
        state = {"messages": [HumanMessage(question)], "company_id": args.company}
        stream = graph.stream(state, config, stream_mode="updates")
        # Each pass ends either with the answer or with an approval request; a
        # resumed run can ask again, so keep going until no interrupt is left.
        while True:
            pending = None
            for update in stream:
                if "__interrupt__" in update:
                    pending = update["__interrupt__"]
                    continue
                _print_update(update)
            if pending is None:
                return
            # One answer per pending interrupt; a single one can be resumed as a plain value.
            answers = {item.id: _review(item.value) for item in pending}
            resume = next(iter(answers.values())) if len(answers) == 1 else answers
            stream = graph.stream(Command(resume=resume), config, stream_mode="updates")

    def _print_update(update: dict) -> None:
        for node, change in update.items():
            if node == "planner":
                print("Plan:", file=sys.stderr)
                for line in change["plan"].splitlines():
                    print(f"  {line}", file=sys.stderr)
                continue
            for message in change.get("messages", []):
                if node == "agent" and getattr(message, "tool_calls", None):
                    names = ", ".join(call["name"] for call in message.tool_calls)
                    print(f"  · {names}", file=sys.stderr)
                elif node == "agent":
                    text = message_text(message)
                    print(text)
                    _warn_invalid_citations(text)

    def _warn_invalid_citations(text: str) -> None:
        bad = invalid_citations(text, args.company)
        if not bad:
            return
        print(f"  ! Citas no verificables en esta respuesta: {', '.join(bad)}", file=sys.stderr)
        audit.record({"company_id": args.company, "tool": "citation_check", "outcome": "invalid_citations", "citations": bad})

    if args.question:
        ask(args.question)
        return

    print("Escribe una pregunta. Ctrl-D o 'salir' para terminar.", file=sys.stderr)
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return
        if question.lower() in {"salir", "exit", "quit"}:
            return
        if question:
            ask(question)

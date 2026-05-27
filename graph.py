"""
graph.py  ·  The Sokosumi LangGraph State Machine
───────────────────────────────────────────────────
Wires all five phases into a single StateGraph.

Flow:
  START
    │
    ▼
  [research]          Phase 1 · Tavily RAG
    │
    ▼
  [classifier]        Phase 2 · Intent classification
    │
    ├─ B2B_Sales    ──▶ [writer_b2b]
    ├─ Partnership  ──▶ [writer_partnership]
    ├─ Grant_Request──▶ [writer_grant]
    └─ Recruitment  ──▶ [writer_recruitment]
                              │
                              ▼
                       [hitl_checkpoint]   Phase 4 · PAUSE
                              │
                    ┌─────────┴──────────┐
                    │ approved            │ revision_requested
                    ▼                    ▼
              [send_email]         [writer_*]  (loop back)
                    │
                   END
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from nodes import (
    research_node,
    classifier_node,
    route_by_intent,
    writer_b2b_node,
    writer_partnership_node,
    writer_grant_node,
    writer_recruitment_node,
    hitl_checkpoint_node,
    route_after_hitl,
    send_email_node,
)


def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph StateGraph.
    Returns a compiled graph ready for .invoke() / .stream().
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("research",          research_node)
    builder.add_node("classifier",        classifier_node)
    builder.add_node("writer_b2b",        writer_b2b_node)
    builder.add_node("writer_partnership",writer_partnership_node)
    builder.add_node("writer_grant",      writer_grant_node)
    builder.add_node("writer_recruitment",writer_recruitment_node)
    builder.add_node("hitl_checkpoint",   hitl_checkpoint_node)
    builder.add_node("send_email",        send_email_node)

    # ── Linear edges ──────────────────────────────────────────────────────────
    builder.add_edge(START,        "research")
    builder.add_edge("research",   "classifier")

    # ── Conditional routing after classifier (Phase 2 → Phase 3) ─────────────
    builder.add_conditional_edges(
        "classifier",
        route_by_intent,
        {
            "writer_b2b":         "writer_b2b",
            "writer_partnership": "writer_partnership",
            "writer_grant":       "writer_grant",
            "writer_recruitment": "writer_recruitment",
        },
    )

    # ── All writers converge on the HITL checkpoint ───────────────────────────
    for writer in ["writer_b2b", "writer_partnership", "writer_grant", "writer_recruitment"]:
        builder.add_edge(writer, "hitl_checkpoint")

    # ── Conditional routing after HITL (Phase 4 → Phase 3 loop OR Phase 5) ───
    builder.add_conditional_edges(
        "hitl_checkpoint",
        route_after_hitl,
        {
            "writer_b2b":         "writer_b2b",
            "writer_partnership": "writer_partnership",
            "writer_grant":       "writer_grant",
            "writer_recruitment": "writer_recruitment",
            "send_email":         "send_email",
            "hitl_checkpoint":    "hitl_checkpoint",   # safety catch
        },
    )

    # ── Terminal edge ─────────────────────────────────────────────────────────
    builder.add_edge("send_email", END)

    # ── Compile with in-memory checkpointer (enables interrupt + resume) ──────
    memory = MemorySaver()
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["hitl_checkpoint"],   # pause BEFORE entering HITL node
    )


# Singleton compiled graph — imported by FastAPI
graph = build_graph()
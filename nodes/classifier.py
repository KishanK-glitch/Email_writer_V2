"""
nodes/classifier.py  ·  Phase 2 — Intent Classification
──────────────────────────────────────────────────────────
Forces the LLM to emit EXACTLY one of the four intent enums.
The graph reads `state.intent` to route to the correct writer node.
This is where V1 failed — we fix it with a strict retry loop.
"""

import os
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState, IntentType

logger = logging.getLogger(__name__)

VALID_INTENTS: list[IntentType] = ["B2B_Sales", "Partnership", "Grant_Request", "Recruitment"]
MAX_RETRIES = 3

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.environ["GROQ_API_KEY"],
        )
    return _llm


CLASSIFIER_SYSTEM = """You are an intent classifier for outreach emails.

Read the user's offering description and output EXACTLY ONE of these four labels:
  B2B_Sales        → User is selling a product, service, or SaaS tool to the company
  Partnership      → User wants a co-marketing, integration, or strategic partnership
  Grant_Request    → User is requesting funding, a grant, or a sponsorship
  Recruitment      → User is offering a job, looking to hire, or proposing a freelance engagement

Rules:
- Output ONLY the label. No punctuation, no explanation, no other text.
- If ambiguous, pick the closest match. Never refuse.
- Valid outputs: B2B_Sales | Partnership | Grant_Request | Recruitment"""


def classify_intent(user_offering: str, company_dna: str) -> IntentType:
    """
    Call the LLM classifier with retry logic.
    Raises ValueError after MAX_RETRIES if we never get a clean enum.
    """
    prompt = (
        f"Company DNA (context only — do NOT classify the company):\n{company_dna}\n\n"
        f"User's offering:\n{user_offering}\n\n"
        f"Output the single intent label:"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        response = _get_llm().invoke([
            SystemMessage(content=CLASSIFIER_SYSTEM),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip().strip('"').strip("'")

        if raw in VALID_INTENTS:
            logger.info(f"[Phase 2] Classified as '{raw}' (attempt {attempt})")
            return raw  # type: ignore[return-value]

        logger.warning(
            f"[Phase 2] Bad classifier output on attempt {attempt}: '{raw}'. Retrying…"
        )

    # Last resort: try a forced re-prompt
    raise ValueError(
        f"Classifier failed after {MAX_RETRIES} attempts. "
        f"Last output: '{raw}'. Valid intents: {VALID_INTENTS}"
    )


# ── Main node function ────────────────────────────────────────────────────────

def classifier_node(state: AgentState) -> dict:
    """
    LangGraph node — Phase 2.
    Inputs:  state.user_offering, state.company_dna
    Outputs: state.intent
    """
    logger.info("[Phase 2] Running intent classifier…")
    try:
        intent = classify_intent(
            user_offering=state["user_offering"],
            company_dna=state["company_dna"] or "No company data available.",
        )
        return {"intent": intent}
    except ValueError as e:
        logger.error(f"[Phase 2] Classification failed: {e}")
        # Default to B2B_Sales rather than crashing the graph
        return {"intent": "B2B_Sales", "error": str(e)}


# ── Router function (called by LangGraph's conditional_edges) ─────────────────

def route_by_intent(state: AgentState) -> str:
    """
    Returns the name of the next node based on classified intent.
    This function is passed to `add_conditional_edges`.
    """
    intent_to_node = {
        "B2B_Sales":    "writer_b2b",
        "Partnership":  "writer_partnership",
        "Grant_Request":"writer_grant",
        "Recruitment":  "writer_recruitment",
    }
    intent = state.get("intent", "B2B_Sales")
    next_node = intent_to_node.get(intent, "writer_b2b")
    logger.info(f"[Router] intent='{intent}' → node='{next_node}'")
    return next_node
"""
nodes/writers.py  ·  Phase 3 — Specialised Email Generation
──────────────────────────────────────────────────────────────
Four distinct writer nodes, one per intent. Each:
  1. Receives the Company DNA from Phase 1
  2. Uses a intent-specific system prompt / template
  3. GUARANTEES the first sentence is a personalised hook
     (drawn from the DNA's PERSONALIZATION_HOOK field)
  4. Respects user_feedback if this is a revision loop (revision_count > 0)
"""

import os
import logging
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState

logger = logging.getLogger(__name__)

MAX_REVISIONS = 5   # Safety cap — after this we force-approve

_llm: ChatGroq | None = None


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.environ["GROQ_API_KEY"],
        )
    return _llm


# ══════════════════════════════════════════════════════════════════════════════
# SHARED BASE WRITER
# ══════════════════════════════════════════════════════════════════════════════

BASE_SYSTEM = """You are an expert cold-email copywriter.
Your output is ONLY the email body (no subject line, no "Here is your email:" preamble).

MANDATORY RULES:
1. FIRST SENTENCE must be a highly specific, researched hook built from the
   PERSONALIZATION_HOOK field in the Company DNA. It must feel like the sender
   spent time on the company — not generic flattery.
2. Keep it under 150 words total. Concise wins.
3. ONE clear call-to-action at the end. Never more than one.
4. Never mention a price or number unless the user explicitly provides one.
5. Sign off with the sender's name.
6. Write in plain text — no markdown, no bullet points, no headers."""


SCRAPE_FAILED_PREFIX = "SCRAPE_FAILED:"


def _build_user_prompt(
    state: AgentState,
    intent_instruction: str,
) -> str:
    """
    Builds the human turn for any writer node.
    Detects the SCRAPE_FAILED sentinel from Phase 1 and switches to
    a no-research mode so writers never fabricate company details.
    """
    company_dna = state["company_dna"] or ""
    scrape_failed = company_dna.startswith(SCRAPE_FAILED_PREFIX)

    revision_context = ""
    if state["revision_count"] > 0 and state.get("user_feedback"):
        revision_context = (
            f"\n\n── REVISION REQUEST ──\n"
            f"Previous draft:\n{state['current_draft']}\n\n"
            f"User feedback: {state['user_feedback']}\n"
            f"Apply this feedback carefully. Keep what worked, fix what was criticised."
        )

    if scrape_failed:
        research_block = (
            "RESEARCH UNAVAILABLE\n"
            "The automated scrape of the target company returned no usable data.\n"
            "You have ZERO verified facts about this company.\n\n"
            "STRICT RULES FOR THIS EMAIL:\n"
            "- Do NOT fabricate or assume any company details, products, or news.\n"
            "- Do NOT use generic flattery as a substitute hook.\n"
            "- Open cold, without a specific hook. Lead with the value of the offering instead."
        )
    else:
        research_block = f"Company DNA (verified research):\n{company_dna}"

    return f"""{research_block}

Sender:         {state['user_name']} <{state['user_email']}>
Their offering: {state['user_offering']}
Recipient:      {state['target_email']}

Intent instruction:
{intent_instruction}
{revision_context}

Write the email now:"""


def _write(state: AgentState, intent_instruction: str) -> dict:
    """Shared writer invocation. Returns state patches."""
    logger.info(
        f"[Phase 3] Writing draft "
        f"(intent={state['intent']}, revision={state['revision_count']})"
    )

    messages = [
        SystemMessage(content=BASE_SYSTEM),
        HumanMessage(content=_build_user_prompt(state, intent_instruction)),
    ]

    response = _get_llm().invoke(messages)
    draft = response.content.strip()

    return {
        "current_draft": draft,
        "draft_history": [draft],   # appended via operator.add in state
        "hitl_status": "awaiting_input",
        "user_feedback": None,       # clear feedback after each write
    }


# ══════════════════════════════════════════════════════════════════════════════
# INTENT-SPECIFIC WRITER NODES
# ══════════════════════════════════════════════════════════════════════════════

def writer_b2b_node(state: AgentState) -> dict:
    """Phase 3 · B2B_Sales writer."""
    instruction = (
        "This is a B2B sales outreach email. "
        "Focus on the business problem the sender's product solves for THIS company specifically. "
        "Tie the value proposition directly to what you see in their Company DNA. "
        "The CTA should be a low-friction ask: a 20-minute call, a quick demo, or a reply."
    )
    return _write(state, instruction)


def writer_partnership_node(state: AgentState) -> dict:
    """Phase 3 · Partnership writer."""
    instruction = (
        "This is a partnership proposal email. "
        "Frame the collaboration as a win-win — explain what the sender brings AND "
        "what the target company gains (not just exposure, but concrete value). "
        "Acknowledge their current objectives from the DNA. "
        "The CTA: a quick exploratory call to see if there's a fit."
    )
    return _write(state, instruction)


def writer_grant_node(state: AgentState) -> dict:
    """Phase 3 · Grant_Request writer."""
    instruction = (
        "This is a grant or sponsorship request email. "
        "The tone should be respectful, mission-aligned, and professional — NOT salesy. "
        "Connect the sender's mission to the company's stated values or CSR goals from the DNA. "
        "Be transparent about what the funding/sponsorship would be used for. "
        "The CTA: request a brief conversation or ask where to submit a formal proposal."
    )
    return _write(state, instruction)


def writer_recruitment_node(state: AgentState) -> dict:
    """Phase 3 · Recruitment writer."""
    instruction = (
        "This is a recruitment or hiring outreach email. "
        "If the sender is offering a role: highlight why this company is exciting, "
        "tie it to the candidate's likely interests, and make the opportunity feel unique. "
        "If the sender IS the company: make the opportunity feel compelling and specific. "
        "The CTA: a quick chat to explore fit, no pressure."
    )
    return _write(state, instruction)


# ── Revision router (called after HITL sends feedback back) ──────────────────

def route_after_hitl(state: AgentState) -> str:
    """
    After the HITL checkpoint, decide what happens next.
    Called by add_conditional_edges from the hitl_checkpoint node.
    """
    status = state.get("hitl_status", "awaiting_input")
    revision_count = state.get("revision_count", 0)

    if status == "approved":
        logger.info("[Router] HITL approved → send_email")
        return "send_email"

    if status == "revision_requested":
        if revision_count >= MAX_REVISIONS:
            logger.warning(f"[Router] Max revisions ({MAX_REVISIONS}) hit → send_email")
            return "send_email"
        # Route back to the correct intent writer
        intent_to_node = {
            "B2B_Sales":    "writer_b2b",
            "Partnership":  "writer_partnership",
            "Grant_Request":"writer_grant",
            "Recruitment":  "writer_recruitment",
        }
        intent = state.get("intent", "B2B_Sales")
        return intent_to_node.get(intent, "writer_b2b")

    # Still awaiting — graph should be paused; this branch shouldn't fire
    return "hitl_checkpoint"
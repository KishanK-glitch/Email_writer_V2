"""
state.py
────────
Central state schema for the Sokosumi outreach agent.
Every node reads from and writes to this TypedDict.
LangGraph's StateGraph propagates it automatically.
"""

from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
import operator


# ── Strict intent enum ────────────────────────────────────────────────────────
IntentType = Literal["B2B_Sales", "Partnership", "Grant_Request", "Recruitment"]

# ── HITL status enum ──────────────────────────────────────────────────────────
HITLStatus = Literal["pending", "awaiting_input", "approved", "revision_requested"]


class AgentState(TypedDict):
    # ── Inputs (set at job start, never mutated) ──────────────────────────────
    job_id: str                        # Unique ID for this outreach job
    target_url: str                    # URL of the company being targeted
    user_offering: str                 # What the user is pitching / offering
    user_name: str                     # Sender's name
    user_email: str                    # Sender's email address
    target_email: str                  # Recipient's email address

    # ── Phase 1 · Research ────────────────────────────────────────────────────
    company_dna: Optional[str]         # Synthesized "Company DNA" block

    # ── Phase 2 · Classification ──────────────────────────────────────────────
    intent: Optional[IntentType]       # Classified email intent

    # ── Phase 3 · Generation ──────────────────────────────────────────────────
    draft_history: Annotated[          # Every draft ever written (append-only)
        list[str], operator.add
    ]
    current_draft: Optional[str]       # The latest draft in flight

    # ── Phase 4 · HITL ───────────────────────────────────────────────────────
    hitl_status: HITLStatus            # Current HITL gate status
    user_feedback: Optional[str]       # Feedback text from UI ("Make it shorter")
    revision_count: int                # Guard against infinite rewrite loops

    # ── Phase 5 · Execution ───────────────────────────────────────────────────
    final_email: Optional[str]         # Approved copy, ready to send
    send_result: Optional[dict]        # Response payload from Resend

    # ── Meta ──────────────────────────────────────────────────────────────────
    error: Optional[str]               # Any fatal error message


def initial_state(
    job_id: str,
    target_url: str,
    user_offering: str,
    user_name: str,
    user_email: str,
    target_email: str,
) -> AgentState:
    """Factory — builds a clean state for a new job."""
    return AgentState(
        job_id=job_id,
        target_url=target_url,
        user_offering=user_offering,
        user_name=user_name,
        user_email=user_email,
        target_email=target_email,
        company_dna=None,
        intent=None,
        draft_history=[],
        current_draft=None,
        hitl_status="pending",
        user_feedback=None,
        revision_count=0,
        final_email=None,
        send_result=None,
        error=None,
    )
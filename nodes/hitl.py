"""
nodes/hitl.py  ·  Phase 4 — Masumi HITL Checkpoint
────────────────────────────────────────────────────
This node PAUSES the graph and waits for human input.

Architecture:
  - When invoked, it sets hitl_status = "awaiting_input"
  - LangGraph's interrupt() is called, suspending execution
  - The FastAPI layer catches the interrupt and exposes the draft via REST
  - The user reviews in the UI, then calls POST /jobs/{id}/review
  - That endpoint resumes the graph with either:
      • {"action": "approve"}
      • {"action": "revise", "feedback": "Make it shorter"}

Masumi integration is behind a clean MasumiClient interface.
Swap _masumi_stub for the real implementation when credentials arrive.
"""

import os
import logging
from typing import Optional
import httpx
from langgraph.types import interrupt

from state import AgentState

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# MASUMI CLIENT — STUB (swap this class for real implementation)
# ══════════════════════════════════════════════════════════════════════════════

class MasumiClientStub:
    """
    Clean interface matching the Masumi HITL API contract.
    Replace the method bodies when you have real credentials.

    Expected Masumi API contract (based on docs Sarthi shared):
      POST /jobs/{job_id}/status  → update job status
      POST /jobs/{job_id}/input   → submit human input decision
    """

    def __init__(self):
        self.api_url = os.environ.get("MASUMI_API_URL", "http://localhost:9000")
        self.api_key = os.environ.get("MASUMI_API_KEY", "stub-key")
        self.enabled = self.api_url != "http://localhost:9000"

    def notify_awaiting_input(self, job_id: str, draft: str, revision_count: int) -> bool:
        """
        Tell Masumi this job is paused and needs human review.
        Returns True on success.

        STUB: logs only. Replace body with real httpx call:

            response = httpx.post(
                f"{self.api_url}/jobs/{job_id}/status",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "status": "awaiting_input",
                    "payload": {"draft": draft, "revision_count": revision_count}
                },
                timeout=10,
            )
            return response.status_code == 200
        """
        if self.enabled:
            try:
                response = httpx.post(
                    f"{self.api_url}/jobs/{job_id}/status",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "status": "awaiting_input",
                        "payload": {"draft": draft, "revision_count": revision_count}
                    },
                    timeout=10,
                )
                ok = response.status_code == 200
                if not ok:
                    logger.warning(f"Masumi notify failed: {response.status_code} {response.text}")
                return ok
            except Exception as e:
                logger.warning(f"Masumi notify error: {e}")
                return False
        else:
            # STUB path
            logger.info(
                f"[Masumi·STUB] notify_awaiting_input "
                f"job_id={job_id} revision={revision_count} "
                f"(real Masumi call skipped — MASUMI_API_URL not configured)"
            )
            return True

    def notify_completed(self, job_id: str, final_email: str, send_result: dict) -> bool:
        """
        Tell Masumi the job is done and the email was sent.

        STUB: logs only. Replace body with real httpx call.
        """
        if self.enabled:
            try:
                response = httpx.post(
                    f"{self.api_url}/jobs/{job_id}/status",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "status": "completed",
                        "payload": {"final_email": final_email, "send_result": send_result}
                    },
                    timeout=10,
                )
                return response.status_code == 200
            except Exception as e:
                logger.warning(f"Masumi complete error: {e}")
                return False
        else:
            logger.info(
                f"[Masumi·STUB] notify_completed "
                f"job_id={job_id} "
                f"(real Masumi call skipped — MASUMI_API_URL not configured)"
            )
            return True


# Singleton
_masumi = MasumiClientStub()


# ══════════════════════════════════════════════════════════════════════════════
# HITL NODE
# ══════════════════════════════════════════════════════════════════════════════

def hitl_checkpoint_node(state: AgentState) -> dict:
    """
    LangGraph node — Phase 4.
    Pauses the graph via interrupt() and waits for human decision.

    On resume, the graph runner injects the human decision into state
    before calling this node again — but because we use interrupt(),
    the node is NOT called again. LangGraph resumes at the NEXT node
    determined by route_after_hitl().

    Inputs:  state.current_draft, state.job_id
    Outputs: state.hitl_status updated to "awaiting_input"
             (then resumed by FastAPI with "approved" or "revision_requested")
    """
    logger.info(f"[Phase 4] HITL checkpoint — job_id={state['job_id']}")

    draft = state["current_draft"]
    job_id = state["job_id"]
    revision_count = state.get("revision_count", 0)

    # Notify Masumi (real or stub)
    _masumi.notify_awaiting_input(job_id, draft, revision_count)

    # ── PAUSE EXECUTION HERE ──────────────────────────────────────────────────
    # interrupt() raises a special LangGraph exception that suspends the graph.
    # The FastAPI server catches this and exposes the draft via REST API.
    # When the user acts (approve / revise), the server calls graph.update_state()
    # and graph.stream(None, ...) to resume.
    human_decision = interrupt({
        "job_id": job_id,
        "current_draft": draft,
        "revision_count": revision_count,
        "message": "Review the draft and either approve or provide revision feedback.",
    })
    # ── RESUMED ──────────────────────────────────────────────────────────────
    # human_decision is the dict passed to graph.update_state() by the API layer
    # e.g. {"action": "approve"} or {"action": "revise", "feedback": "Shorter"}

    action = human_decision.get("action", "approve")

    if action == "approve":
        logger.info(f"[Phase 4] Approved by user — job_id={job_id}")
        return {
            "hitl_status": "approved",
            "final_email": draft,
        }

    elif action == "revise":
        feedback = human_decision.get("feedback", "")
        logger.info(f"[Phase 4] Revision requested: '{feedback}'")
        return {
            "hitl_status": "revision_requested",
            "user_feedback": feedback,
            "revision_count": state.get("revision_count", 0) + 1,
        }

    # Fallback
    logger.warning(f"[Phase 4] Unknown action '{action}' — treating as approve")
    return {"hitl_status": "approved", "final_email": draft}
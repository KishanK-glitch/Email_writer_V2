"""
main.py  ·  Sokosumi Outreach Agent — FastAPI Backend
───────────────────────────────────────────────────────
Endpoints:
  POST /jobs              → Start a new outreach job
  GET  /jobs/{id}         → Poll job status + current draft
  POST /jobs/{id}/review  → Submit human decision (approve / revise)
  GET  /jobs              → List all jobs

Test from terminal (PowerShell):
  $body = @{
      target_url    = "https://stripe.com"
      user_offering = "We built an AI fraud detection layer for payment processors"
      user_name     = "Alex Chen"
      user_email    = "alex@yourcompany.com"
      target_email  = "partnerships@stripe.com"
  } | ConvertTo-Json

  Invoke-RestMethod -Uri http://localhost:8000/jobs -Method POST -Body $body -ContentType "application/json"

  # Poll status:
  Invoke-RestMethod -Uri http://localhost:8000/jobs/<job_id> -Method GET

  # Approve:
  Invoke-RestMethod -Uri http://localhost:8000/jobs/<job_id>/review -Method POST `
    -Body '{"action":"approve"}' -ContentType "application/json"

  # Revise:
  Invoke-RestMethod -Uri http://localhost:8000/jobs/<job_id>/review -Method POST `
    -Body '{"action":"revise","feedback":"Make it shorter and more direct"}' `
    -ContentType "application/json"
"""

import os
import uuid
import logging
import threading
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

from state import initial_state, AgentState
from graph import graph


# ══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY JOB STORE
# (Replace with Redis or Postgres for production)
# ══════════════════════════════════════════════════════════════════════════════

jobs: dict[str, dict] = {}   # job_id → {"state": AgentState, "thread_id": str, "phase": str}
jobs_lock = threading.Lock()


def _get_job(job_id: str) -> dict:
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
        return jobs[job_id]


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class StartJobRequest(BaseModel):
    target_url: str
    user_offering: str
    user_name: str
    user_email: str
    target_email: str


class ReviewRequest(BaseModel):
    action: str                    # "approve" | "revise"
    feedback: Optional[str] = None # required when action == "revise"


class JobStatusResponse(BaseModel):
    job_id: str
    phase: str
    hitl_status: str
    intent: Optional[str]
    current_draft: Optional[str]
    revision_count: int
    send_result: Optional[dict]
    error: Optional[str]


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND GRAPH RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def _run_graph_until_interrupt(job_id: str, state: AgentState, thread_id: str):
    """
    Runs the LangGraph graph in a background thread until it either:
      a) Hits the HITL interrupt (pauses at hitl_checkpoint)
      b) Completes naturally (send_email → END)
    Updates the jobs store with latest state after each step.
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        logger.info(f"[Graph] Starting run — job_id={job_id}")
        with jobs_lock:
            jobs[job_id]["phase"] = "running"

        for chunk in graph.stream(state, config=config, stream_mode="values"):
            # chunk is the full state after each node completes
            with jobs_lock:
                jobs[job_id]["state"] = chunk
                jobs[job_id]["phase"] = _infer_phase(chunk)
            logger.info(
                f"[Graph] Checkpoint — job={job_id} "
                f"phase={jobs[job_id]['phase']} "
                f"hitl_status={chunk.get('hitl_status')}"
            )

        # Stream ended — either interrupted or completed
        final_state = graph.get_state(config).values
        with jobs_lock:
            jobs[job_id]["state"] = final_state
            hitl = final_state.get("hitl_status", "pending")
            if hitl == "awaiting_input":
                jobs[job_id]["phase"] = "awaiting_human_review"
            elif final_state.get("send_result"):
                jobs[job_id]["phase"] = "completed"
            else:
                jobs[job_id]["phase"] = "paused"

        logger.info(f"[Graph] Stream ended — job={job_id} phase={jobs[job_id]['phase']}")

    except Exception as e:
        logger.error(f"[Graph] Error in job {job_id}: {e}", exc_info=True)
        with jobs_lock:
            jobs[job_id]["phase"] = "error"
            jobs[job_id]["state"]["error"] = str(e)


def _resume_graph(job_id: str, thread_id: str, human_input: dict):
    """
    Resumes a paused graph after the user submits a review decision.
    Injects human_input via update_state, then re-streams.
    """
    config = {"configurable": {"thread_id": thread_id}}

    try:
        logger.info(f"[Graph] Resuming — job_id={job_id} action={human_input.get('action')}")
        with jobs_lock:
            jobs[job_id]["phase"] = "running"

        # Inject the human decision into the graph state
        action = human_input.get("action")
        if action == "approve":
            current = graph.get_state(config).values
            draft = current.get("current_draft", "")
            graph.update_state(
                config,
                {"hitl_status": "approved", "final_email": draft},
                as_node="hitl_checkpoint",
            )
        elif action == "revise":
            feedback = human_input.get("feedback", "")
            current = graph.get_state(config).values
            graph.update_state(
                config,
                {
                    "hitl_status": "revision_requested",
                    "user_feedback": feedback,
                    "revision_count": current.get("revision_count", 0) + 1,
                },
                as_node="hitl_checkpoint",
            )

        # Resume streaming from where we paused
        for chunk in graph.stream(None, config=config, stream_mode="values"):
            with jobs_lock:
                jobs[job_id]["state"] = chunk
                jobs[job_id]["phase"] = _infer_phase(chunk)
            logger.info(f"[Graph] Resume chunk — job={job_id} phase={jobs[job_id]['phase']}")

        # Post-resume state
        final_state = graph.get_state(config).values
        with jobs_lock:
            jobs[job_id]["state"] = final_state
            hitl = final_state.get("hitl_status", "pending")
            if hitl == "awaiting_input":
                jobs[job_id]["phase"] = "awaiting_human_review"
            elif final_state.get("send_result"):
                jobs[job_id]["phase"] = "completed"
            else:
                jobs[job_id]["phase"] = "paused"

        logger.info(f"[Graph] Resume complete — job={job_id} phase={jobs[job_id]['phase']}")

    except Exception as e:
        logger.error(f"[Graph] Resume error job={job_id}: {e}", exc_info=True)
        with jobs_lock:
            jobs[job_id]["phase"] = "error"
            jobs[job_id].setdefault("state", {})["error"] = str(e)


def _infer_phase(state: dict) -> str:
    """Derive a human-readable phase label from state."""
    if state.get("send_result"):
        return "completed"
    if state.get("hitl_status") == "awaiting_input":
        return "awaiting_human_review"
    if state.get("current_draft"):
        return "draft_ready"
    if state.get("intent"):
        return "writing"
    if state.get("company_dna"):
        return "classifying"
    return "researching"


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Sokosumi Outreach Agent starting up…")
    yield
    logger.info("Sokosumi Outreach Agent shutting down.")


app = FastAPI(
    title="Sokosumi Outreach Agent",
    description="5-phase LangGraph agent: Research → Classify → Write → HITL → Send",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── POST /jobs ─────────────────────────────────────────────────────────────────
@app.post("/jobs", status_code=202)
async def start_job(req: StartJobRequest, background_tasks: BackgroundTasks):
    """
    Kick off a new outreach job.
    Returns immediately with a job_id. The graph runs in the background.
    Poll GET /jobs/{job_id} for status.
    """
    job_id   = str(uuid.uuid4())
    thread_id = f"thread-{job_id}"

    state = initial_state(
        job_id        = job_id,
        target_url    = req.target_url,
        user_offering = req.user_offering,
        user_name     = req.user_name,
        user_email    = req.user_email,
        target_email  = req.target_email,
    )

    with jobs_lock:
        jobs[job_id] = {
            "state":     state,
            "thread_id": thread_id,
            "phase":     "queued",
        }

    background_tasks.add_task(
        _run_graph_until_interrupt, job_id, state, thread_id
    )

    logger.info(f"[API] Job created — job_id={job_id} target={req.target_url}")

    return {
        "job_id":   job_id,
        "status":   "queued",
        "message":  "Job started. Poll GET /jobs/{job_id} for status.",
        "poll_url": f"/jobs/{job_id}",
    }


# ── GET /jobs/{job_id} ────────────────────────────────────────────────────────
@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    """Poll job status, current draft, and send result."""
    job = _get_job(job_id)
    state = job.get("state", {})

    return JobStatusResponse(
        job_id        = job_id,
        phase         = job.get("phase", "unknown"),
        hitl_status   = state.get("hitl_status", "pending"),
        intent        = state.get("intent"),
        current_draft = state.get("current_draft"),
        revision_count= state.get("revision_count", 0),
        send_result   = state.get("send_result"),
        error         = state.get("error"),
    )


# ── POST /jobs/{job_id}/review ────────────────────────────────────────────────
@app.post("/jobs/{job_id}/review", status_code=202)
async def review_job(job_id: str, req: ReviewRequest, background_tasks: BackgroundTasks):
    """
    Submit human decision to resume a paused graph.

    Body:
      {"action": "approve"}
      {"action": "revise", "feedback": "Make it shorter and more direct"}
    """
    job = _get_job(job_id)

    if job["phase"] not in ("awaiting_human_review", "draft_ready", "paused"):
        raise HTTPException(
            status_code=409,
            detail=f"Job is in phase '{job['phase']}', not awaiting review.",
        )

    if req.action not in ("approve", "revise"):
        raise HTTPException(
            status_code=422,
            detail="action must be 'approve' or 'revise'",
        )

    if req.action == "revise" and not req.feedback:
        raise HTTPException(
            status_code=422,
            detail="feedback is required when action is 'revise'",
        )

    human_input = {"action": req.action, "feedback": req.feedback or ""}

    with jobs_lock:
        jobs[job_id]["phase"] = "resuming"

    background_tasks.add_task(
        _resume_graph, job_id, job["thread_id"], human_input
    )

    logger.info(f"[API] Review submitted — job_id={job_id} action={req.action}")

    return {
        "job_id":  job_id,
        "action":  req.action,
        "message": "Decision received. Graph resuming. Poll GET /jobs/{job_id} for status.",
    }


# ── GET /jobs ─────────────────────────────────────────────────────────────────
@app.get("/jobs")
async def list_jobs():
    """List all jobs and their current phase."""
    with jobs_lock:
        return [
            {
                "job_id": jid,
                "phase":  j.get("phase"),
                "intent": j.get("state", {}).get("intent"),
                "target": j.get("state", {}).get("target_url"),
            }
            for jid, j in jobs.items()
        ]


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.environ.get("APP_HOST", "0.0.0.0"),
        port=int(os.environ.get("APP_PORT", 8000)),
        reload=False,
        log_level="info",
    )
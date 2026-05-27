"""
nodes/sender.py  ·  Phase 5 — Automated Execution via Gmail API
────────────────────────────────────────────────────────────────
Sends the approved email through the Gmail API using OAuth 2.0.

Auth flow (two modes):
  A) Service Account (recommended for servers / CI)
     Set GMAIL_SERVICE_ACCOUNT_JSON to the path of your service account key file.
     Grant the service account "Send as" permission in Google Workspace.

  B) OAuth 2.0 (recommended for local dev)
     Set GMAIL_CREDENTIALS_JSON to the path of your OAuth client secret file.
     On first run, a browser window opens for consent; token is cached to
     GMAIL_TOKEN_JSON (default: gmail_token.json).

  C) Mock mode (no keys set)
     Logs the email to console. Safe default for testing the state machine.

Gmail API scopes required:
  https://www.googleapis.com/auth/gmail.send
"""

import os
import base64
import logging
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from state import AgentState

logger = logging.getLogger(__name__)

# ── Lazy import guards (Gmail deps optional until Phase 5 is wired up) ────────
def _build_gmail_service():
    """
    Returns an authenticated Gmail API service object.
    Tries Service Account first, then OAuth, then raises.
    """
    sa_json_path = os.environ.get("GMAIL_SERVICE_ACCOUNT_JSON", "")
    oauth_creds_path = os.environ.get("GMAIL_CREDENTIALS_JSON", "")

    if sa_json_path and os.path.exists(sa_json_path):
        return _service_account_service(sa_json_path)
    elif oauth_creds_path and os.path.exists(oauth_creds_path):
        return _oauth_service(oauth_creds_path)
    else:
        raise EnvironmentError(
            "No Gmail credentials found. Set either:\n"
            "  GMAIL_SERVICE_ACCOUNT_JSON=/path/to/service-account.json\n"
            "  GMAIL_CREDENTIALS_JSON=/path/to/oauth-client-secret.json"
        )


def _service_account_service(sa_json_path: str):
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
    sender_email = os.environ.get("GMAIL_SENDER_EMAIL", "")
    if not sender_email:
        raise EnvironmentError(
            "GMAIL_SENDER_EMAIL must be set when using a service account "
            "(must match the address that delegated domain-wide authority)."
        )

    creds = service_account.Credentials.from_service_account_file(
        sa_json_path, scopes=SCOPES
    )
    # Impersonate the sender address (requires domain-wide delegation in GWS)
    delegated = creds.with_subject(sender_email)
    return build("gmail", "v1", credentials=delegated, cache_discovery=False)


def _oauth_service(creds_path: str):
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
    token_path = os.environ.get("GMAIL_TOKEN_JSON", "gmail_token.json")
    creds = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            # run_local_server opens browser for consent on first run
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as f:
            f.write(creds.to_json())
        logger.info(f"[Gmail] OAuth token cached to {token_path}")

    return build("gmail", "v1", credentials=creds, cache_discovery=False)


# ── Email builder ─────────────────────────────────────────────────────────────

def _build_mime_message(
    from_email: str,
    from_name: str,
    to_email: str,
    subject: str,
    body_text: str,
) -> dict:
    """
    Build a base64url-encoded RFC 2822 message for the Gmail API.
    Returns the dict Gmail expects: {"raw": "<base64url string>"}
    """
    msg = MIMEMultipart("alternative")
    msg["From"]    = f"{from_name} <{from_email}>"
    msg["To"]      = to_email
    msg["Subject"] = subject

    # Plain text part (primary — keeps deliverability high)
    msg.attach(MIMEText(body_text, "plain", "utf-8"))

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    return {"raw": raw}


def _extract_subject(email_body: str, intent: str) -> str:
    """Derive a subject line from the first sentence of the draft."""
    lines = [l.strip() for l in email_body.splitlines() if l.strip()]
    if lines and len(lines[0]) <= 80:
        return lines[0]

    defaults = {
        "B2B_Sales":    "Quick question about your team",
        "Partnership":  "Partnership opportunity — worth 15 min?",
        "Grant_Request":"Grant inquiry",
        "Recruitment":  "Exciting opportunity — worth a chat?",
    }
    return defaults.get(intent, "Following up")


# ── Core send function ────────────────────────────────────────────────────────

def _send_via_gmail(
    from_email: str,
    from_name: str,
    to_email: str,
    subject: str,
    body_text: str,
) -> dict:
    """
    Send email via Gmail API. Returns a result dict.
    Falls back to mock mode if no credentials are configured.
    """
    sa_set    = bool(os.environ.get("GMAIL_SERVICE_ACCOUNT_JSON", ""))
    oauth_set = bool(os.environ.get("GMAIL_CREDENTIALS_JSON", ""))

    if not sa_set and not oauth_set:
        logger.warning(
            "[Phase 5] Gmail credentials not configured — MOCK mode. "
            "Set GMAIL_SERVICE_ACCOUNT_JSON or GMAIL_CREDENTIALS_JSON to send real emails."
        )
        logger.info(
            f"[Phase 5][MOCK] Would send:\n"
            f"  From:    {from_name} <{from_email}>\n"
            f"  To:      {to_email}\n"
            f"  Subject: {subject}\n"
            f"  Body:\n{body_text}"
        )
        return {
            "id":     "mock-gmail-message-id",
            "status": "mock_sent",
            "to":     to_email,
            "note":   "Set GMAIL_SERVICE_ACCOUNT_JSON or GMAIL_CREDENTIALS_JSON to send real emails.",
        }

    try:
        service = _build_gmail_service()
        message = _build_mime_message(from_email, from_name, to_email, subject, body_text)

        sent = (
            service.users()
            .messages()
            .send(userId="me", body=message)
            .execute()
        )

        logger.info(f"[Phase 5] Email sent via Gmail — message id={sent.get('id')}")
        return {"id": sent.get("id"), "status": "sent", "to": to_email}

    except Exception as e:
        logger.error(f"[Phase 5] Gmail send error: {e}", exc_info=True)
        return {"error": str(e), "status": "failed"}


# ── Main node function ────────────────────────────────────────────────────────

def send_email_node(state: AgentState) -> dict:
    """
    LangGraph node — Phase 5.
    Inputs:  state.final_email, state.user_name, state.user_email,
             state.target_email, state.intent
    Outputs: state.send_result
    """
    logger.info(f"[Phase 5] Sending final email — job_id={state['job_id']}")

    final_email = state.get("final_email") or state.get("current_draft", "")
    intent      = state.get("intent", "B2B_Sales")
    subject     = _extract_subject(final_email, intent)

    # Use GMAIL_SENDER_EMAIL env var if set (service account path requires it),
    # otherwise fall back to the sender's own address from state.
    from_email = os.environ.get("GMAIL_SENDER_EMAIL") or state["user_email"]

    send_result = _send_via_gmail(
        from_email=from_email,
        from_name=state["user_name"],
        to_email=state["target_email"],
        subject=subject,
        body_text=final_email,
    )

    logger.info(f"[Phase 5] send_result={send_result}")
    return {"send_result": send_result, "hitl_status": "approved"}
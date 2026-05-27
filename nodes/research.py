"""
nodes/research.py  ·  Phase 1 — Intake & Deep Research
────────────────────────────────────────────────────────
Uses Firecrawl to scrape the target company's homepage,
/about page, and recent news, then synthesises a "Company DNA"
block that all downstream nodes consume.

Firecrawl advantages over raw scraping:
  - Handles JS-rendered pages (SPAs, Next.js etc.)
  - Returns clean markdown — no HTML noise to strip
  - Built-in crawl mode lets us pull multiple pages in one call
"""

import os
import logging
from firecrawl import FirecrawlApp
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from state import AgentState

logger = logging.getLogger(__name__)

_firecrawl: FirecrawlApp | None = None
_llm: ChatGroq | None = None


def _get_firecrawl() -> FirecrawlApp:
    global _firecrawl
    if _firecrawl is None:
        _firecrawl = FirecrawlApp(api_key=os.environ["FIRECRAWL_API_KEY"])
    return _firecrawl


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.environ["GROQ_API_KEY"],
        )
    return _llm


# ── Helpers ───────────────────────────────────────────────────────────────────

def _firecrawl_scrape(url: str) -> str:
    try:
        result = _get_firecrawl().scrape_url(url)
        # Handle v2 object attributes
        content = getattr(result, "markdown", "") or getattr(result, "content", "")
        # Fallback if it returns a dict
        if not content and isinstance(result, dict):
            content = result.get("markdown", "") or result.get("content", "")
        return str(content)[:6000]
    except Exception as e:
        logger.warning(f"[Firecrawl] scrape failed for {url}: {e}")
        return ""


def _firecrawl_search(query: str, url_hint: str) -> str:
    try:
        results = _get_firecrawl().search(query)
        snippets = []
        
        # Safely extract the data list whether it's a tuple, dict, or object
        data_list = []
        if isinstance(results, tuple):
            data_list = results[0] if len(results) > 0 else []
        elif isinstance(results, dict):
            data_list = results.get("data", [])
        elif hasattr(results, "data"):
            data_list = results.data
        elif isinstance(results, list):
            data_list = results
            
        for r in data_list:
            text = getattr(r, "markdown", "") or getattr(r, "content", "") or getattr(r, "description", "")
            if not text and isinstance(r, dict):
                text = r.get("markdown") or r.get("content") or r.get("description", "")
            if text:
                snippets.append(str(text)[:800])
                
        return "\n\n".join(snippets)[:4000]
    except Exception as e:
        logger.warning(f"[Firecrawl] search failed for '{query}': {e}")
        return ""


DNA_SYNTHESIS_SYSTEM = """You are a business intelligence analyst.
Given raw web content about a company, synthesise a concise "Company DNA" block
in plain text. Structure it as:

TARGET_AUDIENCE: <who they serve>
CORE_MISSION: <what problem they solve>
CURRENT_OBJECTIVES: <what they appear to be focused on right now>
RECENT_MILESTONES: <any notable wins, launches, news, funding rounds>
TONE_AND_CULTURE: <how they present themselves — formal, startup-y, academic, etc.>
PERSONALIZATION_HOOK: <one specific, concrete recent fact that would make a cold email opener feel researched and genuine>

Rules:
- Each field: 1–3 sentences max. No fluff.
- PERSONALIZATION_HOOK must be a real, specific, verifiable data point — not a generic compliment.
- If data is genuinely missing for a field, write "Unknown"."""


# ── Main node function ────────────────────────────────────────────────────────

def research_node(state: AgentState) -> dict:
    """
    LangGraph node — Phase 1.
    Inputs:  state.target_url
    Outputs: state.company_dna (str)
    """
    target_url = state["target_url"]
    logger.info(f"[Phase 1] Researching: {target_url}")

    # 1. Scrape homepage
    homepage_md = _firecrawl_scrape(target_url)

    # 2. Scrape /about (best-effort — may 404, that's fine)
    about_url = target_url.rstrip("/") + "/about"
    about_md = _firecrawl_scrape(about_url)

    # 3. Search for recent news / press
    domain = (
        target_url
        .replace("https://", "")
        .replace("http://", "")
        .split("/")[0]
        .replace("www.", "")
    )
    company_name = domain.split(".")[0].capitalize()
    news_md = _firecrawl_search(
        f"{company_name} news announcement funding launch 2024 2025",
        url_hint=target_url,
    )

    # ── Hard fallback: if every scrape returned empty, do NOT call the LLM ──────
    # An LLM synthesis call on zero data produces confident-sounding hallucinations.
    # Instead, set a canonical sentinel string that every downstream writer checks
    # and handles by switching to a "no-research" generation mode.
    FALLBACK_DNA = (
        "SCRAPE_FAILED: Insufficient data retrieved from target URL. "
        "All three scrape attempts (homepage, /about, news search) returned no content. "
        "Likely causes: bot-blocking, JS-only rendering timeout, or private/gated site. "
        "Writers must rely solely on the user's offering description. "
        "Do NOT fabricate company details. Do NOT use generic flattery as a hook."
    )

    total_content = len(homepage_md) + len(about_md) + len(news_md)
    MIN_USEFUL_CHARS = 150   # anything less is noise (e.g. a single meta tag)

    if total_content < MIN_USEFUL_CHARS:
        logger.warning(
            f"[Phase 1] All scrape attempts returned <{MIN_USEFUL_CHARS} chars total "
            f"({total_content} chars). Setting fallback DNA — skipping LLM synthesis."
        )
        return {"company_dna": FALLBACK_DNA}

    # ── Normal path: synthesise DNA from scraped content ─────────────────────
    combined_raw = f"""
=== HOMEPAGE (scraped via Firecrawl) ===
{homepage_md or 'No content retrieved'}

=== ABOUT PAGE ===
{about_md or 'No content retrieved'}

=== RECENT NEWS / SEARCH ===
{news_md or 'No content retrieved'}
""".strip()

    messages = [
        SystemMessage(content=DNA_SYNTHESIS_SYSTEM),
        HumanMessage(
            content=f"Company URL: {target_url}\n\nRaw scraped content:\n{combined_raw}"
        ),
    ]

    response = _get_llm().invoke(messages)
    company_dna = response.content.strip()

    # ── Secondary check: if Claude still produced a suspiciously short result ─
    # (e.g. partial JS content that looked non-empty but was useless markup)
    if len(company_dna) < 80:
        logger.warning(
            f"[Phase 1] Synthesised DNA is suspiciously short ({len(company_dna)} chars). "
            f"Replacing with fallback to prevent hallucination downstream."
        )
        return {"company_dna": FALLBACK_DNA}

    logger.info(f"[Phase 1] Company DNA synthesised ({len(company_dna)} chars)")
    logger.debug(f"[Phase 1] DNA:\n{company_dna}")

    return {"company_dna": company_dna}
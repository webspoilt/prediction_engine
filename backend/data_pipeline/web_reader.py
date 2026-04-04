"""
Web Reader — Agentic Web Content Extraction
============================================
Inspired by Agent-Reach (Jina Reader) and agent-eyes (semantic HTML parsing).

Provides:
  1. Jina Reader proxy (turns any URL → clean markdown, zero API key)
  2. Direct HTML fetch + BeautifulSoup semantic extraction
  3. Cricket-specific extractors for Cricbuzz/ESPN scorecard pages
"""

import re
import logging
import time
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
JINA_READER_BASE = "https://r.jina.ai"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
]


def _get_ua() -> str:
    """Rotate user agents based on time to reduce fingerprinting."""
    import hashlib
    h = int(hashlib.md5(str(int(time.time()) // 60).encode()).hexdigest(), 16)
    return USER_AGENTS[h % len(USER_AGENTS)]


# ── Jina Reader ──────────────────────────────────────────────────────────────

async def read_url_via_jina(url: str, timeout: int = 15) -> Optional[str]:
    """
    Fetch any URL through Jina Reader proxy → returns clean Markdown.
    Free tier: ~20 req/min, no API key needed.
    """
    jina_url = f"{JINA_READER_BASE}/{url}"
    try:
        from curl_cffi.requests import AsyncSession
        async with AsyncSession(impersonate="chrome110") as session:
            resp = await session.get(
                jina_url,
                headers={"Accept": "text/markdown", "User-Agent": _get_ua()},
                timeout=timeout,
            )
            if resp.status_code == 200 and len(resp.text) > 100:
                logger.info(f"✅ Jina Reader fetched {len(resp.text)} chars from {url}")
                return resp.text
            else:
                logger.warning(f"Jina Reader returned {resp.status_code} for {url}")
    except Exception as e:
        logger.warning(f"Jina Reader failed for {url}: {e}")
    return None


# ── Direct HTML Fetch ────────────────────────────────────────────────────────

async def fetch_raw_html(url: str, timeout: int = 12) -> Optional[str]:
    """Fetch raw HTML from a URL using curl_cffi with Chrome impersonation."""
    try:
        from curl_cffi.requests import AsyncSession
        async with AsyncSession(impersonate="chrome110") as session:
            resp = await session.get(
                url,
                headers={
                    "User-Agent": _get_ua(),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Cache-Control": "no-cache",
                },
                timeout=timeout,
            )
            if resp.status_code == 200:
                return resp.text
    except Exception as e:
        logger.warning(f"Raw HTML fetch failed for {url}: {e}")
    return None


# ── Cricket Score Extraction from HTML ───────────────────────────────────────

def extract_cricket_scores_from_html(html: str) -> List[Dict[str, Any]]:
    """
    Semantic HTML parser for cricket score pages (Cricbuzz/ESPN).
    Extracts match data from the DOM structure like agent-eyes reads accessibility trees.
    """
    matches = []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # ── Strategy 1: Cricbuzz HTML structure ──────────────────────────────
        match_cards = soup.select(".cb-mtch-lst, .cb-lv-scrs-well, .cb-col-100")
        for card in match_cards:
            try:
                title_el = card.select_one(".cb-lv-scrs-well-top, .cb-mtch-crd-itm-ttl, h3")
                if not title_el:
                    continue
                title_text = title_el.get_text(strip=True)
                if "vs" not in title_text.lower():
                    continue

                teams_raw = re.split(r"\s+vs?\s+", title_text, flags=re.IGNORECASE)
                if len(teams_raw) < 2:
                    continue
                team1 = teams_raw[0].strip()
                team2 = teams_raw[1].strip()

                # Try to find score
                score_els = card.select(".cb-scr-wll-chvrn, .cb-lv-scrs-col")
                score_text = ""
                for sel in score_els:
                    t = sel.get_text(strip=True)
                    if "/" in t or re.search(r"\d+", t):
                        score_text = t
                        break

                # Parse score
                total_runs = 0
                total_wickets = 0
                overs = 0.0
                score_match = re.search(r"(\d+)/(\d+)", score_text)
                if score_match:
                    total_runs = int(score_match.group(1))
                    total_wickets = int(score_match.group(2))
                overs_match = re.search(r"\((\d+\.?\d*)\s*ov", score_text)
                if overs_match:
                    overs = float(overs_match.group(1))

                # Status
                status_el = card.select_one(".cb-text-live, .cb-text-complete, .cb-mtch-status")
                status = "scheduled"
                if status_el:
                    st = status_el.get_text(strip=True).lower()
                    if "live" in st or "progress" in st:
                        status = "live"
                    elif "complete" in st or "won" in st or "result" in st:
                        status = "completed"

                matches.append({
                    "teams": [team1, team2],
                    "score": f"{total_runs}/{total_wickets}",
                    "over": overs,
                    "status": status,
                    "source": "html_cricbuzz",
                })
            except Exception:
                continue

        # ── Strategy 2: ESPN Cricinfo HTML structure ─────────────────────────
        espn_cards = soup.select('[class*="match-info"], [class*="MatchCard"], [class*="match-score"]')
        for card in espn_cards:
            try:
                text = card.get_text(" ", strip=True)
                if "vs" not in text.lower() and " v " not in text.lower():
                    continue

                teams_raw = re.split(r"\s+vs?\s+|\s+v\s+", text, flags=re.IGNORECASE)
                if len(teams_raw) < 2:
                    continue

                team1 = re.sub(r"\d+/\d+.*", "", teams_raw[0]).strip()[:40]
                team2 = re.sub(r"\d+/\d+.*", "", teams_raw[1]).strip()[:40]

                if not team1 or not team2:
                    continue

                score_match = re.search(r"(\d+)/(\d+)", text)
                total_runs = int(score_match.group(1)) if score_match else 0
                total_wickets = int(score_match.group(2)) if score_match else 0

                overs_match = re.search(r"\((\d+\.?\d*)\s*ov", text)
                overs = float(overs_match.group(1)) if overs_match else 0.0

                status = "live" if any(w in text.lower() for w in ["live", "batting", "bowling"]) else "scheduled"

                matches.append({
                    "teams": [team1, team2],
                    "score": f"{total_runs}/{total_wickets}",
                    "over": overs,
                    "status": status,
                    "source": "html_espn",
                })
            except Exception:
                continue

    except ImportError:
        logger.error("beautifulsoup4 not installed. pip install beautifulsoup4")
    except Exception as e:
        logger.error(f"HTML score extraction error: {e}")

    return matches


# ── Cricket Score Extraction from Markdown ────────────────────────────────────

def extract_cricket_scores_from_markdown(markdown: str) -> List[Dict[str, Any]]:
    """
    Extract structured match data from Jina Reader markdown output.
    Markdown typically contains team names, scores in patterns like '150/3 (16.2 ov)'.
    """
    matches = []
    if not markdown:
        return matches

    # Find lines with "vs" or "v" between team names
    lines = markdown.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for team matchups
        vs_match = re.search(
            r"([A-Z][a-zA-Z\s]+?)\s+(?:vs?\.?)\s+([A-Z][a-zA-Z\s]+)",
            line,
        )
        if vs_match:
            team1 = vs_match.group(1).strip()[:40]
            team2 = vs_match.group(2).strip()[:40]

            # Look in nearby lines (±3) for score data
            context = " ".join(lines[max(0, i - 2) : min(len(lines), i + 4)])
            score_match = re.search(r"(\d+)/(\d+)", context)
            overs_match = re.search(r"\((\d+\.?\d*)\s*ov", context)

            total_runs = int(score_match.group(1)) if score_match else 0
            total_wickets = int(score_match.group(2)) if score_match else 0
            overs = float(overs_match.group(1)) if overs_match else 0.0

            status = "scheduled"
            ctx_lower = context.lower()
            if any(w in ctx_lower for w in ["live", "batting", "in progress"]):
                status = "live"
            elif any(w in ctx_lower for w in ["won", "result", "completed"]):
                status = "completed"

            matches.append({
                "teams": [team1, team2],
                "score": f"{total_runs}/{total_wickets}",
                "over": overs,
                "status": status,
                "source": "jina_markdown",
            })
        i += 1

    return matches


# ── Full Pipeline: URL → Structured Data ─────────────────────────────────────

async def extract_scores_from_url(url: str) -> List[Dict[str, Any]]:
    """
    Complete pipeline: Try Jina Reader first, then direct HTML fetch + parsing.
    Returns list of extracted match dicts.
    """
    # Strategy 1: Jina Reader → Markdown → Parse
    md = await read_url_via_jina(url)
    if md:
        results = extract_cricket_scores_from_markdown(md)
        if results:
            return results

    # Strategy 2: Direct HTML → BeautifulSoup parse
    html = await fetch_raw_html(url)
    if html:
        results = extract_cricket_scores_from_html(html)
        if results:
            return results

    return []

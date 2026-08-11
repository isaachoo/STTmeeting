"""Web search for questions raised during the meeting.

Tavily is used when a key is configured because it returns extracted page text,
which is what the answering model actually needs. Without a key, search is
simply unavailable and the copilot answers from model knowledge with an explicit
"unverified" flag -- quietly degrading to a guess would be worse.
"""

import logging

import requests

import config

log = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"
TIMEOUT = (5, 20)


def available() -> bool:
    return bool(config.TAVILY_API_KEY)


def search(query: str, max_results: int = 4) -> list[dict]:
    """Return [{title, url, content}]. Empty list if search is off or fails."""
    if not available() or not query.strip():
        return []
    try:
        resp = requests.post(
            TAVILY_URL,
            json={
                "api_key": config.TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
    except (requests.RequestException, ValueError) as exc:
        log.warning("web search failed for %r: %s", query, exc)
        return []

    cleaned = []
    for item in results[:max_results]:
        content = (item.get("content") or "").strip()
        cleaned.append(
            {
                "title": (item.get("title") or "").strip(),
                "url": (item.get("url") or "").strip(),
                "content": content[:1200],
            }
        )
    return cleaned

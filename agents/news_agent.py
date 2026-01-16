# agents/news_agent.py
"""
Ultra-clean StockQuest-style News Agent.
Multi-source → dedupe → relevance AI → rank → deliver.
Compatible with state['news_raw'] and state['news_meta'].
"""

import requests
import feedparser
import hashlib
import time
import random
import re
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from ollama import Client
from bs4 import BeautifulSoup

ollama_client = Client()

# Rotate User-Agents to prevent 403 blocks
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
]

TIMEOUT = 8

def get_headers():
    return {"User-Agent": random.choice(USER_AGENTS)}

# ----------------------------------------------------
# Tier-0 Sources: BSE + NSE + SEBI
# ----------------------------------------------------
def fetch_bse_announcements(query: str) -> List[Dict[str, Any]]:
    """BSE corporate announcements (fast, official)."""
    url = f"https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w?pageno=1&strSearch=({query})&rows=20"
    try:
        r = requests.get(url, timeout=TIMEOUT, headers=get_headers())
        if r.status_code != 200:
            return []
        data = r.json().get("Table", [])
        out = []
        for d in data:
            out.append({
                "title": d.get("Subject") or "BSE Announcement",
                "summary": d.get("Brief") or "",
                "link": d.get("ATTACHMENTNAME") or "",
                "source": "BSE",
                "published": d.get("News_dt")
            })
        return out
    except Exception as e:
        print(f"BSE Fetch Error: {e}")
        return []

def fetch_sebi(query: str) -> List[Dict[str, Any]]:
    """SEBI press announcements."""
    try:
        url = f"https://www.sebi.gov.in/sebiweb/ajax/getNewsByKeyword.jsp?keyword={query}"
        r = requests.get(url, timeout=TIMEOUT, headers=get_headers())
        if r.status_code != 200:
            return []
        data = r.json().get("data", [])
        out = []
        for d in data:
            out.append({
                "title": d.get("title") or "SEBI Update",
                "summary": d.get("summary") or "",
                "link": d.get("url") or "",
                "source": "SEBI",
                "published": d.get("date")
            })
        return out
    except Exception:
        return []

# ----------------------------------------------------
# Tier-1 News APIs (GNews free, Bing optional)
# ----------------------------------------------------
def fetch_gnews(query: str, max_items=10):
    url = f"https://gnews.io/api/v4/search?q={query}&lang=en&country=in&max={max_items}&apikey=demo"
    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code != 200:
            return []
        data = r.json().get("articles", [])
        out = []
        for a in data:
            out.append({
                "title": a.get("title"),
                "summary": a.get("description") or "",
                "link": a.get("url"),
                "source": a.get("source", {}).get("name", "GNews"),
                "published": a.get("publishedAt")
            })
        return out[:max_items]
    except Exception:
        return []

# ----------------------------------------------------
# Tier-2 RSS
# ----------------------------------------------------
def fetch_rss(url: str) -> List[Dict[str, Any]]:
    try:
        # feedparser uses urllib, we can pass agent via headers
        # but feedparser.parse(url, agent='...') is supported
        ua = random.choice(USER_AGENTS)
        feed = feedparser.parse(url, agent=ua)
        out = []
        for e in feed.entries:
            summary = ""
            if e.get("summary"):
                summary = BeautifulSoup(e.get("summary"), "html.parser").get_text()
            
            out.append({
                "title": e.get("title", ""),
                "summary": summary,
                "link": e.get("link", ""),
                "source": url,
                "published": e.get("published", None)
            })
        return out
    except Exception as e:
        print(f"RSS Error ({url}): {e}")
        return []

RSS_SOURCES = [
    "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "https://economictimes.indiatimes.com/feeds/newsdefaultfeeds.cms",
    "https://www.cnbctv18.com/feeds/news/",
    "https://www.livemint.com/rss/news"
]

# ----------------------------------------------------
# Deduplication
# ----------------------------------------------------
def dedupe(items: List[Dict[str, Any]]):
    seen = set()
    out = []
    
    for a in items:
        # Normalize: Lowercase only alphanumeric chars
        raw_title = a.get("title", "") or ""
        norm = re.sub(r'[^a-zA-Z0-9]', '', raw_title.lower())
        
        # Also simple check on link if available
        # But link might vary (ref params), so rely on title content
        if not norm: continue

        if norm not in seen:
            seen.add(norm)
            out.append(a)
    return out

# ----------------------------------------------------
# AI RELEVANCE FILTER
# ----------------------------------------------------
def ai_filter(query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Input: list of raw items
    Output: filtered list sorted by relevance
    """
    if not items:
        return []

    # 1. Pre-filter by Keyword locally to reduce LLM load
    # If we have too many items, prioritizing those containing query words
    q_tokens = set(query.lower().split())
    
    scored_candidates = []
    for item in items:
        text = (item.get("title", "") + " " + item.get("summary", "")).lower()
        score = 0
        if any(t in text for t in q_tokens):
            score += 5
        # Boost recent items if published date present (simple checking if key exists/is not None)
        if item.get("published"):
            score += 2
        scored_candidates.append((score, item))
    
    # Sort and take top 25 for deep analysis
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    candidates = [x[1] for x in scored_candidates[:25]]

    # 2. LLM Relevance Scan (DISABLED FOR SPEED)
    # The keyword scoring in step 1 is sufficient and much faster.
    # LLM filtering adds ~30-60s latency which is unnecessary.
    
    # Just return candidates sorted by the heuristic score
    return candidates

# ----------------------------------------------------
# MAIN NODE
# ----------------------------------------------------
def node_news_fetch(state: Dict[str, Any]) -> Dict[str, Any]:
    print("entry news fetch node")
    query = state.get("company_name") or state.get("ticker", "")
    if not query:
        return {**state, "news_raw": [], "news_meta": {"error": "no query"}}

    items = []

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [
            ex.submit(fetch_bse_announcements, query),
            ex.submit(fetch_sebi, query),
            ex.submit(fetch_gnews, query),
        ] + [ex.submit(fetch_rss, url) for url in RSS_SOURCES]

        for fut in as_completed(futures):
            try:
                part = fut.result()
                if part: items.extend(part)
            except Exception as e:
                print(f"News Fetch Thread Error: {e}")

    # Deduplicate
    items = dedupe(items)
    print(f"News: Deduplicated count: {len(items)}")

    # AI relevance filtering
    # Use top 20 after filter
    items = ai_filter(query, items)
    items = items[:20]

    meta = {
        "count": len(items),
        "fetched_ts": int(time.time()),
        "query": query
    }
    print("exit news fetch node")
    return {**state, "news_raw": items, "news_meta": meta}

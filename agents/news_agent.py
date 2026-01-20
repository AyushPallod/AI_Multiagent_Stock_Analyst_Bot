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
# ----------------------------------------------------
# Tier-1 News APIs (GNews free, Bing optional)
# ----------------------------------------------------
def fetch_google_rss(query: str) -> List[Dict[str, Any]]:
    """
    Fetches news from Google News RSS (No API Key needed).
    Very effective for specific queries like 'ICICI Prudential Life Insurance'.
    """
    # Force sort by date to get recent news
    url = f"https://news.google.com/rss/search?q={query}+when:7d&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        # Use random agent
        ua = random.choice(USER_AGENTS)
        resp = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": ua})
        if resp.status_code != 200:
            return []
            
        feed = feedparser.parse(resp.content)
        out = []
        for e in feed.entries:
            # Clean summary from Google's localized HTML
            summary = e.get("summary", "")
            if summary:
                # Basic cleaning of HTML tags
                summary = re.sub(r'<[^>]+>', '', summary)
            
            out.append({
                "title": e.get("title", ""),
                "summary": summary,
                "link": e.get("link", ""),
                "source": "GoogleNews",  # or e.get('source', {}).get('title')
                "published": e.get("published", None)
            })
        return out
    except Exception as e:
        print(f"Google RSS Error: {e}")
        return []

def fetch_gnews(query: str, max_items=10):
    # Fallback to API if needed, but RSS is better for coverage
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
    # 1. Pre-filter by Keyword locally to reduce LLM load
    # If we have too many items, prioritizing those containing query words
    
    # Generate tokens from Query (Ticker) AND from Company Name if available in state
    # NOTE: The caller (node_news_fetch) needs to pass company_name. 
    # But ai_filter signature only takes query. 
    # We will assume 'query' might contain company name OR we should update signature.
    # Actually, let's just be robust: rely on what's passed, or improve the logic inside node_news_fetch to pass a rich query.
    # However, to avoid changing signature too much, let's extract tokens from the query string itself if it's "TIC CER" 
    # OR better: let's update proper logic.
    
    # Wait, the caller (node_news_fetch) constructs 'query'.
    # But 'node_news_fetch' HAS state['company_name'].
    # So we should update ai_filter to take (items, ticker, company_name).
    pass 
    
def ai_filter(items: List[Dict[str, Any]], ticker: str, company_name: str) -> List[Dict[str, Any]]:
    if not items: return []

    # Safe defaults
    ticker = (ticker or "").lower().replace(".ns", "")
    company = (company_name or "").lower()
    
    # 1. Build Critical Token Set
    critical_tokens = set()
    
    # Ticker tokens (e.g. "reliance")
    if len(ticker) > 2: critical_tokens.add(ticker)
    
    # Company Name tokens (e.g. "ICICI", "Prudential")
    # Remove junk
    junk = ["ltd", "limited", "corporation", "corp", "inc", "india", "industries", "holdings", "enterprise", "company", "life", "insurance", "financial", "services", "finance"]
    clean_company = company
    for j in junk:
        # Case insensitive replace for junk
        clean_company = re.sub(f"(?i)\\b{j}\\b", "", clean_company)
    
    # Split by non-alphanumeric
    c_tokens = re.split(r'[^a-z0-9]', clean_company)
    for t in c_tokens:
        if len(t) > 2:
            critical_tokens.add(t)

    scored_candidates = []
    
    # Keywords for scoring boost
    boost_keywords = set(ticker.split()) | critical_tokens
    
    for item in items:
        # text = title + summary
        text = (item.get("title", "") + " " + item.get("summary", "")).lower()
        score = 0
        
        # STRICT FILTER: Must contain at least one critical token
        if critical_tokens:
             # REGEX WORD BOUNDARY MATCH
             # prevents "life" matching "lifestyle"
             # We check if ANY token exists as a whole word
             found_token = False
             for t in critical_tokens:
                 if re.search(f"\\b{t}\\b", text):
                     found_token = True
                     break
             
             if not found_token:
                 continue 

        # Scoring
        if any(t in text for t in boost_keywords):
            score += 5
        if item.get("published"):
            score += 2
            
        scored_candidates.append((score, item))
    
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    candidates = [x[1] for x in scored_candidates[:25]]
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
            ex.submit(fetch_google_rss, query),  # PRIMARY SOURCE
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
    
    # LOGGING: Print all candidates before AI/Strict Filter
    print("\n--- RAW CANDIDATES (Pre-Filter) ---")
    for i, item in enumerate(items):
        print(f"[{i+1}] {item.get('title')} (Source: {item.get('source')})") 
    print("-----------------------------------\n")

    # AI relevance filtering
    # Use top 20 after filter
    ticker = state.get("ticker", "")
    company = state.get("company_name", "")
    items = ai_filter(items, ticker, company)
    
    # LOGGING: Print what survived the strict filter
    print("\n--- FILTERED CANDIDATES (Post-Filter) ---")
    for i, item in enumerate(items[:20]):
         print(f"[{i+1}] {item.get('title')}")
    print("-----------------------------------------\n")
    
    items = items[:20]

    meta = {
        "count": len(items),
        "fetched_ts": int(time.time()),
        "query": query
    }
    print("exit news fetch node")
    return {**state, "news_raw": items, "news_meta": meta}

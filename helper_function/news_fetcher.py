# news_fetcher.py
"""
News fetcher module — threads for speed, no external APIs.
Fetches Google News RSS and configured site RSS feeds, lightweight article text fetch,
deduplicates and returns a list of normalized article dicts:
  {"title","summary","link","published","source"}
"""

from typing import List, Dict, Any, Optional
import requests
import feedparser
from bs4 import BeautifulSoup
import hashlib
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Minimal logging (no debug)
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 8
USER_AGENT = "Mozilla/5.0 (compatible; MultiAgentNewsFetcher/1.0)"

# RSS sources to include
STANDARD_RSS_FEEDS = [
    # Moneycontrol, Mint, Economic Times, CNBC-TV18 RSS endpoints (public RSS)
    ("Moneycontrol", "https://www.moneycontrol.com/rss/MCtopnews.xml"),
    ("Mint", "https://www.livemint.com/rss/news"),
    ("EconomicTimes", "https://economictimes.indiatimes.com/feeds/newsdefaultfeeds.cms"),
    ("CNBCTV18", "https://www.cnbctv18.com/feeds/news/"),
    # Add more static feeds if needed
]

GOOGLE_NEWS_RSS_TEMPLATE = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
GDELT_RSS_TEMPLATE = "https://api.gdeltproject.org/api/v2/doc/doc?query={q}&mode=artlist&format=rss"

# Helpers
def _safe_text(html: str) -> str:
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return html

def _fetch_rss(url: str, timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, Any]]:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        if resp.status_code != 200:
            return []
        feed = feedparser.parse(resp.content)
        out = []
        for e in feed.entries:
            title = (e.get("title") or "").strip()
            link = (e.get("link") or "").strip()
            summary = _safe_text(e.get("summary", "") or e.get("description", ""))
            published = None
            if e.get("published_parsed"):
                try:
                    published = time.strftime("%Y-%m-%dT%H:%M:%SZ", e.published_parsed)
                except Exception:
                    published = None
            source = (e.get("source", {}).get("title") if e.get("source") else None) or ""
            if title and link:
                out.append({"title": title, "summary": summary, "link": link, "published": published, "source": source})
        return out
    except Exception:
        return []

def _fetch_article_text(url: str, max_chars: int = 1500, timeout: int = 6) -> str:
    try:
        headers = {"User-Agent": USER_AGENT}
        r = requests.get(url, timeout=timeout, headers=headers)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.content, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(p for p in paragraphs if p)
        if not text:
            desc = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
            if desc and desc.get("content"):
                text = desc.get("content")
        if not text:
            # try title tag
            t = soup.title.string if soup.title else ""
            text = t or ""
        return text[:max_chars]
    except Exception:
        return ""

def _dedupe_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for a in articles:
        title = (a.get("title") or "").strip()
        link = (a.get("link") or "").strip()
        key = (link or "") + "||" + title
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(a)
    return out

# Public API
def fetch_google_news(query: str, max_items: int = 10) -> List[Dict[str, Any]]:
    url = GOOGLE_NEWS_RSS_TEMPLATE.format(q=requests.utils.quote(query))
    items = _fetch_rss(url)
    return items[:max_items]

def fetch_gdelt(query: str, max_items: int = 10) -> List[Dict[str, Any]]:
    url = GDELT_RSS_TEMPLATE.format(q=requests.utils.quote(query))
    items = _fetch_rss(url)
    return items[:max_items]

def fetch_standard_feeds(max_items_each: int = 10) -> List[Dict[str, Any]]:
    all_items = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_fetch_rss, url): (name, url) for (name, url) in STANDARD_RSS_FEEDS}
        for fut in as_completed(futures):
            try:
                items = fut.result()
                if items:
                    all_items.extend(items[:max_items_each])
            except Exception:
                pass
    return all_items

def fetch_combined_for_query(query: str, max_items: int = 15) -> List[Dict[str, Any]]:
    # Run Google + GDELT + some standard feeds in parallel
    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        future_map = {
            ex.submit(fetch_google_news, query, max_items): "google",
            ex.submit(fetch_gdelt, query, max_items): "gdelt",
            ex.submit(fetch_standard_feeds, max_items//4): "standard",
        }
        for fut in as_completed(future_map):
            try:
                part = fut.result()
                if part:
                    results.extend(part)
            except Exception:
                pass
    # For any results with short summaries, attempt to fetch article text (fast thread pool)
    short_items = [r for r in results if len((r.get("summary") or "")) < 120 and r.get("link")]
    if short_items:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(_fetch_article_text, it["link"], 1200): it for it in short_items}
            for fut in as_completed(futures):
                try:
                    text = fut.result()
                    if text:
                        futures[fut]["summary"] = text
                except Exception:
                    pass
    deduped = _dedupe_articles(results)
    return deduped[:max_items]

# LangGraph-style node to return news for a ticker or query
def node_news_fetch(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expects state['ticker'] or state['company_name'] (prefer company_name if present)
    Returns state['news_raw'] list and 'news_meta'
    """
    ticker = state.get("ticker") or ""
    company = state.get("company_name") or ""
    query = company or ticker or ""
    if not query:
        return {**state, "news_raw": [], "news_meta": {"error": "no query"}}

    # Use a few derived queries: ticker, company name, ticker + news
    queries = [query, f"{query} stock", f"{query} India"]
    articles = []
    # fetch for each query (but cap)
    for q in queries:
        try:
            items = fetch_combined_for_query(q, max_items=12)
            if items:
                articles.extend(items)
        except Exception:
            pass

    # final dedupe and cap
    articles = _dedupe_articles(articles)
    cap = 20
    articles = articles[:cap]

    meta = {"count": len(articles), "fetched_ts": int(time.time())}
    return {**state, "news_raw": articles, "news_meta": meta}
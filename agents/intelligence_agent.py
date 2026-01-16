from typing import Dict, Any, List
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# import local news fetcher functions
from helper_function.news_fetcher import fetch_combined_for_query

# Minimal logging
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Curated intelligence queries (expandable)
INTELLIGENCE_QUERIES = [
    # Global macro
    "global market news",
    "us interest rate decision",
    "fed rate hike",
    "us inflation news",
    # RBI / India macro
    "Reserve Bank of India policy", "RBI repo rate decision", "RBI circulars", "rbi monetary policy india",
    # Sector & industry
    "banking sector India", "auto sector India", "pharma sector India", "tech sector India",
    # Government & regulation
    "government policy India finance", "budget India", "meen reforms India",
    # Geopolitical / commodity
    "crude oil price", "china economic news", "oil price impact India"
]

def fetch_intel_for_queries(queries: List[str], max_per_query: int = 5) -> List[Dict[str, Any]]:
    """
    Run multiple queries in parallel and return combined deduped set.
    """
    results = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(fetch_combined_for_query, q, max_per_query): q for q in queries}
        for fut in as_completed(futures):
            try:
                part = fut.result()
                if part:
                    results.extend(part)
            except Exception:
                pass
    # Deduplicate using the same logic as news_fetcher
    # Import the dedupe helper locally to keep module lightweight
    from helper_function.news_fetcher import _dedupe_articles
    deduped = _dedupe_articles(results)
    return deduped[:20]  # cap

def node_intelligence_fetch(state: Dict[str, Any]) -> Dict[str, Any]:
    print("entry intelligence node")
    """
    LangGraph-style node to collect intelligence news.
    Returns state['intelligence_raw'] and state['intelligence_meta'].
    """
    # Allow caller to pass additional queries (e.g., sector-specific) via state
    extra_queries = state.get("intelligence_queries") or []
    queries = INTELLIGENCE_QUERIES + extra_queries

    # Fetch in parallel
    items = fetch_intel_for_queries(queries, max_per_query=10)
    meta = {"count": len(items), "fetched_ts": int(time.time()), "queries_used": len(queries)}

    print("exit intelligence node")
    return {**state, "intelligence_raw": items, "intelligence_meta": meta}
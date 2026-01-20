import json
import re
from typing import Dict, List, Any
from ollama import Client

ollama_client = Client()

# ---------------------------------------------------------
# HELPER: Batch Score with Llama 31
# ---------------------------------------------------------
def score_batch(articles: List[Dict[str, Any]], model: str = "llama31") -> List[Dict[str, Any]]:
    """
    Scores a list of articles in ONE LLM call to save massive time.
    """
    if not articles:
        return []

    # 1. Prepare a numbered list for the prompt
    lines = []
    for i, a in enumerate(articles):
        # Use Title + first 50 chars of summary for context
        summary_snippet = (a.get('summary', '') or '')[:50]
        # Clean text to avoid breaking JSON (remove quotes etc)
        text = f"{a.get('title', '')} ({summary_snippet}...)".replace('"', "'").replace('\n', ' ')
        lines.append(f"{i+1}. {text}")
    
    joined_titles = "\n".join(lines)

    # 2. Strict JSON Prompt
    prompt = f"""
Analyze these headlines for Indian market sentiment.

HEADLINES:
{joined_titles}

Return a JSON object with this EXACT structure:
{{
  "results": [
    {{"id": 1, "sentiment": "Positive", "score": 80, "risk_flag": "no"}},
    ...
  ]
}}
Rules:
- score: 0 (Bearish) to 100 (Bullish).
- risk_flag: "yes" IF detection of fraud, SEBI action, default, or resignation. Else "no".
"""

    try:
        # Request JSON format specifically
        response = ollama_client.generate(model=model, prompt=prompt, stream=False, format="json")
        raw_json = response.get("response", "")
        # Attempt to clean potential markdown wrappers
        if "```json" in raw_json:
            raw_json = raw_json.split("```json")[1].split("```")[0]
        elif "```" in raw_json:
            raw_json = raw_json.split("```")[1].split("```")[0]
            
        parsed = json.loads(raw_json)
        results_map = {item['id']: item for item in parsed.get('results', [])}
    except Exception as e:
        print(f"Batch scoring failed: {e}")
        return []

    # 3. Merge results back
    scored_articles = []
    for i, article in enumerate(articles):
        # Default values if LLM misses an ID
        res = results_map.get(i+1, {})
        article['sentiment'] = res.get('sentiment', "Neutral")
        article['score'] = res.get('score', 50)
        article['risk_flag'] = res.get('risk_flag', "no")
        scored_articles.append(article)
        
    return scored_articles

# ---------------------------------------------------------
# LangGraph Node: Smart Filtering & Scoring
# ---------------------------------------------------------
def node_news_score(state: Dict[str, Any]):
    print("entry news score node")
    news = state.get("news_raw", []) or []
    intel = state.get("intelligence_raw", []) or []
    ticker = state.get("ticker", "").replace(".NS", "")

    # Combine all raw items
    all_items = news + intel
    
    if not all_items:
        return {**state, "news_scored": []}

    # --- SMART FILTER: Rank locally to pick best 20 ---
    
    # Build tokens similar to news_agent
    company = state.get("company_name", "").lower()
    ticker_clean = ticker.lower()
    
    tokens = set()
    if len(ticker_clean) > 2: tokens.add(ticker_clean)
    
    # Build tokens similar to news_agent
    company = state.get("company_name", "").lower()
    ticker_clean = ticker.lower()
    
    tokens = set()
    if len(ticker_clean) > 2: tokens.add(ticker_clean)
    
    junk = ["ltd", "limited", "corporation", "corp", "inc", "india", "industries", "holdings", "enterprise", "company", "life", "insurance", "financial", "services", "finance"]
    clean_company = company
    for j in junk:
         clean_company = re.sub(f"(?i)\\b{j}\\b", "", clean_company)
        
    c_parts = re.split(r'[^a-z0-9]', clean_company)
    for t in c_parts:
        if len(t) > 2:
            tokens.add(t)

    ranked_items = []
    for item in all_items:
        text = (item.get("title", "") + " " + item.get("summary", "")).lower()
        relevance = 0
        
        # Priority 1: Direct Match (Ticker OR Company Token)
        is_match = False
        
        # Check Ticker
        if ticker_clean in text:
            relevance += 10
            is_match = True
        
        # Check Company Tokens (if any token is in text)
        if not is_match and tokens:
             for t in tokens:
                 if re.search(f"\\b{t}\\b", text):
                     relevance += 8
                     is_match = True
                     break

        # Priority 2: High-Impact Keywords
        if any(w in text for w in ["fraud", "resignation", "default", "sebi", "profit", "loss", "quarter", "results", "dividend"]):
            relevance += 5
            
        # Priority 3: Recency (prefer items with published dates)
        if item.get("published"):
            relevance += 1
            
        # STRICT RELEVANCE GATE:
        # If it doesn't mention the ticker/company, AND it's not from a trusted specific source (like BSE),
        # we treat it as noise (Trump/Global news) unless we really want macro.
        source = item.get("source", "").lower()
        # NOTE: Only allow non-matches if they are EXPLICITLY from BSE/SEBI.
        # Everything else (even if relevant keywords exists) MUST match the company/ticker.
        if not is_match and "bse" not in source and "sebi" not in source:
             continue 

        ranked_items.append((relevance, item))

    # Sort descending by relevance
    ranked_items.sort(key=lambda x: x[0], reverse=True)

    # HARD CAP increased for max info
    MAX_NEWS = 10  # Reduced from 20 to 10 for SPEED/STABILITY
    top_items = [x[1] for x in ranked_items[:MAX_NEWS]]
    
    scored = score_batch(top_items, model="llama31")
    
    print(f"News: Scored {len(scored)} items")
    print("\n--- FINAL SCORED NEWS ---")
    for s in scored:
        print(f"Title: {s.get('title')}")
        print(f"Summary: {s.get('summary')[:100]}...")
        print(f"Score: {s.get('score')} | Sentiment: {s.get('sentiment')} | Risk: {s.get('risk_flag')}")
        print("-" * 30)
    print("-------------------------\n")
    
    print("exit news score node")
    return {**state, "news_scored": scored}

# ---------------------------------------------------------
# LangGraph Node: Aggregate
# ---------------------------------------------------------
def node_sentiment_aggregate(state: Dict[str, Any]):
    print("entry sent aggregate node")
    scored = state.get("news_scored", []) or []
    
    if not scored:
        return {
            **state,
            "sentiment": {
                "overall": "Neutral",
                "score": 50,
                "risk_flags": [],
                "latest_news_summary": []
            }
        }

    # Simple average
    avg_score = sum(a['score'] for a in scored) / len(scored)
    
    overall = "Neutral"
    if avg_score >= 60: overall = "Bullish"
    elif avg_score <= 40: overall = "Bearish"

    # Robust Risk Check
    risks = []
    for a in scored:
        flag = str(a.get("risk_flag", "no")).lower()
        if flag in ["yes", "true", "1"]:
             risks.append(a.get("title", "Unknown Risk"))

    # Top 8 headlines for the final report
    latest_news = [f"{a['title']} ({a['sentiment']})" for a in scored[:8]]

    print("exit sent aggregate node")
    return {
        **state,
        "sentiment": {
            "overall": overall,
            "score": int(avg_score),
            "risk_flags": risks,
            "latest_news_summary": latest_news
        }
    }
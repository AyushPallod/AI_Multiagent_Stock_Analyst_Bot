import time
import os

# --------------------------
# PROXY & CONFIG (Centralized)
# --------------------------
import config  # This automatically sets/clears proxies from config.py

from typing import Dict, Any, List, TypedDict, Optional, Literal, Annotated
from langgraph.graph import StateGraph, END

# --------------------------
# IMPORT AGENTS
# --------------------------
from agents.technical_agent import node_fetch, node_indicators, node_risk
from helper_function.patterns import node_patterns
from agents.fundamental_agent import node_fundamental_analysis
from agents.intelligence_agent import node_intelligence_fetch
from agents.sentiment_agent import node_news_score, node_sentiment_aggregate

from agents.news_agent import node_news_fetch


from ollama import Client
ollama_client = Client()

# --------------------------
# 0. SMART REDUCER (CRITICAL FIX)
# --------------------------
def smart_overwrite(current, new):
    """
    1. Prevents List Explosion: Replaces lists instead of appending duplicates.
    2. Prevents Stale Overwrite: Ignores empty data from parallel branches.
    """
    # If new value is empty (None, {}, [], ""), keep the existing valid data
    if not new:
        return current
    # Otherwise, update with the new data
    return new

# --------------------------
# 1. STATE SCHEMA
# --------------------------
class GraphState(TypedDict):
    # Apply smart_overwrite to ALL fields.
    # This enables "Last Valid Write Wins" for parallel execution.
    
    ticker: Annotated[str, smart_overwrite]
    company_name: Annotated[Optional[str], smart_overwrite]
    
    price_data: Annotated[Dict[str, Any], smart_overwrite]
    indicators: Annotated[Dict[str, Any], smart_overwrite]
    deterministic_signals: Annotated[Dict[str, Any], smart_overwrite]
    patterns: Annotated[Dict[str, Any], smart_overwrite]
    risk: Annotated[Dict[str, Any], smart_overwrite]
    
    fundamentals: Annotated[Dict[str, Any], smart_overwrite]
    fundamental_score: Annotated[Optional[int], smart_overwrite]
    fundamental_recommendation: Annotated[Optional[str], smart_overwrite]
    fundamental_narrative: Annotated[Optional[str], smart_overwrite]
    
    news_raw: Annotated[List[Dict[str, Any]], smart_overwrite]
    news_meta: Annotated[Dict[str, Any], smart_overwrite]
    intelligence_raw: Annotated[List[Dict[str, Any]], smart_overwrite]
    intelligence_meta: Annotated[Dict[str, Any], smart_overwrite]
    intelligence_queries: Annotated[Optional[List[str]], smart_overwrite]
    news_scored: Annotated[List[Dict[str, Any]], smart_overwrite]
    sentiment: Annotated[Dict[str, Any], smart_overwrite]
    
    analysis: Annotated[str, smart_overwrite]

# --------------------------
# HELPER: LLM CALL (UPDATED TO CHAT)
# --------------------------
def call_llm(system: str, user: str, model: str = "llama31") -> str:
    """
    Uses Ollama's Chat API with strict limits to prevent timeouts.
    """
    try:
        resp = ollama_client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            stream=False,
            options={
                "num_ctx": 4096,       # Strict context limit
                "num_predict": 1024,   # limit output tokens
                "temperature": 0.5     # focused deterministic output
            }
        )
        # Handle response structure (dict or object)
        if isinstance(resp, dict):
            return resp.get("message", {}).get("content", "")
        return resp.message.content
    except Exception as e:
        print(f"LLM Error ({model}):", e)
        return ""

# -------------------------------------------------------
# FINAL COMBINE NODE
# -------------------------------------------------------
def node_combine_reports(state: GraphState) -> GraphState:
    print("entry combine node")
    ticker = state.get("ticker", "Unknown")
    price = state.get("price_data", {})
    latest = price.get("latest", {}) if price else {}
    indicators = state.get("indicators", {})
    patterns = state.get("patterns", {})
    risk = state.get("risk", {})
    
    # Clean fundamentals to remove massive raw dataframes
    raw_funds = state.get("fundamentals", {})
    fundamentals = {k: v for k, v in raw_funds.items() if k != "raw"}
    
    sentiment = state.get("sentiment", {})
    latest_news = sentiment.get("latest_news_summary", [])
    risk_flags = sentiment.get("risk_flags", [])

    # Extract Multiple S/R Levels
    supports = patterns.get("support_resistance", {}).get("supports", [])
    resistances = patterns.get("support_resistance", {}).get("resistances", [])
    
    # Fallback if lists are empty
    if not supports: supports = ["N/A"]
    if not resistances: resistances = ["N/A"]

    # Truncate lists to prevent context overflow
    # Use top 5 news items only, max 100 chars each
    latest_news_str = "\n".join([str(n)[:100] + "..." for n in latest_news[:5]])
    if not latest_news_str: latest_news_str = "No significant news."

    # 1. System Prompt (The Persona & Rules)
    system_prompt = """You are a top-tier Hedge Fund Manager.
Produce a CONCISE, HIGH-IMPACT investment memo.
Focus ONLY on actionable insights. omit generic definitions.
Combine Technicals, Fundamentals, and News into a single narrative."""

    # 2. User Prompt (The Data & Structure)
    user_prompt = f"""
TICKER: {ticker}
Current Price: {latest.get("close")}
Total Score: {fundamentals.get("fundamental_score")} (Fund) | {sentiment.get("score")} (Sent)

### DATA SUMMARY
- **Technical**: EMA20={indicators.get("EMA20")}, RSI={indicators.get("RSI14")}, MACD={indicators.get("MACD_HIST")}
- **Pattern**: {patterns.get("top_pattern",{}).get("name")}
- **Risk**: ATR={risk.get("atr")}, Volatility={risk.get("volatility")}
- **Fundamental**: {fundamentals.get("fundamental_recommendation")}, Growth={fundamentals.get("growth")}, Flags={fundamentals.get("governance_flags")}
- **News**: 
{latest_news_str}

### REQUIRED OUTPUT FORMAT (Markdown)
## 1. Executive Summary & Trade Call
(Single paragraph synthesis + BUY/SELL/HOLD Decision with Confidence %)

## 2. Key Catalysts & Risks
- **Bullish Drivers**: (Key fundamental or technical strengths)
- **Bearish Risks**: (Key warnings, governance flags, or resistance levels)

## 3. Technical & Price Levels
- **Trend**: (Direction)
- **Action Zone**: Buy at {supports}, Stop Loss at {risk.get("stop_loss")}, Target {risk.get("target")}

## 4. Final Verdict
(One punchy sentence summarizing why to take this trade)
"""
    # Call using the new Chat interface
    result = call_llm(system_prompt, user_prompt, model="llama31")
    
    if not result:
        result = "Error: Final analysis could not be generated."

    print("exit combine node")
    return {**state, "analysis": result}


# --------------------------
# CONDITION FUNCTION
# --------------------------
def route_to_combine(state: GraphState) -> Literal["combine_node", "__end__"]:
    has_fundamentals = state.get("fundamentals") is not None
    has_sentiment = state.get("sentiment") is not None

    if has_fundamentals and has_sentiment:
        return "combine_node"
    return "__end__"

# --------------------------
# GRAPH CONSTRUCTION
# --------------------------
def build_trading_graph():
    g = StateGraph(GraphState)

    # NODES
    g.add_node("price_node", node_fetch)
    g.add_node("indicator_node", node_indicators)
    g.add_node("pattern_node", node_patterns)
    g.add_node("risk_node", node_risk)
    g.add_node("fundamental_node", node_fundamental_analysis)
    
    g.add_node("news_node", node_news_fetch)
    g.add_node("intelligence_node", node_intelligence_fetch)
    g.add_node("sent_score_node", node_news_score)
    g.add_node("sent_agg_node", node_sentiment_aggregate)
    
    g.add_node("combine_node", node_combine_reports)

    # EDGES
    g.set_entry_point("price_node")

    # Branch 1 (Tech)
    g.add_edge("price_node", "indicator_node")
    g.add_edge("indicator_node", "pattern_node")
    g.add_edge("pattern_node", "risk_node")
    g.add_edge("risk_node", "fundamental_node")
    
    # Branch 2 (News)
    g.add_edge("price_node", "news_node")
    g.add_edge("news_node", "intelligence_node")
    g.add_edge("intelligence_node", "sent_score_node")
    g.add_edge("sent_score_node", "sent_agg_node")

    # Join
    g.add_conditional_edges("fundamental_node", route_to_combine, {"combine_node": "combine_node", "__end__": END})
    g.add_conditional_edges("sent_agg_node", route_to_combine, {"combine_node": "combine_node", "__end__": END})
    
    g.add_edge("combine_node", END)

    return g.compile()

def run(ticker: str):
    app = build_trading_graph()
    return app.invoke({"ticker": ticker})

if __name__ == "__main__":
    t0 = time.time()
    res = run("ICICIPRULI.NS")
    t1 = time.time()
    print(f"Total Run Time: {round(t1 - t0, 2)} sec")
    print("\n" + "="*50)
    print(res.get("analysis"))
    print("="*50)
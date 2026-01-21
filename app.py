
import streamlit as st
import pandas as pd
import time
import os
import sys
from typing import Dict, Any

# -------------------------------------------------------------------
# CONFIG & PATH SETUP
# -------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import config

# FORCE CLEAR PROXIES for Localhost Ollama
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["ALL_PROXY"] = ""
os.environ["NO_PROXY"] = "*"

# Import Graph
try:
    from graph.graph import run
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

from ollama import Client, ResponseError

# -------------------------------------------------------------------
# APP CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    layout="wide", 
    page_title="Multiagent Trader Bot", 
    page_icon="📈"
)

# Custom CSS for better look
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #00ff00;
    }
    .metric-label {
        font-size: 14px;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

if "agent_state" not in st.session_state:
    st.session_state["agent_state"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# -------------------------------------------------------------------
# HELPER: RAG CONTEXT
# -------------------------------------------------------------------
def build_rag_context(state: Dict[str, Any]) -> str:
    if not state: return ""

    ticker = state.get("ticker", "Unknown")
    price = state.get("price_data", {}).get("latest", {})
    inds = state.get("indicators", {})
    patterns = state.get("patterns", {})
    risk = state.get("risk", {})
    funds = state.get("fundamentals", {})
    sentiment = state.get("sentiment", {})
    
    # Format Technicals
    tech_str = "\n".join([f"- {k}: {v}" for k,v in inds.items() if k in ['EMA20','SMA50','RSI14','MACD_HIST','ADX14']])
    
    # Format News (Top 3 with sentiment)
    news_items = sentiment.get("latest_news_summary", [])
    news_str = "\n".join([f"- {n}" for n in news_items[:3]])

    # KNOWLEDGE BASE INJECTION
    try:
        from knowledge_base import get_definitions_for_context
        kb_str = get_definitions_for_context(state)
    except ImportError:
        kb_str = ""

    return f"""
    STOCK: {ticker}
    PRICE: {price.get('close')} (Vol: {price.get('volume')})
    
    [TECHNICALS]
    {tech_str}
    Trend: {patterns.get('trend_structure', {}).get('trend')}
    Pattern: {patterns.get('top_pattern', {}).get('name')}
    Support: {patterns.get('support_resistance', {}).get('supports')}
    Resistance: {patterns.get('support_resistance', {}).get('resistances')}
    
    [RISK & VOLATILITY]
    Risk Level: {risk.get('risk_level')} (Score: {risk.get('risk_score')})
    Volatility: {risk.get('volatility')}
    ATR (14): {risk.get('atr')}
    ATR %: {risk.get('atr_pct')}% (Higher % = Higher Volatility/Risk)
    Stop Loss: {risk.get('stop_loss')}
    Target: {risk.get('target')}
    
    [FUNDAMENTALS]
    Score: {funds.get('fundamental_score')}
    Recommendation: {funds.get('fundamental_recommendation')}
    Growth Metric: {funds.get('growth')}
    
    [NEWS & SENTIMENT]
    Sentiment Score: {sentiment.get('score')} ({sentiment.get('overall')})
    Recent Headlines:
    {news_str}
    
    [EDUCATIONAL CONTEXT (Use this to explain concepts)]
    {kb_str}
    """

# -------------------------------------------------------------------
# HELPER: SAFETY OLLAMA CALL
# -------------------------------------------------------------------
def safe_ollama_chat(model, messages):
    """
    Wrapper to handle connection errors gracefully.
    """
    client = Client(host='http://127.0.0.1:11434') # Force localhost
    try:
        return client.chat(model=model, messages=messages, stream=True)
    except Exception as e:
        st.error(f"🛑 Connection Error: Could not connect to Ollama at 127.0.0.1:11434.")
        st.info("💡 Troubleshooting:\n1. Open a new terminal.\n2. Run `ollama serve`.\n3. Keep that window open and try again.")
        st.exception(e)
        return None

# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")
    ticker_input = st.text_input("Ticker Symbol", value="RELIANCE.NS").upper()
    if st.button("🚀 Generate Report", type="primary"):
        with st.spinner("🤖 Agents are researching..."):
            try:
                final_state = run(ticker_input)
                st.session_state["agent_state"] = final_state
                st.session_state["messages"] = [] # Reset chat
                st.rerun() # Refresh to show data
            except Exception as e:
                st.error(f"Agent Error: {e}")

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state["messages"] = []

# -------------------------------------------------------------------
# MAIN DASHBOARD
# -------------------------------------------------------------------
state = st.session_state["agent_state"]

if state:
    t_data = state.get("price_data", {}).get("latest", {})
    r_data = state.get("risk", {})
    
    # 1. HEADER METRICS
    st.title(f"{state.get('ticker')} Analysis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"₹{t_data.get('close', 0)}")
    c2.metric("Target", f"₹{r_data.get('target', 0)}")
    c3.metric("Stop Loss", f"₹{r_data.get('stop_loss', 0)}")
    c4.metric("Risk Level", r_data.get('risk_level', 'N/A'))
    
    st.divider()

    # 2. REPORT TABS
    tab_report, tab_tech, tab_fund, tab_news = st.tabs(["📝 Final Report", "📈 Technicals", "🏢 Fundamentals", "📰 News"])
    
    with tab_report:
        st.markdown(state.get("analysis", "No analysis found."))
        
    with tab_tech:
        st.subheader("Technical Indicators")
        st.json(state.get("indicators"))
        st.subheader("detected Patterns")
        st.json(state.get("patterns"))
        
    with tab_fund:
        f = state.get("fundamentals", {})
        st.subheader(f"Fundamental Score: {f.get('fundamental_score')}/100")
        st.markdown(f"**Recommendation:** {f.get('fundamental_recommendation')}")
        st.markdown(f"**Growth:** {f.get('growth')}")
        st.json(f)
        
    with tab_news:
        s = state.get("sentiment", {})
        st.subheader(f"Sentiment: {s.get('overall')} ({s.get('score')})")
        st.write(s.get("latest_news_summary"))
    
    st.divider()

    # 3. RAG CHAT
    st.subheader("💬 Ask the Analyst")
    st.caption(f"Chatting with context from {state.get('ticker')} report.")

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ex: 'Why is the risk high?' or 'What are the support levels?'"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            context = build_rag_context(state)
            system = f"""You are a specialized financial analyst for {state.get('ticker')}.
            
            CRITICAL INSTRUCTIONS:
            1. **Source of Truth**: Answer ONLY using the provided "CONTEXT" block below. Do NOT use outside knowledge or training data.
            2. **Methodology**: 
               - Scan the Context sections ([TECHNICALS], [RISK], [FUNDAMENTALS], etc.) for keywords related to the user's question.
               - Synthesize the answer using *only* the specific numbers and facts found there.
            3. **Handling Missing Data**: If the exact answer is not in the context, state "I do not have that specific data in my report."
            4. **Tone**: Professional, concise, and data-driven.
            
            CONTEXT:
            {context}
            """
            
            stream_gen = safe_ollama_chat(
                config.APP_LLM_MODEL, 
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            )
            
            if stream_gen:
                response = st.write_stream(chunk['message']['content'] for chunk in stream_gen)
                st.session_state["messages"].append({"role": "assistant", "content": response})

else:
    st.info("👈 Enter a ticker in the sidebar and click **Generate Report** to begin.")

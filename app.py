
import streamlit as st
import pandas as pd
import time
from typing import Dict, Any
from ollama import Client

# Import the existing graph run function
# Ensure that c:\PROJECTS\Multiagent_Trader_Bot is in python path or just run from root
import sys
import os

# Fix path to allow imports from root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Load Configuration (Proxies, etc)
import config

try:
    from graph.graph import run
except ImportError as e:
    st.error(f"Could not import 'run' from 'graph.graph'. Error: {e}")
    st.info(f"Current Path: {sys.path}")
    st.stop()

# Initialize Ollama
client = Client()

st.set_page_config(layout="wide", page_title="Trader Bot AI")

# Create a session state to store the "Agent State" (the raw data)
if "agent_state" not in st.session_state:
    st.session_state["agent_state"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ---------------------------------------------------------
# HELPER: Build Context string for RAG
# ---------------------------------------------------------
def build_rag_context(state: Dict[str, Any]) -> str:
    """
    Extracts key structured data from the state to form a context string
    for the LLM. Filters out massive raw data to keep it concise.
    """
    if not state:
        return ""

    ticker = state.get("ticker", "Unknown")
    
    # 1. Prices
    price = state.get("price_data", {})
    latest = price.get("latest", {})
    latest_str = f"Usage: {latest}" if latest else "No Data"

    # 2. Indicators
    inds = state.get("indicators", {})
    ind_str = ", ".join([f"{k}={v}" for k,v in inds.items() if isinstance(v, (int, float, str))])

    # 3. Patterns
    patterns = state.get("patterns", {})
    pat_str = str(patterns.get("top_pattern", "None"))
    sr = patterns.get("support_resistance", {})
    
    # 4. Fundamental
    fund = state.get("fundamentals", {})
    fund_score = fund.get("fundamental_score", "N/A")
    rec = fund.get("fundamental_recommendation", "N/A")
    growth = fund.get("growth", "N/A")
    gov_flags = fund.get("governance_flags", "None")

    # 5. Sentiment & News
    sent = state.get("sentiment", {})
    news_items = sent.get("latest_news_summary", [])
    # Take top 5 news only
    news_str = "\n".join([f"- {n}" for n in news_items[:5]])

    # 6. Risk
    risk = state.get("risk", {})
    risk_str = f"StopLoss={risk.get('stop_loss')}, Target={risk.get('target')}, RiskLevel={risk.get('risk_level')}"

    context = f"""
    [TICKER]: {ticker}
    [LATEST PRICE]: {latest_str}
    [TECHNICAL INDICATORS]: {ind_str}
    [PATTERNS]: {pat_str}
    [SUPPORT/RESISTANCE]: {sr}
    [RISK PROFILE]: {risk_str}
    [FUNDAMENTALS]: Score={fund_score}, Rec={rec}, Growth={growth}, Flags={gov_flags}
    [TOP NEWS]:
    {news_str}
    """
    return context

# ---------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------
st.title("🤖 Multiagent Trader Bot")

with st.sidebar:
    st.header("Configuration")
    ticker_input = st.text_input("Enter Ticker (e.g. RELIANCE.NS)", value="RELIANCE.NS")
    run_btn = st.button("Generate Report", type="primary")

    st.info("Ensure Ollama is running locally.")
    if st.button("Clear Chat"):
        st.session_state["messages"] = []

# MAIN EXECUTION
if run_btn:
    with st.spinner(f"Agents are researching {ticker_input}... (This may take 80-90s)"):
        try:
            # RUN THE GRAPH
            # We assume run() returns the final state dict
            final_state = run(ticker_input)
            
            # STORE STATE
            st.session_state["agent_state"] = final_state
            
            # Clear previous chat on new run
            st.session_state["messages"] = []
            
            st.success("Report Generated!")
        except Exception as e:
            st.error(f"Error running agents: {e}")

# DISPLAY RESULTS
state = st.session_state["agent_state"]

if state:
    # 1. SHOW THE REPORT
    st.subheader(f"Investment Report: {state.get('ticker')}")
    
    analysis_text = state.get("analysis", "No analysis found.")
    st.markdown(analysis_text)
    
    st.divider()
    
    # 2. EXPANDABLE DATA SECTIONS (Optional verification)
    with st.expander("🔍 Inspect Raw Agent Data"):
        tab1, tab2, tab3 = st.tabs(["Technicals", "Fundamentals", "Sentiment"])
        with tab1:
            st.json(state.get("indicators"))
            st.json(state.get("patterns"))
            st.json(state.get("risk"))
        with tab2:
            st.json(state.get("fundamentals"))
        with tab3:
            st.write(state.get("sentiment"))

    st.divider()

    # 3. RAG CHAT INTERFACE
    st.subheader("💬 Chat with the Data")
    
    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a specific question about this stock (e.g. 'What is the stop loss?')"):
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # GENERATE RESPONSE
        with st.chat_message("assistant"):
            # Build Context
            context_str = build_rag_context(state)
            
            # System Prompt (STRICT RAG)
            system_prompt = f"""You are a dedicated financial analyst assisting with THIS SPECIFIC STOCK.
            
            STRICT RULES:
            1. Answer ONLY using the provided CONTEXT DATA below.
            2. If the user asks for a value (e.g., "stop loss", "trend"), give the EXACT value from the context.
            3. Do NOT explain what a term means (e.g., dont explain what RSI is, just give the RSI value).
            4. If the answer is not in the context, state "I do not have that specific data in my report."
            
            CONTEXT DATA:
            {context_str}
            """
            
            # Stream response
            stream = client.chat(
                model=config.APP_LLM_MODEL,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ],
                stream=True,
            )
            
            response = st.write_stream(chunk['message']['content'] for chunk in stream)
            
        # Add assistant message
        st.session_state["messages"].append({"role": "assistant", "content": response})

else:
    st.write("👈 Enter a ticker and click 'Generate Report' to start.")

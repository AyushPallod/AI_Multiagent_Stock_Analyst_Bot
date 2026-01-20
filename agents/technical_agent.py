import os
import pandas as pd
import yfinance as yf

# PROXY_URL = "http://172.31.2.3:8080"
# os.environ["YFINANCE_USE_BACKUP"] = "true"
# os.environ["YFINANCE_IPV4_ONLY"] = "true"
# os.environ["HTTP_PROXY"] = PROXY_URL
# os.environ["HTTPS_PROXY"] = PROXY_URL
# os.environ["ALL_PROXY"] = PROXY_URL
os.environ["HTTPS_PROXY"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["ALL_PROXY"] = ""

from helper_function.indicators_and_signals import compute_indicators_and_signals

# proxy_address = "http://172.31.2.3:8080/"
# proxy_address = "http://mrh2025014:Pooja%2302@172.31.2.3:8080/"

# --------------------------
# PRICE FETCH (force NSE symbol)
# --------------------------
def fetch_price_yahoo(ticker: str):
    os.environ["HTTPS_PROXY"] = ""
    os.environ["HTTP_PROXY"] = ""
    os.environ["ALL_PROXY"] = ""
    # yf.set_config(proxy=PROXY_URL)

    t_in = ticker.strip().upper()
    # ensure NSE suffix
    if not t_in.endswith(".NS"):
        sym = t_in + ".NS"
    else:
        sym = t_in

    try:
        df = yf.Ticker(sym).history(period="8mo", interval="1d")
    except Exception as e:
        print("Exception while fetching from yfinance:", e)
        return {"ticker": t_in, "yahoo_symbol": sym, "error": str(e)}

    if df is None or df.empty:
        return {"ticker": t_in, "yahoo_symbol": sym, "error": f"No NSE data for {sym}"}

    # Fetch Company Name (Info)
    company_name = t_in # fallback
    try:
        # Ticker.info makes a network call, might fail or be slow
        info = yf.Ticker(sym).info
        company_name = info.get("longName") or info.get("shortName") or t_in
    except Exception:
        pass
        
    # Convert into history list
    df = df.reset_index()

    # --------------------------
    # Vectorized Sanitization
    # --------------------------
    # 1. Validate Volume (fill NaN with 1, ensure > 0, cast to int)
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(1)
        # using .loc to avoid chained assignment warnings
        df.loc[df["Volume"] <= 0, "Volume"] = 1
        df["Volume"] = df["Volume"].astype(int)
    else:
        df["Volume"] = 1

    # 2. Format Date (YYYY-MM-DD)
    if "Date" in df.columns:
        # Check if it has .dt accessor (is datetime-like)
        if pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["date"] = df["Date"].dt.strftime("%Y-%m-%d")
        else:
            # Fallback for strings/objects
            df["date"] = df["Date"].astype(str).str.slice(0, 10)
    else:
        # Should not happen with reset_index() on standard yfinance data
        return {"ticker": t_in, "yahoo_symbol": sym, "error": "Date column missing"}

    # 3. Rename OHLC columns to lowercase
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    # 4. Drop Not-a-Number rows in critical columns
    # We need open, high, low, close to be valid
    required_cols = ["open", "high", "low", "close"]
    # Ensure they exist (yfinance might return empty cols if no data)
    for c in required_cols:
        if c not in df.columns:
             return {"ticker": t_in, "yahoo_symbol": sym, "error": f"Missing column {c}"}
    
    df = df.dropna(subset=required_cols)

    if df.empty:
        return {"ticker": t_in, "yahoo_symbol": sym, "error": "All rows missing OHLC after sanitization"}

    # 5. Convert to dict list
    # selecting only the columns we want in the output
    out_cols = ["date", "open", "high", "low", "close", "volume"]
    rows = df[out_cols].to_dict("records")

    latest = rows[-1]
    latest = rows[-1]
    return {"ticker": t_in, "yahoo_symbol": sym, "history": rows, "latest": latest, "company_name": company_name}


# --------------------------
# LANGGRAPH NODES
# --------------------------

# PRICE NODE
def node_fetch(state):
    print("entry price node")
    ticker = state.get("ticker")
    if not ticker:
        return {
            **state,
            "price_data": {"error": "ticker not provided"},
            "ticker": None
        }

    result = fetch_price_yahoo(ticker)
    print(f"exit price node (Company: {result.get('company_name')})")
    return {
        **state,
        "ticker": ticker,
        "company_name": result.get("company_name"),
        "price_data": result    
    }

# INDICATOR NODE
def node_indicators(state):
    print("entry indicator node")
    price_data = state.get("price_data", {}) or {}
    history = price_data.get("history", []) or []

    if not history or len(history) < 10:
        return {
            **state,
            "indicators": {"error": "not enough history"},
            "deterministic_signals": {"error": "not enough history"}
        }

    try:
        out = compute_indicators_and_signals(history)
    except Exception as e:
        return {
            **state,
            "indicators": {"error": str(e)},
            "deterministic_signals": {"error": "indicator calc failed"}
        }

    if "error" in out:
        return {
            **state,
            "indicators": {"error": out.get("error")},
            "deterministic_signals": {"error": "indicator calc failed"}
        }
    print("exit indicator node")
    return {
        **state,
        "indicators": out["indicators"],
        "deterministic_signals": out["signals"]
    }

# RISK NODE
def node_risk(state):
    print("entry risk node")
    price_data = state.get("price_data", {}) or {}
    ind = state.get("indicators", {}) or {}

    latest = price_data.get("latest", {}) or {}
    price = latest.get("close")

    atr = ind.get("ATR14")
    rsi = ind.get("RSI14")
    adx = ind.get("ADX14")
    roc = ind.get("ROC12")

    # If no ATR → cannot compute risk
    if price is None or atr is None:
        return {
            **state,
            "risk": {
                "error": "missing ATR or price"
            }
        }

    # --------------------------
    # ATR-based metrics
    # --------------------------
    atr_pct = (atr / price) * 100

    stop_loss = round(price - atr * 2, 2)          # 2 × ATR
    target = round(price + atr * 3, 2)             # 3 × ATR
    sl_distance = round(price - stop_loss, 2)

    # --------------------------
    # Volatility bucket
    # --------------------------
    if atr_pct > 3:
        vol_bucket = "High"
        allocation = "0.5% of portfolio"
    elif atr_pct > 1.5:
        vol_bucket = "Medium"
        allocation = "1% of portfolio"
    else:
        vol_bucket = "Low"
        allocation = "2% of portfolio"

    # --------------------------
    # Risk Score (0–100)
    # --------------------------
    risk_score = 50

    # Higher ATR% = more risk
    if atr_pct > 3:
        risk_score += 20
    elif atr_pct > 1.5:
        risk_score += 10

    # ADX (trend strength reduces risk)
    if adx:
        if adx >= 25:
            risk_score -= 10
        elif adx < 15:
            risk_score += 5

    # RSI extremes
    if rsi is not None:
        if rsi > 75:
            risk_score += 10
        elif rsi < 25:
            risk_score += 10

    # ROC momentum risk
    if roc is not None:
        if roc > 5:
            risk_score -= 5
        elif roc < -5:
            risk_score += 5

    risk_score = max(0, min(100, risk_score))

    # Map score → category
    if risk_score <= 25:
        risk_level = "Low Risk"
    elif risk_score <= 50:
        risk_level = "Medium Risk"
    elif risk_score <= 75:
        risk_level = "High Risk"
    else:
        risk_level = "Very High Risk"

    print("exit risk node")
    return {
        **state,
        "risk": {
            "price": price,
            "atr": atr,
            "atr_pct": round(atr_pct, 2),
            "volatility": vol_bucket,
            "stop_loss": stop_loss,
            "target": target,
            "recommended_allocation": allocation,
            "risk_score": risk_score,
            "risk_level": risk_level
        }
    }
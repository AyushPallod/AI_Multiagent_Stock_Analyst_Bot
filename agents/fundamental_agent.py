import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import math
import time
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import yfinance as yf

# -------------------------
# Helpers
# -------------------------
def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0: return None
    return float(a) / float(b)

def cagr(start: float, end: float, periods: int) -> Optional[float]:
    if start is None or end is None or start <= 0 or periods <= 0: return None
    return (end / start) ** (1.0 / periods) - 1.0

def number_or_na(x):
    if x is None or (isinstance(x, float) and math.isnan(x)): return "NA"
    return x

def _df_or_empty(raw: Dict[str, Any], key: str) -> pd.DataFrame:
    v = raw.get(key, None)
    if isinstance(v, pd.DataFrame): return v
    return pd.DataFrame()

def _extract_recent_value(df: pd.DataFrame, possible_keys: List[str]) -> Optional[float]:
    """
    Safely extracts the most recent value (iloc[0]) for the first matching column key.
    df: Index=Dates (descending), Columns=Metrics
    """
    if df is None or df.empty: return None
    
    try:
        # Ensure we are looking at the most recent date (row 0)
        recent_row = df.iloc[0]
        
        # 1. Try exact matches first
        for key in possible_keys:
            if key in recent_row.index:
                val = recent_row[key]
                if isinstance(val, (int, float)) and not math.isnan(val):
                    return float(val)

        # 2. Try substring/fuzzy matches
        # Normalize columns once for speed
        cols_map = {c.lower(): c for c in recent_row.index}
        
        for key in possible_keys:
            key_lower = key.lower()
            # Direct loose match
            for col_lower, col_orig in cols_map.items():
                if key_lower in col_lower:
                    val = recent_row[col_orig]
                    if isinstance(val, (int, float)) and not math.isnan(val):
                        return float(val)
    except Exception:
        pass
    return None

# -------------------------
# Data Fetcher (yfinance)
# -------------------------
def fetch_yfinance_financials(ticker: str) -> Dict[str, Any]:
    sym = ticker if ticker.upper().endswith(".NS") else ticker + ".NS"
    t = yf.Ticker(sym)
    
    os.environ["HTTPS_PROXY"] = ""
    os.environ["HTTP_PROXY"] = ""

    info = {}
    try: info = t.info or {}
    except: pass

    # Helper to get table safely and Transpose
    # yfinance returns Columns=Dates (New->Old or Old->New? standard is New->Old columns if not transposed)
    # But usually .financials has columns as dates.
    # We transpose so Index=Dates.
    def get_table(attr):
        try:
            df = getattr(t, attr)
            # If not empty, transpose so Index is Dates (Newest first usually)
            return df.transpose() if not df.empty else pd.DataFrame()
        except: return pd.DataFrame()

    return {
        "info": info,
        "financials": get_table("financials"),
        "balance_sheet": get_table("balance_sheet"),
        "cashflow": get_table("cashflow"),
        "earnings": get_table("earnings")
    }

# -------------------------
# Metric Computations
# -------------------------
def compute_basic_ratios(info: Dict[str, Any], financials: pd.DataFrame, balance: pd.DataFrame) -> Dict[str, Any]:
    price = info.get("currentPrice") or info.get("previousClose")
    eps = info.get("trailingEps") or info.get("ttmEPS")
    
    pe = safe_div(price, eps)
    pb = safe_div(price, info.get("bookValue"))
    
    # ROE Estimate
    # FIX: Use iloc[0] (Recent) instead of iloc[-1] or vals[-1]
    roe = None
    try:
        # Get Net Income from most recent period (iloc[0])
        net_income = None
        if not financials.empty:
            # Filter columns matching 'Net Income'
            ni_cols = financials.filter(like="Net Income")
            if not ni_cols.empty:
                net_income = ni_cols.iloc[0, 0] # First row (recent), first match column

        # Get Equity from most recent period (iloc[0])
        equity = None
        if not balance.empty:
            eq_cols = balance.filter(like="Stockholders Equity")
            if not eq_cols.empty:
                equity = eq_cols.iloc[0, 0]
        
        roe = safe_div(net_income, equity)
    except: pass

    # Debt to Equity
    # FIX: Use recent balance sheet if info is missing
    de = info.get("debtToEquity")
    if de is None:
        try:
            total_debt = None
            total_equity = None
            if not balance.empty:
                # Recent row = iloc[0]
                recent = balance.iloc[0]
                # Fuzzy match for debt
                for idx in recent.index:
                    if "total debt" in str(idx).lower():
                        total_debt = recent[idx]
                        break
                # Fuzzy match for equity
                for idx in recent.index:
                    if "total equity" in str(idx).lower() or "stockholders equity" in str(idx).lower():
                        total_equity = recent[idx]
                        break
            de = safe_div(total_debt, total_equity)
        except: pass

    return {
        "marketCap": number_or_na(info.get("marketCap")),
        "price": number_or_na(price),
        "pe": number_or_na(pe),
        "pb": number_or_na(pb),
        "roe": number_or_na(roe),
        "debtToEquity": number_or_na(de),
        "freeCashflow": number_or_na(info.get("freeCashflow"))
    }

def compute_growth(financials: pd.DataFrame) -> Dict[str, Any]:
    # Simple Revenue Growth check
    growth = {}
    try:
        if not financials.empty:
            # Find Revenue column
            rev_col = None
            for c in financials.columns:
                if "Total Revenue" in str(c) or "Operating Revenue" in str(c):
                    rev_col = c
                    break
            
            if rev_col:
                # vals will be [Recent, Previous, Oldest...]
                vals = financials[rev_col].dropna().tolist()
                
                # Compare Latest (0) vs 3 years ago (3) or oldest available
                if len(vals) >= 2:
                    # FIX: cagr(Start=Old, End=New)
                    # vals[-1] is Oldest, vals[0] is Newest
                    growth["revenue_cagr_3y"] = cagr(vals[-1], vals[0], len(vals)-1)
    except: pass
    return growth

# -------------------------
# Scoring Logic
# -------------------------
def score_fundamentals(basic: Dict[str, Any], growth: Dict[str, Any]) -> Tuple[int, str]:
    score = 50 # Baseline
    
    # 1. Valuation
    pe = basic.get("pe")
    if isinstance(pe, (int, float)):
        if pe < 15: score += 15
        elif pe > 50: score -= 15
        
    # 2. Profitability
    roe = basic.get("roe")
    if isinstance(roe, (int, float)):
        if roe > 0.15: score += 15
        elif roe < 0.05: score -= 10
        
    # 3. Growth
    rev_g = growth.get("revenue_cagr_3y")
    if isinstance(rev_g, (int, float)):
        if rev_g > 0.10: score += 15
        elif rev_g < 0: score -= 10

    # Cap score
    score = max(0, min(100, score))
    
    if score >= 70: rec = "BUY"
    elif score <= 40: rec = "SELL"
    else: rec = "HOLD"
    
    return score, rec

# -------------------------
# Main Node
# -------------------------
def node_fundamental_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    print("entry fundamental node")
    ticker = state.get("ticker")
    if not ticker:
        return {**state, "fundamentals": {"error": "No ticker"}}

    # 1. Fetch Data
    raw = fetch_yfinance_financials(ticker)
    
    # 2. Compute Metrics
    basic = compute_basic_ratios(raw["info"], raw["financials"], raw["balance_sheet"])
    growth = compute_growth(raw["financials"])
    
    # 3. Cashflow Signals
    cf_signals = {"fcf_trend": "Unknown"}
    try:
        fcf = raw["info"].get("freeCashflow")
        if fcf is not None:
            cf_signals["fcf_trend"] = "Positive" if fcf > 0 else "Negative"
        else:
            # Fallback: Check recent operating cash flow from table
            cf_df = raw["cashflow"]
            if not cf_df.empty:
                # Look for standard Cash Flow keys
                recent_cf = _extract_recent_value(cf_df, [
                    "Operating Cash Flow", 
                    "Total Cash From Operating Activities",
                    "Cash Flow From Continuing Operating Activities"
                ])
                if recent_cf is not None:
                    cf_signals["fcf_trend"] = "Positive" if recent_cf > 0 else "Negative"
    except Exception as e:
        print(f"Fundamentals Error (Cashflow): {e}")
        pass

    # 4. Score
    score, rec = score_fundamentals(basic, growth)

    # 5. Governance Flags
    flags = []
    # yfinance often returns debtToEquity as a raw number like 0.5 or 150 (depends on unit)
    # Assuming standard ratio (0.5 to 2.0 range)
    de = basic.get("debtToEquity")
    if isinstance(de, (int, float)):
        # If it's > 200 (often returned as %), or > 2.0 (ratio)
        # yfinance usually returns it as a percentage (e.g. 85.4 means 0.85)
        # But sometimes it's raw. Let's assume if > 200 it's %, if < 10 it's ratio.
        # Safe bet: High debt flag if > 200 (if %) or > 2.5 (if ratio)
        if de > 200: flags.append("High Debt") 
    
    if cf_signals["fcf_trend"] == "Negative":
        flags.append("Negative Free Cashflow")

    fundamentals_block = {
        "ticker": ticker,
        "basic_metrics": basic,
        "growth": growth,
        "cashflow_signals": cf_signals,
        "governance_flags": flags,
        "fundamental_score": score,
        "fundamental_recommendation": rec,
    }

    print("exit fundamental node")
    return {
        **state,
        "fundamentals": fundamentals_block
    }
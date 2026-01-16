import math
import pandas as pd
import numpy as np
from typing import Dict, Any

# ---------- Indicator calculations ----------

def compute_indicators_from_history(history: list) -> Dict[str, Any]:
    """
    history: list of {"date","open","high","low","close","volume"(optional)}
    returns: dict of indicators (last values) and the working dataframe under 'df'
    """
    df = pd.DataFrame(history)
    # Ensure proper types and sorted by date ascending
    if df.empty:
        return {"error": "empty history"}
    # try to coerce columns
    for col in ["open","high","low","close","volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # sort by date ascending if date column exists
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    # basic rolling windows require enough rows; choose min_len
    min_periods = 5

    # SMA & EMA
    df["SMA20"] = df["close"].rolling(20, min_periods=min_periods).mean()
    df["SMA50"] = df["close"].rolling(50, min_periods=min_periods).mean()
    df["EMA20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI14"] = 100 - (100 / (1 + rs))

    # MACD (12,26) and signal(9)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    # Bollinger bands (20, 2 Std)
    ma20 = df["close"].rolling(20, min_periods=min_periods).mean()
    std20 = df["close"].rolling(20, min_periods=min_periods).std()
    df["BB_MID"] = ma20
    df["BB_UP"] = ma20 + 2 * std20
    df["BB_LOW"] = ma20 - 2 * std20

    # ATR (14)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14, min_periods=14).mean()

    # ADX (14) - average directional index
    # Compute directional movement
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    pos_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    neg_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr14 = tr.rolling(14, min_periods=14).sum()
    # smoothed DM
    pos_di = 100 * (pos_dm.rolling(14, min_periods=14).sum() / (tr14 + 1e-9))
    neg_di = 100 * (neg_dm.rolling(14, min_periods=14).sum() / (tr14 + 1e-9))
    dx = (abs(pos_di - neg_di) / (pos_di + neg_di + 1e-9)) * 100
    df["ADX14"] = dx.rolling(14, min_periods=14).mean()

    # Volume indicators
    if "volume" in df.columns:
        df["VOL_SMA20"] = df["volume"].rolling(20, min_periods=5).mean()
        df["VOL_REL"] = df["volume"] / (df["VOL_SMA20"] + 1e-9)
    else:
        df["VOL_SMA20"] = np.nan
        df["VOL_REL"] = np.nan

    # ROC (Rate of change) 12
    df["ROC12"] = df["close"].pct_change(periods=12) * 100

    # last row summary
    last = df.iloc[-1].to_dict()
    # collect into clean dict (only last scalar indicators)
    indicators = {
        "SMA20": _safe_round(last.get("SMA20")),
        "SMA50": _safe_round(last.get("SMA50")),
        "EMA20": _safe_round(last.get("EMA20")),
        "EMA50": _safe_round(last.get("EMA50")),
        "EMA200": _safe_round(last.get("EMA200")),
        "RSI14": _safe_round(last.get("RSI14")),
        "MACD": _safe_round(last.get("MACD")),
        "MACD_SIGNAL": _safe_round(last.get("MACD_SIGNAL")),
        "MACD_HIST": _safe_round(last.get("MACD_HIST")),
        "BB_UP": _safe_round(last.get("BB_UP")),
        "BB_MID": _safe_round(last.get("BB_MID")),
        "BB_LOW": _safe_round(last.get("BB_LOW")),
        "ATR14": _safe_round(last.get("ATR14")),
        "ADX14": _safe_round(last.get("ADX14")),
        "VOL_SMA20": _safe_round(last.get("VOL_SMA20")),
        "VOL_REL": _safe_round(last.get("VOL_REL")),
        "ROC12": _safe_round(last.get("ROC12")),
        # current price & date if present
        "PRICE": _safe_round(last.get("close")),
        "DATE": last.get("date")
    }

    return {"indicators": indicators, "df": df}


def _safe_round(x, ndigits=4):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return None
        return round(float(x), ndigits)
    except Exception:
        return None

# ---------- Signal logic (deterministic) ----------

def compute_signals(ind: Dict[str, Any]) -> Dict[str, Any]:
    """
    ind: indicators dict from compute_indicators_from_history()["indicators"]
    Returns a small structured signals dict.
    """
    # helpers
    price = ind.get("PRICE")
    sma20 = ind.get("SMA20")
    sma50 = ind.get("SMA50")
    ema20 = ind.get("EMA20")
    ema50 = ind.get("EMA50")
    ema200 = ind.get("EMA200")
    rsi = ind.get("RSI14")
    macd = ind.get("MACD")
    macd_sig = ind.get("MACD_SIGNAL")
    macd_hist = ind.get("MACD_HIST")
    adx = ind.get("ADX14")
    vol_rel = ind.get("VOL_REL")
    bb_up = ind.get("BB_UP")
    bb_low = ind.get("BB_LOW")
    atr = ind.get("ATR14")
    roc = ind.get("ROC12")

    signals = {}

    # Trend by moving averages
    # Simple rules: price above EMA50 and EMA200 -> uptrend
    ma_score = 0
    if price and ema50:
        ma_score += 1 if price > ema50 else -1
    if price and ema200:
        ma_score += 1 if price > ema200 else -1

    if ma_score >= 2:
        trend = "Strong Bullish"
    elif ma_score == 1:
        trend = "Bullish"
    elif ma_score == 0:
        trend = "Neutral"
    elif ma_score == -1:
        trend = "Bearish"
    else:
        trend = "Strong Bearish"

    signals["trend"] = trend
    # RSI status
    if rsi is None:
        rsi_status = "NA"
    elif rsi >= 70:
        rsi_status = "Overbought"
    elif rsi <= 30:
        rsi_status = "Oversold"
    else:
        rsi_status = "Neutral"
    signals["rsi_status"] = rsi_status

    # MACD status
    macd_status = "NA"
    if macd is not None and macd_sig is not None:
        # bullish if MACD crosses above signal or hist positive
        if macd_hist is not None:
            if macd_hist > 0:
                macd_status = "Bullish"
            elif macd_hist < 0:
                macd_status = "Bearish"
            else:
                macd_status = "Neutral"
        else:
            macd_status = "Neutral"
    signals["macd_status"] = macd_status

    # Volatility: ATR relative to price
    vol_status = "NA"
    if atr and price:
        atr_pct = (atr / price) * 100
        if atr_pct > 3:
            vol_status = "High"
        elif atr_pct > 1:
            vol_status = "Medium"
        else:
            vol_status = "Low"
        signals["atr_pct"] = _safe_round(atr_pct, 3)
    signals["volatility"] = vol_status

    # Volume spike
    if vol_rel is not None:
        signals["volume_status"] = "Spike" if vol_rel > 2 else ("Above Average" if vol_rel > 1.2 else "Normal")
    else:
        signals["volume_status"] = "NA"

    # Bollinger proximity: price near upper band
    if price and bb_up and bb_low:
        if price >= bb_up:
            bb_pos = "Above Upper"
        elif price <= bb_low:
            bb_pos = "Below Lower"
        else:
            bb_pos = "Inside Bands"
    else:
        bb_pos = "NA"
    signals["bollinger_position"] = bb_pos

    # Momentum ROC
    if roc is not None:
        if roc > 5:
            roc_status = "Strong Positive"
        elif roc > 1:
            roc_status = "Positive"
        elif roc < -5:
            roc_status = "Strong Negative"
        elif roc < -1:
            roc_status = "Negative"
        else:
            roc_status = "Neutral"
    else:
        roc_status = "NA"
    signals["roc_status"] = roc_status

    # ADX strength
    if adx is not None:
        if adx >= 25:
            adx_strength = "Strong"
        elif adx >= 20:
            adx_strength = "Moderate"
        else:
            adx_strength = "Weak"
    else:
        adx_strength = "NA"
    signals["adx_strength"] = adx_strength

    # Compose a numeric confidence score (0-100) from the above heuristics
    score = 50  # baseline
    # trend contribution
    if trend == "Strong Bullish":
        score += 20
    elif trend == "Bullish":
        score += 10
    elif trend == "Neutral":
        score += 0
    elif trend == "Bearish":
        score -= 10
    elif trend == "Strong Bearish":
        score -= 20
    # rsi
    if rsi_status == "Overbought":
        score -= 8
    elif rsi_status == "Oversold":
        score += 8
    # macd
    if macd_status == "Bullish":
        score += 6
    elif macd_status == "Bearish":
        score -= 6
    # adx
    if adx_strength == "Strong":
        score += 6
    elif adx_strength == "Weak":
        score -= 4
    # volume spike
    if signals.get("volume_status") == "Spike":
        score += 4
    # roc
    if roc_status == "Strong Positive":
        score += 4
    if roc_status == "Strong Negative":
        score -= 4

    score = max(0, min(100, score))
    signals["score"] = int(score)

    # final recommendation (simple rules)
    if score >= 65:
        rec = "STRONG BUY"
    elif score >= 55:
        rec = "BUY"
    elif score >= 45:
        rec = "HOLD"
    elif score >= 35:
        rec = "SELL"
    else:
        rec = "STRONG SELL"
    signals["recommendation"] = rec

    # include a few numeric values for the LLM to reference
    signals["summary_numbers"] = {
        "price": price,
        "SMA20": sma20,
        "SMA50": sma50,
        "EMA20": ema20,
        "EMA50": ema50,
        "EMA200": ema200,
        "RSI14": rsi,
        "MACD_HIST": macd_hist,
        "ADX14": adx,
        "VOL_REL": vol_rel,
        "ATR14": atr,
        "ROC12": roc
    }

    return signals

# ---------- Convenience: full compute pipeline ----------

def compute_indicators_and_signals(history: list) -> Dict[str, Any]:
    """
    Return:
    {
      "indicators": {...},
      "signals": {...},
      "df": pandas.DataFrame
    }
    """
    res = compute_indicators_from_history(history)
    if "error" in res:
        return {"error": res["error"]}
    indicators = res["indicators"]
    df = res["df"]
    signals = compute_signals(indicators)
    return {"indicators": indicators, "signals": signals, "df": df}

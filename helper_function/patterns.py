import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

# ---------------------------------------------------
# SAFE HELPERS
# ---------------------------------------------------

def _safe(val):
    try:
        return float(val)
    except Exception:
        return None

def _body(open_, close):
    return abs(close - open_)

def _upper_wick(open_, high, close):
    return high - max(open_, close)

def _lower_wick(open_, low, close):
    return min(open_, close) - low

def _is_doji(open_, close, high, low, thresh=0.12):
    rng = high - low
    if rng <= 0:
        return False
    return (_body(open_, close) / rng) < thresh


# ---------------------------------------------------
# CANDLESTICK PATTERN DETECTION
# ---------------------------------------------------

def detect_candlestick_patterns(
    c: Dict[str, Any],
    prev: Optional[Dict[str, Any]] = None,
    prev2: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:

    patterns = []

    o = _safe(c.get("open"))
    h = _safe(c.get("high"))
    l = _safe(c.get("low"))
    cl = _safe(c.get("close"))
    if None in (o, h, l, cl):
        return patterns

    body = _body(o, cl)
    if body == 0:
        # DOJI CANDLE
        patterns.append({"name": "Doji", "strength": "Moderate", "reason": "open equals close"})
        # Avoid division-by-zero
        body = 1e-6

    rng = max(h - l, 1e-6)
    up_w = _upper_wick(o, h, cl)
    low_w = _lower_wick(o, l, cl)
    body_rel = body / rng

    # -------- Marubozu (very strong candle)
    if body_rel > 0.9 and low_w < (0.02*rng) and up_w < (0.02*rng):
        patterns.append({"name": "Marubozu", "strength": "Strong",
                         "reason": "full body candle with tiny wicks"})

    # -------- Hammer & Hanging Man
    lower_wick_ratio = low_w / body
    upper_wick_ratio = up_w / body

    if lower_wick_ratio >= 2.0 and up_w <= 0.25 * body:
        if cl > o:
            patterns.append({"name": "Hammer", "strength": "Strong",
                             "reason": "long lower wick + bullish body at top"})
        else:
            patterns.append({"name": "Hanging Man", "strength": "Moderate",
                             "reason": "long lower wick + bearish close"})

    # -------- Shooting Star
    if upper_wick_ratio >= 2.0 and low_w <= 0.25 * body and cl < o:
        patterns.append({"name": "Shooting Star", "strength": "Strong",
                         "reason": "long upper wick + bearish body"})

    # ---------------------------------------------------
    # TWO-CANDLE PATTERNS
    # ---------------------------------------------------
    if prev is not None:
        po = _safe(prev.get("open"))
        pcl = _safe(prev.get("close"))
        if None not in (po, pcl):

            # Bullish Engulfing
            if pcl < po and cl > o and (o <= pcl and cl >= po):
                patterns.append({"name": "Bullish Engulfing", "strength": "Strong",
                                 "reason": "bullish candle engulfs previous bearish"})

            # Bearish Engulfing
            if pcl > po and cl < o and (o >= pcl and cl <= po):
                patterns.append({"name": "Bearish Engulfing", "strength": "Strong",
                                 "reason": "bearish candle engulfs previous bullish"})

        # Tweezer Top
        ph = _safe(prev.get("high"))
        if ph is not None and abs(ph - h) / max(ph, h, 1) < 0.003 and pcl > po and cl < o:
            patterns.append({"name": "Tweezer Top", "strength": "Moderate",
                             "reason": "matching highs + bearish reversal"})

        # Tweezer Bottom
        pl = _safe(prev.get("low"))
        if pl is not None and abs(pl - l) / max(pl, l, 1) < 0.003 and pcl < po and cl > o:
            patterns.append({"name": "Tweezer Bottom", "strength": "Moderate",
                             "reason": "matching lows + bullish reversal"})

        # Piercing / Dark Cloud Cover
        prev_mid = (po + pcl) / 2
        if pcl < po and o < pcl and cl > prev_mid:
            patterns.append({"name": "Piercing Pattern", "strength": "Moderate",
                             "reason": "bullish close above mid of previous bearish candle"})

        if pcl > po and o > pcl and cl < prev_mid:
            patterns.append({"name": "Dark Cloud Cover", "strength": "Moderate",
                             "reason": "bearish close below mid of previous bullish candle"})


    # ---------------------------------------------------
    # THREE-CANDLE PATTERNS
    # ---------------------------------------------------
    if prev is not None and prev2 is not None:
        p2o = _safe(prev2.get("open"))
        p2c = _safe(prev2.get("close"))
        po = _safe(prev.get("open"))
        pc = _safe(prev.get("close"))

        if None not in (p2o, p2c, po, pc):

            # Morning Star
            if p2c < p2o and abs(pc - po) < abs(p2c - p2o)*0.4 and cl > o and cl > (p2o + p2c)/2:
                patterns.append({"name":"Morning Star", "strength":"Strong",
                                 "reason":"3-candle bullish reversal"})

            # Evening Star
            if p2c > p2o and abs(pc - po) < abs(p2c - p2o)*0.4 and cl < o and cl < (p2o + p2c)/2:
                patterns.append({"name":"Evening Star", "strength":"Strong",
                                 "reason":"3-candle bearish reversal"})

    # ---------------------------------------------------
    # Inside Bar (consolidation)
    # ---------------------------------------------------
    if prev is not None:
        ph = _safe(prev.get("high"))
        pl = _safe(prev.get("low"))
        if h < ph and l > pl:
            patterns.append({"name":"Inside Bar", "strength":"Weak",
                             "reason":"full candle inside previous range"})

    return patterns


# ---------------------------------------------------
# TREND STRUCTURE
# ---------------------------------------------------

def detect_trend_structure(df: pd.DataFrame, lookback: int = 20) -> Dict[str,Any]:
    out = {"trend":"Unknown", "structure":[]}
    if df is None or df.empty:
        return out

    closes = df["close"].values
    if len(closes) < 5:
        return out

    peaks = []
    troughs = []

    for i in range(1, len(closes)-1):
        if closes[i] > closes[i-1] and closes[i] > closes[i+1]:
            peaks.append((i, closes[i]))
        if closes[i] < closes[i-1] and closes[i] < closes[i+1]:
            troughs.append((i, closes[i]))

    recent_peaks = peaks[-3:]
    recent_troughs = troughs[-3:]

    try:
        if len(recent_peaks)>=2 and len(recent_troughs)>=2:
            p1, p2 = recent_peaks[-2][1], recent_peaks[-1][1]
            t1, t2 = recent_troughs[-2][1], recent_troughs[-1][1]

            if p2>p1 and t2>t1:
                out["trend"] = "Higher Highs / Higher Lows (Uptrend)"

            elif p2<p1 and t2<t1:
                out["trend"] = "Lower Highs / Lower Lows (Downtrend)"

            else:
                out["trend"] = "Sideways / Indecisive"

            out["peaks"] = recent_peaks
            out["troughs"] = recent_troughs
    except:
        pass

    return out



# ---------------------------------------------------
# SUPPORT & RESISTANCE + BREAKOUT
# ---------------------------------------------------

def detect_support_resistance(df: pd.DataFrame, lookback:int=60, n_levels:int=5):
    res = {"supports":[], "resistances":[], "breakout":False, "breakdown":False}

    if df is None or df.empty:
        return res

    df = df.reset_index(drop=True)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    last = float(closes[-1])
    res["last_close"] = last

    segment = df.iloc[-lookback:]

    cand_res = []
    cand_sup = []

    for i in range(1, len(segment)-1):
        if segment["high"].iloc[i] > segment["high"].iloc[i-1] and \
           segment["high"].iloc[i] > segment["high"].iloc[i+1]:
            cand_res.append(float(segment["high"].iloc[i]))

        if segment["low"].iloc[i] < segment["low"].iloc[i-1] and \
           segment["low"].iloc[i] < segment["low"].iloc[i+1]:
            cand_sup.append(float(segment["low"].iloc[i]))

    cand_res = sorted(cand_res, reverse=True)[:n_levels]
    cand_sup = sorted(cand_sup)[:n_levels]

    res["resistances"] = cand_res
    res["supports"] = cand_sup

    if cand_res:
        if last > max(cand_res) * 1.003:
            res["breakout"] = True

    if cand_sup:
        if last < min(cand_sup) * 0.997:
            res["breakdown"] = True

    return res


# ---------------------------------------------------
# TOP-LEVEL PATTERN AGGREGATOR FOR LANGGRAPH
# ---------------------------------------------------

def compute_patterns_from_history(history: List[Dict[str, Any]], lookback_levels=60):
    out = {"patterns": [], "trend_structure": {}, "support_resistance": {}}
    if not history or len(history) < 3:
        return out

    df = pd.DataFrame(history)
    for col in ["open","high","low","close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)

    out["trend_structure"] = detect_trend_structure(df)
    out["support_resistance"] = detect_support_resistance(df)

    patt = []
    n = len(df)

    for i in range(max(2, n-50), n):
        cur = df.iloc[i].to_dict()
        prev = df.iloc[i-1].to_dict() if i-1>=0 else None
        prev2 = df.iloc[i-2].to_dict() if i-2>=0 else None

        found = detect_candlestick_patterns(cur, prev, prev2)
        for f in found:
            f2 = dict(f)
            f2["index"] = i
            f2["date"] = str(cur.get("date"))
            patt.append(f2)

    # Deduplicate
    seen = set()
    uniq = []
    for p in patt:
        key = (p["name"], p["date"])
        if key not in seen:
            seen.add(key)
            uniq.append(p)

    out["patterns"] = uniq

    strength_map = {"Strong":3, "Moderate":2, "Weak":1}
    best = None
    best_score = 0
    for p in uniq:
        sc = strength_map.get(p.get("strength","Weak"), 1)
        if sc > best_score:
            best_score = sc
            best = p

    out["top_pattern"] = best
    out["pattern_count"] = len(uniq)
    return out


# ---------------------------------------------------
# LANGGRAPH NODE
# ---------------------------------------------------

def node_patterns(state):
    price_data = state.get("price_data", {})
    hist = price_data.get("history") or []

    if not hist:
        return {**state, "patterns":{"error":"no history"}}

    patt = compute_patterns_from_history(hist)

    return {**state, "patterns": patt}

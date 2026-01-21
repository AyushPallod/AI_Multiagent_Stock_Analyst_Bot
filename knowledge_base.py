
# Financial Knowledge Base for RAG Context Injection

# ----------------------------------------------
# 1. CANDLESTICK PATTERNS
# ----------------------------------------------
PATTERN_DEFINITIONS = {
    "Doji": "A Doji represents indecision in the market. It forms when the open and close are virtually equal. It often signals a potential reversal or a pause in the current trend.",
    "Marubozu": "A Marubozu is a powerful single-candle pattern with no wicks, indicating that buyers (or sellers) controlled the price from open to close. It suggests strong continuation.",
    "Hammer": "A Hammer is a bullish reversal pattern found at the bottom of a downtrend. It shows that sellers pushed price down, but buyers rejected the lows and pushed it back up.",
    "Hanging Man": "A Hanging Man is a bearish reversal pattern found at the top of an uptrend. It looks like a hammer but indicates that selling pressure is starting to increase.",
    "Shooting Star": "A Shooting Star is a bearish reversal pattern found at the top of an uptrend. It has a long upper wick, showing that buyers tried to push price up but failed.",
    "Bullish Engulfing": "A strong two-candle reversal pattern where a small red candle is followed by a large green candle that completely 'engulfs' the previous one. Strongly Bullish.",
    "Bearish Engulfing": "A strong two-candle reversal pattern where a small green candle is followed by a large red candle that completely 'engulfs' the previous one. Strongly Bearish.",
    "Tweezer Top": "A Tweezer Top occurs when two candlesticks share the same high. It indicates strong resistance and a potential bearish reversal.",
    "Tweezer Bottom": "A Tweezer Bottom occurs when two candlesticks share the same low. It indicates strong support and a potential bullish reversal.",
    "Piercing Pattern": "A bullish two-candle reversal pattern where a green candle closes more than halfway up the previous red candle's body.",
    "Dark Cloud Cover": "A bearish two-candle reversal pattern where a red candle opens above the previous high but closes below the mid-point of the previous green candle.",
    "Morning Star": "A three-candle bullish reversal pattern: a long red candle, a small gap-down doji/spin, and a long green candle. Very reliable bottom signal.",
    "Evening Star": "A three-candle bearish reversal pattern: a long green candle, a small gap-up doji/spin, and a long red candle. Very reliable top signal.",
    "Inside Bar": "An Inside Bar indicates consolidation. The entire range of the candle is within the previous candle's range. It often precedes a breakout."
}

# ----------------------------------------------
# 2. TECHNICAL INDICATORS
# ----------------------------------------------
INDICATOR_DEFINITIONS = {
    "RSI14": "The Relative Strength Index (RSI) measures momentum. Values > 70 suggest Overbought (risk of drop), while < 30 suggest Oversold (potential bounce).",
    "MACD": "MACD is a trend-following momentum indicator. A positive histogram or 'Bullish Crossover' suggests upward momentum. Negative suggests downward.",
    "EMA20": "The 20-period Exponential Moving Average. Price above EMA20 often indicates short-term bullish trend. Price below indicates bearishness.",
    "SMA50": "The 50-day Simple Moving Average. Often acts as a major dynamic support or resistance level for medium-term trends.",
    "ADX14": "The Average Directional Index (ADX) measures trend strength. ADX > 25 means a Strong Trend (doesn't say direction). ADX < 20 means a Weak Trend or ranging market."
}

# ----------------------------------------------
# 3. RISKS & FUNDAMENTALS
# ----------------------------------------------
METRIC_DEFINITIONS = {
    "ATR": "Average True Range (ATR) measures market volatility. It shows the average price movement per candle. Used to set intelligent Stop Losses.",
    "Volatility": "Volatility refers to how wildly the price swings. High volatility means higher risk but potential for higher rewards.",
    "Beta": "A measure of a stock's volatility in relation to the overall market. Beta > 1 means more volatile than the market.",
    "P/E Ratio": "Price-to-Earnings Ratio. A high P/E suggests the stock may be overvalued (or investors expect high growth). Low P/E can mean undervalued."
}

def get_definitions_for_context(state_dict):
    """
    Returns a string of definitions relevant to the current state.
    """
    defs = []
    
    # Check Patterns
    pats = state_dict.get("patterns", {})
    top = pats.get("top_pattern", {})
    if top:
        name = top.get("name")
        if name in PATTERN_DEFINITIONS:
            defs.append(f"Pattern Info ({name}): {PATTERN_DEFINITIONS[name]}")

    # Check Indicators (Always include basics)
    inds = state_dict.get("indicators", {})
    if "RSI14" in inds:
        defs.append(f"Indicator Info (RSI): {INDICATOR_DEFINITIONS['RSI14']}")
    if "ADX14" in inds:
        defs.append(f"Indicator Info (ADX): {INDICATOR_DEFINITIONS['ADX14']}")

    # Check Risk
    risk = state_dict.get("risk", {})
    if risk.get("atr"):
        defs.append(f"Metric Info (ATR): {METRIC_DEFINITIONS['ATR']}")
        
    return "\n".join(defs)

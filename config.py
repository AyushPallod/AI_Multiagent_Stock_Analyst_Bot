import os

# -------------------------------------------------------------------
# PROXY CONFIGURATION
# -------------------------------------------------------------------
# Set to True if you are on a corporate network/VPN using a proxy.
# Set to False to force-clear proxies (needed for localhost/Ollama).
USE_PROXY = False

# If USE_PROXY is True, set your proxy URLs here:
PROXY_URL = ""

def setup_proxy():
    """
    Call this function at the very start of the application.
    """
    if USE_PROXY:
        os.environ["YFINANCE_USE_BACKUP"] = "true"
        os.environ["YFINANCE_IPV4_ONLY"] = "true"
        print(f"CONFIG: Enabling Proxy Configuration ({PROXY_URL})")
        os.environ["HTTP_PROXY"] = PROXY_URL
        os.environ["HTTPS_PROXY"] = PROXY_URL
        os.environ["ALL_PROXY"] = PROXY_URL
        # Remove NO_PROXY if it conflicts, or set it conservatively
        if "NO_PROXY" in os.environ:
            del os.environ["NO_PROXY"]
    else:
        # FORCE CLEAR for Localhost / Home Network Compatibility
        print("CONFIG: Clearing Proxy Settings (Direct Connection Mode)")
        os.environ["HTTP_PROXY"] = ""
        os.environ["HTTPS_PROXY"] = ""
        os.environ["ALL_PROXY"] = ""
        os.environ["NO_PROXY"] = "*"

# Automatically apply on import? 
# Better to let the main entry point call it, but for safety in this project:
setup_proxy()

# -------------------------------------------------------------------
# LLM CONFIGURATION
# -------------------------------------------------------------------
# Model to use for generation (Must be pulled in Ollama)
# Options: "llama31", "phi3", "fin-llama31"
APP_LLM_MODEL = "fin-llama31"

# Specific Model for Sentiment Analysis (Faster/Finetuned)
# User should run `ollama create llama3.2-financial -f Modelfile` if they have the weights
SENTIMENT_LLM_MODEL = "llama3.2-financial" 

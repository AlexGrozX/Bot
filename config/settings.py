"""
IVB Bot — Central Configuration Loader
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from config directory
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

def _get(key: str, default=None, cast=None):
    val = os.environ.get(key, default)
    if val is None:
        return None
    if cast:
        return cast(val)
    return val

# Bloomberg
BLOOMBERG_HOST = _get("BLOOMBERG_HOST", "localhost")
BLOOMBERG_PORT = _get("BLOOMBERG_PORT", 8194, int)

# Polymarket
POLY_API_KEY        = _get("POLY_API_KEY", "")
POLY_ADDRESS        = _get("POLY_ADDRESS", "")
POLY_PRIVATE_KEY    = _get("POLY_PRIVATE_KEY", "")
POLY_API_SECRET     = _get("POLY_API_SECRET", "")
POLY_API_PASSPHRASE = _get("POLY_API_PASSPHRASE", "")
POLY_CHAIN_ID       = _get("POLY_CHAIN_ID", 137, int)

# Instruments
INSTRUMENTS = [t.strip() for t in _get("INSTRUMENTS", "XBT Curncy").split(",")]

# IVB Model Parameters
VOLUME_BREAKOUT_MULTIPLIER = _get("VOLUME_BREAKOUT_MULTIPLIER", 3.0, float)
MIN_DELTA_IMBALANCE        = _get("MIN_DELTA_IMBALANCE", 60.0, float)
BALANCE_LOOKBACK_BARS      = _get("BALANCE_LOOKBACK_BARS", 48, int)
VALUE_AREA_PCT             = _get("VALUE_AREA_PCT", 0.70, float)
LVN_THRESHOLD              = _get("LVN_THRESHOLD", 0.20, float)
BAR_INTERVAL               = _get("BAR_INTERVAL", 5, int)

# Risk
BANKROLL_USDC       = _get("BANKROLL_USDC", 400.0, float)
MAX_KELLY_FRACTION  = _get("MAX_KELLY_FRACTION", 0.25, float)
MAX_POSITION_PCT    = _get("MAX_POSITION_PCT", 0.10, float)
DAILY_LOSS_LIMIT_PCT = _get("DAILY_LOSS_LIMIT_PCT", 0.05, float)

# Session times
NY_SESSION_START    = _get("NY_SESSION_START", "09:30")
NY_SESSION_END      = _get("NY_SESSION_END", "16:00")
LONDON_SESSION_START = _get("LONDON_SESSION_START", "03:00")
LONDON_SESSION_END  = _get("LONDON_SESSION_END", "08:30")

# Dashboard
DASHBOARD_REFRESH = _get("DASHBOARD_REFRESH", 1, int)

# Logging
LOG_LEVEL = _get("LOG_LEVEL", "INFO")
LOG_FILE  = _get("LOG_FILE", "logs/ivb_bot.log")

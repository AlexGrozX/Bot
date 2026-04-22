# IVB Bot — Institutional Volume Breakout Model
### Powered by Bloomberg Terminal + Polymarket CLOB

Implements the **Fabio Valentini AMT/IVB Model** — a professional order flow strategy that identifies institutional volume breakouts using Volume Profile, Market Balance, and Delta analysis, then routes signals to Polymarket prediction markets.

---

## How It Works

The bot applies a 3-step filter on every scan cycle:

| Step | Check | What It Looks For |
|---|---|---|
| **1. Market State** | Balanced or Out-of-Balance? | Is price inside or outside the Value Area? |
| **2. Location** | Is price at a key level? | Is price at an LVN (Low Volume Node)? |
| **3. Aggression** | Institutional confirmation? | Volume ≥ 3× average + Delta imbalance ≥ 60% |

Two setups fire:
- **TREND MODEL** — Out-of-balance market, buy/sell the LVN pullback in trend direction
- **MEAN REVERSION** — Failed breakout, fade the move back toward POC

---

## Setup

### 1. Install Python dependencies
```
pip install -r requirements.txt
```

### 2. Install Bloomberg Desktop API
```
python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
```
Bloomberg Terminal must be **open and logged in** on the same machine.

### 3. Configure credentials
Open `config/.env` and fill in:
```
POLY_PRIVATE_KEY=0xYourPrivateKeyHere
POLY_API_SECRET=YourSecretHere
POLY_API_PASSPHRASE=YourPassphraseHere
```

---

## Running the Bot

| Command | What It Does |
|---|---|
| `python bot.py --scan` | Single scan, prints signals, no trading |
| `python bot.py` | Paper mode — live signals, fake money |
| `python bot.py --live` | Live trading with real USDC |

---

## Dashboard

The live dashboard displays:
- **Open Qty** — Number of active Polymarket positions
- **Open P&L** — Unrealized profit/loss on open positions
- **Daily P&L** — Today's realized + unrealized P&L
- **Live Market Data** — Real-time Bloomberg prices, CVD, market state
- **IVB Signals** — Every signal fired with full reasoning
- **Open Positions** — Current Polymarket bets

---

## Tickers Monitored

| Ticker | Asset |
|---|---|
| `XBT Curncy` | Bitcoin/USD |
| `ES1 Index` | E-mini S&P 500 Futures |
| `NQ1 Index` | E-mini Nasdaq Futures |
| `SPX Index` | S&P 500 Index |

---

## Risk Parameters (config/.env)

| Parameter | Default | Description |
|---|---|---|
| `BANKROLL_USDC` | 400 | Starting bankroll |
| `MAX_KELLY_FRACTION` | 0.25 | Fractional Kelly multiplier |
| `MAX_POSITION_PCT` | 0.10 | Max 10% of bankroll per bet |
| `DAILY_LOSS_LIMIT_PCT` | 0.05 | Stop trading if daily loss > 5% |
| `VOLUME_BREAKOUT_MULTIPLIER` | 3.0 | Volume must be 3× average |
| `MIN_DELTA_IMBALANCE` | 60.0 | 60% of volume must be directional |

---

## Important Notes

- Bloomberg Terminal **must be running** on the same Windows machine as the bot
- The bot defaults to **paper mode** — no real money until you run `--live`
- Minimum 1.5:1 Risk/Reward required for any signal to fire
- 15-minute cooldown between signals on the same ticker

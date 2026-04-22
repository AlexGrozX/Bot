# IVB Bot v2 — Institutional Volume Breakout Paper Trader

**Author:** Alexander Grozovsky  
**Version:** 2.0  
**Mode:** Paper Trading Only (no real money, no broker)  
**Data Source:** Bloomberg Terminal (Desktop API)

---

## What's New in v2

| Feature | v1 | v2 |
|---|---|---|
| Signal setups | 2 (Trend, Mean Rev) | 5 (+CVD Divergence, ORB, VWAP Reclaim) |
| Stops | Fixed % | ATR-based dynamic stops |
| Take-profit | Single TP | Partial TP at 1R + trailing stop after 2R |
| HTF confirmation | ✗ | ✓ 15-min trend alignment |
| Session context | ✗ | ✓ RTH vs ETH, Opening Range |
| CVD divergence | ✗ | ✓ Price/delta divergence detection |
| VWAP | ✗ | ✓ Session VWAP + deviation filter |
| Confidence scoring | Basic | Composite (8 factors) |
| Analytics | Win rate only | Sharpe, Max DD, Profit Factor, Expectancy, R-multiple |
| Equity curve | ✗ | ✓ Tracked + displayed as sparkline |
| Setup breakdown | ✗ | ✓ Per-setup win rate and P&L |
| Backtesting | ✗ | ✓ Full backtest module |

---

## How to Run

### Step 1: Install dependencies

```cmd
pip install -r requirements.txt
```

Bloomberg API must be installed separately:
```cmd
pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
```

### Step 2: Configure

Edit `config/.env`:
```
STARTING_BALANCE=400
MAX_POSITION_PCT=0.10
MAX_DAILY_LOSS_PCT=0.05
VOLUME_BREAKOUT_MULT=3.0
MIN_DELTA_IMBALANCE=60.0
MIN_CONFIDENCE=0.55
```

### Step 3: Start Bloomberg Terminal

Open Bloomberg Terminal and ensure these services are running:
- `//blp/mktdata` (real-time tick data)
- `//blp/mktbar` (5-minute bars)
- `//blp/refdata` (reference data)

### Step 4: Run the bot

```cmd
cd ivb_bot
python bot.py
```

The dashboard will appear in your terminal showing live signals, open positions, and performance stats.

### Step 5: Run a backtest (optional)

```cmd
python -m backtest.backtest_engine --ticker "XBT Curncy" --days 30
```

This pulls 30 days of historical 5-min bars from Bloomberg and runs the IVB engine over them, printing a full performance report.

---

## Signal Setups

| Setup | Trigger | Direction |
|---|---|---|
| **TREND_MODEL** | Out-of-balance market + price at LVN + volume 3x + delta ≥60% | Trend continuation |
| **MEAN_REVERSION** | Failed breakout + price reclaims value area + volume + delta | Toward POC |
| **CVD_DIVERGENCE** | Price makes new high/low but CVD does not confirm | Reversal |
| **ORB_BREAKOUT** | Price breaks above/below 9:30–9:45 opening range with 2x volume | Breakout |
| **VWAP_RECLAIM** | Price crosses VWAP with delta confirmation | Continuation |

---

## Risk Management

- **Position size:** Kelly fraction (confidence × 25% of balance), capped at 10% of balance
- **Stop loss:** 1.5 × ATR below/above entry (dynamic, not fixed)
- **Partial TP:** 50% of position closed at 1R profit; stop moved to breakeven
- **Trailing stop:** Activates after 2R; trails at 1.5 × ATR behind peak price
- **Daily loss halt:** Trading stops if daily loss exceeds 5% of starting balance
- **Cooldown:** 15-minute cooldown per ticker after each signal

---

## File Structure

```
ivb_bot/
├── bot.py                          ← Main entry point
├── requirements.txt
├── README_v2.md
├── config/
│   └── .env                        ← Configuration
├── bloomberg/
│   └── data_feed.py                ← Bloomberg real-time data layer
├── engine/
│   └── ivb_engine.py               ← IVB signal engine v2
├── execution/
│   └── paper_trader.py             ← Paper trading engine v2
├── dashboard/
│   └── dashboard.py                ← Terminal dashboard v2
├── backtest/
│   ├── __init__.py
│   ├── backtest_engine.py          ← Historical backtest runner
│   └── data/                       ← CSV fallback data (optional)
├── logs/
│   ├── trades.json                 ← Trade log (auto-saved)
│   └── equity_curve.json           ← Equity curve (auto-saved)
└── models/
    └── ivb_scorer.pkl              ← ML model (optional, auto-trained)
```

---

## Tickers Monitored

| Bloomberg Ticker | Market |
|---|---|
| `XBT Curncy` | Bitcoin (BTC/USD) |
| `ES1 Index` | S&P 500 E-mini Futures |
| `NQ1 Index` | Nasdaq 100 E-mini Futures |
| `SPX Index` | S&P 500 Index |

---

## Performance Metrics Explained

| Metric | What it means |
|---|---|
| **Win Rate** | % of trades that were profitable |
| **Sharpe Ratio** | Risk-adjusted return (>1.0 is good, >2.0 is excellent) |
| **Max Drawdown** | Largest peak-to-trough loss (keep below 15%) |
| **Profit Factor** | Gross wins / gross losses (>1.5 is good) |
| **Expectancy** | Average dollar profit per trade |
| **R-multiple** | Profit expressed in units of initial risk (>0.5R avg is good) |

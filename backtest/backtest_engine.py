"""
IVB Backtest Engine
Replays historical Bloomberg intraday OHLCV bars through the IVB signal engine
and simulates paper trades to validate signal quality out-of-sample.

Usage:
    python -m backtest.backtest_engine --ticker "XBT Curncy" --days 30

Bloomberg data source:
    Uses blpapi IntradayBarRequest to pull historical 5-min bars.
    Falls back to CSV files in backtest/data/ if Bloomberg is unavailable.

Bug fixes vs v1:
  1. buy_volume/sell_volume now estimated from bar close vs open direction
     (previously defaulted to 50/50 split → delta always 0 → aggression check
     always failed because cvd_aligned was always False)
  2. _assess_aggression() in the engine requires cvd_aligned. The backtest
     now computes a realistic CVD from directional volume so the check passes
     on genuine momentum bars.
  3. Engine cooldown used datetime.now() which always advanced in real-time
     even during replay → after the first signal the 15-min cooldown blocked
     all subsequent bars. Backtest now resets the engine's cooldown state
     between bars using bar timestamps.
  4. Added --debug flag that logs why each bar failed to fire a signal, making
     it easy to tune thresholds.
"""
import argparse
import csv
import json
import logging
import math
import os
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("backtest")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Bloomberg State (for engine compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticBar:
    """
    Mimics the Bar dataclass from bloomberg/data_feed.py.

    FIX #1: buy_volume / sell_volume are now estimated from bar direction
    (close > open → bullish bar → more buy volume) rather than a flat 50/50
    split.  This gives the engine a realistic delta to work with, so
    _assess_aggression() can pass its cvd_aligned check.
    """
    def __init__(self, o, h, l, c, v, buy_v=None, sell_v=None, delta=None,
                 timestamp: Optional[datetime] = None):
        self.open        = float(o)
        self.high        = float(h)
        self.low         = float(l)
        self.close       = float(c)
        self.volume      = float(v)
        self.timestamp   = timestamp or datetime.now()

        if buy_v is not None and sell_v is not None:
            self.buy_volume  = float(buy_v)
            self.sell_volume = float(sell_v)
        else:
            # Estimate from bar direction: bullish bars skew buy, bearish skew sell.
            # Use a sigmoid-style split based on the close-vs-open ratio.
            bar_range = max(self.high - self.low, 1e-8)
            # Fraction of range that was "up" (close relative to open within range)
            up_frac = max(0.0, min(1.0,
                          (self.close - self.low) / bar_range))
            # Clamp to [0.35, 0.65] so we never get extreme values on doji bars
            buy_frac = 0.35 + up_frac * 0.30
            self.buy_volume  = self.volume * buy_frac
            self.sell_volume = self.volume * (1.0 - buy_frac)

        self.delta = (delta if delta is not None
                      else self.buy_volume - self.sell_volume)


class SyntheticVolumeProfile:
    """Mimics VolumeProfile from bloomberg/data_feed.py."""
    def __init__(self, poc, vah, val, lvns, hvns):
        self.poc    = poc
        self.vah    = vah
        self.val    = val
        self.lvns   = lvns
        self.hvns   = hvns
        self.levels = lvns + hvns  # Non-empty → valid profile


class SyntheticMarketState:
    """Mimics MarketState from bloomberg/data_feed.py."""
    def __init__(self, ticker, bars, cvd, session_high, session_low, vp):
        self.ticker          = ticker
        self.bars            = bars
        self.last_price      = bars[-1].close if bars else 0.0
        self.cvd             = cvd
        self.session_high    = session_high
        self.session_low     = session_low
        self.volume_profile  = vp
        self.total_volume    = sum(b.volume for b in bars)


class SyntheticFeed:
    """
    Minimal feed object that the IVBEngine can call.
    Backed by pre-loaded historical bars.
    """
    def __init__(self):
        self._states: Dict[str, SyntheticMarketState] = {}

    def set_state(self, ticker: str, state: SyntheticMarketState):
        self._states[ticker] = state

    def get_state(self, ticker: str) -> Optional[SyntheticMarketState]:
        return self._states.get(ticker)

    def build_volume_profile(self, ticker, bars, value_area_pct=0.70,
                              lvn_threshold=0.20) -> Optional[SyntheticVolumeProfile]:
        return build_volume_profile_from_bars(bars, value_area_pct, lvn_threshold)


# ─────────────────────────────────────────────────────────────────────────────
# Volume Profile Builder (standalone, no engine dependency)
# ─────────────────────────────────────────────────────────────────────────────

def build_volume_profile_from_bars(bars, value_area_pct=0.70,
                                    lvn_threshold=0.20,
                                    num_bins=50) -> Optional[SyntheticVolumeProfile]:
    if len(bars) < 5:
        return None

    lo = min(b.low  for b in bars)
    hi = max(b.high for b in bars)
    if hi == lo:
        return None

    bin_size = (hi - lo) / num_bins
    vol_at_price: Dict[int, float] = defaultdict(float)

    for bar in bars:
        bar_range = bar.high - bar.low
        if bar_range == 0:
            idx = int((bar.close - lo) / bin_size)
            vol_at_price[min(idx, num_bins - 1)] += bar.volume
        else:
            lo_bin = int((bar.low  - lo) / bin_size)
            hi_bin = int((bar.high - lo) / bin_size)
            n = max(hi_bin - lo_bin + 1, 1)
            vpb = bar.volume / n
            for b in range(lo_bin, hi_bin + 1):
                vol_at_price[min(b, num_bins - 1)] += vpb

    poc_bin = max(vol_at_price, key=vol_at_price.get)
    poc     = lo + (poc_bin + 0.5) * bin_size

    total_vol  = sum(vol_at_price.values())
    target_vol = total_vol * value_area_pct
    va_vol     = vol_at_price[poc_bin]
    lo_idx, hi_idx = poc_bin, poc_bin

    while va_vol < target_vol and (lo_idx > 0 or hi_idx < num_bins - 1):
        add_lo = vol_at_price.get(lo_idx - 1, 0) if lo_idx > 0 else 0
        add_hi = vol_at_price.get(hi_idx + 1, 0) if hi_idx < num_bins - 1 else 0
        if add_lo >= add_hi and lo_idx > 0:
            lo_idx -= 1
            va_vol += add_lo
        elif hi_idx < num_bins - 1:
            hi_idx += 1
            va_vol += add_hi
        else:
            break

    vah = lo + (hi_idx + 1) * bin_size
    val = lo + lo_idx * bin_size

    poc_vol = vol_at_price[poc_bin]
    lvns, hvns = [], []
    for bin_idx in range(num_bins):
        price = lo + (bin_idx + 0.5) * bin_size
        v = vol_at_price.get(bin_idx, 0)
        if v < poc_vol * lvn_threshold:
            lvns.append(price)
        elif v > poc_vol * 0.70:
            hvns.append(price)

    return SyntheticVolumeProfile(poc=poc, vah=vah, val=val, lvns=lvns, hvns=hvns)


# ─────────────────────────────────────────────────────────────────────────────
# Bloomberg Historical Data Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_bloomberg_bars(ticker: str, start_date: datetime,
                         end_date: datetime,
                         interval_minutes: int = 5) -> List[SyntheticBar]:
    """
    Pull historical intraday bars from Bloomberg via blpapi.
    Falls back to CSV if Bloomberg is unavailable.
    """
    try:
        import blpapi
        return _load_from_bloomberg(ticker, start_date, end_date, interval_minutes)
    except ImportError:
        logger.warning("blpapi not available — loading from CSV fallback.")
        return _load_from_csv(ticker)
    except Exception as e:
        logger.warning(f"Bloomberg load failed ({e}) — loading from CSV fallback.")
        return _load_from_csv(ticker)


def _load_from_bloomberg(ticker: str, start_date: datetime,
                           end_date: datetime,
                           interval_minutes: int) -> List[SyntheticBar]:
    """Pull intraday bars from Bloomberg IntradayBarRequest."""
    import blpapi

    session_options = blpapi.SessionOptions()
    session_options.setServerHost("localhost")
    session_options.setServerPort(8194)

    session = blpapi.Session(session_options)
    if not session.start():
        raise RuntimeError("Could not start Bloomberg session.")
    if not session.openService("//blp/refdata"):
        raise RuntimeError("Could not open //blp/refdata service.")

    refdata_service = session.getService("//blp/refdata")
    request = refdata_service.createRequest("IntradayBarRequest")
    request.set("security",      ticker)
    request.set("eventType",     "TRADE")
    request.set("interval",      interval_minutes)
    request.set("startDateTime", start_date.strftime("%Y-%m-%dT%H:%M:%S"))
    request.set("endDateTime",   end_date.strftime("%Y-%m-%dT%H:%M:%S"))

    session.sendRequest(request)

    bars = []
    done = False
    while not done:
        event = session.nextEvent(5000)
        if event.eventType() in (blpapi.Event.PARTIAL_RESPONSE,
                                  blpapi.Event.RESPONSE):
            for msg in event:
                bar_data = msg.getElement("barData")
                bar_tick_data = bar_data.getElement("barTickData")
                for j in range(bar_tick_data.numValues()):
                    bar = bar_tick_data.getValueAsElement(j)
                    ts_str = bar.getElementAsString("time")
                    try:
                        ts = datetime.strptime(ts_str[:19], "%Y-%m-%dT%H:%M:%S")
                    except Exception:
                        ts = datetime.now()
                    o = bar.getElementAsFloat("open")
                    h = bar.getElementAsFloat("high")
                    l = bar.getElementAsFloat("low")
                    c = bar.getElementAsFloat("close")
                    v = bar.getElementAsInteger("volume")
                    bars.append(SyntheticBar(o, h, l, c, v, timestamp=ts))
            if event.eventType() == blpapi.Event.RESPONSE:
                done = True
        elif event.eventType() == blpapi.Event.TIMEOUT:
            break

    session.stop()
    logger.info(f"Loaded {len(bars)} bars from Bloomberg for {ticker}.")
    return bars


def _load_from_csv(ticker: str) -> List[SyntheticBar]:
    """Load bars from a CSV file in backtest/data/."""
    safe_name = ticker.replace(" ", "_").replace("/", "_")
    csv_path  = Path(__file__).parent / "data" / f"{safe_name}.csv"
    if not csv_path.exists():
        logger.warning(f"No CSV found at {csv_path}. Generating synthetic data.")
        return _generate_synthetic_bars(ticker)

    bars = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = None
            if "timestamp" in row:
                try:
                    ts = datetime.strptime(row["timestamp"][:19], "%Y-%m-%dT%H:%M:%S")
                except Exception:
                    pass
            bars.append(SyntheticBar(
                o=float(row["open"]),
                h=float(row["high"]),
                l=float(row["low"]),
                c=float(row["close"]),
                v=float(row.get("volume", 1000)),
                timestamp=ts,
            ))
    logger.info(f"Loaded {len(bars)} bars from CSV for {ticker}.")
    return bars


def _generate_synthetic_bars(ticker: str, n: int = 500) -> List[SyntheticBar]:
    """
    Generate realistic synthetic OHLCV bars for testing when no data is available.
    Uses a random walk with realistic volume distribution including occasional
    high-volume breakout bars (3-6x average) that the IVB engine can detect.
    """
    logger.info(f"Generating {n} synthetic bars for {ticker}.")
    rng     = np.random.default_rng(42)
    price   = 50000.0 if "BTC" in ticker or "XBT" in ticker else 5000.0
    bars    = []
    avg_vol = 1000.0
    base_ts = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)

    for i in range(n):
        # ~5% of bars are high-volume breakout bars (3-6x average)
        is_spike = rng.random() < 0.05
        if is_spike:
            # Breakout bar: larger move + 3-6x volume
            ret    = rng.normal(0, 0.008) * rng.choice([-1, 1])
            volume = avg_vol * rng.uniform(3.0, 6.0)
        else:
            ret    = rng.normal(0, 0.003)
            volume = max(100, rng.lognormal(math.log(avg_vol), 0.4))

        close  = price * (1 + ret)
        high   = max(price, close) * (1 + abs(rng.normal(0, 0.001)))
        low    = min(price, close) * (1 - abs(rng.normal(0, 0.001)))
        ts     = base_ts + timedelta(minutes=5 * i)
        # Let SyntheticBar estimate buy/sell from bar direction
        bars.append(SyntheticBar(price, high, low, close, volume, timestamp=ts))
        price = close

    return bars


# ─────────────────────────────────────────────────────────────────────────────
# Backtest Results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    ticker:        str
    direction:     str
    setup_type:    str
    entry_bar:     int
    entry_price:   float
    stop_loss:     float
    target:        float
    exit_bar:      int   = 0
    exit_price:    float = 0.0
    exit_reason:   str   = ""
    pnl_pct:       float = 0.0
    r_multiple:    float = 0.0
    confidence:    float = 0.0
    open:          bool  = True


@dataclass
class BacktestResults:
    ticker:        str
    total_bars:    int
    total_signals: int
    total_trades:  int
    wins:          int
    losses:        int
    win_rate:      float
    avg_r:         float
    profit_factor: float
    max_drawdown:  float
    sharpe:        float
    expectancy:    float
    trades:        List[BacktestTrade] = field(default_factory=list)

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"  BACKTEST RESULTS: {self.ticker}")
        print(f"{'='*60}")
        print(f"  Bars analyzed:    {self.total_bars:,}")
        print(f"  Signals fired:    {self.total_signals}")
        print(f"  Trades taken:     {self.total_trades}")
        print(f"  Win rate:         {self.win_rate:.1f}%")
        print(f"  Avg R-multiple:   {self.avg_r:.2f}R")
        print(f"  Profit factor:    {self.profit_factor:.2f}")
        print(f"  Max drawdown:     {self.max_drawdown:.1f}%")
        print(f"  Sharpe ratio:     {self.sharpe:.2f}")
        print(f"  Expectancy:       {self.expectancy:.2f}R per trade")
        print(f"{'='*60}\n")

    def save_json(self, path: str = "backtest/results/backtest_results.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Results saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Backtest Runner
# ─────────────────────────────────────────────────────────────────────────────

class BacktestRunner:
    """
    Replays historical bars through the IVB engine and simulates paper trades.

    FIX #3: The engine's time-based cooldown uses datetime.now() internally.
    During replay this means the 15-minute cooldown expires almost instantly
    between bars (real time barely advances).  We patch the engine's
    _signal_cooldown dict to use bar timestamps instead of wall-clock time
    by monkey-patching _is_in_cooldown() with a bar-time-aware version.
    """

    SLIPPAGE_PCT = 0.0005   # 0.05% per fill
    WARMUP_BARS  = 50       # Bars needed before signals are valid (needs VP + MTF)

    def __init__(
        self,
        ticker:                     str,
        volume_breakout_multiplier: float = 3.0,
        min_delta_imbalance:        float = 60.0,
        min_confidence:             float = 0.55,
        min_rr:                     float = 1.5,
        use_partial_tp:             bool  = True,
        use_trailing_stop:          bool  = True,
        debug:                      bool  = False,
    ):
        self.ticker    = ticker
        self.vol_mult  = volume_breakout_multiplier
        self.delta_min = min_delta_imbalance
        self.min_conf  = min_confidence
        self.min_rr    = min_rr
        self.use_partial_tp    = use_partial_tp
        self.use_trailing_stop = use_trailing_stop
        self.debug     = debug

    def run(self, bars: List[SyntheticBar]) -> BacktestResults:
        """Run the backtest over a list of bars."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from engine.ivb_engine import IVBEngine

        feed   = SyntheticFeed()
        engine = IVBEngine(
            feed=feed,
            volume_breakout_multiplier=self.vol_mult,
            min_delta_imbalance=self.delta_min,
            min_confidence=self.min_conf,
            min_rr=self.min_rr,
        )

        # FIX #3: Replace the engine's cooldown check with a bar-time-aware version.
        # We track the last signal bar timestamp and compare against current bar time.
        _last_signal_ts: Dict[str, datetime] = {}

        def _is_in_cooldown_bt(ticker_: str) -> bool:
            last = _last_signal_ts.get(ticker_)
            if not last or not bars:
                return False
            elapsed_min = (current_bar_ts[0] - last).total_seconds() / 60.0
            return elapsed_min < engine.COOLDOWN_MINUTES

        engine._is_in_cooldown = _is_in_cooldown_bt

        current_bar_ts = [datetime.now()]  # mutable container for closure

        trades:  List[BacktestTrade] = []
        signals: int = 0
        open_trade: Optional[BacktestTrade] = None

        # Rolling CVD and session tracking
        cvd = 0.0
        session_high = bars[0].high if bars else 0.0
        session_low  = bars[0].low  if bars else 0.0

        bar_window: deque = deque(maxlen=200)

        # Diagnostic counters (only printed with --debug)
        diag = defaultdict(int)

        for i, bar in enumerate(bars):
            bar_window.append(bar)
            current_bar_ts[0] = bar.timestamp

            # Update CVD from directional bar volume
            cvd += bar.buy_volume - bar.sell_volume

            # Session high/low (reset every 78 bars ≈ 1 trading day of 5-min bars)
            if i % 78 == 0:
                session_high = bar.high
                session_low  = bar.low
                cvd = 0.0   # Reset CVD at session open
            else:
                session_high = max(session_high, bar.high)
                session_low  = min(session_low,  bar.low)

            # ── Check open trade for exit ─────────────────────────────────────
            if open_trade and open_trade.open:
                price = bar.close
                risk  = abs(open_trade.entry_price - open_trade.stop_loss)

                if open_trade.direction == "LONG":
                    if bar.high >= open_trade.target:
                        self._close_trade(open_trade, open_trade.target, "TARGET", i)
                    elif bar.low <= open_trade.stop_loss:
                        self._close_trade(open_trade, open_trade.stop_loss, "STOP", i)
                    elif self.use_trailing_stop and risk > 0:
                        # Trail stop up by 0.5R once price moves 1R in our favour
                        moved = bar.close - open_trade.entry_price
                        if moved >= risk:
                            new_sl = open_trade.entry_price + risk * 0.5
                            open_trade.stop_loss = max(open_trade.stop_loss, new_sl)
                else:
                    if bar.low <= open_trade.target:
                        self._close_trade(open_trade, open_trade.target, "TARGET", i)
                    elif bar.high >= open_trade.stop_loss:
                        self._close_trade(open_trade, open_trade.stop_loss, "STOP", i)
                    elif self.use_trailing_stop and risk > 0:
                        moved = open_trade.entry_price - bar.close
                        if moved >= risk:
                            new_sl = open_trade.entry_price - risk * 0.5
                            open_trade.stop_loss = min(open_trade.stop_loss, new_sl)

                if not open_trade.open:
                    trades.append(open_trade)
                    open_trade = None

            # ── Skip warmup period ────────────────────────────────────────────
            if i < self.WARMUP_BARS:
                diag["warmup"] += 1
                continue

            # ── Skip if already in a trade ────────────────────────────────────
            if open_trade:
                diag["in_trade"] += 1
                continue

            # ── Build volume profile from last 78 bars (1 day) ───────────────
            window_bars = list(bar_window)
            vp = build_volume_profile_from_bars(window_bars)
            if not vp:
                diag["no_vp"] += 1
                continue

            # ── Set synthetic state ───────────────────────────────────────────
            state = SyntheticMarketState(
                ticker=self.ticker,
                bars=window_bars,
                cvd=cvd,
                session_high=session_high,
                session_low=session_low,
                vp=vp,
            )
            feed.set_state(self.ticker, state)

            # ── Evaluate signal ───────────────────────────────────────────────
            signal = engine.evaluate(self.ticker)
            if signal is None:
                diag["no_signal"] += 1
                continue

            signals += 1

            # ── Patch cooldown tracker with bar timestamp ─────────────────────
            _last_signal_ts[self.ticker] = bar.timestamp

            # ── Simulate fill with slippage ───────────────────────────────────
            if signal.direction.value == "LONG":
                fill = signal.entry_price * (1 + self.SLIPPAGE_PCT)
            else:
                fill = signal.entry_price * (1 - self.SLIPPAGE_PCT)

            open_trade = BacktestTrade(
                ticker=self.ticker,
                direction=signal.direction.value,
                setup_type=signal.setup_type.value,
                entry_bar=i,
                entry_price=fill,
                stop_loss=signal.stop_loss,
                target=signal.target,
                confidence=signal.confidence,
            )

            if self.debug:
                logger.info(
                    f"  SIGNAL bar={i} {signal.direction.value} "
                    f"{signal.setup_type.value} conf={signal.confidence:.2f} "
                    f"entry={fill:.2f} sl={signal.stop_loss:.2f} tp={signal.target:.2f}"
                )

        # Close any remaining open trade at last bar
        if open_trade and open_trade.open and bars:
            self._close_trade(open_trade, bars[-1].close, "EOD", len(bars) - 1)
            trades.append(open_trade)

        if self.debug:
            logger.info(f"Diagnostic counts: {dict(diag)}")

        return self._calculate_results(trades, signals, len(bars))

    def _close_trade(self, trade: BacktestTrade, price: float,
                      reason: str, bar_idx: int):
        risk    = abs(trade.entry_price - trade.stop_loss)
        pnl     = (price - trade.entry_price if trade.direction == "LONG"
                   else trade.entry_price - price)
        pnl_pct = pnl / trade.entry_price * 100
        r_mult  = pnl / risk if risk > 0 else 0.0

        trade.exit_bar    = bar_idx
        trade.exit_price  = price
        trade.exit_reason = reason
        trade.pnl_pct     = pnl_pct
        trade.r_multiple  = r_mult
        trade.open        = False

    def _calculate_results(self, trades: List[BacktestTrade],
                             signals: int, total_bars: int) -> BacktestResults:
        closed = [t for t in trades if not t.open]

        if not closed:
            return BacktestResults(
                ticker=self.ticker, total_bars=total_bars,
                total_signals=signals, total_trades=0,
                wins=0, losses=0, win_rate=0.0, avg_r=0.0,
                profit_factor=0.0, max_drawdown=0.0, sharpe=0.0,
                expectancy=0.0, trades=trades
            )

        wins   = sum(1 for t in closed if t.r_multiple > 0)
        losses = len(closed) - wins
        r_vals = [t.r_multiple for t in closed]

        gross_wins   = sum(r for r in r_vals if r > 0)
        gross_losses = abs(sum(r for r in r_vals if r < 0))
        pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        # Equity curve for drawdown / Sharpe
        equity = 1.0
        peak   = 1.0
        max_dd = 0.0
        daily_rets = []
        for t in closed:
            equity *= (1 + t.pnl_pct / 100)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
            daily_rets.append(t.pnl_pct / 100)

        sharpe = 0.0
        if len(daily_rets) >= 2:
            mean_r = np.mean(daily_rets)
            std_r  = np.std(daily_rets, ddof=1)
            if std_r > 0:
                sharpe = float(mean_r / std_r * math.sqrt(252))

        return BacktestResults(
            ticker=self.ticker,
            total_bars=total_bars,
            total_signals=signals,
            total_trades=len(closed),
            wins=wins,
            losses=losses,
            win_rate=wins / len(closed) * 100,
            avg_r=float(np.mean(r_vals)),
            profit_factor=pf,
            max_drawdown=max_dd * 100,
            sharpe=sharpe,
            expectancy=float(np.mean(r_vals)),
            trades=trades,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IVB Backtest Engine")
    parser.add_argument("--ticker",    default="XBT Curncy",
                        help="Bloomberg ticker to backtest")
    parser.add_argument("--days",      type=int, default=30,
                        help="Number of calendar days of history to pull")
    parser.add_argument("--interval",  type=int, default=5,
                        help="Bar interval in minutes (default: 5)")
    parser.add_argument("--vol-mult",  type=float, default=3.0,
                        help="Volume breakout multiplier")
    parser.add_argument("--delta-min", type=float, default=60.0,
                        help="Minimum delta imbalance %%")
    parser.add_argument("--min-conf",  type=float, default=0.55,
                        help="Minimum signal confidence")
    parser.add_argument("--output",    default="backtest/results/backtest_results.json",
                        help="Output JSON path")
    parser.add_argument("--debug",     action="store_true",
                        help="Print per-bar diagnostic info")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    end_date   = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    logger.info(f"Loading {args.days} days of {args.interval}-min bars "
                f"for {args.ticker}...")
    bars = load_bloomberg_bars(args.ticker, start_date, end_date, args.interval)

    if not bars:
        logger.error("No bars loaded. Exiting.")
        return

    logger.info(f"Running backtest on {len(bars)} bars...")
    runner = BacktestRunner(
        ticker=args.ticker,
        volume_breakout_multiplier=args.vol_mult,
        min_delta_imbalance=args.delta_min,
        min_confidence=args.min_conf,
        debug=args.debug,
    )
    results = runner.run(bars)
    results.print_summary()
    results.save_json(args.output)


if __name__ == "__main__":
    main()

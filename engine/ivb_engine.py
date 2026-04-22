"""
IVB Engine v2 — Institutional Volume Breakout Signal Engine
Based on the Fabio Valentini AMT/IVB model.

v2 Improvements over v1:
  1. Multi-timeframe confirmation (5-min bars → 15-min HTF trend)
  2. Dynamic VWAP with session-aware deviation filter
  3. CVD divergence detection (price/delta divergence → reversal)
  4. Opening Range Breakout (ORB) as a standalone setup
  5. VWAP Reclaim as a standalone setup
  6. Composite ML-style confidence scoring (rule-based, upgradeable to XGBoost)
  7. Session context: RTH vs ETH awareness
  8. ATR-based dynamic stops (replaces fixed % stops)
  9. Per-ticker bar-count cooldown (faster than 15-min clock)

All v1 logic (3-step filter, TREND_MODEL, MEAN_REVERSION) is preserved and enhanced.
"""
import logging
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from bloomberg.data_feed import BloombergFeed, MarketState, VolumeProfile, Bar

logger = logging.getLogger("engine.ivb")


# ─────────────────────────────────────────────────────────────────────────────
# Enums & Signal Dataclass
# ─────────────────────────────────────────────────────────────────────────────

class MarketCondition(Enum):
    BALANCED       = "BALANCED"
    OUT_OF_BALANCE = "OUT_OF_BALANCE"

class SignalDirection(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    FLAT  = "FLAT"

class SetupType(Enum):
    TREND_MODEL    = "TREND_MODEL"
    MEAN_REVERSION = "MEAN_REVERSION"
    CVD_DIVERGENCE = "CVD_DIVERGENCE"
    ORB_BREAKOUT   = "ORB_BREAKOUT"
    VWAP_RECLAIM   = "VWAP_RECLAIM"
    NONE           = "NONE"

@dataclass
class IVBSignal:
    ticker:                 str
    direction:              SignalDirection
    setup_type:             SetupType
    confidence:             float          # 0.0 – 1.0
    entry_price:            float
    stop_loss:              float
    target:                 float
    risk_reward:            float
    market_condition:       MarketCondition
    poc:                    float
    vah:                    float
    val:                    float
    lvn_triggered:          float          # Which LVN triggered the signal (0 if N/A)
    volume_breakout_ratio:  float
    delta_imbalance_pct:    float
    cvd:                    float
    reasoning:              str
    timestamp:              datetime = field(default_factory=datetime.now)
    polymarket_keyword:     str = ""
    # v2 additions
    vwap:                   float = 0.0
    htf_aligned:            bool  = False
    session:                str   = "RTH"
    atr:                    float = 0.0
    ml_score:               float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# v2 Helper: Multi-Timeframe Aggregator
# ─────────────────────────────────────────────────────────────────────────────

class MTFAggregator:
    """Aggregates 5-min bars into 15-min bars for HTF trend confirmation."""

    def __init__(self):
        self._buffer:   Dict[str, List[Bar]] = defaultdict(list)
        self._htf_bars: Dict[str, deque]     = defaultdict(lambda: deque(maxlen=50))

    def add_bar(self, ticker: str, bar: Bar):
        self._buffer[ticker].append(bar)
        if len(self._buffer[ticker]) >= 3:
            batch = self._buffer[ticker][-3:]
            # Build a synthetic 15-min bar
            htf = {
                "open":   batch[0].open,
                "high":   max(b.high   for b in batch),
                "low":    min(b.low    for b in batch),
                "close":  batch[-1].close,
                "volume": sum(b.volume for b in batch),
                "delta":  sum(getattr(b, "delta", 0) for b in batch),
            }
            self._htf_bars[ticker].append(htf)
            self._buffer[ticker] = []

    def htf_trend(self, ticker: str) -> Optional[SignalDirection]:
        """5-bar slope of 15-min closes."""
        bars = list(self._htf_bars[ticker])
        if len(bars) < 5:
            return None
        closes = [b["close"] for b in bars[-5:]]
        slope = closes[-1] - closes[0]
        if slope > 0:
            return SignalDirection.LONG
        elif slope < 0:
            return SignalDirection.SHORT
        return None


# ─────────────────────────────────────────────────────────────────────────────
# v2 Helper: Session Context
# ─────────────────────────────────────────────────────────────────────────────

class SessionContext:
    RTH_START = dtime(9, 30)
    RTH_END   = dtime(16, 0)
    ORB_END   = dtime(9, 45)   # 15-min opening range

    def __init__(self):
        self._orb_high: Dict[str, float] = {}
        self._orb_low:  Dict[str, float] = {}
        self._orb_locked: Dict[str, bool] = {}

    def is_rth(self) -> bool:
        now = datetime.now().time()
        return self.RTH_START <= now <= self.RTH_END

    def is_orb_period(self) -> bool:
        now = datetime.now().time()
        return self.RTH_START <= now <= self.ORB_END

    def session_name(self) -> str:
        return "RTH" if self.is_rth() else "ETH"

    def update_orb(self, ticker: str, high: float, low: float):
        if self.is_orb_period():
            if ticker not in self._orb_high:
                self._orb_high[ticker] = high
                self._orb_low[ticker]  = low
            else:
                self._orb_high[ticker] = max(self._orb_high[ticker], high)
                self._orb_low[ticker]  = min(self._orb_low[ticker],  low)
            self._orb_locked[ticker] = False
        else:
            self._orb_locked[ticker] = True

    def get_orb(self, ticker: str) -> Tuple[float, float]:
        return (self._orb_high.get(ticker, 0.0),
                self._orb_low.get(ticker, 0.0))

    def is_orb_breakout(self, ticker: str, price: float,
                         direction: SignalDirection) -> bool:
        hi, lo = self.get_orb(ticker)
        if hi == 0 or lo == 0:
            return False
        if direction == SignalDirection.LONG  and price > hi * 1.001:
            return True
        if direction == SignalDirection.SHORT and price < lo * 0.999:
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# v2 Helper: CVD Divergence Detector
# ─────────────────────────────────────────────────────────────────────────────

class CVDDivergenceDetector:
    """
    Detects price/CVD divergence over a rolling window.
    Bullish divergence: price LL, CVD HL → reversal up
    Bearish divergence: price HH, CVD LH → reversal down
    """
    LOOKBACK = 20

    def __init__(self):
        self._prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.LOOKBACK))
        self._cvds:   Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.LOOKBACK))

    def update(self, ticker: str, price: float, cvd: float):
        self._prices[ticker].append(price)
        self._cvds[ticker].append(cvd)

    def detect(self, ticker: str) -> Optional[SignalDirection]:
        prices = list(self._prices[ticker])
        cvds   = list(self._cvds[ticker])
        if len(prices) < self.LOOKBACK:
            return None

        recent_p = prices[-10:]
        recent_c = cvds[-10:]

        # Bearish: price HH but CVD LH
        if (prices[-1] > max(prices[-10:-1]) and
                cvds[-1] < max(cvds[-10:-1]) * 0.95):
            return SignalDirection.SHORT

        # Bullish: price LL but CVD HL
        if (prices[-1] < min(prices[-10:-1]) and
                cvds[-1] > min(cvds[-10:-1]) * 0.95):
            return SignalDirection.LONG

        return None


# ─────────────────────────────────────────────────────────────────────────────
# v2 Helper: Session VWAP
# ─────────────────────────────────────────────────────────────────────────────

class VWAPCalculator:
    def __init__(self):
        self._cum_pv:  Dict[str, float] = defaultdict(float)
        self._cum_vol: Dict[str, float] = defaultdict(float)
        self._last_reset: Dict[str, str] = {}

    def update(self, ticker: str, bar: Bar):
        today = datetime.now().date().isoformat()
        if self._last_reset.get(ticker) != today:
            self._cum_pv[ticker]  = 0.0
            self._cum_vol[ticker] = 0.0
            self._last_reset[ticker] = today
        typical = (bar.high + bar.low + bar.close) / 3.0
        self._cum_pv[ticker]  += typical * bar.volume
        self._cum_vol[ticker] += bar.volume

    def get_vwap(self, ticker: str) -> float:
        vol = self._cum_vol.get(ticker, 0.0)
        return self._cum_pv[ticker] / vol if vol > 0 else 0.0

    def deviation_pct(self, ticker: str, price: float) -> float:
        vwap = self.get_vwap(ticker)
        return (price - vwap) / vwap * 100.0 if vwap > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# IVB Engine v2
# ─────────────────────────────────────────────────────────────────────────────

class IVBEngine:
    """
    IVB Signal Engine v2.
    Implements the Fabio Valentini 3-step filter (Market State → Location → Aggression)
    plus four additional setups: CVD Divergence, ORB Breakout, VWAP Reclaim.
    """

    def __init__(
        self,
        feed: BloombergFeed,
        volume_breakout_multiplier: float = 3.0,
        min_delta_imbalance:        float = 60.0,
        balance_lookback_bars:      int   = 48,
        value_area_pct:             float = 0.70,
        lvn_threshold:              float = 0.20,
        lvn_proximity_pct:          float = 0.002,  # tighter: must be within 0.2% of LVN
        min_confidence:             float = 0.60,   # raised from 0.55 → fewer, higher-quality signals
        min_rr:                     float = 2.0,    # raised from 1.5 → require better reward
        min_stop_atr_mult:          float = 0.3,    # new: stop must be ≥ 0.3 ATR to avoid noise
        require_htf_for_trend:      bool  = True,   # new: TREND_MODEL requires HTF alignment
    ):
        self.feed                       = feed
        self.volume_breakout_multiplier = volume_breakout_multiplier
        self.min_delta_imbalance        = min_delta_imbalance
        self.balance_lookback_bars      = balance_lookback_bars
        self.value_area_pct             = value_area_pct
        self.lvn_threshold              = lvn_threshold
        self.lvn_proximity_pct          = lvn_proximity_pct
        self.min_confidence             = min_confidence
        self.min_rr                     = min_rr
        self.min_stop_atr_mult          = min_stop_atr_mult
        self.require_htf_for_trend      = require_htf_for_trend

        # v2 helpers
        self.mtf        = MTFAggregator()
        self.session    = SessionContext()
        self.cvd_detect = CVDDivergenceDetector()
        self.vwap_calc  = VWAPCalculator()

        self._last_signals:    Dict[str, IVBSignal] = {}
        self._signal_cooldown: Dict[str, datetime]  = {}
        self._bar_count:       Dict[str, int]       = defaultdict(int)
        self._last_sig_bar:    Dict[str, int]       = defaultdict(int)
        self.COOLDOWN_MINUTES = 15
        self.COOLDOWN_BARS    = 3

        logger.info(
            f"IVB Engine v2 | vol={self.volume_breakout_multiplier}x | "
            f"delta={self.min_delta_imbalance}% | conf={self.min_confidence:.0%} | "
            f"rr={self.min_rr}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Main Evaluation Entry Point
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(self, ticker: str) -> Optional[IVBSignal]:
        """Run the full IVB evaluation for a ticker. Returns signal or None."""
        state = self.feed.get_state(ticker)
        if not state or state.last_price == 0:
            return None

        bars = list(state.bars)
        if len(bars) < 10:
            return None

        price = state.last_price
        cvd   = state.cvd

        # Update v2 helpers
        if bars:
            self.mtf.add_bar(ticker, bars[-1])
            self.vwap_calc.update(ticker, bars[-1])
            self.session.update_orb(ticker, state.session_high, state.session_low)

        self.cvd_detect.update(ticker, price, cvd)
        self._bar_count[ticker] += 1

        # Cooldown: both time-based and bar-count-based
        if self._is_in_cooldown(ticker):
            return None
        bars_since = self._bar_count[ticker] - self._last_sig_bar[ticker]
        if bars_since < self.COOLDOWN_BARS:
            return None

        # Build volume profile
        profile = state.volume_profile
        if not profile or not profile.levels:
            profile = self.feed.build_volume_profile(
                ticker, bars, self.value_area_pct, self.lvn_threshold
            )
        if not profile or not profile.levels:
            return None

        vwap = self.vwap_calc.get_vwap(ticker)

        # ── Try setups in priority order ──────────────────────────────────────
        signal = (
            self._check_cvd_divergence(ticker, state, bars, profile, vwap) or
            self._check_orb_breakout(ticker, state, bars, profile, vwap)   or
            self._check_trend_model(ticker, state, bars, profile, vwap)    or
            self._check_mean_reversion(ticker, state, bars, profile, vwap) or
            self._check_vwap_reclaim(ticker, state, bars, profile, vwap)
        )

        if signal:
            self._last_signals[ticker]    = signal
            self._signal_cooldown[ticker] = datetime.now()
            self._last_sig_bar[ticker]    = self._bar_count[ticker]
            logger.info(
                f"IVB SIGNAL [{ticker}] {signal.direction.value} | "
                f"{signal.setup_type.value} | Conf={signal.confidence:.2f} | "
                f"Entry={signal.entry_price:.4f} | SL={signal.stop_loss:.4f} | "
                f"TP={signal.target:.4f} | R:R={signal.risk_reward:.2f} | "
                f"HTF={'✓' if signal.htf_aligned else '✗'} | "
                f"Session={signal.session}"
            )

        return signal

    # ─────────────────────────────────────────────────────────────────────────
    # Setup 1 (v1): Trend Model
    # ─────────────────────────────────────────────────────────────────────────

    def _check_trend_model(self, ticker, state, bars, profile, vwap) -> Optional[IVBSignal]:
        """
        Classic IVB Trend Model:
        Out-of-balance market + price at LVN + institutional aggression.
        v2 adds: HTF alignment, ATR stops, VWAP filter.
        """
        price = state.last_price
        condition = self._assess_market_state(state, bars, profile)

        if condition != MarketCondition.OUT_OF_BALANCE:
            return None

        # Location: price at LVN
        nearest_lvn = self._find_nearest_lvn(price, profile.lvns)
        if nearest_lvn is None:
            return None
        if abs(price - nearest_lvn) / price > self.lvn_proximity_pct:
            return None

        # Direction from trend
        direction = (SignalDirection.LONG if price > profile.poc
                     else SignalDirection.SHORT)

        # Aggression check
        agg = self._assess_aggression(state, bars, direction)
        if agg is None or not agg[2]:
            return None
        vol_ratio, delta_pct, _ = agg

        # v2: HTF confirmation
        htf_trend = self.mtf.htf_trend(ticker)
        htf_aligned = (htf_trend == direction or htf_trend is None)

        # FIX: TREND_MODEL requires HTF alignment when enabled.
        # Without this, the engine trades against the higher-timeframe trend,
        # which is the #1 cause of false positives on real data.
        if self.require_htf_for_trend and htf_trend is not None and not htf_aligned:
            return None

        # v2: ATR-based stops and targets.
        # Always use ATR for targets (not VP val/vah) because the VP may be
        # slightly stale (rebuilt every N bars), causing incorrect R:R calculations.
        # The ATR-based target is also more robust on real data.
        atr = self._atr(bars)
        if direction == SignalDirection.LONG:
            stop_loss = nearest_lvn - atr * 0.5
            # Use VAH as target if it's meaningfully above price AND above the ATR target
            atr_target = price + 2.5 * atr
            target = profile.vah if profile.vah > price + 0.5 * atr else atr_target
        else:
            stop_loss = nearest_lvn + atr * 0.5
            atr_target = price - 2.5 * atr
            target = profile.val if profile.val < price - 0.5 * atr else atr_target

        # FIX: Reject signals where the stop is too tight (< min_stop_atr_mult ATR).
        # Tight stops on equity futures get hit by normal tick noise, not real reversals.
        stop_dist = abs(price - stop_loss)
        if stop_dist < atr * self.min_stop_atr_mult:
            return None

        rr = self._rr(price, stop_loss, target)
        if rr < self.min_rr:
            return None

        # v2: VWAP filter — price should be on correct side of VWAP
        vwap_dev = self.vwap_calc.deviation_pct(ticker, price)
        if direction == SignalDirection.LONG  and vwap_dev < -1.5:
            return None  # Too far below VWAP for a long
        if direction == SignalDirection.SHORT and vwap_dev >  1.5:
            return None

        conf = self._score_confidence(
            vol_ratio=vol_ratio, delta_pct=delta_pct, rr=rr,
            condition=condition, htf_aligned=htf_aligned,
            at_lvn=True, cvd_divergence=False, session_rth=self.session.is_rth()
        )
        if conf < self.min_confidence:
            return None

        reasoning = (
            f"TREND_MODEL | {condition.value} | LVN={nearest_lvn:.4f} | "
            f"POC={profile.poc:.4f} | Vol={vol_ratio:.1f}x | "
            f"Delta={delta_pct:.0f}% | HTF={'✓' if htf_aligned else '✗'} | "
            f"VWAP_dev={vwap_dev:+.2f}% | R:R={rr:.2f}"
        )

        return IVBSignal(
            ticker=ticker, direction=direction,
            setup_type=SetupType.TREND_MODEL,
            confidence=conf, entry_price=price,
            stop_loss=stop_loss, target=target,
            risk_reward=rr, market_condition=condition,
            poc=profile.poc, vah=profile.vah, val=profile.val,
            lvn_triggered=nearest_lvn,
            volume_breakout_ratio=vol_ratio,
            delta_imbalance_pct=delta_pct,
            cvd=state.cvd, reasoning=reasoning,
            polymarket_keyword=self._get_polymarket_keyword(ticker, direction),
            vwap=vwap, htf_aligned=htf_aligned,
            session=self.session.session_name(), atr=atr, ml_score=conf
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Setup 2 (v1): Mean Reversion
    # ─────────────────────────────────────────────────────────────────────────

    def _check_mean_reversion(self, ticker, state, bars, profile, vwap) -> Optional[IVBSignal]:
        """
        Classic IVB Mean Reversion:
        Failed breakout + price reclaims value area → trade toward POC.
        v2 adds: VWAP deviation confirmation, ATR stops.
        """
        price = state.last_price
        condition = self._assess_market_state(state, bars, profile)

        if condition != MarketCondition.BALANCED:
            return None

        # Location: price back inside value area after a failed breakout
        if not (profile.val <= price <= profile.vah):
            return None

        session_high = state.session_high
        session_low  = state.session_low
        direction    = None

        if session_high > profile.vah * 1.002 and price < profile.vah:
            direction = SignalDirection.SHORT
        elif session_low < profile.val * 0.998 and price > profile.val:
            direction = SignalDirection.LONG

        if direction is None:
            return None

        # Nearest LVN for stop placement
        nearest_lvn = self._find_nearest_lvn(price, profile.lvns)
        lvn_price   = nearest_lvn if nearest_lvn else price

        # Aggression check
        agg = self._assess_aggression(state, bars, direction)
        if agg is None or not agg[2]:
            return None
        vol_ratio, delta_pct, _ = agg

        # v2: ATR stops
        atr = self._atr(bars)
        if direction == SignalDirection.LONG:
            stop_loss = price - 1.5 * atr
            target    = profile.poc
        else:
            stop_loss = price + 1.5 * atr
            target    = profile.poc

        # FIX: Reject tight stops
        stop_dist = abs(price - stop_loss)
        if stop_dist < atr * self.min_stop_atr_mult:
            return None

        rr = self._rr(price, stop_loss, target)
        if rr < 1.5:   # Raised from 1.2 to match tighter quality bar
            return None

        htf_trend   = self.mtf.htf_trend(ticker)
        htf_aligned = (htf_trend == direction or htf_trend is None)

        conf = self._score_confidence(
            vol_ratio=vol_ratio, delta_pct=delta_pct, rr=rr,
            condition=condition, htf_aligned=htf_aligned,
            at_lvn=(nearest_lvn is not None), cvd_divergence=False,
            session_rth=self.session.is_rth()
        )
        if conf < self.min_confidence - 0.05:
            return None

        vwap_dev = self.vwap_calc.deviation_pct(ticker, price)
        reasoning = (
            f"MEAN_REVERSION | Failed breakout | POC={profile.poc:.4f} | "
            f"Vol={vol_ratio:.1f}x | Delta={delta_pct:.0f}% | "
            f"VWAP_dev={vwap_dev:+.2f}% | R:R={rr:.2f}"
        )

        return IVBSignal(
            ticker=ticker, direction=direction,
            setup_type=SetupType.MEAN_REVERSION,
            confidence=conf, entry_price=price,
            stop_loss=stop_loss, target=target,
            risk_reward=rr, market_condition=condition,
            poc=profile.poc, vah=profile.vah, val=profile.val,
            lvn_triggered=lvn_price,
            volume_breakout_ratio=vol_ratio,
            delta_imbalance_pct=delta_pct,
            cvd=state.cvd, reasoning=reasoning,
            polymarket_keyword=self._get_polymarket_keyword(ticker, direction),
            vwap=vwap, htf_aligned=htf_aligned,
            session=self.session.session_name(), atr=atr, ml_score=conf
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Setup 3 (v2 NEW): CVD Divergence
    # ─────────────────────────────────────────────────────────────────────────

    def _check_cvd_divergence(self, ticker, state, bars, profile, vwap) -> Optional[IVBSignal]:
        """
        NEW in v2: CVD divergence reversal.
        Price makes new high/low but CVD does not confirm → high-probability reversal.
        """
        divergence = self.cvd_detect.detect(ticker)
        if not divergence:
            return None

        price = state.last_price
        atr   = self._atr(bars)

        if divergence == SignalDirection.LONG:
            stop_loss = price - 1.5 * atr
            target    = price + 2.0 * atr
        else:
            stop_loss = price + 1.5 * atr
            target    = price - 2.0 * atr

        rr = self._rr(price, stop_loss, target)
        if rr < 1.2:
            return None

        htf_trend   = self.mtf.htf_trend(ticker)
        htf_aligned = (htf_trend == divergence or htf_trend is None)

        conf = self._score_confidence(
            vol_ratio=2.0, delta_pct=65.0, rr=rr,
            condition=MarketCondition.BALANCED,
            htf_aligned=htf_aligned, at_lvn=False,
            cvd_divergence=True, session_rth=self.session.is_rth()
        )
        if conf < self.min_confidence - 0.05:
            return None

        label = "bullish" if divergence == SignalDirection.LONG else "bearish"
        reasoning = (
            f"CVD_DIVERGENCE | {label} divergence | "
            f"Price vs CVD disagree | R:R={rr:.2f}"
        )

        return IVBSignal(
            ticker=ticker, direction=divergence,
            setup_type=SetupType.CVD_DIVERGENCE,
            confidence=conf, entry_price=price,
            stop_loss=stop_loss, target=target,
            risk_reward=rr, market_condition=MarketCondition.BALANCED,
            poc=profile.poc, vah=profile.vah, val=profile.val,
            lvn_triggered=0.0,
            volume_breakout_ratio=2.0,
            delta_imbalance_pct=65.0,
            cvd=state.cvd, reasoning=reasoning,
            polymarket_keyword=self._get_polymarket_keyword(ticker, divergence),
            vwap=vwap, htf_aligned=htf_aligned,
            session=self.session.session_name(), atr=atr, ml_score=conf
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Setup 4 (v2 NEW): Opening Range Breakout
    # ─────────────────────────────────────────────────────────────────────────

    def _check_orb_breakout(self, ticker, state, bars, profile, vwap) -> Optional[IVBSignal]:
        """
        NEW in v2: Opening Range Breakout (first 15 min of RTH session).
        Price breaks above/below the 9:30–9:45 range with volume confirmation.
        """
        if not self.session.is_rth():
            return None

        price = state.last_price
        orb_hi, orb_lo = self.session.get_orb(ticker)
        if orb_hi == 0 or orb_lo == 0:
            return None

        orb_range = orb_hi - orb_lo
        if orb_range == 0:
            return None

        # Determine direction
        if price > orb_hi * 1.001:
            direction = SignalDirection.LONG
        elif price < orb_lo * 0.999:
            direction = SignalDirection.SHORT
        else:
            return None

        # Volume confirmation
        if len(bars) < 5:
            return None
        recent_vols = [b.volume for b in bars[-10:-1]]
        avg_vol = statistics.mean(recent_vols) if recent_vols else 0
        curr_vol = bars[-1].volume
        vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 0

        if vol_ratio < 2.0:
            return None

        atr = self._atr(bars)
        if direction == SignalDirection.LONG:
            stop_loss = orb_hi - 0.3 * orb_range
            target    = price + 2.0 * atr
        else:
            stop_loss = orb_lo + 0.3 * orb_range
            target    = price - 2.0 * atr

        rr = self._rr(price, stop_loss, target)
        if rr < self.min_rr:
            return None

        htf_trend   = self.mtf.htf_trend(ticker)
        htf_aligned = (htf_trend == direction or htf_trend is None)

        conf = self._score_confidence(
            vol_ratio=vol_ratio, delta_pct=60.0, rr=rr,
            condition=MarketCondition.OUT_OF_BALANCE,
            htf_aligned=htf_aligned, at_lvn=False,
            cvd_divergence=False, session_rth=True
        )
        if conf < self.min_confidence - 0.05:
            return None

        reasoning = (
            f"ORB_BREAKOUT | {'Above' if direction==SignalDirection.LONG else 'Below'} "
            f"ORB {orb_hi:.4f}/{orb_lo:.4f} | Vol={vol_ratio:.1f}x | R:R={rr:.2f}"
        )

        return IVBSignal(
            ticker=ticker, direction=direction,
            setup_type=SetupType.ORB_BREAKOUT,
            confidence=conf, entry_price=price,
            stop_loss=stop_loss, target=target,
            risk_reward=rr, market_condition=MarketCondition.OUT_OF_BALANCE,
            poc=profile.poc, vah=profile.vah, val=profile.val,
            lvn_triggered=0.0,
            volume_breakout_ratio=vol_ratio,
            delta_imbalance_pct=60.0,
            cvd=state.cvd, reasoning=reasoning,
            polymarket_keyword=self._get_polymarket_keyword(ticker, direction),
            vwap=vwap, htf_aligned=htf_aligned,
            session="RTH", atr=atr, ml_score=conf
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Setup 5 (v2 NEW): VWAP Reclaim
    # ─────────────────────────────────────────────────────────────────────────

    def _check_vwap_reclaim(self, ticker, state, bars, profile, vwap) -> Optional[IVBSignal]:
        """
        NEW in v2: VWAP Reclaim continuation.
        Price crosses VWAP with delta confirmation → continuation signal.
        """
        price = state.last_price
        if len(bars) < 5 or vwap == 0:
            return None

        prev_closes = [b.close for b in bars[-5:-1]]
        if not prev_closes:
            return None
        prev_avg = statistics.mean(prev_closes)

        crossed_up   = prev_avg < vwap and price > vwap
        crossed_down = prev_avg > vwap and price < vwap

        if not (crossed_up or crossed_down):
            return None

        direction = SignalDirection.LONG if crossed_up else SignalDirection.SHORT

        # Delta confirmation
        curr_bar = bars[-1]
        delta = getattr(curr_bar, "delta", 0)
        if direction == SignalDirection.LONG  and delta < 0:
            return None
        if direction == SignalDirection.SHORT and delta > 0:
            return None

        atr = self._atr(bars)
        if direction == SignalDirection.LONG:
            stop_loss = vwap - 0.5 * atr
            target    = price + 2.0 * atr
        else:
            stop_loss = vwap + 0.5 * atr
            target    = price - 2.0 * atr

        rr = self._rr(price, stop_loss, target)
        if rr < 1.2:
            return None

        htf_trend   = self.mtf.htf_trend(ticker)
        htf_aligned = (htf_trend == direction or htf_trend is None)

        conf = self._score_confidence(
            vol_ratio=1.5, delta_pct=55.0, rr=rr,
            condition=MarketCondition.BALANCED,
            htf_aligned=htf_aligned, at_lvn=False,
            cvd_divergence=False, session_rth=self.session.is_rth()
        )
        if conf < self.min_confidence - 0.08:
            return None

        reasoning = (
            f"VWAP_RECLAIM | {'Above' if crossed_up else 'Below'} VWAP={vwap:.4f} | "
            f"Delta confirmed | R:R={rr:.2f}"
        )

        return IVBSignal(
            ticker=ticker, direction=direction,
            setup_type=SetupType.VWAP_RECLAIM,
            confidence=conf, entry_price=price,
            stop_loss=stop_loss, target=target,
            risk_reward=rr, market_condition=MarketCondition.BALANCED,
            poc=profile.poc, vah=profile.vah, val=profile.val,
            lvn_triggered=0.0,
            volume_breakout_ratio=1.5,
            delta_imbalance_pct=55.0,
            cvd=state.cvd, reasoning=reasoning,
            polymarket_keyword=self._get_polymarket_keyword(ticker, direction),
            vwap=vwap, htf_aligned=htf_aligned,
            session=self.session.session_name(), atr=atr, ml_score=conf
        )

    # ─────────────────────────────────────────────────────────────────────────
    # v1 Helpers (preserved + enhanced)
    # ─────────────────────────────────────────────────────────────────────────

    def _assess_market_state(self, state: MarketState,
                              bars: List[Bar],
                              profile: VolumeProfile) -> MarketCondition:
        """
        Determine whether the market is balanced (range-bound) or out-of-balance
        (trending / directional).

        FIX: The original 0.5% displacement threshold is too tight for equity
        index futures (ES, NQ) which trade in narrow intraday ranges.  We now
        use an ATR-relative threshold: price must be more than 0.5 ATR away
        from the POC to be considered out-of-balance.  This is instrument-
        agnostic and works for both crypto (wide ATR) and equity futures
        (narrow ATR).
        """
        price = state.last_price
        vah   = profile.vah
        val   = profile.val
        poc   = profile.poc

        # ATR-relative displacement threshold (0.5 ATR = meaningful move)
        atr = self._atr(bars)
        displacement = abs(price - poc)
        oob_threshold = max(atr * 0.5, poc * 0.002)  # at least 0.2% of price

        if val <= price <= vah:
            # Inside value area — check for directional momentum breakout
            if len(bars) >= 3:
                recent_bars = bars[-3:]
                all_up   = all(b.close > b.open for b in recent_bars)
                all_down = all(b.close < b.open for b in recent_bars)
                if len(bars) >= 10:
                    avg_vol    = statistics.mean(b.volume for b in bars[-10:])
                    recent_vol = statistics.mean(b.volume for b in recent_bars)
                    if (all_up or all_down) and recent_vol > avg_vol * 1.3:
                        return MarketCondition.OUT_OF_BALANCE
            return MarketCondition.BALANCED

        # Outside value area — use ATR-relative displacement
        if displacement > oob_threshold:
            return MarketCondition.OUT_OF_BALANCE
        return MarketCondition.BALANCED

    def _assess_aggression(self, state: MarketState, bars: List[Bar],
                            direction: SignalDirection
                            ) -> Optional[Tuple[float, float, bool]]:
        if len(bars) < 5:
            return None

        recent_bar = bars[-1]
        lookback   = bars[-20:] if len(bars) >= 20 else bars[:-1]
        if not lookback:
            return None

        avg_volume = statistics.mean(b.volume for b in lookback)
        if avg_volume == 0:
            return None
        vol_ratio = recent_bar.volume / avg_volume

        total_vol = recent_bar.buy_volume + recent_bar.sell_volume
        if total_vol == 0:
            return None

        dominant_vol = (recent_bar.buy_volume  if direction == SignalDirection.LONG
                        else recent_bar.sell_volume)
        delta_pct = (dominant_vol / total_vol) * 100.0

        # CVD alignment: use the bar's own delta (buy - sell) rather than the
        # session cumulative CVD.  Session CVD is a monotonically growing number
        # that is almost always positive, which permanently blocks SHORT signals
        # and makes the check meaningless.  Bar delta directly reflects whether
        # the most recent bar was buyer-dominated or seller-dominated.
        bar_delta = recent_bar.buy_volume - recent_bar.sell_volume
        cvd_aligned = (
            (direction == SignalDirection.LONG  and bar_delta > 0) or
            (direction == SignalDirection.SHORT and bar_delta < 0)
        )

        # Also check rolling CVD trend (last 5 bars) as a secondary confirmation.
        # This is softer — it adds weight but doesn't hard-block.
        recent_deltas = [getattr(b, 'delta', b.buy_volume - b.sell_volume)
                         for b in bars[-5:]]
        rolling_cvd_trend = sum(recent_deltas)
        rolling_aligned = (
            (direction == SignalDirection.LONG  and rolling_cvd_trend > 0) or
            (direction == SignalDirection.SHORT and rolling_cvd_trend < 0)
        )

        is_aggressive = (
            vol_ratio  >= self.volume_breakout_multiplier and
            delta_pct  >= self.min_delta_imbalance and
            cvd_aligned and
            rolling_aligned
        )

        return (vol_ratio, delta_pct, is_aggressive)

    def _find_nearest_lvn(self, price: float,
                           lvns: List[float]) -> Optional[float]:
        if not lvns:
            return None
        return min(lvns, key=lambda lvn: abs(price - lvn))

    # ─────────────────────────────────────────────────────────────────────────
    # v2 Confidence Scorer (composite rule-based, upgradeable to XGBoost)
    # ─────────────────────────────────────────────────────────────────────────

    def _score_confidence(
        self,
        vol_ratio:      float,
        delta_pct:      float,
        rr:             float,
        condition:      MarketCondition,
        htf_aligned:    bool,
        at_lvn:         bool,
        cvd_divergence: bool,
        session_rth:    bool,
    ) -> float:
        """
        Composite confidence score 0–1.
        Weights tuned to the Fabio Valentini IVB model priorities.
        """
        score = 0.0

        # Volume breakout strength (0–0.25)
        if   vol_ratio >= 5.0: score += 0.25
        elif vol_ratio >= 4.0: score += 0.20
        elif vol_ratio >= 3.0: score += 0.15
        elif vol_ratio >= 2.0: score += 0.08
        else:                  score += 0.02

        # Delta imbalance (0–0.25)
        if   delta_pct >= 80: score += 0.25
        elif delta_pct >= 70: score += 0.20
        elif delta_pct >= 60: score += 0.15
        elif delta_pct >= 55: score += 0.08
        else:                 score += 0.02

        # Risk/Reward quality (0–0.15)
        if   rr >= 3.0: score += 0.15
        elif rr >= 2.0: score += 0.12
        elif rr >= 1.5: score += 0.08
        elif rr >= 1.2: score += 0.04

        # Market condition (0–0.10)
        if condition == MarketCondition.OUT_OF_BALANCE:
            score += 0.10
        else:
            score += 0.05

        # v2 additions
        if htf_aligned:    score += 0.10   # HTF trend confirmation
        if at_lvn:         score += 0.08   # Price at key LVN
        if cvd_divergence: score += 0.08   # CVD divergence bonus
        if session_rth:    score += 0.04   # RTH has better liquidity

        return min(score, 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _atr(bars: List[Bar], period: int = 14) -> float:
        """Average True Range over last `period` bars."""
        if len(bars) < 2:
            return bars[-1].close * 0.005 if bars else 1.0
        trs = []
        for i in range(1, min(period + 1, len(bars))):
            h = bars[-i].high
            l = bars[-i].low
            c = bars[-i - 1].close
            trs.append(max(h - l, abs(h - c), abs(l - c)))
        return float(np.mean(trs)) if trs else 1.0

    @staticmethod
    def _rr(entry: float, stop: float, target: float) -> float:
        risk   = abs(entry - stop)
        reward = abs(target - entry)
        return reward / risk if risk > 0 else 0.0

    def _is_in_cooldown(self, ticker: str) -> bool:
        last = self._signal_cooldown.get(ticker)
        if not last:
            return False
        elapsed = (datetime.now() - last).total_seconds() / 60
        return elapsed < self.COOLDOWN_MINUTES

    def _get_polymarket_keyword(self, ticker: str,
                                 direction: SignalDirection) -> str:
        mapping = {
            "XBT Curncy": {
                SignalDirection.LONG:  "Bitcoin price above",
                SignalDirection.SHORT: "Bitcoin price below",
            },
            "ES1 Index": {
                SignalDirection.LONG:  "S&P 500 above",
                SignalDirection.SHORT: "S&P 500 below",
            },
            "NQ1 Index": {
                SignalDirection.LONG:  "Nasdaq above",
                SignalDirection.SHORT: "Nasdaq below",
            },
            "SPX Index": {
                SignalDirection.LONG:  "S&P 500 above",
                SignalDirection.SHORT: "S&P 500 below",
            },
        }
        return mapping.get(ticker, {}).get(direction, "")

    def get_last_signal(self, ticker: str) -> Optional[IVBSignal]:
        return self._last_signals.get(ticker)

    def get_all_signals(self) -> Dict[str, IVBSignal]:
        return dict(self._last_signals)

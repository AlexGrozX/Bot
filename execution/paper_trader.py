"""
IVB Bot — Paper Trading Engine v2
Improvements over v1:
  - ATR-based trailing stops (moves stop to breakeven at 1R, trails after 2R)
  - Partial take-profits: scale out 50% at 1R, let rest run with trailing stop
  - Advanced P&L analytics: Sharpe ratio, max drawdown, profit factor, expectancy
  - Equity curve tracking (balance history for charting)
  - Per-setup-type performance breakdown
  - Realistic slippage model (wider in ETH, tighter in RTH)
"""
import json
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("execution.paper_trader")


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """A completed (or open) trade record."""
    id:              int
    ticker:          str
    direction:       str       # LONG / SHORT
    setup_type:      str
    entry_price:     float
    stop_loss:       float     # Initial stop
    target:          float     # Full target (2R)
    size_usd:        float
    entry_time:      str
    exit_price:      float = 0.0
    exit_time:       str   = ""
    pnl:             float = 0.0
    pnl_pct:         float = 0.0
    exit_reason:     str   = ""   # TARGET / PARTIAL_TP / STOP / TRAILING_STOP / MANUAL
    open:            bool  = True
    reasoning:       str   = ""
    # v2 additions
    atr:             float = 0.0
    confidence:      float = 0.0
    session:         str   = "RTH"
    partial_closed:  bool  = False   # True if 50% already taken at 1R
    partial_pnl:     float = 0.0     # P&L from the partial close
    trailing_stop:   float = 0.0     # Current trailing stop level
    peak_price:      float = 0.0     # Highest (LONG) or lowest (SHORT) price seen

    @property
    def risk_reward(self) -> float:
        risk   = abs(self.entry_price - self.stop_loss)
        reward = abs(self.target - self.entry_price)
        return reward / risk if risk > 0 else 0.0

    @property
    def r_value(self) -> float:
        """P&L expressed in R units."""
        risk = abs(self.entry_price - self.stop_loss)
        return self.pnl / (self.size_usd * risk / self.entry_price) if risk > 0 and self.size_usd > 0 else 0.0


@dataclass
class DailyStats:
    date:       str
    trades:     int   = 0
    wins:       int   = 0
    losses:     int   = 0
    gross_pnl:  float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0

    @property
    def win_rate(self) -> float:
        return (self.wins / self.trades * 100) if self.trades > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Paper Trading Engine v2
# ─────────────────────────────────────────────────────────────────────────────

class PaperTrader:
    """
    Paper trading engine with trailing stops, partial TPs, and advanced analytics.
    """

    SLIPPAGE_RTH = 0.0003   # 0.03% in regular hours
    SLIPPAGE_ETH = 0.0007   # 0.07% in extended hours
    COMMISSION   = 0.0

    def __init__(
        self,
        starting_balance:   float = 400.0,
        max_position_pct:   float = 0.10,
        max_daily_loss_pct: float = 0.05,
        log_dir:            str   = "logs",
        use_trailing_stop:  bool  = True,
        use_partial_tp:     bool  = True,
    ):
        self.starting_balance   = starting_balance
        self.balance            = starting_balance
        self.max_position_pct   = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.use_trailing_stop  = use_trailing_stop
        self.use_partial_tp     = use_partial_tp

        self._trades:       List[TradeRecord]       = []
        self._open:         Dict[str, TradeRecord]  = {}
        self._trade_counter = 0
        self._daily_stats:  Dict[str, DailyStats]   = {}
        self._equity_curve: List[Tuple[str, float]] = []  # (timestamp, balance)

        self._session_start_balance = starting_balance
        self._daily_loss_halt       = False

        self._log_dir        = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._trade_log_path = self._log_dir / "trades.json"
        self._equity_path    = self._log_dir / "equity_curve.json"
        self._load_trades()

        # Record starting equity
        self._record_equity()

        logger.info(
            f"Paper Trader v2 initialized | Balance=${self.balance:.2f} | "
            f"TrailingStop={'ON' if use_trailing_stop else 'OFF'} | "
            f"PartialTP={'ON' if use_partial_tp else 'OFF'}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def can_trade(self, ticker: str) -> Tuple[bool, str]:
        if self._daily_loss_halt:
            return False, "Daily loss limit reached — trading halted for today."
        if ticker in self._open:
            return False, f"Already have an open position on {ticker}."
        if self.balance <= 0:
            return False, "Balance is zero."
        return True, "OK"

    def open_trade(self, signal) -> Optional[TradeRecord]:
        """Open a paper trade from an IVB signal."""
        ok, reason = self.can_trade(signal.ticker)
        if not ok:
            logger.info(f"Trade rejected [{signal.ticker}]: {reason}")
            return None

        # Position sizing: Kelly fraction, capped at max_position_pct
        kelly_size = self.balance * signal.confidence * 0.25
        max_size   = self.balance * self.max_position_pct
        size_usd   = min(kelly_size, max_size)
        size_usd   = max(size_usd, 1.0)

        # Session-aware slippage
        session = getattr(signal, "session", "RTH")
        slip = self.SLIPPAGE_RTH if session == "RTH" else self.SLIPPAGE_ETH

        if signal.direction.value == "LONG":
            fill_price = signal.entry_price * (1 + slip)
        else:
            fill_price = signal.entry_price * (1 - slip)

        atr = getattr(signal, "atr", abs(signal.entry_price - signal.stop_loss))

        self._trade_counter += 1
        trade = TradeRecord(
            id=self._trade_counter,
            ticker=signal.ticker,
            direction=signal.direction.value,
            setup_type=signal.setup_type.value,
            entry_price=fill_price,
            stop_loss=signal.stop_loss,
            target=signal.target,
            size_usd=size_usd,
            entry_time=datetime.now().isoformat(),
            reasoning=signal.reasoning,
            open=True,
            atr=atr,
            confidence=signal.confidence,
            session=session,
            trailing_stop=signal.stop_loss,
            peak_price=fill_price,
        )

        self._open[signal.ticker] = trade
        self._trades.append(trade)
        self._save_trades()

        logger.info(
            f"📈 PAPER TRADE OPENED #{trade.id} | {trade.direction} {trade.ticker} | "
            f"Entry={fill_price:.4f} | SL={trade.stop_loss:.4f} | "
            f"TP={trade.target:.4f} | Size=${size_usd:.2f} | "
            f"R:R={trade.risk_reward:.1f} | Session={session}"
        )
        return trade

    def update_prices(self, prices: Dict[str, float]):
        """
        Called on every price update.
        Handles: partial TP at 1R, trailing stop adjustment, SL/TP hits.
        """
        closed_tickers = []

        for ticker, trade in list(self._open.items()):
            price = prices.get(ticker)
            if not price:
                continue

            # Update peak price for trailing stop
            if trade.direction == "LONG":
                trade.peak_price = max(trade.peak_price, price)
            else:
                trade.peak_price = min(trade.peak_price, price)

            # ── Partial TP at 1R ──────────────────────────────────────────────
            if self.use_partial_tp and not trade.partial_closed:
                one_r_target = self._one_r_target(trade)
                if (trade.direction == "LONG"  and price >= one_r_target) or \
                   (trade.direction == "SHORT" and price <= one_r_target):
                    self._partial_close(trade, price)
                    # Move stop to breakeven after partial TP
                    if trade.direction == "LONG":
                        trade.trailing_stop = max(trade.trailing_stop, trade.entry_price)
                    else:
                        trade.trailing_stop = min(trade.trailing_stop, trade.entry_price)

            # ── Trailing Stop Update (after 2R) ───────────────────────────────
            if self.use_trailing_stop:
                two_r_target = self._two_r_target(trade)
                at_2r = (
                    (trade.direction == "LONG"  and price >= two_r_target) or
                    (trade.direction == "SHORT" and price <= two_r_target)
                )
                if at_2r:
                    # Trail at 1.5 × ATR behind peak
                    trail_dist = 1.5 * trade.atr if trade.atr > 0 else abs(trade.entry_price - trade.stop_loss)
                    if trade.direction == "LONG":
                        new_trail = trade.peak_price - trail_dist
                        trade.trailing_stop = max(trade.trailing_stop, new_trail)
                    else:
                        new_trail = trade.peak_price + trail_dist
                        trade.trailing_stop = min(trade.trailing_stop, new_trail)

            # ── Check Exit Conditions ─────────────────────────────────────────
            hit_target        = False
            hit_stop          = False
            hit_trailing_stop = False

            if trade.direction == "LONG":
                if price >= trade.target:
                    hit_target = True
                elif price <= trade.trailing_stop:
                    if trade.trailing_stop > trade.stop_loss:
                        hit_trailing_stop = True
                    else:
                        hit_stop = True
            else:
                if price <= trade.target:
                    hit_target = True
                elif price >= trade.trailing_stop:
                    if trade.trailing_stop < trade.stop_loss:
                        hit_trailing_stop = True
                    else:
                        hit_stop = True

            if hit_target:
                self._close_trade(trade, trade.target, "TARGET")
                closed_tickers.append(ticker)
            elif hit_trailing_stop:
                self._close_trade(trade, price, "TRAILING_STOP")
                closed_tickers.append(ticker)
            elif hit_stop:
                self._close_trade(trade, trade.stop_loss, "STOP")
                closed_tickers.append(ticker)

        for t in closed_tickers:
            del self._open[t]

        self._check_daily_loss()

    def close_all(self, prices: Dict[str, float]):
        """Manually close all open positions at current prices."""
        for ticker, trade in list(self._open.items()):
            price = prices.get(ticker, trade.entry_price)
            self._close_trade(trade, price, "MANUAL")
        self._open.clear()
        self._save_trades()

    # ── Properties ────────────────────────────────────────────────────────────

    def get_open_pnl(self, prices: Dict[str, float]) -> float:
        total = 0.0
        for ticker, trade in self._open.items():
            price = prices.get(ticker)
            if not price:
                continue
            pnl = self._calc_pnl(trade, price)
            total += pnl + trade.partial_pnl
        return total

    @property
    def daily_pnl(self) -> float:
        today = date.today().isoformat()
        return self._daily_stats.get(today, DailyStats(today)).gross_pnl

    @property
    def total_pnl(self) -> float:
        return self.balance - self.starting_balance

    @property
    def total_trades(self) -> int:
        return len([t for t in self._trades if not t.open])

    @property
    def win_rate(self) -> float:
        closed = [t for t in self._trades if not t.open]
        if not closed:
            return 0.0
        wins = sum(1 for t in closed if t.pnl > 0)
        return wins / len(closed) * 100

    @property
    def open_positions(self) -> List[TradeRecord]:
        return list(self._open.values())

    @property
    def recent_trades(self) -> List[TradeRecord]:
        closed = [t for t in self._trades if not t.open]
        return sorted(closed, key=lambda t: t.exit_time, reverse=True)[:20]

    @property
    def open_count(self) -> int:
        return len(self._open)

    # ── Advanced Analytics ────────────────────────────────────────────────────

    def sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        """Annualized Sharpe ratio from daily P&L."""
        daily_pnls = [ds.gross_pnl for ds in self._daily_stats.values()]
        if len(daily_pnls) < 2:
            return 0.0
        daily_returns = [p / self.starting_balance for p in daily_pnls]
        mean_r = np.mean(daily_returns)
        std_r  = np.std(daily_returns, ddof=1)
        if std_r == 0:
            return 0.0
        daily_rf = risk_free_rate / 252
        return float((mean_r - daily_rf) / std_r * math.sqrt(252))

    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown as a percentage of peak balance."""
        if not self._equity_curve:
            return 0.0
        balances = [b for _, b in self._equity_curve]
        peak = balances[0]
        max_dd = 0.0
        for b in balances:
            if b > peak:
                peak = b
            dd = (peak - b) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd * 100  # as percentage

    def profit_factor(self) -> float:
        """Gross profits / gross losses."""
        closed = [t for t in self._trades if not t.open]
        gross_wins   = sum(t.pnl for t in closed if t.pnl > 0)
        gross_losses = abs(sum(t.pnl for t in closed if t.pnl < 0))
        return gross_wins / gross_losses if gross_losses > 0 else float("inf")

    def expectancy(self) -> float:
        """Expected P&L per trade in dollars."""
        closed = [t for t in self._trades if not t.open]
        if not closed:
            return 0.0
        return sum(t.pnl for t in closed) / len(closed)

    def avg_r_multiple(self) -> float:
        """Average R-multiple of closed trades."""
        closed = [t for t in self._trades if not t.open]
        if not closed:
            return 0.0
        r_vals = [t.r_value for t in closed]
        return float(np.mean(r_vals))

    def setup_breakdown(self) -> Dict[str, dict]:
        """Performance breakdown by setup type."""
        closed = [t for t in self._trades if not t.open]
        breakdown: Dict[str, dict] = {}
        for t in closed:
            st = t.setup_type
            if st not in breakdown:
                breakdown[st] = {"trades": 0, "wins": 0, "pnl": 0.0}
            breakdown[st]["trades"] += 1
            if t.pnl > 0:
                breakdown[st]["wins"] += 1
            breakdown[st]["pnl"] += t.pnl
        for st, d in breakdown.items():
            d["win_rate"] = d["wins"] / d["trades"] * 100 if d["trades"] > 0 else 0.0
        return breakdown

    def full_stats(self) -> dict:
        """Return a complete performance statistics dictionary."""
        return {
            "balance":       round(self.balance, 2),
            "starting":      round(self.starting_balance, 2),
            "total_pnl":     round(self.total_pnl, 2),
            "total_pnl_pct": round(self.total_pnl / self.starting_balance * 100, 2),
            "daily_pnl":     round(self.daily_pnl, 2),
            "total_trades":  self.total_trades,
            "open_trades":   self.open_count,
            "win_rate":      round(self.win_rate, 1),
            "sharpe":        round(self.sharpe_ratio(), 2),
            "max_drawdown":  round(self.max_drawdown(), 2),
            "profit_factor": round(self.profit_factor(), 2),
            "expectancy":    round(self.expectancy(), 2),
            "avg_r_multiple": round(self.avg_r_multiple(), 2),
            "setup_breakdown": self.setup_breakdown(),
        }

    def summary(self) -> str:
        """One-line performance summary."""
        return (
            f"Balance=${self.balance:.2f} | "
            f"Total P&L=${self.total_pnl:+.2f} | "
            f"Daily P&L=${self.daily_pnl:+.2f} | "
            f"Trades={self.total_trades} | "
            f"Win Rate={self.win_rate:.0f}% | "
            f"Sharpe={self.sharpe_ratio():.2f} | "
            f"MaxDD={self.max_drawdown():.1f}% | "
            f"Open={self.open_count}"
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _one_r_target(self, trade: TradeRecord) -> float:
        """Price level at 1R profit."""
        risk = abs(trade.entry_price - trade.stop_loss)
        if trade.direction == "LONG":
            return trade.entry_price + risk
        else:
            return trade.entry_price - risk

    def _two_r_target(self, trade: TradeRecord) -> float:
        """Price level at 2R profit."""
        risk = abs(trade.entry_price - trade.stop_loss)
        if trade.direction == "LONG":
            return trade.entry_price + 2 * risk
        else:
            return trade.entry_price - 2 * risk

    def _partial_close(self, trade: TradeRecord, price: float):
        """Close 50% of the position at 1R."""
        half_size = trade.size_usd * 0.5
        pnl = self._calc_pnl_for_size(trade, price, half_size)
        trade.partial_pnl    = pnl
        trade.partial_closed = True
        trade.size_usd      *= 0.5   # Remaining half
        self.balance         += pnl

        today = date.today().isoformat()
        if today not in self._daily_stats:
            self._daily_stats[today] = DailyStats(today)
        self._daily_stats[today].gross_pnl += pnl

        self._record_equity()
        logger.info(
            f"🔀 PARTIAL TP #{trade.id} | {trade.direction} {trade.ticker} | "
            f"50% closed at {price:.4f} | Partial P&L=${pnl:+.2f} | "
            f"Stop moved to breakeven | Balance=${self.balance:.2f}"
        )

    def _calc_pnl(self, trade: TradeRecord, exit_price: float) -> float:
        return self._calc_pnl_for_size(trade, exit_price, trade.size_usd)

    def _calc_pnl_for_size(self, trade: TradeRecord,
                            exit_price: float, size: float) -> float:
        price_move = exit_price - trade.entry_price
        if trade.direction == "SHORT":
            price_move = -price_move
        pct_move = price_move / trade.entry_price if trade.entry_price > 0 else 0
        return size * pct_move

    def _close_trade(self, trade: TradeRecord,
                     exit_price: float, reason: str):
        pnl     = self._calc_pnl(trade, exit_price)
        total_pnl = pnl + trade.partial_pnl
        pnl_pct = (total_pnl / (trade.size_usd + trade.size_usd) * 100) if trade.size_usd > 0 else 0

        trade.exit_price  = exit_price
        trade.exit_time   = datetime.now().isoformat()
        trade.pnl         = total_pnl
        trade.pnl_pct     = pnl_pct
        trade.exit_reason = reason
        trade.open        = False

        self.balance += pnl

        today = date.today().isoformat()
        if today not in self._daily_stats:
            self._daily_stats[today] = DailyStats(today)
        ds = self._daily_stats[today]
        ds.trades    += 1
        ds.gross_pnl += total_pnl
        if total_pnl > 0:
            ds.wins      += 1
            ds.best_trade = max(ds.best_trade, total_pnl)
        else:
            ds.losses     += 1
            ds.worst_trade = min(ds.worst_trade, total_pnl)

        self._record_equity()

        emoji = "✅" if total_pnl > 0 else "❌"
        logger.info(
            f"{emoji} PAPER TRADE CLOSED #{trade.id} | {reason} | "
            f"{trade.direction} {trade.ticker} | "
            f"Entry={trade.entry_price:.4f} → Exit={exit_price:.4f} | "
            f"P&L=${total_pnl:+.2f} ({pnl_pct:+.1f}%) | "
            f"Balance=${self.balance:.2f}"
        )
        self._save_trades()

    def _check_daily_loss(self):
        daily_loss_pct = abs(self.daily_pnl) / self.starting_balance
        if self.daily_pnl < 0 and daily_loss_pct >= self.max_daily_loss_pct:
            if not self._daily_loss_halt:
                self._daily_loss_halt = True
                logger.warning(
                    f"⛔ Daily loss limit reached: ${self.daily_pnl:.2f} "
                    f"({daily_loss_pct*100:.1f}%). Trading halted for today."
                )

    def reset_daily_halt(self):
        self._daily_loss_halt       = False
        self._session_start_balance = self.balance
        logger.info("Daily halt reset. New trading day started.")

    def _record_equity(self):
        self._equity_curve.append((datetime.now().isoformat(), round(self.balance, 2)))
        # Save equity curve
        try:
            with open(self._equity_path, "w") as f:
                json.dump(self._equity_curve, f)
        except:
            pass

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_trades(self):
        try:
            data = {
                "balance":       self.balance,
                "trades":        [asdict(t) for t in self._trades],
                "trade_counter": self._trade_counter,
            }
            with open(self._trade_log_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save trades: {e}")

    def _load_trades(self):
        if not self._trade_log_path.exists():
            return
        try:
            with open(self._trade_log_path) as f:
                data = json.load(f)
            self.balance        = data.get("balance", self.starting_balance)
            self._trade_counter = data.get("trade_counter", 0)
            for td in data.get("trades", []):
                # Handle v1 trade records that lack v2 fields
                td.setdefault("atr", 0.0)
                td.setdefault("confidence", 0.0)
                td.setdefault("session", "RTH")
                td.setdefault("partial_closed", False)
                td.setdefault("partial_pnl", 0.0)
                td.setdefault("trailing_stop", td.get("stop_loss", 0.0))
                td.setdefault("peak_price", td.get("entry_price", 0.0))
                self._trades.append(TradeRecord(**td))
            logger.info(
                f"Loaded {len(self._trades)} trades from log. "
                f"Balance: ${self.balance:.2f}"
            )
        except Exception as e:
            logger.warning(f"Could not load trade log: {e}")

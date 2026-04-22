"""
IVB Bot — Paper Trading Dashboard v2
Real-time terminal UI showing:
  - Balance | Open P&L | Daily P&L | Win Rate | Sharpe | Max DD
  - Live Bloomberg market data (price, CVD, VWAP, session hi/lo)
  - IVB signals as they fire (with setup type, session, HTF alignment)
  - Open positions with trailing stop and partial TP status
  - Recent closed trades with R-multiple
  - Performance stats panel: Sharpe, Max DD, Profit Factor, Expectancy
  - Equity curve (ASCII sparkline)
  - Setup-type breakdown table
"""
import logging
import time
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

if TYPE_CHECKING:
    from bloomberg.data_feed import BloombergFeed
    from execution.paper_trader import PaperTrader, TradeRecord
    from engine.ivb_engine import IVBSignal

logger = logging.getLogger("dashboard")
console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pnl_color(v: float) -> str:
    return "bright_green" if v > 0 else ("bright_red" if v < 0 else "white")

def dir_color(d: str) -> str:
    return "bright_green" if d == "LONG" else "bright_red"

def sparkline(values: list, width: int = 20) -> str:
    """ASCII sparkline from a list of floats."""
    if len(values) < 2:
        return "─" * width
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    rng = hi - lo
    if rng == 0:
        return "─" * width
    # Sample to `width` points
    step = max(1, len(values) // width)
    sampled = values[::step][-width:]
    result = ""
    for v in sampled:
        idx = int((v - lo) / rng * (len(blocks) - 1))
        result += blocks[idx]
    return result.ljust(width, "─")


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────

class IVBDashboard:

    MAX_SIGNALS = 50
    TICKERS     = ["XBT Curncy", "ES1 Index", "NQ1 Index", "SPX Index"]

    def __init__(self, paper_trader: "PaperTrader", feed: "BloombergFeed"):
        self.trader  = paper_trader
        self.feed    = feed
        self._signals: List["IVBSignal"] = []

    def add_signal(self, signal: "IVBSignal"):
        self._signals.insert(0, signal)
        if len(self._signals) > self.MAX_SIGNALS:
            self._signals.pop()

    # ── Renderers ─────────────────────────────────────────────────────────────

    def _header(self) -> Panel:
        now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        t   = self.trader

        prices   = self._get_prices()
        open_pnl = t.get_open_pnl(prices)

        # Advanced stats (v2 methods, fall back gracefully)
        sharpe  = getattr(t, "sharpe_ratio",  lambda: 0.0)()
        max_dd  = getattr(t, "max_drawdown",  lambda: 0.0)()
        pf      = getattr(t, "profit_factor", lambda: 0.0)()
        exp     = getattr(t, "expectancy",    lambda: 0.0)()

        txt = Text(justify="center")
        txt.append("  IVB BOT v2  —  Institutional Volume Breakout Model\n",
                   style="bold white on dark_blue")
        txt.append(f"  PAPER TRADING MODE   {now}\n\n", style="dim")

        # Row 1: balance and P&L
        txt.append("  Balance: ",  style="dim")
        txt.append(f"${t.balance:>10,.2f}  ", style="bold white")
        txt.append("Open P&L: ",   style="dim")
        txt.append(f"${open_pnl:>+10,.2f}  ", style=f"bold {pnl_color(open_pnl)}")
        txt.append("Daily P&L: ",  style="dim")
        txt.append(f"${t.daily_pnl:>+10,.2f}  ", style=f"bold {pnl_color(t.daily_pnl)}")
        txt.append("Total P&L: ",  style="dim")
        txt.append(f"${t.total_pnl:>+10,.2f}\n", style=f"bold {pnl_color(t.total_pnl)}")

        # Row 2: performance metrics
        wr = t.win_rate
        txt.append("  Win Rate: ", style="dim")
        txt.append(f"{wr:>5.1f}%  ", style=f"bold {'bright_green' if wr >= 50 else 'bright_red'}")
        txt.append("Trades: ",      style="dim")
        txt.append(f"{t.total_trades}  ", style="bold white")
        txt.append("Sharpe: ",      style="dim")
        txt.append(f"{sharpe:>+.2f}  ", style=f"bold {'bright_green' if sharpe > 0 else 'bright_red'}")
        txt.append("Max DD: ",      style="dim")
        txt.append(f"{max_dd:.1f}%  ", style=f"bold {'bright_red' if max_dd > 5 else 'bright_green'}")
        txt.append("PF: ",          style="dim")
        txt.append(f"{pf:.2f}  ",   style=f"bold {'bright_green' if pf > 1 else 'bright_red'}")
        txt.append("Expectancy: ",  style="dim")
        txt.append(f"${exp:+.2f}",  style=f"bold {pnl_color(exp)}")

        return Panel(txt, border_style="dark_blue", padding=(0, 1))

    def _market_table(self) -> Panel:
        tbl = Table(box=box.SIMPLE_HEAD, show_header=True,
                    header_style="bold dim", expand=True)
        tbl.add_column("Ticker",   style="bold cyan",  width=18)
        tbl.add_column("Price",    justify="right",     width=12)
        tbl.add_column("Chg %",    justify="right",     width=8)
        tbl.add_column("VWAP",     justify="right",     width=12)
        tbl.add_column("CVD",      justify="right",     width=10)
        tbl.add_column("Volume",   justify="right",     width=12)
        tbl.add_column("Hi",       justify="right",     width=12)
        tbl.add_column("Lo",       justify="right",     width=12)
        tbl.add_column("Bars",     justify="right",     width=6)

        for ticker in self.TICKERS:
            state = self.feed.get_state(ticker)
            if not state or state.last_price == 0:
                tbl.add_row(ticker, "—", "—", "—", "—", "—", "—", "—", "—")
                continue

            chg       = state.price_change_pct
            chg_style = pnl_color(chg)
            cvd_style = "bright_green" if state.cvd >= 0 else "bright_red"
            vwap      = getattr(state, "vwap", 0.0)

            tbl.add_row(
                ticker,
                f"{state.last_price:,.2f}",
                Text(f"{chg:+.2f}%", style=chg_style),
                f"{vwap:,.2f}" if vwap else "—",
                Text(f"{state.cvd:+,.0f}", style=cvd_style),
                f"{state.total_volume:,.0f}",
                f"{state.session_high:,.2f}",
                f"{state.session_low:,.2f}" if state.session_low < 1e10 else "—",
                str(len(state.bars))
            )

        return Panel(tbl, title="[bold]Live Market Data[/bold]",
                     border_style="blue", padding=(0, 1))

    def _signals_table(self) -> Panel:
        tbl = Table(box=box.SIMPLE_HEAD, show_header=True,
                    header_style="bold dim", expand=True)
        tbl.add_column("Time",    width=10)
        tbl.add_column("Ticker",  width=14)
        tbl.add_column("Dir",     width=6)
        tbl.add_column("Setup",   width=16)
        tbl.add_column("Conf",    justify="right", width=6)
        tbl.add_column("HTF",     width=4)
        tbl.add_column("Sess",    width=4)
        tbl.add_column("Entry",   justify="right", width=10)
        tbl.add_column("SL",      justify="right", width=10)
        tbl.add_column("TP",      justify="right", width=10)
        tbl.add_column("R:R",     justify="right", width=5)

        for sig in self._signals[:12]:
            risk  = abs(sig.entry_price - sig.stop_loss)
            rew   = abs(sig.target - sig.entry_price)
            rr    = f"{rew/risk:.1f}" if risk > 0 else "—"
            t_str = sig.timestamp.strftime("%H:%M:%S") if hasattr(sig, "timestamp") else "—"
            htf   = "✓" if getattr(sig, "htf_aligned", False) else "✗"
            sess  = getattr(sig, "session", "—")

            tbl.add_row(
                t_str,
                sig.ticker,
                Text(sig.direction.value, style=dir_color(sig.direction.value)),
                sig.setup_type.value,
                f"{sig.confidence:.0%}",
                Text(htf, style="bright_green" if htf == "✓" else "dim"),
                sess,
                f"{sig.entry_price:,.2f}",
                f"{sig.stop_loss:,.2f}",
                f"{sig.target:,.2f}",
                rr,
            )

        return Panel(tbl, title="[bold]IVB Signals[/bold]",
                     border_style="yellow", padding=(0, 1))

    def _positions_table(self) -> Panel:
        tbl = Table(box=box.SIMPLE_HEAD, show_header=True,
                    header_style="bold dim", expand=True)
        tbl.add_column("ID",       width=4)
        tbl.add_column("Ticker",   width=14)
        tbl.add_column("Dir",      width=6)
        tbl.add_column("Entry",    justify="right", width=10)
        tbl.add_column("Current",  justify="right", width=10)
        tbl.add_column("Trail SL", justify="right", width=10)
        tbl.add_column("TP",       justify="right", width=10)
        tbl.add_column("Partial",  width=8)
        tbl.add_column("Size",     justify="right", width=8)
        tbl.add_column("P&L",      justify="right", width=10)

        for pos in self.trader.open_positions:
            state = self.feed.get_state(pos.ticker)
            curr  = state.last_price if state else 0.0

            if curr > 0:
                move = curr - pos.entry_price
                if pos.direction == "SHORT":
                    move = -move
                pnl = pos.size_usd * (move / pos.entry_price)
            else:
                pnl = 0.0

            trail_sl  = getattr(pos, "trailing_stop", pos.stop_loss)
            partial   = "✓" if getattr(pos, "partial_closed", False) else "—"

            tbl.add_row(
                str(pos.id),
                pos.ticker,
                Text(pos.direction, style=dir_color(pos.direction)),
                f"{pos.entry_price:,.2f}",
                f"{curr:,.2f}" if curr else "—",
                f"{trail_sl:,.2f}",
                f"{pos.target:,.2f}",
                Text(partial, style="bright_green" if partial == "✓" else "dim"),
                f"${pos.size_usd:.2f}",
                Text(f"${pnl:+.2f}", style=pnl_color(pnl))
            )

        return Panel(tbl,
                     title=f"[bold]Open Positions ({self.trader.open_count})[/bold]",
                     border_style="cyan", padding=(0, 1))

    def _trades_table(self) -> Panel:
        tbl = Table(box=box.SIMPLE_HEAD, show_header=True,
                    header_style="bold dim", expand=True)
        tbl.add_column("ID",     width=4)
        tbl.add_column("Ticker", width=14)
        tbl.add_column("Dir",    width=6)
        tbl.add_column("Setup",  width=16)
        tbl.add_column("Entry",  justify="right", width=10)
        tbl.add_column("Exit",   justify="right", width=10)
        tbl.add_column("Reason", width=14)
        tbl.add_column("P&L",    justify="right", width=10)
        tbl.add_column("R",      justify="right", width=6)

        for t in self.trader.recent_trades[:12]:
            r_val = getattr(t, "r_value", 0.0)
            tbl.add_row(
                str(t.id),
                t.ticker,
                Text(t.direction, style=dir_color(t.direction)),
                t.setup_type,
                f"{t.entry_price:,.2f}",
                f"{t.exit_price:,.2f}",
                t.exit_reason,
                Text(f"${t.pnl:+.2f}", style=pnl_color(t.pnl)),
                Text(f"{r_val:+.2f}R",  style=pnl_color(r_val))
            )

        return Panel(tbl, title="[bold]Recent Closed Trades[/bold]",
                     border_style="green", padding=(0, 1))

    def _stats_panel(self) -> Panel:
        """Performance stats + equity sparkline + setup breakdown."""
        t = self.trader

        # Equity sparkline
        eq_curve = getattr(t, "_equity_curve", [])
        balances = [b for _, b in eq_curve] if eq_curve else [t.balance]
        spark    = sparkline(balances, width=30)

        # Setup breakdown
        breakdown = {}
        if hasattr(t, "setup_breakdown"):
            breakdown = t.setup_breakdown()

        txt = Text()
        txt.append("  Equity Curve: ", style="dim")
        txt.append(spark + "\n", style="bright_cyan")

        if breakdown:
            txt.append("\n  Setup Breakdown:\n", style="bold dim")
            for setup, d in breakdown.items():
                wr_color = "bright_green" if d.get("win_rate", 0) >= 50 else "bright_red"
                txt.append(f"    {setup:<20}", style="dim")
                txt.append(f"  {d['trades']:>3} trades  ", style="white")
                txt.append(f"WR={d.get('win_rate',0):.0f}%  ", style=wr_color)
                txt.append(f"P&L=${d.get('pnl',0):+.2f}\n", style=pnl_color(d.get("pnl", 0)))

        avg_r = getattr(t, "avg_r_multiple", lambda: 0.0)()
        txt.append(f"\n  Avg R-multiple: {avg_r:+.2f}R", style="bold white")

        return Panel(txt, title="[bold]Performance Analytics[/bold]",
                     border_style="magenta", padding=(0, 1))

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._header(),          name="header",    size=8),
            Layout(self._market_table(),    name="markets",   size=9),
            Layout(name="middle",           ratio=2),
            Layout(name="bottom",           ratio=2),
            Layout(self._stats_panel(),     name="stats",     size=10),
        )
        layout["middle"].split_row(
            Layout(self._signals_table(),   name="signals",   ratio=2),
            Layout(self._positions_table(), name="positions", ratio=1),
        )
        layout["bottom"].split_row(
            Layout(self._trades_table(),    name="trades"),
        )
        return layout

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run_live(self, on_refresh, refresh_interval: int = 30):
        """
        Run the live dashboard.
        Calls on_refresh() every refresh_interval seconds to scan for signals.
        """
        last_refresh = 0.0
        with Live(self._build_layout(), console=console,
                  refresh_per_second=2, screen=True) as live:
            while True:
                now = time.time()

                # Update prices for open position P&L monitoring
                prices = self._get_prices()
                if prices:
                    self.trader.update_prices(prices)

                # Refresh layout
                live.update(self._build_layout())

                # Run scan cycle
                if now - last_refresh >= refresh_interval:
                    on_refresh()
                    last_refresh = now

                time.sleep(0.5)

    def _get_prices(self) -> dict:
        prices = {}
        for ticker in self.TICKERS:
            state = self.feed.get_state(ticker)
            if state and state.last_price > 0:
                prices[ticker] = state.last_price
        return prices

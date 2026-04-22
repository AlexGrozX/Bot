"""
IVB Bot — Main Orchestrator (Paper Trading)
Fabio Valentini Institutional Volume Breakout Model
Powered by Bloomberg Terminal Desktop API
"""
import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Logging ───────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/ivb_bot.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("bot.main")

# ── Env ───────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / "config" / ".env")

BANKROLL          = float(os.getenv("BANKROLL_USDC", "400"))
SCAN_INTERVAL     = int(os.getenv("DASHBOARD_REFRESH", "30"))
MAX_POSITION_PCT  = float(os.getenv("MAX_POSITION_PCT", "0.10"))
DAILY_LOSS_LIMIT  = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.05"))

TICKERS = [t.strip() for t in
           os.getenv("INSTRUMENTS", "XBT Curncy,ES1 Index,NQ1 Index,SPX Index").split(",")]


class IVBBot:

    def __init__(self):
        self.running = False

        logger.info("=" * 60)
        logger.info("  IVB BOT — Institutional Volume Breakout Model")
        logger.info("  Mode: PAPER TRADING (no real money)")
        logger.info(f"  Starting Balance: ${BANKROLL:.2f}")
        logger.info(f"  Tickers: {', '.join(TICKERS)}")
        logger.info("=" * 60)

        self._init_bloomberg()
        self._init_paper_trader()
        self._init_engine()
        self._init_dashboard()

        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_bloomberg(self):
        try:
            from bloomberg.data_feed import BloombergFeed
            self.feed = BloombergFeed(tickers=TICKERS)
            if not self.feed.start():
                logger.error("Bloomberg connection failed. Is Terminal open?")
                sys.exit(1)
            logger.info(f"Bloomberg connected. Streaming {len(TICKERS)} tickers.")
        except ImportError:
            logger.error(
                "blpapi not installed.\n"
                "Run: python -m pip install "
                "--index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi"
            )
            sys.exit(1)
        except Exception as e:
            logger.error(f"Bloomberg init error: {e}")
            sys.exit(1)

    def _init_paper_trader(self):
        from execution.paper_trader import PaperTrader
        self.trader = PaperTrader(
            starting_balance=BANKROLL,
            max_position_pct=MAX_POSITION_PCT,
            max_daily_loss_pct=DAILY_LOSS_LIMIT,
            log_dir="logs"
        )
        logger.info(f"Paper trader ready. Balance: ${self.trader.balance:.2f}")

    def _init_engine(self):
        from engine.ivb_engine import IVBEngine
        self.engine = IVBEngine(
            feed=self.feed,
            volume_breakout_multiplier=float(os.getenv("VOLUME_BREAKOUT_MULTIPLIER", "3.0")),
            min_delta_imbalance=float(os.getenv("MIN_DELTA_IMBALANCE", "60.0")),
            lvn_proximity_pct=float(os.getenv("LVN_PROXIMITY_PCT", "0.003"))
        )
        logger.info("IVB signal engine ready.")

    def _init_dashboard(self):
        from dashboard.dashboard import IVBDashboard
        self.dashboard = IVBDashboard(self.trader, self.feed)
        logger.info("Dashboard ready.")

    # ── Scan Cycle ────────────────────────────────────────────────────────────

    def _scan_cycle(self):
        """Evaluate all tickers for IVB signals and open paper trades."""
        # Update open positions with current prices first
        prices = {}
        for ticker in TICKERS:
            state = self.feed.get_state(ticker)
            if state and state.last_price > 0:
                prices[ticker] = state.last_price
        if prices:
            self.trader.update_prices(prices)

        # Reset daily halt at start of new day
        today = datetime.now().date().isoformat()
        if not hasattr(self, "_last_day") or self._last_day != today:
            self.trader.reset_daily_halt()
            self._last_day = today

        # Evaluate each ticker
        new_signals = 0
        for ticker in TICKERS:
            try:
                sig = self.engine.evaluate(ticker)
                if sig:
                    self.dashboard.add_signal(sig)
                    trade = self.trader.open_trade(sig)
                    if trade:
                        new_signals += 1
            except Exception as e:
                logger.error(f"Error evaluating {ticker}: {e}")

        if new_signals:
            logger.info(f"Scan complete: {new_signals} new trade(s) opened.")

    # ── Scan-only mode ────────────────────────────────────────────────────────

    def scan_once(self):
        """Print current market state and any signals — no trading."""
        logger.info("Running single scan (no trades)...")
        time.sleep(4)  # Let Bloomberg populate

        for ticker in TICKERS:
            state = self.feed.get_state(ticker)
            if state and state.last_price > 0:
                logger.info(
                    f"  {ticker:20s} | price={state.last_price:>10,.2f} | "
                    f"cvd={state.cvd:>+8,.0f} | vol={state.total_volume:>10,.0f} | "
                    f"bars={len(state.bars)}"
                )
            else:
                logger.info(f"  {ticker:20s} | waiting for data...")

            sig = self.engine.evaluate(ticker)
            if sig:
                logger.info(
                    f"  *** SIGNAL *** {sig.direction.value} | {sig.setup_type.value} | "
                    f"conf={sig.confidence:.0%} | entry={sig.entry_price:.2f} | "
                    f"sl={sig.stop_loss:.2f} | tp={sig.target:.2f}"
                )
                logger.info(f"  Reason: {sig.reasoning}")
            else:
                logger.info(f"  No IVB signal on {ticker} right now.")

        logger.info(self.trader.summary())

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self):
        self.running = True
        self._last_day = datetime.now().date().isoformat()
        logger.info(f"Bot running. Scan every {SCAN_INTERVAL}s. Ctrl+C to stop.")

        # Let Bloomberg data populate before first scan
        time.sleep(5)

        def on_refresh():
            if self.running:
                self._scan_cycle()

        try:
            self.dashboard.run_live(on_refresh, refresh_interval=SCAN_INTERVAL)
        except KeyboardInterrupt:
            self._shutdown()
        except Exception as e:
            logger.error(f"Dashboard error: {e}. Falling back to headless mode.")
            while self.running:
                self._scan_cycle()
                logger.info(self.trader.summary())
                time.sleep(SCAN_INTERVAL)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _shutdown(self, *args):
        logger.info("Shutting down...")
        self.running = False
        try:
            # Close all open positions at last known price
            prices = {}
            for ticker in TICKERS:
                state = self.feed.get_state(ticker)
                if state and state.last_price > 0:
                    prices[ticker] = state.last_price
            self.trader.close_all(prices)
        except:
            pass
        try:
            self.feed.stop()
        except:
            pass
        logger.info("=" * 60)
        logger.info(self.trader.summary())
        logger.info("=" * 60)
        sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IVB Bot — Institutional Volume Breakout (Paper Trading)"
    )
    parser.add_argument("--scan", action="store_true",
                        help="Single scan: print signals and exit")
    args = parser.parse_args()

    bot = IVBBot()

    if args.scan:
        bot.scan_once()
    else:
        bot.run()


if __name__ == "__main__":
    main()

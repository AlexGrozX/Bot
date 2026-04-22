"""
IVB Bot — Bloomberg Data Feed
Real-time tick streaming, intraday bars, and order flow data.
Connects to Bloomberg Desktop API via BBComm on localhost:8194.
"""
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger("bloomberg.data_feed")


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Bar:
    """A single OHLCV bar with order flow data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float  = 0.0   # Estimated from tick data
    sell_volume: float = 0.0

    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


@dataclass
class VolumeProfile:
    """Volume Profile for a session."""
    levels: Dict[float, float] = field(default_factory=dict)  # price → volume
    poc: float = 0.0    # Point of Control
    vah: float = 0.0    # Value Area High
    val: float = 0.0    # Value Area Low
    lvns: List[float] = field(default_factory=list)  # Low Volume Nodes


@dataclass
class MarketState:
    """Current state of a market."""
    ticker: str
    last_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    volume: float = 0.0
    total_volume: float = 0.0
    session_high: float = 0.0
    session_low: float = float("inf")
    open_price: float = 0.0
    prev_close: float = 0.0
    cvd: float = 0.0             # Cumulative Volume Delta
    bars: deque = field(default_factory=lambda: deque(maxlen=200))
    volume_profile: Optional[VolumeProfile] = None
    last_update: datetime = field(default_factory=datetime.now)

    @property
    def price_change_pct(self) -> float:
        if self.prev_close and self.prev_close > 0:
            return ((self.last_price - self.prev_close) / self.prev_close) * 100
        return 0.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid if self.ask > self.bid else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Bloomberg Feed
# ─────────────────────────────────────────────────────────────────────────────

class BloombergFeed:
    """
    Connects to Bloomberg Desktop API and streams real-time data.
    Maintains MarketState for each ticker.
    """

    TICK_FIELDS = [
        "LAST_PRICE", "BID", "ASK", "VOLUME",
        "LAST_TRADE_SIZE", "LAST_AT_TRADE_COND_CODE_RT",
        "OPEN", "HIGH", "LOW", "PREV_CLOSE_VALUE_REALTIME",
        "LAST_UPDATE_ALL_SESSIONS_RT"
    ]

    BAR_INTERVAL = 5  # 5-minute bars

    def __init__(self, tickers: List[str],
                 host: str = "localhost", port: int = 8194):
        self.tickers = tickers
        self.host = host
        self.port = port

        self._states: Dict[str, MarketState] = {
            t: MarketState(ticker=t) for t in tickers
        }
        self._session = None
        self._subs = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        # Current bar accumulator per ticker
        self._current_bar: Dict[str, Optional[Bar]] = {t: None for t in tickers}
        self._bar_start: Dict[str, Optional[datetime]] = {t: None for t in tickers}

    def start(self) -> bool:
        try:
            import blpapi
        except ImportError:
            logger.error("blpapi not installed. Run the Bloomberg pip install command.")
            return False

        try:
            import blpapi
            opts = blpapi.SessionOptions()
            opts.setServerHost(self.host)
            opts.setServerPort(self.port)

            self._session = blpapi.Session(opts, self._event_handler)
            if not self._session.start():
                logger.error("Bloomberg session failed to start.")
                return False

            logger.info(f"Bloomberg session started → {self.host}:{self.port}")

            # Open services
            for svc in ["//blp/mktdata", "//blp/mktbar", "//blp/refdata"]:
                if not self._session.openService(svc):
                    logger.warning(f"Could not open service: {svc}")
                else:
                    logger.info(f"Service opened: {svc}")

            # Subscribe to real-time ticks
            self._subscribe_ticks()

            # Subscribe to 5-min bars
            self._subscribe_bars()

            self._running = True
            return True

        except Exception as e:
            logger.error(f"Bloomberg start error: {e}")
            return False

    def _subscribe_ticks(self):
        import blpapi
        subs = blpapi.SubscriptionList()
        for ticker in self.tickers:
            corr = blpapi.CorrelationId(f"tick:{ticker}")
            subs.add(
                topic=ticker,
                fields=self.TICK_FIELDS,
                correlationId=corr
            )
            logger.info(f"Subscribing ticks: {ticker}")
        self._session.subscribe(subs)

    def _subscribe_bars(self):
        """
        Subscribe to real-time intraday bars via //blp/mktbar.

        The correct Bloomberg Desktop API approach is to use a SubscriptionList
        with a topic string in the form:

            //blp/mktbar/ticker?eventType=TRADE&interval=5

        NOT createRequest("BarsSubscriptionRequest") — that operation does not
        exist on the mktbar service and raises error 0x0006000d.
        """
        import blpapi
        subs = blpapi.SubscriptionList()
        for ticker in self.tickers:
            # URL-encode spaces in the ticker for the topic string
            safe_ticker = ticker.replace(" ", "%20")
            topic = (f"//blp/mktbar/{safe_ticker}"
                     f"?eventType=TRADE&interval={self.BAR_INTERVAL}")
            corr = blpapi.CorrelationId(f"bar:{ticker}")
            subs.add(topic=topic, correlationId=corr)
            logger.info(f"Subscribing bars: {ticker} ({self.BAR_INTERVAL}min)")
        self._session.subscribe(subs)

    def _event_handler(self, event, session):
        import blpapi
        try:
            for msg in event:
                corr_str = str(msg.correlationId().value()) if msg.correlationId() else ""

                if event.eventType() == blpapi.Event.SUBSCRIPTION_DATA:
                    if corr_str.startswith("tick:"):
                        ticker = corr_str[5:]
                        self._process_tick(ticker, msg)
                    elif corr_str.startswith("bar:"):
                        ticker = corr_str[4:]
                        self._process_bar(ticker, msg)

                elif event.eventType() == blpapi.Event.SUBSCRIPTION_STATUS:
                    logger.debug(f"Subscription status: {msg}")

        except Exception as e:
            logger.error(f"Event handler error: {e}")

    def _process_tick(self, ticker: str, msg):
        """Process a real-time tick message."""
        with self._lock:
            state = self._states.get(ticker)
            if not state:
                return

            def safe_float(field_name):
                try:
                    if msg.hasElement(field_name):
                        v = msg.getElementAsFloat(field_name)
                        return v if v and v != 0.0 else None
                except:
                    pass
                return None

            price = safe_float("LAST_PRICE")
            bid   = safe_float("BID")
            ask   = safe_float("ASK")
            vol   = safe_float("VOLUME")
            size  = safe_float("LAST_TRADE_SIZE")
            prev  = safe_float("PREV_CLOSE_VALUE_REALTIME")
            open_ = safe_float("OPEN")
            high  = safe_float("HIGH")
            low   = safe_float("LOW")

            if price:
                state.last_price = price
                state.last_update = datetime.now()

                # Session high/low
                if price > state.session_high:
                    state.session_high = price
                if price < state.session_low:
                    state.session_low = price

                # Estimate buy/sell volume from bid/ask comparison
                if size and bid and ask:
                    mid = (bid + ask) / 2
                    if price >= mid:
                        state.cvd += size   # Buy aggression
                    else:
                        state.cvd -= size   # Sell aggression

                # Update bar accumulator
                self._update_current_bar(ticker, price, size or 0, bid, ask)

            if bid:  state.bid = bid
            if ask:  state.ask = ask
            if vol:  state.total_volume = vol
            if prev: state.prev_close = prev
            if open_: state.open_price = open_
            if high and high > state.session_high:
                state.session_high = high
            if low and low < state.session_low:
                state.session_low = low

    def _update_current_bar(self, ticker: str, price: float,
                             size: float, bid: float, ask: float):
        """Accumulate ticks into 5-minute bars."""
        now = datetime.now()
        bar_minute = (now.minute // self.BAR_INTERVAL) * self.BAR_INTERVAL
        bar_start = now.replace(minute=bar_minute, second=0, microsecond=0)

        current = self._current_bar.get(ticker)
        last_start = self._bar_start.get(ticker)

        if current is None or last_start != bar_start:
            # Finalize previous bar
            if current is not None:
                state = self._states[ticker]
                state.bars.append(current)
                # Rebuild volume profile periodically
                if len(state.bars) % 12 == 0:  # Every hour
                    bars_list = list(state.bars)
                    state.volume_profile = self.build_volume_profile(
                        ticker, bars_list, 0.70, 0.20)

            # Start new bar
            self._current_bar[ticker] = Bar(
                timestamp=bar_start,
                open=price, high=price, low=price, close=price,
                volume=size
            )
            self._bar_start[ticker] = bar_start
        else:
            # Update existing bar
            current.high  = max(current.high, price)
            current.low   = min(current.low, price)
            current.close = price
            current.volume += size

            # Estimate buy/sell split
            if bid and ask:
                mid = (bid + ask) / 2
                if price >= mid:
                    current.buy_volume += size
                else:
                    current.sell_volume += size

    def _process_bar(self, ticker: str, msg):
        """Process a Bloomberg mktbar subscription event.

        The //blp/mktbar service delivers a BarData element containing:
            OPEN, HIGH, LOW, CLOSE, VOLUME, NUMBER_OF_TICKS, TIME
        The element name is 'BarData' (not the field names directly on msg).
        """
        with self._lock:
            state = self._states.get(ticker)
            if not state:
                return
            try:
                # mktbar delivers fields inside a 'BarData' sub-element.
                # Fall back to reading directly from msg if BarData is absent
                # (some versions surface fields at the top level).
                if msg.hasElement("BarData"):
                    bd = msg.getElement("BarData")
                    o  = bd.getElementAsFloat("OPEN")
                    h  = bd.getElementAsFloat("HIGH")
                    l  = bd.getElementAsFloat("LOW")
                    c  = bd.getElementAsFloat("CLOSE")
                    v  = bd.getElementAsFloat("VOLUME")
                    ts_str = (bd.getElementAsString("TIME")
                              if bd.hasElement("TIME") else None)
                else:
                    o  = msg.getElementAsFloat("OPEN")
                    h  = msg.getElementAsFloat("HIGH")
                    l  = msg.getElementAsFloat("LOW")
                    c  = msg.getElementAsFloat("CLOSE")
                    v  = msg.getElementAsFloat("VOLUME")
                    ts_str = None

                try:
                    ts = (datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
                          if ts_str else datetime.now())
                except ValueError:
                    ts = datetime.now()

                bar = Bar(
                    timestamp=ts,
                    open=o, high=h, low=l, close=c, volume=v
                )
                state.bars.append(bar)
                logger.debug(f"Bar [{ticker}] O={bar.open} H={bar.high} "
                             f"L={bar.low} C={bar.close} V={bar.volume}")
            except Exception as e:
                logger.debug(f"Bar parse error [{ticker}]: {e}")

    def build_volume_profile(self, ticker: str, bars: List[Bar],
                              value_area_pct: float = 0.70,
                              lvn_threshold: float = 0.20) -> Optional[VolumeProfile]:
        """Build a Volume Profile from a list of bars."""
        if not bars:
            return None

        # Build price → volume map (round to 0.1% increments)
        price_vol: Dict[float, float] = {}
        total_vol = 0.0

        for bar in bars:
            if bar.volume <= 0:
                continue
            # Distribute volume across the bar's range
            price_range = bar.high - bar.low
            if price_range == 0:
                key = round(bar.close, 1)
                price_vol[key] = price_vol.get(key, 0) + bar.volume
            else:
                # Simple distribution: 3 levels per bar
                for frac in [0.25, 0.50, 0.75]:
                    p = bar.low + frac * price_range
                    key = round(p, 1)
                    price_vol[key] = price_vol.get(key, 0) + bar.volume / 3
            total_vol += bar.volume

        if not price_vol or total_vol == 0:
            return None

        # POC = price with highest volume
        poc = max(price_vol, key=price_vol.get)

        # Value Area: 70% of volume around POC
        sorted_prices = sorted(price_vol.keys())
        poc_idx = sorted_prices.index(poc) if poc in sorted_prices else len(sorted_prices) // 2

        va_vol = price_vol[poc]
        target_vol = total_vol * value_area_pct
        lo_idx, hi_idx = poc_idx, poc_idx

        while va_vol < target_vol:
            lo_candidate = price_vol.get(sorted_prices[lo_idx - 1], 0) if lo_idx > 0 else 0
            hi_candidate = price_vol.get(sorted_prices[hi_idx + 1], 0) if hi_idx < len(sorted_prices) - 1 else 0

            if lo_candidate == 0 and hi_candidate == 0:
                break
            if lo_candidate >= hi_candidate and lo_idx > 0:
                lo_idx -= 1
                va_vol += lo_candidate
            elif hi_idx < len(sorted_prices) - 1:
                hi_idx += 1
                va_vol += hi_candidate
            else:
                break

        val = sorted_prices[lo_idx]
        vah = sorted_prices[hi_idx]

        # LVNs: prices with volume < threshold × average
        avg_vol = total_vol / len(price_vol)
        lvns = [p for p, v in price_vol.items() if v < avg_vol * lvn_threshold]

        profile = VolumeProfile(
            levels=price_vol,
            poc=poc, vah=vah, val=val,
            lvns=sorted(lvns)
        )

        with self._lock:
            state = self._states.get(ticker)
            if state:
                state.volume_profile = profile

        return profile

    def get_state(self, ticker: str) -> Optional[MarketState]:
        with self._lock:
            return self._states.get(ticker)

    def stop(self):
        self._running = False
        if self._session:
            try:
                self._session.stop()
                logger.info("Bloomberg session stopped.")
            except:
                pass

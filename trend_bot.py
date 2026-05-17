"""
Trend Following Trading Bot
----------------------------
Strategy: Buy when price is above 50-day moving average (uptrend)
          Exit when price closes below 50-day moving average
Watchlist: SPY, QQQ
Risk: 2% per trade max
Reward/Risk Target: 3:1
Black swan protection: VIX filter + 15% account circuit breaker

Requirements:
    pip install alpaca-trade-api pandas pandas-ta requests

Setup:
    1. Create free account at alpaca.markets
    2. Get your API keys from the Alpaca dashboard (Paper Trading section)
    3. Replace API_KEY and SECRET_KEY below with your actual keys
    4. Keep PAPER = True for 30-60 days before going live
"""

import time
import logging
from datetime import datetime, timedelta

import pandas as pd
import pandas_ta as ta
import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ─────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ─────────────────────────────────────────────

import os
API_KEY    = os.environ.get("ALPACA_API_KEY", "YOUR_ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET_KEY")

PAPER = True                             # True = paper trading (safe), False = live real money

WATCHLIST             = ["SPY", "QQQ"]  # Stocks to monitor
RISK_PER_TRADE_PCT    = 0.02            # Risk 2% of account per trade
MAX_ACCOUNT_LOSS_PCT  = 0.15            # Circuit breaker: stop if account drops 15%
VIX_PAUSE_LEVEL       = 30             # Pause new trades if VIX >= 30
VIX_STOP_LEVEL        = 40             # Stop all trades if VIX >= 40
REWARD_RISK_RATIO     = 3.0            # Target = 3x your risk (3R)
MA_PERIOD             = 50             # 50-day moving average

# ─────────────────────────────────────────────
# LOGGING — writes to file and screen
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("trend_bot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────

trade_client = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)
data_client  = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# ─────────────────────────────────────────────
# VIX FILTER — black swan protection
# ─────────────────────────────────────────────

def get_vix() -> float:
    """Fetch current VIX (market fear index) from Yahoo Finance. Free, no key needed."""
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1d&range=1d"
        r = requests.get(url, timeout=10)
        data = r.json()
        vix = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        log.info(f"VIX: {vix}")
        return float(vix)
    except Exception as e:
        log.warning(f"Could not fetch VIX: {e}. Defaulting to 0 (no block).")
        return 0.0


def vix_allows_trading(vix: float) -> bool:
    """
    VIX < 30  → trade normally
    VIX 30-40 → pause new entries (market fear elevated)
    VIX >= 40 → stop all trading (crisis/panic mode)
    """
    if vix >= VIX_STOP_LEVEL:
        log.warning(f"VIX {vix} >= {VIX_STOP_LEVEL}. EMERGENCY STOP. All trading halted.")
        return False
    if vix >= VIX_PAUSE_LEVEL:
        log.warning(f"VIX {vix} >= {VIX_PAUSE_LEVEL}. Market fear elevated. Pausing new entries.")
        return False
    log.info(f"VIX {vix} — market calm. Trading allowed.")
    return True

# ─────────────────────────────────────────────
# ACCOUNT CIRCUIT BREAKER
# ─────────────────────────────────────────────

def get_account_info() -> dict:
    account = trade_client.get_account()
    return {
        "equity":          float(account.equity),
        "cash":            float(account.cash),
        "starting_equity": float(account.last_equity),
    }


def circuit_breaker_triggered(account: dict) -> bool:
    """Stop all trading if account has dropped more than MAX_ACCOUNT_LOSS_PCT."""
    if account["starting_equity"] <= 0:
        return False
    drop = (account["starting_equity"] - account["equity"]) / account["starting_equity"]
    if drop >= MAX_ACCOUNT_LOSS_PCT:
        log.warning(
            f"CIRCUIT BREAKER TRIGGERED: Account down {drop:.1%}. "
            f"Stopping all trading. Please review manually before resuming."
        )
        return True
    log.info(f"Account down {drop:.1%} from start — within safe range.")
    return False

# ─────────────────────────────────────────────
# MARKET DATA
# ─────────────────────────────────────────────

def get_daily_bars(symbol: str, days: int = 120) -> pd.DataFrame:
    """Fetch daily OHLCV price bars for a symbol. Uses IEX feed."""
    end   = datetime.now()
    start = end - timedelta(days=days)
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed="iex",
    )
    bars = data_client.get_stock_bars(req).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")
    bars = bars.sort_index()
    return bars

# ─────────────────────────────────────────────
# TREND DETECTION — the core strategy logic
# ─────────────────────────────────────────────

def is_in_uptrend(bars: pd.DataFrame) -> bool:
    """
    Uptrend = current price is above the 50-day moving average.
    Also checks that the MA itself is sloping upward (trending, not flat).
    """
    if len(bars) < MA_PERIOD:
        log.info(f"  Not enough data for {MA_PERIOD}-day MA.")
        return False

    bars["ma50"] = bars["close"].rolling(window=MA_PERIOD).mean()

    current_price = bars["close"].iloc[-1]
    current_ma    = bars["ma50"].iloc[-1]
    prev_ma       = bars["ma50"].iloc[-5]   # MA 5 days ago

    price_above_ma = current_price > current_ma
    ma_sloping_up  = current_ma > prev_ma   # MA trending upward

    log.info(f"  Price: ${current_price:.2f} | 50-day MA: ${current_ma:.2f}")
    log.info(f"  Price above MA: {price_above_ma} | MA sloping up: {ma_sloping_up}")

    return price_above_ma and ma_sloping_up


def is_below_ma(bars: pd.DataFrame) -> bool:
    """Exit signal: price has closed below the 50-day moving average."""
    bars["ma50"] = bars["close"].rolling(window=MA_PERIOD).mean()
    current_price = bars["close"].iloc[-1]
    current_ma    = bars["ma50"].iloc[-1]
    return current_price < current_ma

# ─────────────────────────────────────────────
# POSITION SIZING — always 2% risk max
# ─────────────────────────────────────────────

def calculate_position(account_equity: float, entry: float, stop: float) -> dict:
    """
    Calculate number of shares based on 2% account risk rule.
    Stop is placed just below the 50-day MA.
    """
    risk_per_share   = entry - stop
    if risk_per_share <= 0:
        return {}

    max_risk_dollars = account_equity * RISK_PER_TRADE_PCT
    shares           = int(max_risk_dollars / risk_per_share)

    if shares < 1:
        log.info("  Position too small for current account size. Skipping.")
        return {}

    target = round(entry + (risk_per_share * REWARD_RISK_RATIO), 2)

    return {
        "entry":          round(entry, 2),
        "stop":           round(stop, 2),
        "target":         target,
        "shares":         shares,
        "risk_dollars":   round(shares * risk_per_share, 2),
        "reward_dollars": round(shares * risk_per_share * REWARD_RISK_RATIO, 2),
    }

# ─────────────────────────────────────────────
# ORDER EXECUTION
# ─────────────────────────────────────────────

def place_buy_order(symbol: str, position: dict):
    """Place a market buy order with bracket (stop-loss + take-profit)."""
    try:
        order = trade_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=position["shares"],
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                order_class="bracket",
                stop_loss={"stop_price": position["stop"]},
                take_profit={"limit_price": position["target"]},
            )
        )
        log.info(
            f"  ✅ BUY ORDER PLACED: {symbol} | "
            f"Shares: {position['shares']} | "
            f"Entry: ~${position['entry']} | "
            f"Stop: ${position['stop']} | "
            f"Target: ${position['target']} | "
            f"Max Risk: ${position['risk_dollars']} | "
            f"Potential Gain: ${position['reward_dollars']}"
        )
        return order
    except Exception as e:
        log.error(f"  ❌ Buy order failed for {symbol}: {e}")
        return None


def place_sell_order(symbol: str, qty: float):
    """Exit position — sell all shares at market price."""
    try:
        order = trade_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
        )
        log.info(f"  ✅ SELL ORDER PLACED: {symbol} | Shares: {qty} | Reason: Price below 50-day MA")
        return order
    except Exception as e:
        log.error(f"  ❌ Sell order failed for {symbol}: {e}")
        return None

# ─────────────────────────────────────────────
# POSITION CHECKER
# ─────────────────────────────────────────────

def get_position(symbol: str):
    """Return current position for a symbol, or None if not held."""
    try:
        positions = trade_client.get_all_positions()
        for p in positions:
            if p.symbol == symbol:
                return p
    except Exception:
        pass
    return None

# ─────────────────────────────────────────────
# MAIN SCAN — runs once per trading day
# ─────────────────────────────────────────────

def run_scan():
    log.info("=" * 60)
    log.info(f"Trend Bot Scan — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Mode: {'PAPER TRADING' if PAPER else '⚠️  LIVE TRADING'}")

    # 1. VIX check — black swan filter
    vix = get_vix()
    if not vix_allows_trading(vix):
        return

    # 2. Account check — circuit breaker
    account = get_account_info()
    log.info(f"Account equity: ${account['equity']:,.2f} | Cash: ${account['cash']:,.2f}")
    if circuit_breaker_triggered(account):
        return

    # 3. Scan each symbol
    for symbol in WATCHLIST:
        log.info(f"\n--- {symbol} ---")

        try:
            bars = get_daily_bars(symbol)
        except Exception as e:
            log.error(f"  Could not fetch data for {symbol}: {e}")
            continue

        position = get_position(symbol)

        # ── EXIT LOGIC ──
        # If we're in a position, check if we should exit
        if position:
            qty = float(position.qty)
            log.info(f"  Currently holding {qty} shares of {symbol}")
            if is_below_ma(bars):
                log.info(f"  Price crossed below 50-day MA — EXIT SIGNAL")
                place_sell_order(symbol, qty)
            else:
                log.info(f"  Still in uptrend — holding position")
            continue

        # ── ENTRY LOGIC ──
        # If we're not in a position, check if we should enter
        if is_in_uptrend(bars):
            log.info(f"  Uptrend confirmed — checking entry...")

            entry = bars["close"].iloc[-1]
            ma50  = bars["close"].rolling(window=MA_PERIOD).mean().iloc[-1]
            stop  = round(ma50 * 0.99, 2)   # Stop just below the 50-day MA

            pos = calculate_position(account["equity"], entry, stop)
            if pos:
                place_buy_order(symbol, pos)
        else:
            log.info(f"  No uptrend detected — no trade.")

    log.info("\nScan complete. Next scan tomorrow at market open.")


# ─────────────────────────────────────────────
# SCHEDULER — waits for 9:35 AM ET then scans
# ─────────────────────────────────────────────

def wait_for_market_open():
    """Wait until 9:35 AM ET on a weekday."""
    while True:
        now = datetime.now()
        if now.weekday() < 5 and now.hour == 9 and now.minute == 35:
            return
        time.sleep(60)


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Trend Following Bot — Starting Up")
    log.info(f"Mode: {'PAPER TRADING ✓' if PAPER else '⚠️  LIVE TRADING'}")
    log.info(f"Watchlist: {WATCHLIST}")
    log.info(f"Strategy: Buy above {MA_PERIOD}-day MA | Sell below {MA_PERIOD}-day MA")
    log.info(f"Risk per trade: {RISK_PER_TRADE_PCT:.0%} | Reward target: {REWARD_RISK_RATIO}R")
    log.info(f"VIX pause: {VIX_PAUSE_LEVEL} | VIX stop: {VIX_STOP_LEVEL}")
    log.info(f"Circuit breaker: stops if account drops {MAX_ACCOUNT_LOSS_PCT:.0%}")
    log.info("=" * 60)

    while True:
        wait_for_market_open()
        run_scan()
        time.sleep(23 * 60 * 60)   # Sleep 23 hours before checking again

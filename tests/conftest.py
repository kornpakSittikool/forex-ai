"""
Pytest configuration and shared fixtures for Gold Day Trade Assistant test suite.
"""

import datetime
import math
import pytest
import pytz
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from main import app, state
import analyzer
from analyzer import THAI_TZ


@pytest.fixture
def client():
    """
    FastAPI TestClient fixture with isolated state reset.
    """
    initial_user_settings = {"investment": 10.0, "multiplier": 200}
    state["latest_analysis"] = None
    state["history"] = []
    state["last_triggered_date"] = None
    state["last_triggered_slots"] = set()
    state["user_settings"] = initial_user_settings.copy()
    state["new_alert_pending"] = False

    with TestClient(app) as test_client:
        yield test_client

    # Clean up after test
    state["latest_analysis"] = None
    state["history"] = []
    state["last_triggered_date"] = None
    state["last_triggered_slots"] = set()
    state["user_settings"] = initial_user_settings.copy()
    state["new_alert_pending"] = False


def generate_candles(
    count: int = 150,
    base_price: float = 2650.0,
    start_dt: datetime.datetime = None,
    trend: str = "flat",
    volatility: float = 1.0
) -> pd.DataFrame:
    """
    Helper to generate synthetic tz-aware 1m candle DataFrames.
    """
    if start_dt is None:
        now = datetime.datetime.now(THAI_TZ)
        start_dt = now - datetime.timedelta(minutes=1 * count)

    dates = [start_dt + datetime.timedelta(minutes=1 * i) for i in range(count)]
    records = []
    current = base_price

    for i in range(count):
        if trend == "bullish":
            delta = 0.5 + (i * 0.05)
        elif trend == "bearish":
            delta = -0.5 - (i * 0.05)
        elif trend == "flat":
            delta = 0.0
        else:
            delta = np.sin(i / 3.0) * 0.3

        open_p = current
        close_p = current + delta
        high_p = max(open_p, close_p) + (volatility if trend != "flat" else 0.0)
        low_p = min(open_p, close_p) - (volatility if trend != "flat" else 0.0)
        current = close_p

        records.append({
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(close_p, 2),
            "volume": 100.0
        })

    df = pd.DataFrame(records, index=pd.to_datetime(dates))
    df.index.name = "timestamp"
    return df


@pytest.fixture
def bullish_df():
    """
    Generates a DataFrame with strong bullish breakout:
    - 60 candles leading up to today 15:00 Thai Time
    - Asian session range: $38.00 (within $35-$45 healthy bounds, 5-7x ATR)
    - RSI at 15:00: ~61.4 (momentum sweet spot between 55 and 62)
    - 15:00 entry candle: solid body (62.5% >= 60%), small opposing upper rejection wick (12.5% < 25%)
    - EMA spread: clear separation between EMA20 and EMA50 (> $2.00, EMA20 > EMA50)
    - Passes genuine evaluate_trade_quality() with score = 100 (EXCELLENT / APPROVED)
    """
    now = datetime.datetime.now(THAI_TZ)
    target_1500 = now.replace(hour=15, minute=0, second=0, microsecond=0)
    start_dt = target_1500 - datetime.timedelta(minutes=15 * 60)
    dates = [start_dt + datetime.timedelta(minutes=15 * i) for i in range(60)]

    # Pre-Asian session (candles 0..27, 00:00 to 06:45): gentle oscillation around 2648
    closes = []
    for i in range(28):
        closes.append(2648.0 + 1.5 * np.sin(i * 0.5))

    # Asian session early run (candles 28..45): trend up from 2652 to 2684
    for i in range(28, 46):
        progress = (i - 28) / 17.0
        val = 2652.0 + progress * 32.0 + (0.5 if i % 2 == 0 else -0.5)
        closes.append(round(val, 2))

    # Candles 46..59: 14 deltas calibrated for RSI momentum in sweet spot (RSI ~61.4)
    deltas = [
        +2.0, -1.5, +1.8, -1.4, +2.0, -1.6, +1.8, -1.5, +1.7, -1.5, +1.5, -1.5, +1.0, +2.5
    ]
    current = closes[-1]
    for d in deltas:
        current += d
        closes.append(round(current, 2))

    records = []
    for i in range(60):
        c = closes[i]
        prev_c = closes[i - 1] if i > 0 else c
        o = prev_c
        if i == 59:
            # 15:00 breakout candle: body 62.5% (>=60%), upper wick 12.5% (<25%)
            o = round(c - 2.5, 2)
            h = round(c + 0.5, 2)
            l = round(c - 3.5, 2)
        elif 28 <= i < 56:
            # Constrain Asian session range to exactly $38.00 (Low 2650.0 to High 2688.0)
            if i == 28:
                l = 2650.0
                h = max(o, c) + 1.5
            elif i == 45:
                h = 2688.0
                l = min(o, c) - 1.5
            else:
                h = min(2687.5, max(o, c) + 2.0)
                l = max(2650.5, min(o, c) - 2.0)
        else:
            h = max(o, c) + 2.0
            l = min(o, c) - 2.0
        records.append({
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "volume": 150.0
        })

    df = pd.DataFrame(records, index=pd.to_datetime(dates))
    df.index.name = "timestamp"
    return df


@pytest.fixture
def bearish_df():
    """
    Generates a DataFrame with strong bearish breakout:
    - 60 candles leading up to today 15:00 Thai Time
    - Asian session range: $38.00 (within $35-$45 healthy bounds, 5-7x ATR)
    - RSI at 15:00: ~38.6 (momentum sweet spot between 38 and 45)
    - 15:00 entry candle: solid body (62.5% >= 60%), small opposing lower rejection wick (12.5% < 25%)
    - EMA spread: clear separation between EMA20 and EMA50 (> $2.00, EMA20 < EMA50)
    - Passes genuine evaluate_trade_quality() with score = 100 (EXCELLENT / APPROVED)
    """
    now = datetime.datetime.now(THAI_TZ)
    target_1500 = now.replace(hour=15, minute=0, second=0, microsecond=0)
    start_dt = target_1500 - datetime.timedelta(minutes=15 * 60)
    dates = [start_dt + datetime.timedelta(minutes=15 * i) for i in range(60)]

    # Pre-Asian session (candles 0..27, 00:00 to 06:45): gentle oscillation around 2702
    closes = []
    for i in range(28):
        closes.append(2702.0 - 1.5 * np.sin(i * 0.5))

    # Asian session early run (candles 28..45): trend down from 2698 to 2666
    for i in range(28, 46):
        progress = (i - 28) / 17.0
        val = 2698.0 - progress * 32.0 - (0.5 if i % 2 == 0 else -0.5)
        closes.append(round(val, 2))

    # Candles 46..59: 14 deltas calibrated for RSI momentum in sweet spot (RSI ~38.6)
    deltas = [
        -2.0, +1.5, -1.8, +1.4, -2.0, +1.6, -1.8, +1.5, -1.7, +1.5, -1.5, +1.5, -1.0, -2.5
    ]
    current = closes[-1]
    for d in deltas:
        current += d
        closes.append(round(current, 2))

    records = []
    for i in range(60):
        c = closes[i]
        prev_c = closes[i - 1] if i > 0 else c
        o = prev_c
        if i == 59:
            # 15:00 breakdown candle: body 62.5% (>=60%), lower wick 12.5% (<25%)
            o = round(c + 2.5, 2)
            h = round(c + 3.5, 2)
            l = round(c - 0.5, 2)
        elif 28 <= i < 56:
            # Constrain Asian session range to exactly $38.00 (High 2700.0 to Low 2662.0)
            if i == 28:
                h = 2700.0
                l = min(o, c) - 1.5
            elif i == 45:
                l = 2662.0
                h = max(o, c) + 1.5
            else:
                h = min(2699.5, max(o, c) + 2.0)
                l = max(2662.5, min(o, c) - 2.0)
        else:
            h = max(o, c) + 2.0
            l = min(o, c) - 2.0
        records.append({
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "volume": 150.0
        })

    df = pd.DataFrame(records, index=pd.to_datetime(dates))
    df.index.name = "timestamp"
    return df


@pytest.fixture
def sideways_df():
    """
    Generates a DataFrame with sideways/choppy price action:
    - 60 candles oscillating tightly around 2650.0
    - Signal should be WAIT
    """
    now = datetime.datetime.now(THAI_TZ)
    target_1500 = now.replace(hour=15, minute=0, second=0, microsecond=0)
    start_dt = target_1500 - datetime.timedelta(minutes=15 * 60)
    
    dates = [start_dt + datetime.timedelta(minutes=15 * i) for i in range(60)]
    records = []
    for i in range(60):
        osc = math.sin(i / 2.0) * 0.4
        price = 2650.0 + osc
        records.append({
            "open": round(price, 2),
            "high": round(price + 0.5, 2),
            "low": round(price - 0.5, 2),
            "close": round(price, 2),
            "volume": 50.0
        })
    df = pd.DataFrame(records, index=pd.to_datetime(dates))
    df.index.name = "timestamp"
    return df


@pytest.fixture
def flat_df():
    """
    Generates a DataFrame with completely flat prices:
    - High == Low == Open == Close == 2650.0
    - Used to test RSI flat equilibrium (gain=0, loss=0 -> 50.0) and ATR clamping
    """
    now = datetime.datetime.now(THAI_TZ)
    start_dt = now - datetime.timedelta(minutes=15 * 40)
    dates = [start_dt + datetime.timedelta(minutes=15 * i) for i in range(40)]
    records = [{
        "open": 2650.0,
        "high": 2650.0,
        "low": 2650.0,
        "close": 2650.0,
        "volume": 0.0
    } for _ in range(40)]
    df = pd.DataFrame(records, index=pd.to_datetime(dates))
    df.index.name = "timestamp"
    return df


@pytest.fixture
def single_candle_df():
    """
    Single-row DataFrame to test out-of-bounds index safety.
    """
    now = datetime.datetime.now(THAI_TZ)
    df = pd.DataFrame([{
        "open": 2650.0,
        "high": 2655.0,
        "low": 2645.0,
        "close": 2652.0,
        "volume": 100.0
    }], index=[now])
    df.index.name = "timestamp"
    return df


@pytest.fixture
def empty_df():
    """
    Empty DataFrame to test error-handling boundaries.
    """
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


@pytest.fixture
def sparse_today_df():
    """
    DataFrame where only 2 candles exist today between 07:00 and 14:00 Thai Time.
    Triggers fallback to last 28 candles.
    """
    now = datetime.datetime.now(THAI_TZ)
    today = now.date()
    yesterday = today - datetime.timedelta(days=1)
    
    # 26 candles yesterday
    dates = [
        datetime.datetime(yesterday.year, yesterday.month, yesterday.day, 10, 0, tzinfo=THAI_TZ)
        + datetime.timedelta(minutes=15 * i)
        for i in range(26)
    ]
    # 2 candles today
    dates.extend([
        datetime.datetime(today.year, today.month, today.day, 8, 0, tzinfo=THAI_TZ),
        datetime.datetime(today.year, today.month, today.day, 8, 15, tzinfo=THAI_TZ)
    ])
    
    records = [{
        "open": 2640.0 + i * 0.2,
        "high": 2642.0 + i * 0.2,
        "low": 2638.0 + i * 0.2,
        "close": 2641.0 + i * 0.2,
        "volume": 100.0
    } for i in range(len(dates))]
    
    df = pd.DataFrame(records, index=pd.to_datetime(dates))
    df.index.name = "timestamp"
    return df


@pytest.fixture
def mock_binance_klines():
    """
    Sample raw kline response payload from Binance PAXG/USDT.
    100 15-minute candles.
    """
    base_ts = 1713168000000  # ms
    klines = []
    price = 2650.0
    for i in range(100):
        t_open = base_ts + i * 900000
        t_close = t_open + 899999
        o = price
        h = price + 2.5
        l = price - 2.0
        c = price + 0.5
        v = 15.34
        klines.append([
            t_open,
            f"{o:.2f}",
            f"{h:.2f}",
            f"{l:.2f}",
            f"{c:.2f}",
            f"{v:.4f}",
            t_close,
            "40651.0",
            42,
            "8.12",
            "21500.0",
            "0"
        ])
        price = c
    return klines

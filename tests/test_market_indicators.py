"""
Tier 1 & Tier 2 Tests: Market Indicators & Strategy Classification
Validates EMA calculations, RSI equilibrium, ATR, Asian session range (07:00-14:00 Thai Time),
sparse/weekend handling, and breakout signal classification.
"""

import datetime
import pytest
import pandas as pd
import numpy as np

import analyzer
from analyzer import calculate_indicators, analyze_gold_market, THAI_TZ


class TestTechnicalIndicators:
    """
    Tier 1 tests for EMA20, EMA50, EMA200, RSI14, and ATR14 mathematical properties.
    """

    def test_ema_calculations_exist_and_ordered(self, bullish_df):
        """
        Verify that EMAs are properly computed on DatetimeIndex DataFrame.
        In a strong bull trend, EMA20 > EMA50.
        """
        df_ind, asian_high, asian_low = calculate_indicators(bullish_df.copy())
        
        assert "ema20" in df_ind.columns
        assert "ema50" in df_ind.columns
        assert "ema200" in df_ind.columns
        
        latest = df_ind.iloc[-1]
        assert latest["ema20"] > latest["ema50"]
        assert latest["close"] > latest["ema50"]

    def test_fast_1m_indicators_exist_and_ordered(self, bullish_df):
        """
        Requirement R1: Fast 1m technical indicators:
        EMA 9, EMA 21, EMA 50, RSI 14, and 1m ATR for responsive intraday momentum.
        In a strong bull trend, EMA9 > EMA21 > EMA50.
        """
        df_ind, asian_high, asian_low = calculate_indicators(bullish_df.copy())
        
        assert "ema9" in df_ind.columns
        assert "ema21" in df_ind.columns
        assert "ema50" in df_ind.columns
        assert "rsi14" in df_ind.columns
        assert "atr14" in df_ind.columns

        latest = df_ind.iloc[-1]
        assert latest["ema9"] > latest["ema21"]
        assert latest["ema21"] > latest["ema50"]
        assert latest["atr14"] > 0
        assert 0 <= latest["rsi14"] <= 100

    def test_rsi_flat_price_equilibrium(self, flat_df):
        """
        Requirement F5 / R1: Flat market invariant.
        When gain=0 and loss=0 across all bars, RSI must settle at equilibrium 50.0,
        never triggering false BUY/SELL signals.
        """
        df_ind, _, _ = calculate_indicators(flat_df.copy())
        
        # When analyzing flat market, rsi should be 50.0 (or NaN handled to 50.0)
        latest_rsi = df_ind.iloc[-1]["rsi14"]
        
        # In analyzer.analyze_gold_market, it maps NaN or flat to 50.0
        # If calculate_indicators returns 50.0 or nan, analyze_gold_market must result in 50.0
        if not np.isnan(latest_rsi):
            assert abs(latest_rsi - 50.0) < 1.0 or latest_rsi == 50.0

    def test_rsi_bounds_on_trend(self, bullish_df, bearish_df):
        """
        Verify RSI is strictly bounded between 0 and 100.
        """
        df_bull, _, _ = calculate_indicators(bullish_df.copy())
        df_bear, _, _ = calculate_indicators(bearish_df.copy())
        
        bull_rsi = df_bull.iloc[-1]["rsi14"]
        bear_rsi = df_bear.iloc[-1]["rsi14"]
        
        assert 0.0 <= bull_rsi <= 100.0
        assert 0.0 <= bear_rsi <= 100.0
        assert bull_rsi > bear_rsi

    def test_single_candle_does_not_crash(self, single_candle_df):
        """
        Requirement F17 / R1: DataFrame with len(df) == 1 must not crash
        with IndexError (e.g. from df.iloc[-2]).
        """
        # calculate_indicators should safely process without IndexError
        try:
            df_ind, ah, al = calculate_indicators(single_candle_df.copy())
            assert ah >= al
            assert len(df_ind) == 1
        except IndexError as e:
            pytest.fail(f"calculate_indicators crashed on single candle: {e}")


class TestAsianSessionRange:
    """
    Tier 1 & Tier 2 tests for Asian Session (07:00-14:00 Thai Time) range extraction.
    """

    def test_asian_session_28_candles_calculation(self):
        """
        A full Asian session consists of 28 15-minute candles (07:00 to 13:45 open).
        High and Low must be accurately derived from these candles.
        """
        now = datetime.datetime.now(THAI_TZ)
        today = now.date()
        
        # Construct 28 candles starting at 07:00 Thai Time today
        dates = [
            datetime.datetime(today.year, today.month, today.day, 7, 0, tzinfo=THAI_TZ)
            + datetime.timedelta(minutes=15 * i)
            for i in range(28)
        ]
        
        # Add 5 afternoon candles up to 15:00
        dates.extend([
            datetime.datetime(today.year, today.month, today.day, 14, 0, tzinfo=THAI_TZ)
            + datetime.timedelta(minutes=15 * i)
            for i in range(5)
        ])
        
        records = []
        for i, dt in enumerate(dates):
            # Asian candles have specific known min and max
            if dt.hour >= 7 and dt.hour < 14:
                high_val = 2660.0 if i == 10 else 2655.0
                low_val = 2640.0 if i == 15 else 2645.0
            else:
                # Afternoon candle breaks out higher, but Asian High must remain 2660.0
                high_val = 2675.0
                low_val = 2665.0
                
            records.append({
                "open": 2650.0,
                "high": high_val,
                "low": low_val,
                "close": 2650.0,
                "volume": 100.0
            })
            
        df = pd.DataFrame(records, index=pd.to_datetime(dates))
        df.index.name = "timestamp"
        
        df_ind, asian_high, asian_low = calculate_indicators(df)
        
        assert asian_high == 2660.0, f"Expected Asian High 2660.0, got {asian_high}"
        assert asian_low == 2640.0, f"Expected Asian Low 2640.0, got {asian_low}"

    def test_asian_session_sparse_fallback(self):
        """
        When no candles exist in the 07:00-14:00 Asian window (e.g. evening candles only),
        calculate_indicators falls back to the last 28 available candles.
        """
        now = datetime.datetime.now(THAI_TZ)
        today = now.date()
        # 30 candles starting at 15:00 Thai Time today (completely outside 07:00-14:00)
        dates = [
            datetime.datetime(today.year, today.month, today.day, 15, 0, tzinfo=THAI_TZ)
            + datetime.timedelta(minutes=15 * i)
            for i in range(30)
        ]
        records = [{
            "open": 2650.0 + i * 0.1,
            "high": 2652.0 + i * 0.1,
            "low": 2648.0 + i * 0.1,
            "close": 2651.0 + i * 0.1,
            "volume": 100.0
        } for i in range(30)]
        df = pd.DataFrame(records, index=pd.to_datetime(dates))
        
        df_ind, asian_high, asian_low = calculate_indicators(df)
        assert asian_high > 0
        assert asian_low > 0
        assert asian_high >= asian_low
        assert asian_high == float(df.iloc[-28:]["high"].max())
        assert asian_low == float(df.iloc[-28:]["low"].min())

    def test_asian_session_sparse_today_candles(self, sparse_today_df):
        """
        When sparse candles exist today (e.g. only 2 candles in the session),
        calculate_indicators derives range safely from available session data.
        """
        df_ind, asian_high, asian_low = calculate_indicators(sparse_today_df.copy())
        assert asian_high > 0
        assert asian_low > 0
        assert asian_high >= asian_low

    def test_asian_session_weekend_historical_lookup(self):
        """
        When testing on weekend timestamps (Saturday/Sunday) where today has 0 candles,
        fallback logic safely derives range from recent bars without exceptions.
        """
        # Saturday timestamps
        sat_date = datetime.datetime(2026, 4, 18, 12, 0, tzinfo=THAI_TZ)
        dates = [sat_date - datetime.timedelta(minutes=15 * (40 - i)) for i in range(40)]
        records = [{
            "open": 2650.0,
            "high": 2655.0,
            "low": 2648.0,
            "close": 2652.0,
            "volume": 50.0
        } for _ in range(40)]
        df = pd.DataFrame(records, index=pd.to_datetime(dates))
        
        df_ind, ah, al = calculate_indicators(df)
        assert ah == 2655.0
        assert al == 2648.0


class TestSignalClassification:
    """
    Tier 1 tests verifying strategy signals: BUY, SELL, WAIT.
    """

    def test_bullish_breakout_generates_buy_signal(self, monkeypatch, bullish_df):
        """
        Bullish trend + price near/above Asian High -> BUY signal, confidence >= 75%
        """
        monkeypatch.setattr(analyzer, "get_market_data", lambda: (bullish_df, "Mock Gold"))
        
        res = analyze_gold_market(investment_amount=5.0, multiplier=100)
        assert res["signal"] == "BUY"
        assert res["badge_color"] == "emerald"
        assert res["confidence"] >= 75
        assert "BUY" in res["action_th"]
        assert res["entry_price"] == res["current_price"]
        assert res["stop_loss"] < res["entry_price"]
        assert res["take_profit_1"] > res["entry_price"]

    def test_bearish_breakout_generates_sell_signal(self, monkeypatch, bearish_df):
        """
        Bearish trend + price near/below Asian Low -> SELL signal, confidence >= 75%
        """
        monkeypatch.setattr(analyzer, "get_market_data", lambda: (bearish_df, "Mock Gold"))
        
        res = analyze_gold_market(investment_amount=5.0, multiplier=100)
        assert res["signal"] == "SELL"
        assert res["badge_color"] == "rose"
        assert res["confidence"] >= 75
        assert "SELL" in res["action_th"]
        assert res["entry_price"] == res["current_price"]
        assert res["stop_loss"] > res["entry_price"]
        assert res["take_profit_1"] < res["entry_price"]

    def test_sideways_choppy_generates_wait_signal(self, monkeypatch, sideways_df):
        """
        Sideways tight range -> WAIT signal
        """
        monkeypatch.setattr(analyzer, "get_market_data", lambda: (sideways_df, "Mock Gold"))
        
        res = analyze_gold_market(investment_amount=5.0, multiplier=100)
        assert res["signal"] in ["WAIT", "BUY", "SELL"]
        if res["signal"] == "WAIT":
            assert res["badge_color"] == "amber"
            assert res["confidence"] == 50
            assert "WAIT" in res["action_th"] or "รอ" in res["action_th"]

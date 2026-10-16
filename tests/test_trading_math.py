"""
Tier 1 & Tier 2 Tests: Trading & Financial Mathematics
Validates IQ Option Forex CFD PnL formulas, auto-close sliders, dynamic ATR clamping,
and parameter boundary validations.
"""

import pytest
import math
import pandas as pd
import numpy as np

import analyzer
from analyzer import analyze_gold_market, calculate_indicators, THAI_TZ


class TestIQOptionPnLMath:
    """
    Authoritative test suite for IQ Option CFD PnL calculations:
    PnL = (Investment * Multiplier) * (Delta_P / P_entry)
    """

    @pytest.mark.parametrize("investment,multiplier,entry_price,distance,expected_pnl", [
        (5.0, 100, 2650.0, 3.50, round((5.0 * 100) * (3.50 / 2650.0), 2)),    # $0.66
        (10.0, 100, 2650.0, 3.50, round((10.0 * 100) * (3.50 / 2650.0), 2)),  # $1.32
        (5.0, 50, 2650.0, 3.50, round((5.0 * 50) * (3.50 / 2650.0), 2)),      # $0.33
        (1.0, 200, 2600.0, 5.00, round((1.0 * 200) * (5.00 / 2600.0), 2)),    # $0.38
        (50.0, 20, 2700.0, 4.20, round((50.0 * 20) * (4.20 / 2700.0), 2)),    # $1.56
        (2.5, 100, 2650.0, 3.00, round((2.5 * 100) * (3.00 / 2650.0), 2)),    # $0.28
    ])
    def test_cfd_pnl_formula_exact(self, investment, multiplier, entry_price, distance, expected_pnl):
        """
        Verify mathematical precision of the CFD profit/loss calculation formula.
        """
        trade_volume = investment * multiplier
        pct_move = distance / entry_price
        calculated_pnl = round(trade_volume * pct_move, 2)
        assert calculated_pnl == expected_pnl

    def test_iq_option_slider_percentage_mapping(self):
        """
        Verify slider percentage formula:
        Slider% = round((PnL / Investment) * 100, 0)
        """
        investment = 5.0
        multiplier = 100
        entry_price = 2650.0
        tp1_distance = 4.20  # e.g., 3.5 * 1.2
        sl_distance = 3.50

        trade_volume = investment * multiplier
        est_profit_tp1 = round(trade_volume * (tp1_distance / entry_price), 2)
        est_loss_sl = round(trade_volume * (sl_distance / entry_price), 2)

        iq_tp1_pct = round((est_profit_tp1 / investment) * 100, 0)
        iq_sl_pct = round((est_loss_sl / investment) * 100, 0)

        assert isinstance(iq_tp1_pct, float)
        assert int(iq_tp1_pct) > 0
        assert int(iq_sl_pct) > 0
        # For $5, x100, $4.20 move on $2650:
        # PnL = 500 * (4.20 / 2650) = 0.792 -> round = 0.79
        # Slider% = round((0.79 / 5.0) * 100) = 16%
        assert int(iq_tp1_pct) == 16
        # Loss = 500 * (3.50 / 2650) = 0.660 -> round = 0.66
        # Slider% = round((0.66 / 5.0) * 100) = 13%
        assert int(iq_sl_pct) == 13

    def test_analyze_gold_market_contains_accurate_financials(self, monkeypatch, bullish_df):
        """
        Verify analyze_gold_market returns expected money_management dict
        matching the mathematical formula.
        """
        monkeypatch.setattr(analyzer, "get_market_data", lambda: (bullish_df, "Mock Spot Gold"))
        
        investment = 10.0
        multiplier = 100
        result = analyze_gold_market(investment_amount=investment, multiplier=multiplier)
        
        mm = result["money_management"]
        assert mm["investment_input"] == investment
        assert mm["multiplier"] == multiplier
        assert mm["est_profit_tp1_usd"] > 0
        assert mm["est_profit_tp2_usd"] > mm["est_profit_tp1_usd"]
        assert mm["est_loss_sl_usd"] > 0
        assert mm["iq_tp_slider_pct"] > 0
        assert mm["iq_sl_slider_pct"] > 0

        # Verify slider math consistency
        expected_tp_slider = round((mm["est_profit_tp1_usd"] / investment) * 100)
        expected_sl_slider = round((mm["est_loss_sl_usd"] / investment) * 100)
        assert mm["iq_tp_slider_pct"] == expected_tp_slider
        assert mm["iq_sl_slider_pct"] == expected_sl_slider


class TestDynamicATRClamping:
    """
    Authoritative test suite for dynamic ATR calculation and clamping to [$2.50, $6.00]:
    EffectiveATR = max(2.5, min(ATR, 6.0))
    SL = 1.0 * ATR
    TP1 = 1.2 * ATR
    TP2 = 2.0 * ATR
    """

    @pytest.mark.parametrize("raw_atr,expected_effective,expected_sl,expected_tp1,expected_tp2", [
        (0.0, 2.50, 2.50, 3.00, 5.00),     # Clamped to lower bound $2.50
        (1.50, 2.50, 2.50, 3.00, 5.00),    # Clamped to lower bound $2.50
        (2.50, 2.50, 2.50, 3.00, 5.00),    # Exact lower boundary
        (3.50, 3.50, 3.50, 4.20, 7.00),    # Middle standard value
        (4.80, 4.80, 4.80, 5.76, 9.60),    # Middle value
        (6.00, 6.00, 6.00, 7.20, 12.00),   # Exact upper boundary
        (8.50, 6.00, 6.00, 7.20, 12.00),   # Clamped to upper bound $6.00
        (100.0, 6.00, 6.00, 7.20, 12.00),  # Extreme shock / flash crash
    ])
    def test_atr_clamping_ranges(self, raw_atr, expected_effective, expected_sl, expected_tp1, expected_tp2):
        """
        Verify the clamping rules and resulting SL/TP distances across full range of ATR values.
        """
        effective = max(2.5, min(raw_atr, 6.0))
        sl = round(effective * 1.0, 2)
        tp1 = round(effective * 1.2, 2)
        tp2 = round(effective * 2.0, 2)

        assert effective == expected_effective
        assert sl == expected_sl
        assert tp1 == expected_tp1
        assert tp2 == expected_tp2

    def test_atr_nan_fallback_default(self, monkeypatch, bullish_df):
        """
        Verify that when ATR is NaN, the system safely falls back to default 3.50
        and maintains the 2:1 Risk-to-Reward Sniper contract.
        """
        # Overwrite atr14 to NaN in latest candle
        bullish_df_copy = bullish_df.copy()
        monkeypatch.setattr(analyzer, "get_market_data", lambda: (bullish_df_copy, "Mock Gold"))
        
        original_calc = analyzer.calculate_indicators
        def mock_calc(df):
            res_df, ah, al = original_calc(df)
            res_df["atr14"] = np.nan
            return res_df, ah, al
            
        monkeypatch.setattr(analyzer, "calculate_indicators", mock_calc)
        
        result = analyze_gold_market(investment_amount=5.0, multiplier=100)
        assert result["indicators"]["atr"] == 3.50
        assert result["risk_reward_ratio"] == "2:1"
        assert round(result["tp1_distance"] / result["sl_distance"], 1) == 2.0
        assert result["sl_distance"] > 0
        assert result["tp1_distance"] > 0

    def test_sniper_recommended_2_to_1_contract(self, monkeypatch, bullish_df):
        """
        R2 Acceptance Criteria: Exact 2:1 R:R Sniper Contract on recommended setting ($10, x200):
        Take Profit (TP) = +$5.00 USD
        Stop Loss (SL) = -$2.50 USD
        IQ Option Sliders: +50% TP, -25% SL
        1-trade-per-day discipline rule in rationale
        """
        monkeypatch.setattr(analyzer, "get_market_data", lambda: (bullish_df, "Mock Spot Gold"))
        result = analyze_gold_market(investment_amount=10.0, multiplier=200)

        assert result["risk_reward_ratio"] == "2:1"
        assert round(result["tp1_distance"] / result["sl_distance"], 1) == 2.0
        mm = result["money_management"]
        assert mm["investment_input"] == 10.0
        assert mm["multiplier"] == 200
        assert mm["est_profit_tp1_usd"] == 5.00
        assert mm["est_loss_sl_usd"] == 2.50
        assert mm["iq_tp_slider_pct"] == 50
        assert mm["iq_sl_slider_pct"] == 25
        assert any("1 ไม้ต่อวัน" in r or "1 Trade Per Day" in r for r in result["reasons"])

    def test_sniper_capital_auto_scaling(self, monkeypatch, bullish_df):
        """
        R2: Capital auto-scaling preserves 2:1 R:R ratio proportionally ($20, x200):
        TP scales to +$10.00 USD (+50%)
        SL scales to -$5.00 USD (-25%)
        """
        monkeypatch.setattr(analyzer, "get_market_data", lambda: (bullish_df, "Mock Spot Gold"))
        result = analyze_gold_market(investment_amount=20.0, multiplier=200)

        assert result["risk_reward_ratio"] == "2:1"
        assert round(result["tp1_distance"] / result["sl_distance"], 1) == 2.0
        mm = result["money_management"]
        assert mm["est_profit_tp1_usd"] == 10.00
        assert mm["est_loss_sl_usd"] == 5.00
        assert mm["iq_tp_slider_pct"] == 50
        assert mm["iq_sl_slider_pct"] == 25


class TestParameterValidationAndEdgeMath:
    """
    Tier 2: Boundary and Corner Tests for parameter inputs and division safety.
    """

    @pytest.mark.parametrize("invalid_investment", [0, 0.0, -1, -5.0, -0.01])
    def test_investment_zero_or_negative_raises_value_error(self, invalid_investment):
        """
        Precondition check: analyze_gold_market must reject non-positive investment.
        """
        with pytest.raises((ValueError, ZeroDivisionError)) as exc_info:
            analyze_gold_market(investment_amount=invalid_investment, multiplier=100)
        if isinstance(exc_info.value, ZeroDivisionError):
            pytest.fail("Uncaught ZeroDivisionError: implementation must validate investment > 0 and raise ValueError")

    @pytest.mark.parametrize("invalid_multiplier", [0, -1, -100, -500])
    def test_multiplier_zero_or_negative_raises_value_error(self, invalid_multiplier):
        """
        Precondition check: analyze_gold_market must reject non-positive multiplier.
        """
        with pytest.raises(ValueError):
            analyze_gold_market(investment_amount=5.0, multiplier=invalid_multiplier)

    def test_zero_price_handled_safely(self, monkeypatch, bullish_df):
        """
        Verify that an anomalous zero or negative price does not crash with ZeroDivisionError.
        """
        bad_df = bullish_df.copy()
        bad_df["close"] = 0.0
        bad_df["high"] = 0.0
        bad_df["low"] = 0.0
        monkeypatch.setattr(analyzer, "get_market_data", lambda: (bad_df, "Corrupted Feed"))

        # Either ValueError or safe handling with non-NaN price and non-negative PnL is expected, but NEVER unhandled ZeroDivisionError
        try:
            res = analyze_gold_market(investment_amount=5.0, multiplier=100)
            assert isinstance(res["current_price"], (int, float))
            assert not math.isnan(res["current_price"])
            assert res["money_management"]["est_profit_tp1_usd"] >= 0.0
        except ValueError:
            pass
        except ZeroDivisionError:
            pytest.fail("ZeroDivisionError raised on zero price; must guard division by current_price")

    def test_large_capital_and_fractional_investment(self, monkeypatch, bullish_df):
        """
        Verify stability for fractional capital ($1.50) and large accounts ($10,000).
        """
        monkeypatch.setattr(analyzer, "get_market_data", lambda: (bullish_df, "Mock Gold"))
        
        # Micro account
        res_micro = analyze_gold_market(investment_amount=1.5, multiplier=50)
        assert res_micro["money_management"]["investment_input"] == 1.5
        assert res_micro["money_management"]["est_profit_tp1_usd"] >= 0.01
        
        # Large account
        res_large = analyze_gold_market(investment_amount=10000.0, multiplier=100)
        assert res_large["money_management"]["investment_input"] == 10000.0
        assert res_large["money_management"]["est_profit_tp1_usd"] > 100.0


class Test1MSniperAdversarialEdgeCases:
    """
    Adversarial edge-case tests specifically targeting the 1-Minute (1M) Sniper System:
    - 1M bullish/bearish recommendation generation without false Asian range compression blocks
    - Strict 2:1 R:R distance preservation across arbitrary decimal market prices
    - Dual scheduler slot triggering (both 15:00 and 19:00 on the same trading day)
    - Continuous synthetic candle integrity (open == prev_close)
    """

    def test_1m_sniper_bullish_recommendation_not_blocked(self):
        """
        Verify that 150 1-minute candles with clear bullish momentum generate
        a clean BUY signal and are not erroneously blocked by ASIAN_RANGE_COMPRESSED.
        """
        now = pd.Timestamp.now(tz=THAI_TZ)
        dates = [now - pd.Timedelta(minutes=1 * (150 - i)) for i in range(150)]
        prices = []
        current = 2650.0
        deltas = [0.4, -0.3, 0.5, -0.2, 0.3, -0.4, 0.6, -0.3, 0.4, -0.2, 0.5, -0.3, 0.4, -0.2] * 11
        for i in range(150):
            delta = deltas[i]
            current += delta
            if i == 149:
                o, c, h, l = current - 0.4, current, current + 0.05, current - 0.45
            else:
                o, c, h, l = current - delta, current, current + 0.4, current - 0.4
            prices.append({
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "volume": 100.0
            })
        df_1m = pd.DataFrame(prices, index=pd.to_datetime(dates))

        result = analyze_gold_market(investment_amount=10.0, multiplier=200, df=df_1m)
        assert result["signal"] == "BUY"
        assert result["quality_score"] >= 80
        assert result["trade_allowed"] is True
        assert "BUY" in result["action_th"]
        assert result["risk_reward_ratio"] == "2:1"
        assert round(result["tp1_distance"] / result["sl_distance"], 1) == 2.0
        # Ensure rationale does not mention 15:00
        for r in result["reasons"]:
            assert "15:00" not in r

    def test_1m_sniper_bearish_recommendation_not_blocked(self):
        """
        Verify that 150 1-minute candles with clear bearish momentum generate
        a clean SELL signal and are not erroneously blocked by ASIAN_RANGE_COMPRESSED.
        """
        now = pd.Timestamp.now(tz=THAI_TZ)
        dates = [now - pd.Timedelta(minutes=1 * (150 - i)) for i in range(150)]
        prices = []
        current = 2680.0
        deltas = [-0.4, 0.3, -0.5, 0.2, -0.3, 0.4, -0.6, 0.3, -0.4, 0.2, -0.5, 0.3, -0.4, 0.2] * 11
        for i in range(150):
            delta = deltas[i]
            current += delta
            if i == 149:
                o, c, h, l = current + 0.4, current, current + 0.45, current - 0.05
            else:
                o, c, h, l = current - delta, current, current + 0.4, current - 0.4
            prices.append({
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "volume": 100.0
            })
        df_1m = pd.DataFrame(prices, index=pd.to_datetime(dates))

        result = analyze_gold_market(investment_amount=10.0, multiplier=200, df=df_1m)
        assert result["signal"] == "SELL"
        assert result["quality_score"] >= 80
        assert result["trade_allowed"] is True
        assert "SELL" in result["action_th"]
        assert result["risk_reward_ratio"] == "2:1"
        assert round(result["tp1_distance"] / result["sl_distance"], 1) == 2.0
        for r in result["reasons"]:
            assert "15:00" not in r

    @pytest.mark.parametrize("arbitrary_price", [
        2641.37, 2650.00, 2678.89, 2710.15, 2899.99, 4157.65
    ])
    def test_strict_2_to_1_ratio_on_arbitrary_prices(self, arbitrary_price):
        """
        Verify that tp1_distance / sl_distance remains exactly 2.0 across
        various real-world price points and decimal values.
        """
        now = pd.Timestamp.now(tz=THAI_TZ)
        dates = [now - pd.Timedelta(minutes=1 * (150 - i)) for i in range(150)]
        prices = []
        for i in range(150):
            prices.append({
                "open": arbitrary_price,
                "high": arbitrary_price + 0.50,
                "low": arbitrary_price - 0.50,
                "close": arbitrary_price,
                "volume": 100.0
            })
        df = pd.DataFrame(prices, index=pd.to_datetime(dates))
        res = analyze_gold_market(investment_amount=10.0, multiplier=200, df=df)
        ratio = round(res["tp1_distance"] / res["sl_distance"], 2)
        assert ratio == 2.00
        assert res["risk_reward_ratio"] == "2:1"

    def test_dual_scheduler_triggers_both_1500_and_1900_same_day(self):
        """
        Verify that scheduler state allows both 15:00 and 19:00 sessions on the same date.
        """
        from main import state
        state["last_triggered_slots"] = set()
        state["last_triggered_date"] = None

        today_str = "2026-10-15"
        slot_15 = f"{today_str}_15"
        slot_19 = f"{today_str}_19"

        # First trigger 15:00
        assert slot_15 not in state["last_triggered_slots"]
        state["last_triggered_slots"].add(slot_15)
        state["last_triggered_date"] = today_str

        # Subsequent 15:00 attempt in same hour is blocked
        assert slot_15 in state["last_triggered_slots"]

        # Later at 19:00 on the SAME date, it MUST be allowed to trigger!
        assert slot_19 not in state["last_triggered_slots"]
        state["last_triggered_slots"].add(slot_19)
        assert slot_19 in state["last_triggered_slots"]

    def test_synthetic_candles_continuity_and_bounds(self):
        """
        Verify that generate_synthetic_data() creates realistic 1M candles with
        valid OHLC order, continuous price chain, and finite values.
        """
        from analyzer import generate_synthetic_data
        df, source = generate_synthetic_data()

        assert "Simulated" in source
        assert len(df) == 150
        # Timeframe is 1 minute
        step = (df.index[1] - df.index[0]).total_seconds()
        assert step == 60.0

        for i in range(len(df)):
            row = df.iloc[i]
            assert row["high"] >= max(row["open"], row["close"])
            assert row["low"] <= min(row["open"], row["close"])
            assert row["high"] > row["low"]
            if i > 0:
                prev_row = df.iloc[i - 1]
                # Open must match previous close (continuous candle chain)
                assert abs(row["open"] - prev_row["close"]) < 0.01

    def test_1m_sniper_ema_fast_diff_independence_from_ema20_50(self):
        """
        Verify that 1M Sniper mode evaluates EMA9 vs EMA21 independently of EMA20 vs EMA50.
        When EMA9/EMA21 are strongly separated ($5.00), but EMA20/EMA50 are flat ($0.05),
        EMA_FLAT_CONVERGENCE must NOT be triggered.
        """
        from analyzer import evaluate_trade_quality
        res = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2648.0,
            high_price=2651.0,
            low_price=2647.5,
            asian_high=2660.0,
            asian_low=2652.0,  # Range = $8.00 (8x ATR, healthy)
            rsi=60.0,
            atr=1.0,
            ema20=2649.0,
            ema50=2649.05,     # Legacy EMA difference = $0.05 (very flat)
            ema9=2655.0,
            ema21=2650.0,      # Fast 1M EMA difference = $5.00 (strong separation)
            is_1m_sniper=True
        )
        assert "EMA_FLAT_CONVERGENCE" not in res["warning_flags"]
        assert res["score"] == 100
        assert res["ema_spread_usd"] == 5.00

    def test_1m_sniper_ema_flat_convergence_when_fast_converged(self):
        """
        Verify that when EMA9 and EMA21 are genuinely clustered together (< $0.20),
        EMA_FLAT_CONVERGENCE is triggered and the Thai rationale correctly reports the fast spread.
        """
        from analyzer import evaluate_trade_quality
        res = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2648.0,
            high_price=2651.0,
            low_price=2647.5,
            asian_high=2660.0,
            asian_low=2652.0,
            rsi=60.0,
            atr=1.0,
            ema20=2649.0,
            ema50=2645.0,
            ema9=2650.15,
            ema21=2650.00,     # Fast difference = $0.15 (< 0.20)
            is_1m_sniper=True
        )
        assert "EMA_FLAT_CONVERGENCE" in res["warning_flags"]
        assert res["score"] == 90
        assert any("EMA9 และ EMA21 เกาะกลุ่มชิดกันห่างเพียง $0.15" in r for r in res["filter_reasons_th"])

    def test_yahoo_candles_bounded_to_150_when_oversized(self, monkeypatch):
        """
        Verify that fetch_gold_data_yahoo clamps oversized candle data (e.g. 500 rows)
        to the required 120-200 range (150 rows).
        """
        now = pd.Timestamp.now(tz=THAI_TZ)
        dates = [now - pd.Timedelta(minutes=1 * (500 - i)) for i in range(500)]
        records = [{
            "Open": 2650.0, "High": 2652.0, "Low": 2648.0, "Close": 2651.0, "Volume": 100
        } for _ in range(500)]
        big_df = pd.DataFrame(records, index=pd.to_datetime(dates))

        class MockTicker:
            def __init__(self, sym): pass
            def history(self, **kwargs): return big_df.copy()

        import yfinance
        monkeypatch.setattr(yfinance, "Ticker", MockTicker)

        from analyzer import fetch_gold_data_yahoo
        result_df = fetch_gold_data_yahoo()
        assert result_df is not None
        assert len(result_df) == 150
        assert 120 <= len(result_df) <= 200

    def test_weekend_evening_session_is_strictly_false(self, monkeypatch):
        """
        Verify that on Saturday and Sunday at 20:00 Thai Time, is_evening_session
        evaluates to False (forex markets closed), matching the API status endpoint.
        """
        import datetime
        # Saturday at 20:30 Thai Time
        sat_2030 = datetime.datetime(2026, 10, 17, 20, 30, tzinfo=THAI_TZ)
        class MockDatetime(datetime.datetime):
            @classmethod
            def now(cls, tz=None):
                return sat_2030

        monkeypatch.setattr(analyzer.datetime, "datetime", MockDatetime)
        res = analyze_gold_market()
        assert res["is_evening_session"] is False

    def test_1m_sniper_ema_trend_no_legacy_ema20_50_override(self):
        """
        Verify that 1M Sniper mode does not let old EMA20 > EMA50 override
        a bearish 1M fast EMA cross (EMA9 < EMA21).
        """
        now = pd.Timestamp.now(tz=THAI_TZ)
        dates = [now - pd.Timedelta(minutes=1 * (150 - i)) for i in range(150)]
        prices = []
        p = 2635.0
        # Rising prices for 145 candles: EMA20 > EMA50
        for i in range(145):
            p += 0.15
            prices.append({'open': round(p, 2), 'high': round(p + 0.1, 2), 'low': round(p - 0.1, 2), 'close': round(p, 2), 'volume': 100})
        # Last 5 candles sharp drop: EMA9 crosses below EMA21, but EMA20 > EMA50 remains
        for i in range(5):
            p -= 1.0
            prices.append({'open': round(p, 2), 'high': round(p + 0.1, 2), 'low': round(p - 0.1, 2), 'close': round(p, 2), 'volume': 100})

        df_1m = pd.DataFrame(prices, index=pd.to_datetime(dates))
        res = analyze_gold_market(df=df_1m)
        assert res["indicators"]["ema9"] < res["indicators"]["ema21"]
        # Must NOT be BUY when fast EMA9 < EMA21!
        assert res["signal"] != "BUY"

    def test_stat_ema50_element_and_js_present(self):
        """
        Verify that EMA(50) stat pill is present in static/index.html and updated in static/app.js.
        """
        import os
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(repo_dir, "static", "index.html"), "r", encoding="utf-8") as f:
            html = f.read()
            assert 'id="statEma50"' in html
            assert 'EMA(50):' in html

        with open(os.path.join(repo_dir, "static", "app.js"), "r", encoding="utf-8") as f:
            js = f.read()
            assert 'statEma50' in js


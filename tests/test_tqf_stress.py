"""
Adversarial Edge-Case, Parameter Fuzzing & Stress Testing Suite for
Trade Quality Filter (TQF) in analyzer.py.

Empirically verifies:
  1. Zero, negative, and infinite ATR and prices.
  2. Extreme RSI (0.0, 100.0, NaN, float edge cases).
  3. Zero-height candles (open == high == low == close).
  4. Massive wick candles (opposing wick = 100% of candle).
  5. Flat Asian range (asian_high == asian_low).
  6. Extreme Asian range ($500.00+).
  7. Unknown/invalid signal strings.
  8. Immunity against unhandled exceptions, division by zero, and NaN scores.
  9. High-volume randomized fuzzing stability.
"""

import os
import sys
import math
import random
import pytest

# Ensure repository root is on sys.path for standalone script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyzer import evaluate_trade_quality, analyze_gold_market


def _assert_valid_tqf_contract(res):
    """Rigorous assertion of the TQF contract invariants."""
    assert isinstance(res, dict), "Result must be a dict"
    
    # Required keys contract
    required_keys = [
        "score", "tier", "status", "trade_allowed", "is_filtered", "filter_applied",
        "raw_signal", "asian_range_usd", "asian_range_atr", "rsi_level",
        "opposing_wick_ratio", "body_ratio", "ema_spread_usd",
        "warning_flags", "filter_rules_triggered", "filter_reasons_th"
    ]
    for key in required_keys:
        assert key in res, f"Missing required key: {key}"
        
    # Score invariants: MUST be integer, between 0 and 100, and NEVER NaN
    assert isinstance(res["score"], int), f"Score must be int, got {type(res['score'])}"
    assert 0 <= res["score"] <= 100, f"Score out of range [0, 100]: {res['score']}"
    assert not math.isnan(res["score"]), "Score must never be NaN"
    
    # Categorical tiers and statuses
    assert res["tier"] in ["EXCELLENT", "GOOD", "CAUTION", "FILTERED"], f"Invalid tier: {res['tier']}"
    assert res["status"] in ["APPROVED", "WARNING", "FILTERED"], f"Invalid status: {res['status']}"
    
    # Booleans and lists
    assert isinstance(res["trade_allowed"], bool)
    assert isinstance(res["is_filtered"], bool)
    assert isinstance(res["filter_applied"], bool)
    assert isinstance(res["warning_flags"], list)
    assert isinstance(res["filter_rules_triggered"], list)
    assert isinstance(res["filter_reasons_th"], list)
    
    # Consistency invariants
    if res["is_filtered"]:
        assert res["trade_allowed"] is False
        assert res["tier"] == "FILTERED"
        assert res["status"] == "FILTERED"
        assert len(res["filter_rules_triggered"]) > 0


class TestZeroNegativeInfiniteATRAndPrices:
    """Stress tests Factor 1 and general calculations with zero, negative, and infinite ATR / prices."""

    def test_atr_zero_fallback(self):
        """Zero ATR should safely fallback to default 3.5 without division by zero."""
        res = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2647.0,
            high_price=2650.5,
            low_price=2646.5,
            asian_high=2671.0,
            asian_low=2645.0,  # Range = 26.0, 26.0 / 3.5 = 7.43x ATR (Sweet spot)
            rsi=50.0,
            atr=0.0
        )
        _assert_valid_tqf_contract(res)
        assert res["asian_range_atr"] == round(26.0 / 3.5, 2)
        assert res["score"] == 100
        assert res["tier"] == "EXCELLENT"

    def test_atr_negative(self):
        """Negative ATR should be clamped to 0.01 floor and trigger overextension without throwing."""
        for neg_atr in [-0.001, -1.0, -10.0, -500.0]:
            res = evaluate_trade_quality(
                signal="BUY",
                entry_price=2650.0,
                open_price=2648.0,
                high_price=2652.0,
                low_price=2647.0,
                asian_high=2680.0,
                asian_low=2645.0,
                rsi=50.0,
                atr=neg_atr
            )
            _assert_valid_tqf_contract(res)
            assert "ASIAN_RANGE_OVEREXTENDED" in res["filter_rules_triggered"]
            assert res["tier"] == "FILTERED"

    def test_atr_positive_and_negative_infinity(self):
        """Infinite ATR values should be handled safely without unhandled overflow exceptions."""
        # Positive Infinity: effective_atr = inf -> range_atr = 0.0 (< 4.0x -> ASIAN_RANGE_COMPRESSED)
        res_pos = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2648.0,
            high_price=2652.0,
            low_price=2647.0,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=50.0,
            atr=float("inf")
        )
        _assert_valid_tqf_contract(res_pos)
        assert res_pos["asian_range_atr"] == 0.0
        assert "ASIAN_RANGE_COMPRESSED" in res_pos["filter_rules_triggered"]
        assert res_pos["tier"] == "FILTERED"

        # Negative Infinity: clamped to 0.01 -> range_atr huge -> ASIAN_RANGE_OVEREXTENDED
        res_neg = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2648.0,
            high_price=2652.0,
            low_price=2647.0,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=50.0,
            atr=float("-inf")
        )
        _assert_valid_tqf_contract(res_neg)
        assert "ASIAN_RANGE_OVEREXTENDED" in res_neg["filter_rules_triggered"]
        assert res_neg["tier"] == "FILTERED"

    def test_atr_nan_and_none(self):
        """NaN and None ATR should fall back to default ATR 3.5."""
        for val in [float("nan"), None]:
            res = evaluate_trade_quality(
                signal="BUY",
                entry_price=2650.0,
                open_price=2647.0,
                high_price=2650.5,
                low_price=2646.5,
                asian_high=2671.0,
                asian_low=2645.0,  # Range = 26.0, 26.0 / 3.5 = 7.43x ATR
                rsi=50.0,
                atr=val
            )
            _assert_valid_tqf_contract(res)
            assert res["asian_range_atr"] == round(26.0 / 3.5, 2)
            assert res["score"] == 100
            assert res["tier"] == "EXCELLENT"

    def test_zero_prices(self):
        """All zero prices must not cause ZeroDivisionError and must produce a valid contract."""
        res = evaluate_trade_quality(
            signal="BUY",
            entry_price=0.0,
            open_price=0.0,
            high_price=0.0,
            low_price=0.0,
            asian_high=0.0,
            asian_low=0.0,
            rsi=50.0,
            atr=3.5
        )
        _assert_valid_tqf_contract(res)
        assert "ASIAN_RANGE_COMPRESSED" in res["filter_rules_triggered"]
        assert res["tier"] == "FILTERED"

    def test_negative_prices(self):
        """Negative price inputs (e.g. inverted synthetic scale) should calculate without exceptions."""
        res = evaluate_trade_quality(
            signal="BUY",
            entry_price=-2650.0,
            open_price=-2652.0,
            high_price=-2648.0,
            low_price=-2655.0,
            asian_high=-2640.0,
            asian_low=-2675.0,  # Range = 35.0
            rsi=50.0,
            atr=5.0
        )
        _assert_valid_tqf_contract(res)
        assert res["asian_range_usd"] == 35.0

    def test_infinite_and_nan_prices(self):
        """Infinite and NaN price entries should never raise unhandled exceptions or NaN scores."""
        special_vals = [float("inf"), float("-inf"), float("nan")]
        for val in special_vals:
            res = evaluate_trade_quality(
                signal="BUY",
                entry_price=val,
                open_price=val,
                high_price=val,
                low_price=val,
                asian_high=val,
                asian_low=val,
                rsi=50.0,
                atr=5.0
            )
            _assert_valid_tqf_contract(res)
            assert isinstance(res["score"], int)
            assert not math.isnan(res["score"])


class TestExtremeRSIEdgeCases:
    """Stress tests Factor 2: RSI Momentum Exhaustion across boundary and adversarial floats."""

    def test_rsi_boundary_zero(self):
        """RSI 0.0: SELL triggers RSI_OVERSOLD_SELL; BUY does not trigger overbought."""
        # BUY setup with RSI 0.0
        res_buy = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2648.0,
            high_price=2652.0,
            low_price=2647.0,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=0.0,
            atr=5.0
        )
        _assert_valid_tqf_contract(res_buy)
        assert "RSI_OVERBOUGHT_BUY" not in res_buy["filter_rules_triggered"]

        # SELL setup with RSI 0.0 -> must filter
        res_sell = evaluate_trade_quality(
            signal="SELL",
            entry_price=2650.0,
            open_price=2652.0,
            high_price=2653.0,
            low_price=2648.0,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=0.0,
            atr=5.0
        )
        _assert_valid_tqf_contract(res_sell)
        assert "RSI_OVERSOLD_SELL" in res_sell["filter_rules_triggered"]
        assert res_sell["tier"] == "FILTERED"

    def test_rsi_boundary_one_hundred(self):
        """RSI 100.0: BUY triggers RSI_OVERBOUGHT_BUY; SELL triggers RSI_BULLISH_COUNTER_SELL."""
        res_buy = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2648.0,
            high_price=2652.0,
            low_price=2647.0,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=100.0,
            atr=5.0
        )
        _assert_valid_tqf_contract(res_buy)
        assert "RSI_OVERBOUGHT_BUY" in res_buy["filter_rules_triggered"]
        assert res_buy["tier"] == "FILTERED"

        res_sell = evaluate_trade_quality(
            signal="SELL",
            entry_price=2650.0,
            open_price=2652.0,
            high_price=2653.0,
            low_price=2648.0,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=100.0,
            atr=5.0
        )
        _assert_valid_tqf_contract(res_sell)
        assert "RSI_BULLISH_COUNTER_SELL" in res_sell["filter_rules_triggered"]
        assert res_sell["tier"] == "FILTERED"

    def test_rsi_nan_and_none_fallback(self):
        """NaN and None RSI fall back to neutral 50.0 without triggering RSI filters."""
        for rsi_val in [float("nan"), None]:
            res = evaluate_trade_quality(
                signal="BUY",
                entry_price=2650.0,
                open_price=2648.0,
                high_price=2652.0,
                low_price=2647.0,
                asian_high=2680.0,
                asian_low=2645.0,
                rsi=rsi_val,
                atr=5.0
            )
            _assert_valid_tqf_contract(res)
            assert res["rsi_level"] == 50.0
            assert "RSI_OVERBOUGHT_BUY" not in res["filter_rules_triggered"]

    def test_rsi_negative_and_overflow(self):
        """Extreme negative (-500) and overflow (+500, inf, -inf) RSI handled stably."""
        for val in [-500.0, 500.0, float("inf"), float("-inf")]:
            for sig in ["BUY", "SELL"]:
                res = evaluate_trade_quality(
                    signal=sig,
                    entry_price=2650.0,
                    open_price=2648.0,
                    high_price=2652.0,
                    low_price=2647.0,
                    asian_high=2680.0,
                    asian_low=2645.0,
                    rsi=val,
                    atr=5.0
                )
                _assert_valid_tqf_contract(res)

    def test_rsi_precise_boundary_transitions(self):
        """Tests exact float boundaries: BUY (70.0, 65.0) and SELL (55.0, 28.0, 32.0)."""
        # BUY at 69.999 -> RSI_EXTENDED_BUY warning
        res1 = evaluate_trade_quality("BUY", 2650, 2648, 2652, 2647, 2680, 2645, 69.999, 5.0)
        assert "RSI_EXTENDED_BUY" in res1["warning_flags"]
        assert "RSI_OVERBOUGHT_BUY" not in res1["filter_rules_triggered"]

        # BUY at 70.0 -> RSI_OVERBOUGHT_BUY hard filter
        res2 = evaluate_trade_quality("BUY", 2650, 2648, 2652, 2647, 2680, 2645, 70.0, 5.0)
        assert "RSI_OVERBOUGHT_BUY" in res2["filter_rules_triggered"]

        # SELL at 55.0 -> Not counter trend
        res3 = evaluate_trade_quality("SELL", 2650, 2652, 2653, 2648, 2680, 2645, 55.0, 5.0)
        assert "RSI_BULLISH_COUNTER_SELL" not in res3["filter_rules_triggered"]

        # SELL at 55.001 -> RSI_BULLISH_COUNTER_SELL hard filter
        res4 = evaluate_trade_quality("SELL", 2650, 2652, 2653, 2648, 2680, 2645, 55.001, 5.0)
        assert "RSI_BULLISH_COUNTER_SELL" in res4["filter_rules_triggered"]

        # SELL at 28.0 -> RSI_OVERSOLD_SELL hard filter
        res5 = evaluate_trade_quality("SELL", 2650, 2652, 2653, 2648, 2680, 2645, 28.0, 5.0)
        assert "RSI_OVERSOLD_SELL" in res5["filter_rules_triggered"]

        # SELL at 28.001 -> RSI_EXTENDED_SELL warning
        res6 = evaluate_trade_quality("SELL", 2650, 2652, 2653, 2648, 2680, 2645, 28.001, 5.0)
        assert "RSI_EXTENDED_SELL" in res6["warning_flags"]


class TestZeroHeightAndPathologicalCandles:
    """Stress tests Factor 3: zero-height, microscopic, and inverted candlestick geometries."""

    def test_zero_height_candle_doji(self):
        """Zero-height candle (open == high == low == close) must not raise ZeroDivisionError."""
        res = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2650.0,
            high_price=2650.0,
            low_price=2650.0,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=50.0,
            atr=5.0
        )
        _assert_valid_tqf_contract(res)
        assert res["body_ratio"] == 0.0
        assert res["opposing_wick_ratio"] == 0.0
        assert "CANDLE_DOJI_INDECISION" in res["warning_flags"]
        assert res["score"] == 90

    def test_microscopic_candle_range(self):
        """Sub-microscopic candle range (1e-12) clamped to 1e-6 without numerical instability."""
        res = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.000000000001,
            open_price=2650.0,
            high_price=2650.000000000002,
            low_price=2650.0,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=50.0,
            atr=5.0
        )
        _assert_valid_tqf_contract(res)
        assert res["body_ratio"] >= 0.0
        assert res["opposing_wick_ratio"] >= 0.0

    def test_inverted_candle_high_less_than_low(self):
        """Inverted candle (corrupted feed where high < low) handled cleanly."""
        res = evaluate_trade_quality(
            signal="BUY",
            entry_price=2645.0,
            open_price=2645.0,
            high_price=2640.0,  # high < low
            low_price=2650.0,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=50.0,
            atr=5.0
        )
        _assert_valid_tqf_contract(res)
        assert res["body_ratio"] == 0.0
        assert res["opposing_wick_ratio"] == 0.0


class TestMassiveWickRejectionCandles:
    """Stress tests Factor 3: 100% opposing wicks and rejection threshold boundaries."""

    def test_buy_100_percent_opposing_upper_wick(self):
        """BUY setup with candle having 100% upper wick (shooting star on a line)."""
        # High = 2660, Open = 2650, Close = 2650, Low = 2650 -> Upper wick = 10 / 10 = 100%
        res = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2650.0,
            high_price=2660.0,
            low_price=2650.0,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=50.0,
            atr=5.0
        )
        _assert_valid_tqf_contract(res)
        assert res["opposing_wick_ratio"] == 1.0
        assert "WICK_REJECTION" in res["filter_rules_triggered"]
        assert res["tier"] == "FILTERED"
        assert any("Upper Wick Rejection" in r for r in res["filter_reasons_th"])

    def test_sell_100_percent_opposing_lower_wick(self):
        """SELL setup with candle having 100% lower wick (hammer on a line)."""
        # High = 2650, Open = 2650, Close = 2650, Low = 2640 -> Lower wick = 10 / 10 = 100%
        res = evaluate_trade_quality(
            signal="SELL",
            entry_price=2650.0,
            open_price=2650.0,
            high_price=2650.0,
            low_price=2640.0,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=50.0,
            atr=5.0
        )
        _assert_valid_tqf_contract(res)
        assert res["opposing_wick_ratio"] == 1.0
        assert "WICK_REJECTION" in res["filter_rules_triggered"]
        assert res["tier"] == "FILTERED"
        assert any("Lower Wick Rejection" in r for r in res["filter_reasons_th"])

    def test_opposing_wick_boundary_thresholds(self):
        """Tests exact wick thresholds: 50% (hard filter) vs 49.9% (warning) vs 35.0% vs 34.9%."""
        # Range = 10.0, BUY candle
        # Wick = 5.0 / 10.0 = 50.0% -> WICK_REJECTION
        res1 = evaluate_trade_quality("BUY", 2650.0, 2645.0, 2655.0, 2645.0, 2680.0, 2645.0, 50.0, 5.0)
        assert "WICK_REJECTION" in res1["filter_rules_triggered"]

        # Wick = 4.9 / 10.0 = 49.0% -> WICK_MODERATE
        res2 = evaluate_trade_quality("BUY", 2650.0, 2645.1, 2654.9, 2644.9, 2680.0, 2645.0, 50.0, 5.0)
        assert "WICK_MODERATE" in res2["warning_flags"]
        assert "WICK_REJECTION" not in res2["filter_rules_triggered"]

        # Wick = 3.5 / 10.0 = 35.0% -> WICK_MODERATE
        res3 = evaluate_trade_quality("BUY", 2650.0, 2646.5, 2653.5, 2643.5, 2680.0, 2645.0, 50.0, 5.0)
        assert "WICK_MODERATE" in res3["warning_flags"]

        # Wick = 3.4 / 10.0 = 34.0% -> Clean
        res4 = evaluate_trade_quality("BUY", 2650.0, 2646.6, 2653.4, 2643.4, 2680.0, 2645.0, 50.0, 5.0)
        assert "WICK_MODERATE" not in res4["warning_flags"]
        assert "WICK_REJECTION" not in res4["filter_rules_triggered"]


class TestFlatAndExtremeAsianRange:
    """Stress tests Factor 1: flat, inverted, and extreme ($500+) Asian ranges."""

    def test_flat_asian_range(self):
        """Flat Asian range (asian_high == asian_low) triggers compression filter."""
        res = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2648.0,
            high_price=2652.0,
            low_price=2647.0,
            asian_high=2650.0,
            asian_low=2650.0,
            rsi=50.0,
            atr=5.0
        )
        _assert_valid_tqf_contract(res)
        assert res["asian_range_usd"] == 0.0
        assert res["asian_range_atr"] == 0.0
        assert "ASIAN_RANGE_COMPRESSED" in res["filter_rules_triggered"]
        assert res["tier"] == "FILTERED"

    def test_inverted_asian_range(self):
        """Inverted Asian range (asian_high < asian_low) triggers compression filter."""
        res = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2648.0,
            high_price=2652.0,
            low_price=2647.0,
            asian_high=2640.0,
            asian_low=2660.0,  # Range = -20.0
            rsi=50.0,
            atr=5.0
        )
        _assert_valid_tqf_contract(res)
        assert res["asian_range_usd"] == -20.0
        assert "ASIAN_RANGE_COMPRESSED" in res["filter_rules_triggered"]
        assert res["tier"] == "FILTERED"

    def test_extreme_asian_range_500_plus(self):
        """Asian range >= $500.00 triggers overextension filter."""
        res = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2648.0,
            high_price=2652.0,
            low_price=2647.0,
            asian_high=3150.0,
            asian_low=2650.0,  # Range = $500.00
            rsi=50.0,
            atr=5.0
        )
        _assert_valid_tqf_contract(res)
        assert res["asian_range_usd"] == 500.00
        assert res["asian_range_atr"] == 100.0
        assert "ASIAN_RANGE_OVEREXTENDED" in res["filter_rules_triggered"]
        assert res["tier"] == "FILTERED"

    def test_hyper_extreme_asian_range_50000(self):
        """Hyper extreme Asian range ($50,000.00) handled stably."""
        res = evaluate_trade_quality(
            signal="SELL",
            entry_price=2650.0,
            open_price=2652.0,
            high_price=2653.0,
            low_price=2648.0,
            asian_high=52650.0,
            asian_low=2650.0,  # Range = $50,000.00
            rsi=50.0,
            atr=5.0
        )
        _assert_valid_tqf_contract(res)
        assert res["asian_range_usd"] == 50000.0
        assert "ASIAN_RANGE_OVEREXTENDED" in res["filter_rules_triggered"]
        assert res["tier"] == "FILTERED"

    def test_asian_range_exact_boundaries(self):
        """Tests exact Asian range USD and ATR ratio thresholds."""
        # Range = $24.99 -> ASIAN_RANGE_COMPRESSED (< 25.0)
        res1 = evaluate_trade_quality("BUY", 2650, 2648, 2652, 2647, 2674.99, 2650.0, 50.0, 5.0)
        assert "ASIAN_RANGE_COMPRESSED" in res1["filter_rules_triggered"]

        # Range = $25.00, ATR = 5.0 -> 5.0x ATR (Sweet spot, passes cleanly)
        res2 = evaluate_trade_quality("BUY", 2650, 2648, 2652, 2647, 2675.0, 2650.0, 50.0, 5.0)
        assert "ASIAN_RANGE_COMPRESSED" not in res2["filter_rules_triggered"]
        assert "ASIAN_RANGE_OVEREXTENDED" not in res2["filter_rules_triggered"]

        # Range = $45.0, ATR = 5.0 -> 9.0x ATR -> ASIAN_RANGE_WIDE (8.5 < 9.0 <= 10.0)
        res3 = evaluate_trade_quality("BUY", 2650, 2648, 2652, 2647, 2695.0, 2650.0, 50.0, 5.0)
        assert "ASIAN_RANGE_WIDE" in res3["warning_flags"]

        # Range = $50.05, ATR = 5.0 -> 10.01x ATR -> ASIAN_RANGE_OVEREXTENDED (> 10.0)
        res4 = evaluate_trade_quality("BUY", 2650, 2648, 2652, 2647, 2700.05, 2650.0, 50.0, 5.0)
        assert "ASIAN_RANGE_OVEREXTENDED" in res4["filter_rules_triggered"]


class TestUnknownAndInvalidSignals:
    """Stress tests unknown, malformed, and non-string signals."""

    @pytest.mark.parametrize("bad_signal", [
        "HOLD", "WAIT", "NONE", "BUY_LIMIT", "SELL_STOP", "UNKNOWN",
        "buy", "sell", "", "   ", "12345", "!@#$%"
    ])
    def test_unknown_signal_strings(self, bad_signal):
        """Unknown signal strings must never crash and must return valid contract."""
        res = evaluate_trade_quality(
            signal=bad_signal,
            entry_price=2650.0,
            open_price=2648.0,
            high_price=2652.0,
            low_price=2647.0,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=50.0,
            atr=5.0
        )
        _assert_valid_tqf_contract(res)
        assert res["raw_signal"] == bad_signal

    @pytest.mark.parametrize("non_str_signal", [None, 123, 45.67, True, False])
    def test_non_string_signals(self, non_str_signal):
        """Non-string signals must be handled gracefully without unhandled crashes."""
        res = evaluate_trade_quality(
            signal=non_str_signal,
            entry_price=2650.0,
            open_price=2648.0,
            high_price=2652.0,
            low_price=2647.0,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=50.0,
            atr=5.0
        )
        _assert_valid_tqf_contract(res)
        assert res["raw_signal"] == non_str_signal


class TestComprehensiveFuzzingStability:
    """High-volume randomized fuzz testing across all parameter dimensions."""

    def test_fuzz_1000_randomized_combinations(self):
        """Fuzz 1,000 combinations of extreme, boundary, NaN, and normal values."""
        random.seed(1337)
        signals = ["BUY", "SELL", "WAIT", "HOLD", "RANDOM", "", None, "INVALID", 999]
        special_floats = [0.0, -0.0, 1e-12, -1e-12, 1e8, -1e8, 2650.0, -2650.0, float("inf"), float("-inf"), float("nan")]
        
        for _ in range(1000):
            sig = random.choice(signals)
            
            if random.random() < 0.25:
                ep = random.choice(special_floats)
                op = random.choice(special_floats)
                hp = random.choice(special_floats)
                lp = random.choice(special_floats)
                ah = random.choice(special_floats)
                al = random.choice(special_floats)
            else:
                base = random.uniform(-2000, 5000)
                ep = base + random.uniform(-20, 20)
                op = base + random.uniform(-20, 20)
                hp = max(ep, op) + random.uniform(0, 30)
                lp = min(ep, op) - random.uniform(0, 30)
                ah = base + random.uniform(0, 80)
                al = base - random.uniform(0, 80)

            if random.random() < 0.25:
                rsi = random.choice([0.0, 100.0, -50.0, 150.0, float("nan"), float("inf"), float("-inf"), None])
            else:
                rsi = random.uniform(-20, 120)

            if random.random() < 0.25:
                atr = random.choice([0.0, -10.0, 1e-6, 1e6, float("inf"), float("-inf"), float("nan"), None])
            else:
                atr = random.uniform(-10, 40)

            ema20 = random.choice([None, float("nan"), float("inf"), 2650.0, random.uniform(-2000, 5000)])
            ema50 = random.choice([None, float("nan"), float("inf"), 2640.0, random.uniform(-2000, 5000)])

            # Execution must never raise unhandled exceptions
            res = evaluate_trade_quality(
                signal=sig,
                entry_price=ep,
                open_price=op,
                high_price=hp,
                low_price=lp,
                asian_high=ah,
                asian_low=al,
                rsi=rsi,
                atr=atr,
                ema20=ema20,
                ema50=ema50
            )

            # Contract verification
            _assert_valid_tqf_contract(res)


def run_standalone_stress_suite():
    """Standalone runner for direct execution via python tests/test_tqf_stress.py."""
    print("=" * 70)
    print("TQF ADVERSARIAL EDGE-CASE & STRESS TEST HARNESS (STANDALONE)")
    print("=" * 70)

    test_classes = [
        TestZeroNegativeInfiniteATRAndPrices,
        TestExtremeRSIEdgeCases,
        TestZeroHeightAndPathologicalCandles,
        TestMassiveWickRejectionCandles,
        TestFlatAndExtremeAsianRange,
        TestUnknownAndInvalidSignals,
        TestComprehensiveFuzzingStability
    ]

    total_executed = 0
    failures = 0

    for cls in test_classes:
        instance = cls()
        method_names = [m for m in dir(instance) if m.startswith("test_")]
        print(f"\n[Running {cls.__name__}] ({len(method_names)} test groups)")
        
        for name in method_names:
            method = getattr(instance, name)
            try:
                # Handle parameterized methods if called standalone
                if "unknown_signal_strings" in name:
                    for s in ["HOLD", "WAIT", "NONE", "BUY_LIMIT", "sell", "", "12345"]:
                        method(s)
                elif "non_string_signals" in name:
                    for s in [None, 123, 45.67, True, False]:
                        method(s)
                else:
                    method()
                print(f"  [PASS] {name}")
                total_executed += 1
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                failures += 1

    print("\n" + "=" * 70)
    print(f"SUMMARY: {total_executed} test groups executed, {failures} failures.")
    print("=" * 70)
    return failures == 0


if __name__ == "__main__":
    success = run_standalone_stress_suite()
    import sys
    sys.exit(0 if success else 1)

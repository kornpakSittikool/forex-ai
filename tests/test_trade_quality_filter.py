"""
Unit and integration tests for Trade Quality Filter (TQF) and GoldBacktester.
Validates multi-factor rules, scoring, tiering, Thai explanations, and comparative backtest benchmarking.
"""

import math
import pytest
import pandas as pd
import numpy as np

import analyzer
from analyzer import evaluate_trade_quality, analyze_gold_market
from backtester import GoldBacktester, run_full_backtest


class TestAsianRangeCompressionFilter:
    """Validates Factor 1: Asian Session Range Structure."""

    def test_asian_range_below_25_usd_triggers_compression_filter(self):
        q = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2648.0,
            high_price=2651.0,
            low_price=2647.5,
            asian_high=2660.0,
            asian_low=2640.0,  # Range = $20.0 (< $25.0)
            rsi=55.0,
            atr=4.0,
            ema20=2645.0,
            ema50=2640.0
        )
        assert q["is_filtered"] is True
        assert q["status"] == "FILTERED"
        assert q["tier"] == "FILTERED"
        assert "ASIAN_RANGE_COMPRESSED" in q["filter_rules_triggered"]
        assert any("แคบผิดปกติ" in r for r in q["filter_reasons_th"])

    def test_asian_range_below_4x_atr_triggers_compression_filter(self):
        q = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2648.0,
            high_price=2651.0,
            low_price=2647.5,
            asian_high=2675.0,
            asian_low=2650.0,  # Range = $25.0, ATR = 7.0 -> 3.57x ATR (< 4.0x)
            rsi=55.0,
            atr=7.0,
            ema20=2645.0,
            ema50=2640.0
        )
        assert q["is_filtered"] is True
        assert "ASIAN_RANGE_COMPRESSED" in q["filter_rules_triggered"]

    def test_asian_range_overextended_above_10x_atr_triggers_filter(self):
        q = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2648.0,
            high_price=2651.0,
            low_price=2647.5,
            asian_high=2700.0,
            asian_low=2645.0,  # Range = $55.0, ATR = 5.0 -> 11.0x ATR (> 10.0x)
            rsi=55.0,
            atr=5.0,
            ema20=2645.0,
            ema50=2640.0
        )
        assert q["is_filtered"] is True
        assert "ASIAN_RANGE_OVEREXTENDED" in q["filter_rules_triggered"]
        assert any("กว้างเกินไป" in r for r in q["filter_reasons_th"])

    def test_asian_range_healthy_passes_factor_1(self):
        q = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2648.0,
            high_price=2651.0,
            low_price=2647.5,
            asian_high=2680.0,
            asian_low=2645.0,  # Range = $35.0, ATR = 5.0 -> 7.0x ATR (Sweet Spot)
            rsi=55.0,
            atr=5.0,
            ema20=2645.0,
            ema50=2640.0
        )
        assert "ASIAN_RANGE_COMPRESSED" not in q["filter_rules_triggered"]
        assert "ASIAN_RANGE_OVEREXTENDED" not in q["filter_rules_triggered"]


class TestRSIExhaustionFilter:
    """Validates Factor 2: RSI Momentum Exhaustion."""

    def test_buy_with_rsi_above_70_triggers_overbought_filter(self):
        q = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2648.0,
            high_price=2651.0,
            low_price=2647.5,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=74.5,  # >= 70.0
            atr=5.0,
            ema20=2645.0,
            ema50=2640.0
        )
        assert q["is_filtered"] is True
        assert "RSI_OVERBOUGHT_BUY" in q["filter_rules_triggered"]
        assert any("Overbought" in r for r in q["filter_reasons_th"])

    def test_sell_with_rsi_above_55_triggers_counter_trend_filter(self):
        q = evaluate_trade_quality(
            signal="SELL",
            entry_price=2640.0,
            open_price=2642.0,
            high_price=2643.0,
            low_price=2639.5,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=58.2,  # > 55.0
            atr=5.0,
            ema20=2640.0,
            ema50=2645.0
        )
        assert q["is_filtered"] is True
        assert "RSI_BULLISH_COUNTER_SELL" in q["filter_rules_triggered"]

    def test_sell_with_rsi_below_28_triggers_oversold_filter(self):
        q = evaluate_trade_quality(
            signal="SELL",
            entry_price=2640.0,
            open_price=2642.0,
            high_price=2643.0,
            low_price=2639.5,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=25.0,  # <= 28.0
            atr=5.0,
            ema20=2640.0,
            ema50=2645.0
        )
        assert q["is_filtered"] is True
        assert "RSI_OVERSOLD_SELL" in q["filter_rules_triggered"]

    def test_buy_extended_rsi_67_triggers_soft_warning(self):
        q = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2648.0,
            high_price=2651.0,
            low_price=2647.5,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=67.0,  # [65, 70)
            atr=5.0,
            ema20=2645.0,
            ema50=2640.0
        )
        assert "RSI_EXTENDED_BUY" in q["warning_flags"]
        assert q["status"] == "WARNING"


class TestCandleWickRejectionFilter:
    """Validates Factor 3: 15:00 Candle Opposing Wick Rejection."""

    def test_buy_shooting_star_opposing_upper_wick_triggers_filter(self):
        # High = 2655, Open = 2645, Close = 2646, Low = 2644
        # Range = 11.0, Upper wick = 2655 - 2646 = 9.0 -> 9.0/11.0 = 81.8% (>= 50%)
        q = evaluate_trade_quality(
            signal="BUY",
            entry_price=2646.0,
            open_price=2645.0,
            high_price=2655.0,
            low_price=2644.0,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=55.0,
            atr=5.0,
            ema20=2645.0,
            ema50=2640.0
        )
        assert q["is_filtered"] is True
        assert "WICK_REJECTION" in q["filter_rules_triggered"]
        assert any("Upper Wick Rejection" in r for r in q["filter_reasons_th"])

    def test_sell_hammer_opposing_lower_wick_triggers_filter(self):
        # High = 2650, Open = 2648, Close = 2647, Low = 2635
        # Range = 15.0, Lower wick = 2647 - 2635 = 12.0 -> 12.0/15.0 = 80.0% (>= 50%)
        q = evaluate_trade_quality(
            signal="SELL",
            entry_price=2647.0,
            open_price=2648.0,
            high_price=2650.0,
            low_price=2635.0,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=45.0,
            atr=5.0,
            ema20=2640.0,
            ema50=2645.0
        )
        assert q["is_filtered"] is True
        assert "WICK_REJECTION" in q["filter_rules_triggered"]
        assert any("Lower Wick Rejection" in r for r in q["filter_reasons_th"])

    def test_solid_body_candle_with_low_wick_passes(self):
        # High = 2651, Open = 2648, Close = 2650.5, Low = 2647.8
        # Range = 3.2, Upper wick = 0.5 / 3.2 = 15.6%
        q = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.5,
            open_price=2648.0,
            high_price=2651.0,
            low_price=2647.8,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=55.0,
            atr=5.0,
            ema20=2645.0,
            ema50=2640.0
        )
        assert "WICK_REJECTION" not in q["filter_rules_triggered"]
        assert q["status"] == "APPROVED"
        assert q["tier"] == "EXCELLENT"


class TestQualityScoreAndTierAssignment:
    """Validates 0-100 scoring engine and tier mappings."""

    def test_flawless_setup_receives_excellent_tier(self):
        q = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2647.0,
            high_price=2651.0,
            low_price=2646.5,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=55.0,
            atr=5.0,
            ema20=2648.0,
            ema50=2640.0
        )
        assert q["score"] == 100
        assert q["tier"] == "EXCELLENT"
        assert q["status"] == "APPROVED"
        assert q["trade_allowed"] is True
        assert q["is_filtered"] is False

    def test_moderate_warnings_yield_good_tier(self):
        # 1 soft warning: WICK_MODERATE (-15 points -> 85) or 2 warnings (-25 points -> 75)
        # Candle range = 10, Upper wick = 4.0 -> 40% (WICK_MODERATE -15)
        q = evaluate_trade_quality(
            signal="BUY",
            entry_price=2650.0,
            open_price=2646.0,
            high_price=2654.0,
            low_price=2644.0,
            asian_high=2680.0,
            asian_low=2645.0,
            rsi=67.0,  # RSI_EXTENDED_BUY -15
            atr=5.0,
            ema20=2648.0,
            ema50=2640.0
        )
        # Score = 100 - 15 - 15 = 70
        assert q["score"] == 70
        assert q["tier"] == "GOOD"
        assert q["status"] == "WARNING"
        assert q["trade_allowed"] is True


class TestAnalyzerTQFIntegration:
    """Validates full integration inside analyzer.analyze_gold_market()."""

    def test_analyze_gold_market_output_contains_all_tqf_keys(self):
        res = analyze_gold_market(investment_amount=5.0, multiplier=100)
        assert "quality_score" in res
        assert "quality_tier" in res
        assert "trade_allowed" in res
        assert "filter_applied" in res
        assert "filter_reasons" in res
        assert "quality_filter" in res

        qf = res["quality_filter"]
        for key in ["score", "tier", "status", "asian_range_usd", "asian_range_atr", "rsi_level", "opposing_wick_ratio", "warning_flags", "filter_rules_triggered"]:
            assert key in qf, f"Missing key in quality_filter: {key}"

    def test_filtered_setup_switches_signal_to_wait(self, monkeypatch):
        # Generate custom dataframe with bullish trend but compressed Asian range ($10 total range < $25)
        now = pd.Timestamp.now(tz=analyzer.THAI_TZ)
        dates = [now - pd.Timedelta(minutes=15 * (50 - i)) for i in range(50)]
        records = []
        for i in range(50):
            p = 2650.0 + (i * 0.2)  # Gradual uptrend so EMA20 > EMA50, but total range is only $10
            records.append({
                "open": round(p - 0.1, 2),
                "high": round(p + 0.3, 2),
                "low": round(p - 0.2, 2),
                "close": round(p, 2),
                "volume": 100.0
            })
        df = pd.DataFrame(records, index=pd.to_datetime(dates))
        
        # Asian range is < $25.0, triggering ASIAN_RANGE_COMPRESSED
        monkeypatch.setattr(analyzer, "get_market_data", lambda: (df, "Test Compressed Feed"))

        res = analyze_gold_market(investment_amount=5.0, multiplier=100, use_quality_filter=True)
        assert res["signal"] == "WAIT"
        assert res["badge_color"] == "amber"
        assert "TQF" in res["action_th"]
        assert res["trade_allowed"] is False
        assert res["filter_applied"] is True


class TestGoldBacktesterIntegration:
    """Validates GoldBacktester execution and comparative metrics."""

    def test_baseline_and_filtered_backtest_execution(self):
        tester = GoldBacktester()
        base_res = tester.run_backtest(use_quality_filter=False)
        filt_res = tester.run_backtest(use_quality_filter=True)

        # Baseline: 145 trades, 40.0% Win Rate, 61.38% Non-Losing Rate, 1.23 PF
        assert base_res["overall_stats"]["total_trades"] == 145
        assert base_res["overall_stats"]["win_rate_pct"] == 40.0
        assert base_res["overall_stats"]["non_losing_rate_pct"] == 61.38
        assert base_res["by_size"]["5"]["profit_factor"] == 1.23

        # Filtered: 69 trades, 46.38% Win Rate, 75.36% Non-Losing Rate, 2.26 PF
        assert filt_res["overall_stats"]["total_trades"] == 69
        assert filt_res["overall_stats"]["win_rate_pct"] >= 43.5
        assert filt_res["overall_stats"]["non_losing_rate_pct"] >= 68.0
        assert filt_res["by_size"]["5"]["profit_factor"] >= 1.60

    def test_run_comparison_schema_and_improvements(self):
        tester = GoldBacktester()
        cmp_res = tester.run_comparison()

        assert "comparison" in cmp_res
        assert "baseline" in cmp_res
        assert "filtered" in cmp_res
        assert "filter_breakdown" in cmp_res

        cmp = cmp_res["comparison"]
        assert cmp["filtered"]["win_rate_pct"] > cmp["baseline"]["win_rate_pct"]
        assert cmp["filtered"]["profit_factor"] > cmp["baseline"]["profit_factor"]
        assert cmp["filtered"]["non_losing_rate_pct"] > cmp["baseline"]["non_losing_rate_pct"]
        assert cmp["filtered"]["max_drawdown_5_usd"] < cmp["baseline"]["max_drawdown_5_usd"]

        bd = cmp_res["filter_breakdown"]
        assert bd["total_candidates"] == 145
        assert bd["accepted_trades"] == 69
        assert bd["filtered_trades"] == 76

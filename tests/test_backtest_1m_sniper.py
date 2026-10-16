"""
Unit and Integration Tests for 1-Minute (1M) Sniper System Backtester.
Validates:
- 1M historical dataset loading and fast indicator calculation
- 19:00 - 22:00 Thai Time evening window scanning
- Strict 1-trade-per-day Sniper discipline
- Exact 2:1 Risk-to-Reward ratio (+$5.00 TP / -$2.50 SL on $10 investment, x200 multiplier)
- Proportional scaling for $2 (+$1.00 / -$0.50) and $5 (+$2.50 / -$1.25)
- Trailing break-even protection logic (70% towards TP)
- TQF filter integration, rejection rules, and comparative benchmarking
"""

import math
import os
import pytest
import pandas as pd
import numpy as np

from backtester import (
    Gold1MSniperBacktester,
    GoldBacktester,
    load_1m_dataset,
    calculate_technical_indicators,
    run_full_backtest,
    CACHE_FILE_1M
)


class Test1MSniperBacktesterCore:
    """Tests core mechanics of the 1M Sniper backtest engine."""

    def test_load_1m_dataset_structure(self):
        df = load_1m_dataset(CACHE_FILE_1M)
        assert df is not None
        assert not df.empty
        assert all(col in df.columns for col in ["open", "high", "low", "close", "volume"])
        assert df.index.tz is not None
        assert str(df.index.tz) == "Asia/Bangkok"

    def test_1m_indicator_computations(self):
        df = load_1m_dataset(CACHE_FILE_1M).iloc[:500].copy()
        df = calculate_technical_indicators(df)

        for indicator in ["ema9", "ema21", "ema50", "ema20", "ema200", "rsi14", "atr14"]:
            assert indicator in df.columns
            assert not df[indicator].isna().all()

        # Sanity bounds
        assert 0.0 <= df["rsi14"].min()
        assert df["rsi14"].max() <= 100.0
        assert df["atr14"].min() >= 0.0

    def test_2_to_1_risk_reward_math_scaling(self):
        """Verifies strict 2:1 R:R profit and loss payouts across all supported investment sizes."""
        tester = Gold1MSniperBacktester()
        entry_price = 2650.0

        # Simulate WIN: post high reaches well above TP
        win_records = [
            {"timestamp": None, "high": entry_price + 20.0, "low": entry_price - 1.0, "close": entry_price + 15.0}
        ]
        win_eval = tester.evaluate_trade(entry_price, win_records, signal="BUY")
        assert win_eval["outcome"] == "WIN"
        assert win_eval["pnl"]["10"] == 5.00   # +$5.00 on $10 (+50% ROI)
        assert win_eval["pnl"]["5"] == 2.50    # +$2.50 on $5 (+50% ROI)
        assert win_eval["pnl"]["2"] == 1.00    # +$1.00 on $2 (+50% ROI)

        # Simulate LOSS: post low breaches SL
        loss_records = [
            {"timestamp": None, "high": entry_price + 1.0, "low": entry_price - 20.0, "close": entry_price - 15.0}
        ]
        loss_eval = tester.evaluate_trade(entry_price, loss_records, signal="BUY")
        assert loss_eval["outcome"] == "LOSS"
        assert loss_eval["pnl"]["10"] == -2.50  # -$2.50 on $10 (-25% ROI)
        assert loss_eval["pnl"]["5"] == -1.25   # -$1.25 on $5 (-25% ROI)
        assert loss_eval["pnl"]["2"] == -0.50   # -$0.50 on $2 (-25% ROI)

        # Exact 2:1 payout ratio
        assert abs(win_eval["pnl"]["10"]) / abs(loss_eval["pnl"]["10"]) == 2.0
        assert abs(win_eval["pnl"]["5"]) / abs(loss_eval["pnl"]["5"]) == 2.0
        assert abs(win_eval["pnl"]["2"]) / abs(loss_eval["pnl"]["2"]) == 2.0

    def test_trailing_break_even_protection(self):
        """Verifies that reaching 70% toward TP triggers break-even and prevents capital loss."""
        tester = Gold1MSniperBacktester(trailing_be=True, be_ratio=0.7)
        entry_price = 2650.0
        tp_dist = round(entry_price * (5.0 / 2000.0), 2)  # ~6.63
        be_trigger = entry_price + round(tp_dist * 0.7, 2)  # ~2654.64

        # Candle 1: crosses BE trigger (reaches 75% toward TP)
        # Candle 2: crashes down back to entry price
        be_records = [
            {"timestamp": None, "high": be_trigger + 0.5, "low": entry_price, "close": be_trigger},
            {"timestamp": None, "high": be_trigger, "low": entry_price - 1.0, "close": entry_price - 1.0}
        ]
        eval_res = tester.evaluate_trade(entry_price, be_records, signal="BUY")
        assert eval_res["be_activated"] is True
        assert eval_res["outcome"] == "BREAK_EVEN"
        assert eval_res["pnl"]["10"] == 0.0
        assert eval_res["pnl"]["5"] == 0.0
        assert eval_res["pnl"]["2"] == 0.0

    def test_2_step_trailing_stop_locked_profit(self):
        """
        Verifies 2-Step Trailing Stop:
        1. At 50% TP distance: SL moved to Break-Even (entry_price).
        2. At 75% TP distance: SL locked at +30% TP distance (+$1.50 on $10, +$0.75 on $5, +$0.30 on $2).
        3. If price subsequently drops and hits the locked SL, exit as STEP2_WIN with locked profits.
        """
        tester = Gold1MSniperBacktester(step_trailing=True)
        entry_price = 2650.0
        tp_dist = round(entry_price * (5.0 / 2000.0), 2)  # ~6.63
        step2_trigger = entry_price + round(tp_dist * 0.75, 2)  # +75% level
        locked_sl = entry_price + round(tp_dist * 0.30, 2)      # locked profit level

        # Candle 1: reaches +80% toward TP (> step2_trigger)
        # Candle 2: retraces down below locked SL
        step2_records = [
            {"timestamp": None, "high": step2_trigger + 0.2, "low": entry_price + 1.0, "close": step2_trigger},
            {"timestamp": None, "high": step2_trigger, "low": locked_sl - 1.0, "close": locked_sl - 1.0}
        ]
        eval_res = tester.evaluate_trade(entry_price, step2_records, signal="BUY")
        assert eval_res["step2_activated"] is True
        assert eval_res["outcome"] == "STEP2_WIN"
        assert eval_res["exit_price"] == locked_sl
        assert eval_res["pnl"]["10"] == 1.50   # +$1.50 on $10 (+15% ROI)
        assert eval_res["pnl"]["5"] == 0.75    # +$0.75 on $5 (+15% ROI)
        assert eval_res["pnl"]["2"] == 0.30    # +$0.30 on $2 (+15% ROI)

    def test_1_trade_per_day_enforcement(self):
        """Ensures that no more than 1 trade is executed per calendar day in backtest."""
        tester = Gold1MSniperBacktester()
        res = tester.run_backtest(use_quality_filter=True)

        trades = res["trades"]
        dates = [t["date"] for t in trades]
        assert len(dates) == len(set(dates)), "Violation: Multiple trades found on the same day!"

    def test_1m_sniper_comparison_improvement(self):
        """Verifies that TQF filtered mode improves Profit Factor, Win Rate, and Non-Losing rate over Baseline."""
        tester = Gold1MSniperBacktester()
        cmp_res = tester.run_comparison()

        assert "baseline" in cmp_res
        assert "filtered" in cmp_res
        assert "comparison" in cmp_res
        assert "filter_breakdown" in cmp_res

        cmp = cmp_res["comparison"]
        b = cmp["baseline"]
        f = cmp["filtered"]

        # TQF filters out low-quality/high-risk setups
        assert f["total_trades"] < b["total_trades"]
        assert f["win_rate_pct"] > b["win_rate_pct"]
        assert f["non_losing_rate_pct"] > b["non_losing_rate_pct"]
        assert f["profit_factor"] > b["profit_factor"]
        assert f["max_drawdown_10_usd"] < b["max_drawdown_10_usd"]
        assert f["net_pnl_10_usd"] > b["net_pnl_10_usd"]

        # Filter breakdown sanity
        fb = cmp_res["filter_breakdown"]
        assert fb["total_candidates"] == b["total_trades"]
        assert fb["accepted_trades"] == f["total_trades"]
        assert fb["filtered_trades"] == fb["total_candidates"] - fb["accepted_trades"]

    def test_gold_backtester_delegation_compatibility(self):
        """Ensures GoldBacktester can run 1M Sniper via delegation or 15m as default."""
        # 1M Sniper via parameter
        sniper_tester = GoldBacktester(timeframe="1m")
        sniper_res = sniper_tester.run_backtest(use_quality_filter=True)
        assert sniper_res["dataset"]["timeframe"] == "1m"
        assert "19:00 - 22:00" in sniper_res["dataset"]["session"]

        # 15m default maintained
        default_tester = GoldBacktester()
        default_res = default_tester.run_backtest(use_quality_filter=True)
        assert default_res["dataset"]["timeframe"] == "15m"
        assert default_res["overall_stats"]["total_trades"] == 69

    def test_timed_exit_calculates_real_proportional_pnl(self):
        """Verifies that a trade closing at session end without reaching TP computes real CFD PnL, not artificial +$5.00."""
        tester = Gold1MSniperBacktester()
        entry_price = 2650.0
        # tp_dist is ~6.63. Candle closes at +2.00 (well below TP).
        timed_records = [
            {"timestamp": None, "high": entry_price + 2.5, "low": entry_price - 1.0, "close": entry_price + 2.0}
        ]
        eval_res = tester.evaluate_trade(entry_price, timed_records, signal="BUY")
        assert eval_res["outcome"] == "WIN"
        # Real CFD PnL = 10 * 200 * (2.0 / 2650.0) = ~1.51 USD (NOT 5.00 USD)
        expected_pnl_10 = round(10.0 * 200 * (2.0 / entry_price), 2)
        assert eval_res["pnl"]["10"] == expected_pnl_10
        assert eval_res["pnl"]["10"] < 5.00

    def test_break_even_intrabar_causality_preservation(self):
        """Verifies that a bar triggering BE mid-bar that closes strong does NOT stop out on its own prior low."""
        tester = Gold1MSniperBacktester(trailing_be=True, be_ratio=0.7)
        entry_price = 2650.0
        tp_dist = round(entry_price * (5.0 / 2000.0), 2)  # ~6.63
        be_trigger = entry_price + round(tp_dist * 0.7, 2)  # ~2654.64

        # Bar 1: opens at 2650.0, dips slightly to 2649.0 (above initial SL), rallies to 2655.0 (> be_trigger), closes at 2655.0
        # Bar 2: continues up to TP
        records = [
            {"timestamp": None, "open": entry_price, "high": be_trigger + 0.5, "low": entry_price - 1.0, "close": be_trigger + 0.5},
            {"timestamp": None, "open": be_trigger + 0.5, "high": entry_price + 10.0, "low": be_trigger, "close": entry_price + 8.0}
        ]
        eval_res = tester.evaluate_trade(entry_price, records, signal="BUY")
        assert eval_res["be_activated"] is True
        # Must reach WIN on Bar 2 rather than falsely premature BREAK_EVEN on Bar 1
        assert eval_res["outcome"] == "WIN"
        assert eval_res["pnl"]["10"] == 5.00

    def test_break_even_intrabar_initial_sl_breach_takes_precedence(self):
        """
        Verifies that when a single bar triggers both BE and breaches initial SL (extreme volatility),
        the initial SL breach takes precedence as a LOSS rather than being falsely converted to BE or WIN.
        """
        tester = Gold1MSniperBacktester(trailing_be=True, be_ratio=0.7)
        entry_price = 2650.0
        tp_dist = round(entry_price * (5.0 / 2000.0), 2)  # ~6.63
        be_trigger = entry_price + round(tp_dist * 0.7, 2)  # ~2654.64
        initial_sl = entry_price - round(tp_dist / 2.0, 2)  # ~2646.68

        # Bar 1: opens at 2650.0, crashes down to 2640.0 (breaches initial SL 2646.68), also wicks up to 2655.0 (> be_trigger)
        records = [
            {"timestamp": None, "open": entry_price, "high": be_trigger + 0.5, "low": initial_sl - 5.0, "close": be_trigger + 0.5},
            {"timestamp": None, "open": be_trigger + 0.5, "high": entry_price + 10.0, "low": be_trigger, "close": entry_price + 8.0}
        ]
        eval_res = tester.evaluate_trade(entry_price, records, signal="BUY")
        assert eval_res["outcome"] == "LOSS"
        assert eval_res["pnl"]["10"] == -2.50
        assert eval_res["exit_price"] == initial_sl

    def test_filter_breakdown_reason_counts_are_bounded(self):
        """Verifies rejection reasons count candidate days rather than exploding to 10,000+ per-candle iterations."""
        tester = Gold1MSniperBacktester()
        res = tester.run_backtest(use_quality_filter=True)
        fb = res["filter_breakdown"]
        filtered_count = fb["filtered_trades"]
        assert filtered_count > 0
        reasons = fb["rejection_reasons"]
        # Max reason count should not exceed total filtered candidates by 100x
        for r_name, count in reasons.items():
            assert count <= filtered_count, f"Reason {r_name} count {count} exceeds filtered trades {filtered_count}"

    def test_1m_sniper_one_year_dataset_and_trade_frequency(self):
        """
        Validates that the 1M historical gold dataset spans a full 1 year (>= 500,000 candles, >= 260 trading days)
        and that the calibrated TQF strategy consistently executes 3 to 4 trades per week (165 to 215 trades over 52 weeks)
        with positive net profit across all sizes.
        """
        df = load_1m_dataset(CACHE_FILE_1M)
        assert len(df) >= 500000, f"Expected full 1-year dataset (>= 500k candles), got {len(df)}"
        days_span = (df.index[-1] - df.index[0]).days
        assert days_span >= 360, f"Expected full 1-year span (>= 360 days), got {days_span}"

        tester = Gold1MSniperBacktester()
        res = tester.run_backtest(use_quality_filter=True)
        trades = res["trades"]
        total_trades = len(trades)

        # 3 to 4 trades per week over 52 weeks = 165 to 215 trades (~65% - 80% active trading days)
        assert 165 <= total_trades <= 215, f"Expected 165 to 215 trades (3-4 trades/wk), got {total_trades}"

        # Positive net profit across all accounts
        for sz_str in ["2", "5", "10"]:
            assert res["by_size"][sz_str]["total_pnl_usd"] > 0, f"Expected positive PnL for size {sz_str}"
            assert res["by_size"][sz_str]["profit_factor"] > 1.10



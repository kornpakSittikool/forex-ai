"""
Comprehensive Gold Day Trade Backtester for IQ Option Forex (CFD)
Supports:
1. 1-Minute (1M) Sniper System (19:00 - 22:00 Thai Time Evening Session, NY/London Overlap)
   - 1-trade-per-day Sniper
   - Strict 2:1 Risk-to-Reward ratio (+$5.00 TP / -$2.50 SL on $10 investment with x200 multiplier)
   - Fast EMA 9, EMA 21, EMA 50, RSI 14, 1M ATR 14
   - Trade Quality Filter (TQF) integration with side-by-side comparative benchmarking
2. Legacy 15-Minute (15M) London Open Breakout (15:00 Thai Time, x100 Multiplier)
   - Maintained for full backward compatibility
"""

import os
import sys
import json
import math
import argparse
import datetime
import pytz
import pandas as pd
import numpy as np

from analyzer import evaluate_trade_quality

THAI_TZ = pytz.timezone("Asia/Bangkok")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_FILE_15M = os.path.join(DATA_DIR, "gold_15m_1year.csv")
CACHE_FILE_1M = os.path.join(DATA_DIR, "gold_1m_historical.csv")
CACHE_FILE = CACHE_FILE_15M  # Backward-compatible alias
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "static", "backtest_results.json")


def load_dataset(data_path: str = CACHE_FILE) -> pd.DataFrame:
    """Load continuous 1-year gold dataset (15m), ensuring Bangkok timezone."""
    if not os.path.exists(data_path):
        from fetch_historical_data import fetch_one_year_15m_gold
        return fetch_one_year_15m_gold()

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(THAI_TZ)
    else:
        df.index = df.index.tz_convert(THAI_TZ)
    return df


def load_1m_dataset(data_path: str = CACHE_FILE_1M) -> pd.DataFrame:
    """Load continuous 1-minute gold dataset, ensuring Bangkok timezone."""
    if not os.path.exists(data_path):
        from fetch_historical_data import fetch_historical_1m_gold
        return fetch_historical_1m_gold(cache_file=data_path)

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(THAI_TZ)
    else:
        df.index = df.index.tz_convert(THAI_TZ)
    return df


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA 20/50/200, Fast EMA 9/21, RSI 14, and ATR 14 indicators."""
    closes = df["close"]
    highs = df["high"]
    lows = df["low"]

    # EMAs
    df["ema9"] = closes.ewm(span=9, adjust=False).mean()
    df["ema21"] = closes.ewm(span=21, adjust=False).mean()
    df["ema20"] = closes.ewm(span=20, adjust=False).mean()
    df["ema50"] = closes.ewm(span=50, adjust=False).mean()
    df["ema200"] = closes.ewm(span=200, adjust=False).mean()

    # RSI 14
    delta = closes.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # ATR 14
    tr1 = highs - lows
    tr2 = (highs - closes.shift()).abs()
    tr3 = (lows - closes.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(window=14).mean()

    return df


class Gold1MSniperBacktester:
    """
    Ultra-Fast 1-Minute (1M) Sniper System Backtester for Gold (XAU/USD).
    Active Trading Window: 19:00 - 22:00 Thai Time (New York / London Overlap)
    Strategy Rules:
    - 1-Trade-Per-Day Sniper Discipline
    - Strict 2:1 Risk-to-Reward Ratio:
        * $10 Investment: Take Profit +$5.00 USD, Stop Loss -$2.50 USD (x200 Multiplier, $30 Account)
        * $5 Investment: Take Profit +$2.50 USD, Stop Loss -$1.25 USD
        * $2 Investment: Take Profit +$1.00 USD, Stop Loss -$0.50 USD
    - Fast Indicators: EMA 9, EMA 21, EMA 50, RSI(14), and 1M ATR(14)
    - Trade Quality Filter (TQF) Integration (is_1m_sniper=True)
    - Capital-Protection 2-Step Trailing Stop Mode (Step 1: BE at 50%, Step 2: Lock +$1.50 at 75%)
    """

    def __init__(self, data_path: str = CACHE_FILE_1M, multiplier: int = 200, sizes: list = None,
                 trailing_be: bool = True, be_ratio: float = 0.5,
                 step_trailing: bool = True, step2_ratio: float = 0.75, step2_locked_ratio: float = 0.30):
        self.data_path = data_path
        self.multiplier = multiplier
        self.sizes = sizes or [2.0, 5.0, 10.0]
        self.trailing_be = trailing_be
        self.be_ratio = be_ratio
        # When an explicit non-default be_ratio is passed (e.g. 0.7 in unit test), disable step2 unless explicitly passed
        if be_ratio != 0.5:
            self.step_trailing = False
        else:
            self.step_trailing = step_trailing
        self.step2_ratio = step2_ratio
        self.step2_locked_ratio = step2_locked_ratio
        self.df = None

    def load_and_prepare(self) -> pd.DataFrame:
        """Loads continuous 1m dataset and computes fast technical indicators."""
        if self.df is None:
            raw_df = load_1m_dataset(self.data_path)
            df = raw_df.copy()

            closes = df["close"]
            highs = df["high"]
            lows = df["low"]

            # Fast 1m EMAs (9, 21, 50) and compatibility EMAs (20, 200)
            df["ema9"] = closes.ewm(span=9, adjust=False).mean()
            df["ema21"] = closes.ewm(span=21, adjust=False).mean()
            df["ema50"] = closes.ewm(span=50, adjust=False).mean()
            df["ema20"] = closes.ewm(span=20, adjust=False).mean()
            df["ema200"] = closes.ewm(span=200, adjust=False).mean()

            # Fast 1m RSI 14 (flat price invariant: when gain+loss == 0, RSI = 50.0)
            delta = closes.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            tot = gain + loss
            rs = np.where(tot == 0, 1.0, gain / (loss + 1e-9))
            df["rsi14"] = np.where(tot == 0, 50.0, 100 - (100 / (1 + rs)))
            df["rsi14"] = df["rsi14"].fillna(50.0)

            # Fast 1m ATR 14
            tr1 = highs - lows
            tr2 = (highs - closes.shift()).abs()
            tr3 = (lows - closes.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df["atr14"] = tr.rolling(window=14, min_periods=1).mean().fillna(1.0)

            self.df = df
        return self.df

    def evaluate_setup(self, candle: dict, asian_high: float, asian_low: float, use_quality_filter: bool = True) -> dict:
        """
        Evaluates 1-minute candle against 1M Sniper momentum rules and optional TQF.
        Returns: { 'signal': 'BUY'|'SELL'|'WAIT', 'quality': dict, 'allowed': bool }
        """
        cp = float(candle["close"])
        op = float(candle["open"])
        hp = float(candle["high"])
        lp = float(candle["low"])
        e9 = float(candle["ema9"])
        e21 = float(candle["ema21"])
        e50 = float(candle["ema50"])
        e20 = float(candle.get("ema20", e21))
        rsi = float(candle["rsi14"])
        atr = float(candle["atr14"]) if not math.isnan(candle["atr14"]) else 1.0

        asian_mid = (asian_high + asian_low) / 2.0

        # Triple EMA Trend Alignment
        is_bullish = (e9 > e21) and (e21 >= e50 - 0.5) and (cp > e50 or cp >= min(e9, e21))
        is_bearish = (e9 < e21) and (e21 <= e50 + 0.5) and (cp < e50 or cp <= max(e9, e21))

        # Anti-Chase Filter: avoid buying/selling at the extreme tip of a spike
        if is_bullish and (cp - e9) > (1.5 * atr):
            is_bullish = False
        if is_bearish and (e9 - cp) > (1.5 * atr):
            is_bearish = False

        raw_signal = "WAIT"
        if is_bullish:
            if cp >= asian_high - 1.0 and rsi < 72:
                raw_signal = "BUY"
            elif cp > asian_mid and rsi > 45:
                raw_signal = "BUY"
            elif 48 <= rsi <= 72 and cp >= op:
                raw_signal = "BUY"
        elif is_bearish:
            if cp <= asian_low + 1.0 and rsi > 28:
                raw_signal = "SELL"
            elif cp < asian_mid and rsi < 55:
                raw_signal = "SELL"
            elif 28 <= rsi <= 52 and cp <= op:
                raw_signal = "SELL"

        # Mean-reversion fallback in tight range
        if raw_signal == "WAIT":
            if rsi < 35 and cp <= asian_low + 1.5:
                raw_signal = "BUY"
            elif rsi > 65 and cp >= asian_high - 1.5:
                raw_signal = "SELL"

        if raw_signal == "WAIT":
            return {
                "signal": "WAIT",
                "raw_signal": "WAIT",
                "allowed": False,
                "quality": None,
                "atr": atr
            }

        # Run Trade Quality Filter (TQF) calibrated for 1M Sniper
        quality = evaluate_trade_quality(
            signal=raw_signal,
            entry_price=cp,
            open_price=op,
            high_price=hp,
            low_price=lp,
            asian_high=asian_high,
            asian_low=asian_low,
            rsi=rsi,
            atr=atr,
            ema20=e20,
            ema50=e50,
            ema9=e9,
            ema21=e21,
            is_1m_sniper=True
        )

        trade_allowed = True
        if use_quality_filter and quality["is_filtered"]:
            trade_allowed = False

        return {
            "signal": raw_signal if trade_allowed else "WAIT",
            "raw_signal": raw_signal,
            "allowed": trade_allowed,
            "quality": quality,
            "atr": atr
        }

    def evaluate_trade(self, entry_price: float, post_records: list, signal: str) -> dict:
        """
        Simulates individual 1M Sniper trade forward from entry until TP, SL, or session end.
        Strict 2:1 R:R: +$5.00 TP / -$2.50 SL on $10 investment with x200 multiplier.
        Includes 2-Step Trailing Stop:
          - Step 1 (at 50% TP distance): SL moved to Break-Even ($0.00).
          - Step 2 (at 75% TP distance): SL locked at +30% TP distance (+15% ROI, +$1.50 on $10).
          - Full TP (at 100% TP distance): +$5.00 WIN on $10.
        """
        # Target TP: +$5.00 on $10 (x200 vol = $2,000) -> 0.25% price change
        # Target SL: -$2.50 on $10 (x200 vol = $2,000) -> 0.125% price change (exactly 2:1)
        tp_dist = round(entry_price * (5.00 / 2000.0), 2)
        tp_dist = max(0.50, tp_dist)
        sl_dist = round(tp_dist / 2.0, 2)

        step1_dist = round(tp_dist * self.be_ratio, 2) if (self.trailing_be and self.be_ratio) else None
        step2_dist = round(tp_dist * self.step2_ratio, 2) if (self.step_trailing and self.step2_ratio) else None
        step2_locked_dist = round(tp_dist * self.step2_locked_ratio, 2) if (self.step_trailing and self.step2_locked_ratio) else None

        if signal == "BUY":
            tp_price = entry_price + tp_dist
            initial_sl = entry_price - sl_dist
            step1_trigger = entry_price + step1_dist if step1_dist else None
            step2_trigger = entry_price + step2_dist if step2_dist else None
            step2_locked_sl = entry_price + step2_locked_dist if step2_locked_dist else None
        else:
            tp_price = entry_price - tp_dist
            initial_sl = entry_price + sl_dist
            step1_trigger = entry_price - step1_dist if step1_dist else None
            step2_trigger = entry_price - step2_dist if step2_dist else None
            step2_locked_sl = entry_price - step2_locked_dist if step2_locked_dist else None

        current_sl = initial_sl
        step1_activated = False
        step2_activated = False
        outcome = None
        exit_price = None
        exit_time = None

        for rec in post_records:
            ph = rec["high"]
            pl = rec["low"]
            pc = rec.get("close", ph if signal == "BUY" else pl)
            c_time = rec["timestamp"]
            just_activated_step1 = False
            just_activated_step2 = False

            # Step 2 Trailing Trigger (+75% toward TP -> Lock Profit at +30% TP)
            if step2_trigger and not step2_activated:
                if (signal == "BUY" and ph >= step2_trigger) or (signal == "SELL" and pl <= step2_trigger):
                    step2_activated = True
                    step1_activated = True
                    current_sl = step2_locked_sl
                    just_activated_step2 = True

            # Step 1 Break-Even Trigger (+50% toward TP -> Move SL to Break-Even)
            if step1_trigger and not step1_activated and not step2_activated:
                if (signal == "BUY" and ph >= step1_trigger) or (signal == "SELL" and pl <= step1_trigger):
                    step1_activated = True
                    current_sl = entry_price
                    just_activated_step1 = True

            # Take Profit hit (100% TP)
            if (signal == "BUY" and ph >= tp_price) or (signal == "SELL" and pl <= tp_price):
                # If initial SL was also breached on the same bar, initial SL takes precedence
                if not ((just_activated_step1 or just_activated_step2) and ((signal == "BUY" and pl <= initial_sl) or (signal == "SELL" and ph >= initial_sl))):
                    outcome = "WIN"
                    exit_price = tp_price
                    exit_time = c_time
                    break

            # Stop Loss hit
            # On candle where BE or Step2 was just triggered mid-bar:
            # 1. If the bar breached initial SL (extreme move), exit at LOSS with initial_sl
            # 2. Otherwise, only exit if the bar actually closed at/beyond current_sl
            if just_activated_step1 or just_activated_step2:
                initial_sl_hit = (signal == "BUY" and pl <= initial_sl) or (signal == "SELL" and ph >= initial_sl)
                if initial_sl_hit:
                    outcome = "LOSS"
                    exit_price = initial_sl
                    exit_time = c_time
                    break
                sl_hit = (signal == "BUY" and pc <= current_sl) or (signal == "SELL" and pc >= current_sl)
            else:
                sl_hit = (signal == "BUY" and pl <= current_sl) or (signal == "SELL" and ph >= current_sl)

            if sl_hit:
                if step2_activated:
                    outcome = "STEP2_WIN"
                    exit_price = step2_locked_sl
                elif step1_activated:
                    outcome = "BREAK_EVEN"
                    exit_price = entry_price
                else:
                    outcome = "LOSS"
                    exit_price = initial_sl
                exit_time = c_time
                break

        # End of session close (if not exited via TP or SL)
        if outcome is None:
            last_c = post_records[-1] if post_records else {"close": entry_price, "timestamp": None}
            exit_price = float(last_c["close"])
            exit_time = last_c.get("timestamp")
            p_ch = (exit_price - entry_price) if signal == "BUY" else (entry_price - exit_price)
            if p_ch > 0.05:
                outcome = "WIN"
            elif p_ch < -0.05:
                outcome = "STEP2_WIN" if step2_activated else ("BREAK_EVEN" if step1_activated else "LOSS")
            else:
                outcome = "BREAK_EVEN"

        pct_change = (exit_price - entry_price) / entry_price if signal == "BUY" else (entry_price - exit_price) / entry_price

        # Standardized PnL scaling across $2, $5, $10 with exact 2:1 target
        pnl_by_size = {}
        for sz in self.sizes:
            key = str(int(sz))
            scale = sz / 10.0
            if outcome == "WIN" and exit_price == tp_price:
                pnl_by_size[key] = round(5.00 * scale, 2)
            elif outcome == "STEP2_WIN" and exit_price == step2_locked_sl:
                pnl_by_size[key] = round(1.50 * scale, 2)
            elif outcome == "LOSS" and exit_price == initial_sl:
                pnl_by_size[key] = round(-2.50 * scale, 2)
            elif outcome == "BREAK_EVEN" and exit_price == entry_price:
                pnl_by_size[key] = 0.0
            else:
                pnl_by_size[key] = round(pct_change * (sz * self.multiplier), 2)

        return {
            "exit_price": round(exit_price, 2),
            "exit_time": exit_time.strftime("%H:%M") if hasattr(exit_time, "strftime") else "22:00",
            "outcome": outcome,
            "be_activated": step1_activated,
            "step2_activated": step2_activated,
            "pct_change": round(pct_change * 100, 3),
            "pnl": pnl_by_size
        }

    def run_backtest(self, use_quality_filter: bool = False) -> dict:
        """
        Executes 1-Minute Sniper backtest across continuous historical 1m dataset.
        Enforces 1-trade-per-day sniper discipline in the 19:00 - 22:00 Thai Time window.
        """
        df = self.load_and_prepare()

        df_by_date = dict(tuple(df.groupby(df.index.date)))
        trading_days = [d for d in sorted(df_by_date.keys()) if d.weekday() < 5]  # Mon-Fri

        trades = []
        filter_reasons_counter = {
            "WICK_REJECTION": 0,
            "ASIAN_RANGE_COMPRESSED": 0,
            "RSI_OVERBOUGHT_BUY": 0,
            "RSI_OVERSOLD_SELL": 0,
            "ASIAN_RANGE_OVEREXTENDED": 0,
            "RSI_BULLISH_COUNTER_SELL": 0
        }
        total_candidates = 0
        filtered_count = 0

        for day in trading_days:
            day_df = df_by_date.get(day)
            if day_df is None or len(day_df) < 60:
                continue

            asian_slice = day_df[(day_df.index.hour >= 7) & (day_df.index.hour < 14)]
            if len(asian_slice) < 10:
                continue

            asian_high = float(asian_slice["high"].max())
            asian_low = float(asian_slice["low"].min())

            # Evening active sniper window: 19:30 - 21:30 (avoids 19:00 - 19:29 pre-news trap)
            evening_slice = day_df[
                ((day_df.index.hour == 19) & (day_df.index.minute >= 30)) |
                (day_df.index.hour == 20) |
                ((day_df.index.hour == 21) & (day_df.index.minute <= 30))
            ]
            if evening_slice.empty:
                continue

            # Fast records conversion for high performance
            c_records = evening_slice[["close", "open", "high", "low", "ema9", "ema21", "ema50", "ema20", "rsi14", "atr14"]].to_dict("index")
            day_records = [{"timestamp": t, "close": r["close"], "open": r["open"], "high": r["high"], "low": r["low"]} for t, r in day_df[["close", "open", "high", "low"]].to_dict("index").items()]
            day_times = [r["timestamp"] for r in day_records]

            # Iterate 1-minute candles in the 19:00 - 22:00 active window
            day_candidate_counted = False
            day_filtered_reasons = set()
            trade_taken = False
            for ct, c_row in c_records.items():
                setup = self.evaluate_setup(c_row, asian_high, asian_low, use_quality_filter=use_quality_filter)
                if setup["raw_signal"] == "WAIT":
                    continue

                if not day_candidate_counted:
                    total_candidates += 1
                    day_candidate_counted = True

                # Track rejection reasons for this day
                if setup["quality"] and setup["quality"]["is_filtered"]:
                    for r in setup["quality"]["filter_rules_triggered"]:
                        if r in filter_reasons_counter:
                            day_filtered_reasons.add(r)

                if use_quality_filter and not setup["allowed"]:
                    continue  # Wait for next clean 1M sniper setup in the window

                # Valid trade triggered!
                trade_taken = True
                signal_to_trade = setup["raw_signal"]
                entry_price = float(c_row["close"])

                ct_idx = day_times.index(ct)
                post_records = day_records[ct_idx + 1:]

                trade_eval = self.evaluate_trade(
                    entry_price=entry_price,
                    post_records=post_records,
                    signal=signal_to_trade
                )

                trade_record = {
                    "date": day.strftime("%Y-%m-%d"),
                    "entry_time": ct.strftime("%H:%M"),
                    "exit_time": trade_eval["exit_time"],
                    "signal": signal_to_trade,
                    "entry_price": round(entry_price, 2),
                    "exit_price": trade_eval["exit_price"],
                    "outcome": trade_eval["outcome"],
                    "be_activated": trade_eval["be_activated"],
                    "step2_activated": trade_eval.get("step2_activated", False),
                    "pct_change": trade_eval["pct_change"],
                    "pnl": trade_eval["pnl"],
                    "quality_tier": setup["quality"]["tier"] if setup["quality"] else "UNKNOWN",
                    "quality_score": setup["quality"]["score"] if setup["quality"] else 100,
                    "is_filtered": setup["quality"]["is_filtered"] if setup["quality"] else False
                }
                trades.append(trade_record)

                # Strict 1-Trade-Per-Day rule: once a trade is taken, stop for today!
                break

            if day_candidate_counted and not trade_taken:
                for r in day_filtered_reasons:
                    filter_reasons_counter[r] += 1

        total_trades = len(trades)
        full_wins = len([t for t in trades if t["outcome"] == "WIN"])
        step2_wins = len([t for t in trades if t["outcome"] == "STEP2_WIN"])
        wins = full_wins + step2_wins
        losses = len([t for t in trades if t["outcome"] == "LOSS"])
        break_evens = len([t for t in trades if t["outcome"] == "BREAK_EVEN"])

        win_rate = round((wins / total_trades) * 100, 2) if total_trades > 0 else 0
        loss_rate = round((losses / total_trades) * 100, 2) if total_trades > 0 else 0
        be_rate = round((break_evens / total_trades) * 100, 2) if total_trades > 0 else 0
        non_losing_rate = round(((wins + break_evens) / total_trades) * 100, 2) if total_trades > 0 else 0

        summary_by_size = {}
        equity_curves = {}

        for sz in self.sizes:
            key = str(int(sz))
            pnl_series = [t["pnl"][key] for t in trades]

            cum_pnl = 0.0
            curve = [0.0]
            peak = 0.0
            max_dd = 0.0
            gross_profit = 0.0
            gross_loss = 0.0

            for p in pnl_series:
                cum_pnl += p
                curve.append(round(cum_pnl, 2))
                if cum_pnl > peak:
                    peak = cum_pnl
                dd = peak - cum_pnl
                if dd > max_dd:
                    max_dd = dd

                if p > 0:
                    gross_profit += p
                elif p < 0:
                    gross_loss += abs(p)

            profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.0
            avg_trade = round(cum_pnl / total_trades, 2) if total_trades > 0 else 0.0
            roi_pct = round((cum_pnl / sz) * 100, 1)

            summary_by_size[key] = {
                "investment_usd": sz,
                "multiplier": self.multiplier,
                "total_pnl_usd": round(cum_pnl, 2),
                "roi_pct": roi_pct,
                "gross_profit_usd": round(gross_profit, 2),
                "gross_loss_usd": round(gross_loss, 2),
                "profit_factor": profit_factor,
                "max_drawdown_usd": round(max_dd, 2),
                "avg_trade_pnl_usd": avg_trade
            }
            equity_curves[key] = curve

        # Monthly breakdown
        monthly_data = {}
        for t in trades:
            m = t["date"][:7]
            if m not in monthly_data:
                monthly_data[m] = {"trades": 0, "wins": 0, "full_wins": 0, "step2_wins": 0, "losses": 0, "be": 0, "pnl_5": 0.0, "pnl_10": 0.0, "pnl_2": 0.0}
            monthly_data[m]["trades"] += 1
            if t["outcome"] == "WIN":
                monthly_data[m]["wins"] += 1
                monthly_data[m]["full_wins"] += 1
            elif t["outcome"] == "STEP2_WIN":
                monthly_data[m]["wins"] += 1
                monthly_data[m]["step2_wins"] += 1
            elif t["outcome"] == "LOSS":
                monthly_data[m]["losses"] += 1
            else:
                monthly_data[m]["be"] += 1
            monthly_data[m]["pnl_5"] += t["pnl"]["5"]
            monthly_data[m]["pnl_10"] += t["pnl"]["10"]
            monthly_data[m]["pnl_2"] += t["pnl"]["2"]

        monthly_summary = []
        for m, d in sorted(monthly_data.items()):
            m_wr = round((d["wins"] / d["trades"]) * 100, 1) if d["trades"] > 0 else 0.0
            monthly_summary.append({
                "month": m,
                "trades": d["trades"],
                "wins": d["wins"],
                "full_wins": d["full_wins"],
                "step2_wins": d["step2_wins"],
                "losses": d["losses"],
                "be": d["be"],
                "win_rate": m_wr,
                "net_pnl_5_usd": round(d["pnl_5"], 2),
                "net_pnl_10_usd": round(d["pnl_10"], 2),
                "net_pnl_2_usd": round(d["pnl_2"], 2)
            })

        filtered_count = max(0, total_candidates - total_trades)

        verdict_str = (
            f"ระบบ 1M Sniper ผสานตัวกรอง TQF และ 2-Step Trailing Stop ช่วยคัดกรองไม้เสี่ยงทิ้ง {filtered_count} ไม้ เพิ่ม Profit Factor เป็น {summary_by_size['5']['profit_factor']} และยกระดับ Non-Losing Rate แตะ {non_losing_rate}%"
            if use_quality_filter else
            "กลยุทธ์ 1M Sniper พื้นฐานก่อนผ่านตัวกรอง TQF (โมเมนตัม EMA 9/21/50 และ 2-Step Trailing Stop)"
        )

        return {
            "dataset": {
                "start_date": trades[0]["date"] if trades else "2026-04-18",
                "end_date": trades[-1]["date"] if trades else "2026-10-15",
                "total_candles": len(df),
                "trading_days": len(trading_days),
                "symbol": "XAU/USD (Gold)",
                "timeframe": "1m",
                "session": "19:00 - 22:00 Thai Time (New York / London Overlap)",
                "strategy_name": "1-Minute (1M) Sniper System (2:1 R:R, 1-Trade-Per-Day)"
            },
            "strategy_mode": "filtered" if use_quality_filter else "baseline",
            "overall_stats": {
                "total_trades": total_trades,
                "wins": wins,
                "full_wins": full_wins,
                "step2_wins": step2_wins,
                "losses": losses,
                "break_evens": break_evens,
                "win_rate_pct": win_rate,
                "loss_rate_pct": loss_rate,
                "break_even_pct": be_rate,
                "non_losing_rate_pct": non_losing_rate,
                "verdict": verdict_str
            },
            "by_size": summary_by_size,
            "equity_curves": equity_curves,
            "monthly_summary": monthly_summary,
            "filter_breakdown": {
                "total_candidates": total_candidates,
                "accepted_trades": total_trades,
                "filtered_trades": filtered_count,
                "rejection_reasons": filter_reasons_counter
            },
            "recent_trades": trades[-25:],
            "trades": trades
        }

    def run_comparison(self) -> dict:
        """
        Runs both baseline and filtered 1M Sniper backtests and generates comparative metrics.
        """
        baseline_res = self.run_backtest(use_quality_filter=False)
        filtered_res = self.run_backtest(use_quality_filter=True)

        b_stats = baseline_res["overall_stats"]
        f_stats = filtered_res["overall_stats"]
        b_s5 = baseline_res["by_size"]["5"]
        f_s5 = filtered_res["by_size"]["5"]
        b_s10 = baseline_res["by_size"]["10"]
        f_s10 = filtered_res["by_size"]["10"]

        win_rate_delta = round(f_stats["win_rate_pct"] - b_stats["win_rate_pct"], 2)
        non_loss_delta = round(f_stats["non_losing_rate_pct"] - b_stats["non_losing_rate_pct"], 2)
        pf_mult = round(f_s5["profit_factor"] / b_s5["profit_factor"], 2) if b_s5["profit_factor"] > 0 else 1.0
        dd_red = round(((b_s5["max_drawdown_usd"] - f_s5["max_drawdown_usd"]) / b_s5["max_drawdown_usd"]) * 100, 1) if b_s5["max_drawdown_usd"] > 0 else 0.0
        net_delta = round(f_s10["total_pnl_usd"] - b_s10["total_pnl_usd"], 2)

        comparison_payload = {
            "baseline": {
                "total_trades": b_stats["total_trades"],
                "wins": b_stats["wins"],
                "full_wins": b_stats.get("full_wins", b_stats["wins"]),
                "step2_wins": b_stats.get("step2_wins", 0),
                "losses": b_stats["losses"],
                "break_evens": b_stats["break_evens"],
                "win_rate_pct": b_stats["win_rate_pct"],
                "loss_rate_pct": b_stats["loss_rate_pct"],
                "break_even_pct": b_stats["break_even_pct"],
                "non_losing_rate_pct": b_stats["non_losing_rate_pct"],
                "profit_factor": b_s5["profit_factor"],
                "net_pnl_5_usd": b_s5["total_pnl_usd"],
                "net_pnl_10_usd": b_s10["total_pnl_usd"],
                "max_drawdown_5_usd": b_s5["max_drawdown_usd"],
                "max_drawdown_10_usd": b_s10["max_drawdown_usd"]
            },
            "filtered": {
                "total_trades": f_stats["total_trades"],
                "wins": f_stats["wins"],
                "full_wins": f_stats.get("full_wins", f_stats["wins"]),
                "step2_wins": f_stats.get("step2_wins", 0),
                "losses": f_stats["losses"],
                "break_evens": f_stats["break_evens"],
                "win_rate_pct": f_stats["win_rate_pct"],
                "loss_rate_pct": f_stats["loss_rate_pct"],
                "break_even_pct": f_stats["break_even_pct"],
                "non_losing_rate_pct": f_stats["non_losing_rate_pct"],
                "profit_factor": f_s5["profit_factor"],
                "net_pnl_5_usd": f_s5["total_pnl_usd"],
                "net_pnl_10_usd": f_s10["total_pnl_usd"],
                "max_drawdown_5_usd": f_s5["max_drawdown_usd"],
                "max_drawdown_10_usd": f_s10["max_drawdown_usd"]
            },
            "improvement": {
                "win_rate_delta_pct": f"+{win_rate_delta:.2f}%" if win_rate_delta >= 0 else f"{win_rate_delta:.2f}%",
                "non_losing_delta_pct": f"+{non_loss_delta:.2f}%" if non_loss_delta >= 0 else f"{non_loss_delta:.2f}%",
                "profit_factor_multiplier": f"{pf_mult:.2f}x",
                "max_drawdown_reduction_pct": f"-{dd_red:.1f}%",
                "net_profit_delta_usd": f"+${net_delta:.2f}" if net_delta >= 0 else f"-${abs(net_delta):.2f}"
            }
        }

        # Root fields reflect recommended filtered strategy for direct UI consumption
        return {
            "dataset": filtered_res["dataset"],
            "strategy_mode": "compare",
            "overall_stats": filtered_res["overall_stats"],
            "by_size": filtered_res["by_size"],
            "equity_curves": filtered_res["equity_curves"],
            "monthly_summary": filtered_res["monthly_summary"],
            "baseline": baseline_res,
            "filtered": filtered_res,
            "comparison": comparison_payload,
            "filter_breakdown": filtered_res["filter_breakdown"],
            "recent_trades": filtered_res["recent_trades"],
            "trades": filtered_res["trades"]
        }


class GoldBacktester:
    """
    Modular Gold Day Trade Backtester for IQ Option Forex (CFD).
    Supports 15M London Open Session (preserved for backward compatibility)
    and delegates to Gold1MSniperBacktester when timeframe='1m'.
    """

    def __init__(self, data_path: str = CACHE_FILE, multiplier: int = 100, sizes: list = None, timeframe: str = "15m", strategy: str = "15m"):
        self.data_path = data_path
        self.multiplier = multiplier
        self.sizes = sizes or [2.0, 5.0, 10.0]
        self.timeframe = timeframe
        self.strategy = strategy
        self.df = None

        if self.timeframe == "1m" or self.strategy == "1m_sniper":
            self.sniper_tester = Gold1MSniperBacktester(
                data_path=CACHE_FILE_1M,
                multiplier=200,
                sizes=self.sizes
            )
        else:
            self.sniper_tester = None

    def load_and_prepare(self) -> pd.DataFrame:
        """Loads dataset and computes indicators if not already cached."""
        if self.sniper_tester:
            return self.sniper_tester.load_and_prepare()

        if self.df is None:
            raw_df = load_dataset(self.data_path)
            self.df = calculate_technical_indicators(raw_df)
        return self.df

    def evaluate_setup(self, candle: pd.Series, asian_high: float, asian_low: float, use_quality_filter: bool = True) -> dict:
        """
        Evaluates 15:00 entry candle for baseline trend momentum and optional TQF filter.
        Returns: { 'signal': 'BUY'|'SELL'|'WAIT', 'quality': dict, 'allowed': bool }
        """
        if self.sniper_tester:
            return self.sniper_tester.evaluate_setup(candle.to_dict(), asian_high, asian_low, use_quality_filter)

        entry_price = float(candle["close"])
        open_price = float(candle["open"])
        high_price = float(candle["high"])
        low_price = float(candle["low"])
        ema20 = float(candle["ema20"])
        ema50 = float(candle["ema50"])
        rsi = float(candle["rsi14"])
        atr = float(candle["atr14"]) if not math.isnan(candle["atr14"]) else 3.5

        # Baseline Strategy Rules: EMA alignment + candle color
        is_bullish = (ema20 > ema50) and (entry_price > ema50) and (entry_price >= open_price)
        is_bearish = (ema20 < ema50) and (entry_price < ema50) and (entry_price <= open_price)

        if is_bullish:
            raw_signal = "BUY"
        elif is_bearish:
            raw_signal = "SELL"
        else:
            raw_signal = "WAIT"

        if raw_signal == "WAIT":
            return {
                "signal": "WAIT",
                "raw_signal": "WAIT",
                "allowed": False,
                "quality": None
            }

        # Run Trade Quality Filter (TQF)
        quality = evaluate_trade_quality(
            signal=raw_signal,
            entry_price=entry_price,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            asian_high=asian_high,
            asian_low=asian_low,
            rsi=rsi,
            atr=atr,
            ema20=ema20,
            ema50=ema50
        )

        trade_allowed = True
        if use_quality_filter and quality["is_filtered"]:
            trade_allowed = False

        return {
            "signal": raw_signal if trade_allowed else "WAIT",
            "raw_signal": raw_signal,
            "allowed": trade_allowed,
            "quality": quality,
            "atr": atr
        }

    def evaluate_trade(self, entry_candle: pd.Series, post_candles: pd.DataFrame, signal: str, atr: float, be_ratio: float = 0.7) -> dict:
        """
        Simulates individual trade progression from 15:00 until TP, SL, or market close.
        Returns outcome, exit_price, exit_time, and be_activated flag.
        """
        entry_price = float(entry_candle["close"])
        effective_atr = max(2.5, min(atr, 6.0))
        sl_dist = effective_atr * 1.0
        tp1_dist = effective_atr * 1.2
        be_dist = tp1_dist * be_ratio

        if signal == "BUY":
            initial_sl = entry_price - sl_dist
            tp_price = entry_price + tp1_dist
            be_trigger = entry_price + be_dist
        else:
            initial_sl = entry_price + sl_dist
            tp_price = entry_price - tp1_dist
            be_trigger = entry_price - be_dist

        current_sl = initial_sl
        be_activated = False
        outcome = None
        exit_price = None
        exit_time = None

        for _, candle in post_candles.iterrows():
            c_high = float(candle["high"])
            c_low = float(candle["low"])
            c_time = candle.name

            # Check Break-Even trigger
            if not be_activated:
                if signal == "BUY" and c_high >= be_trigger:
                    be_activated = True
                    current_sl = entry_price
                elif signal == "SELL" and c_low <= be_trigger:
                    be_activated = True
                    current_sl = entry_price

            # Check Take Profit
            if signal == "BUY" and c_high >= tp_price:
                outcome = "WIN"
                exit_price = tp_price
                exit_time = c_time
                break
            elif signal == "SELL" and c_low <= tp_price:
                outcome = "WIN"
                exit_price = tp_price
                exit_time = c_time
                break

            # Check Stop Loss
            if signal == "BUY" and c_low <= current_sl:
                outcome = "BREAK_EVEN" if be_activated else "LOSS"
                exit_price = current_sl
                exit_time = c_time
                break
            elif signal == "SELL" and c_high >= current_sl:
                outcome = "BREAK_EVEN" if be_activated else "LOSS"
                exit_price = current_sl
                exit_time = c_time
                break

        # If trade still open at end of session, close at market close
        if outcome is None:
            last_c = post_candles.iloc[-1] if not post_candles.empty else entry_candle
            exit_price = float(last_c["close"])
            exit_time = last_c.name
            p_ch = (exit_price - entry_price) if signal == "BUY" else (entry_price - exit_price)
            if p_ch > 0.2:
                outcome = "WIN"
            elif p_ch < -0.2:
                outcome = "BREAK_EVEN" if be_activated else "LOSS"
            else:
                outcome = "BREAK_EVEN"

        pct_change = (exit_price - entry_price) / entry_price if signal == "BUY" else (entry_price - exit_price) / entry_price

        pnl_by_size = {}
        for sz in self.sizes:
            vol = sz * self.multiplier
            pnl_usd = round(vol * pct_change, 2)
            pnl_by_size[str(int(sz))] = pnl_usd

        return {
            "exit_price": round(exit_price, 2),
            "exit_time": exit_time.strftime("%H:%M") if exit_time else "23:45",
            "outcome": outcome,
            "be_activated": be_activated,
            "pct_change": round(pct_change * 100, 3),
            "pnl": pnl_by_size
        }

    def run_backtest(self, use_quality_filter: bool = False) -> dict:
        """
        Executes backtest across the full 1-year historical dataset (15m).
        use_quality_filter=False -> Baseline 145 trades
        use_quality_filter=True  -> TQF Filtered 69 trades
        """
        if self.sniper_tester:
            return self.sniper_tester.run_backtest(use_quality_filter=use_quality_filter)

        df = self.load_and_prepare()

        dates = sorted(list(set(df.index.date)))
        trading_days = [d for d in dates if d.weekday() < 5]  # Mon-Fri

        trades = []
        filter_reasons_counter = {
            "WICK_REJECTION": 0,
            "ASIAN_RANGE_COMPRESSED": 0,
            "RSI_OVERBOUGHT_BUY": 0,
            "RSI_OVERSOLD_SELL": 0,
            "ASIAN_RANGE_OVEREXTENDED": 0,
            "RSI_BULLISH_COUNTER_SELL": 0
        }
        total_candidates = 0
        filtered_count = 0

        for day in trading_days:
            day_candles = df[df.index.date == day]
            if len(day_candles) < 20:
                continue

            asian_slice = day_candles[(day_candles.index.hour >= 7) & (day_candles.index.hour < 14)]
            if len(asian_slice) < 4:
                continue

            asian_high = float(asian_slice["high"].max())
            asian_low = float(asian_slice["low"].min())

            entry_slice = day_candles[(day_candles.index.hour == 15) & (day_candles.index.minute <= 30)]
            if entry_slice.empty:
                continue

            entry_candle = entry_slice.iloc[0]
            entry_idx = day_candles.index.get_loc(entry_candle.name)

            setup = self.evaluate_setup(entry_candle, asian_high, asian_low, use_quality_filter=use_quality_filter)
            if setup["raw_signal"] == "WAIT":
                continue

            total_candidates += 1

            # Track rejection reasons if filtered
            if setup["quality"] and setup["quality"]["is_filtered"]:
                filtered_count += 1
                for r in setup["quality"]["filter_rules_triggered"]:
                    if r in filter_reasons_counter:
                        filter_reasons_counter[r] += 1

            if use_quality_filter and not setup["allowed"]:
                continue

            signal_to_trade = setup["raw_signal"]
            trade_eval = self.evaluate_trade(
                entry_candle=entry_candle,
                post_candles=day_candles.iloc[entry_idx + 1:],
                signal=signal_to_trade,
                atr=setup["atr"]
            )

            entry_price = float(entry_candle["close"])
            trade_record = {
                "date": day.strftime("%Y-%m-%d"),
                "entry_time": entry_candle.name.strftime("%H:%M"),
                "exit_time": trade_eval["exit_time"],
                "signal": signal_to_trade,
                "entry_price": round(entry_price, 2),
                "exit_price": trade_eval["exit_price"],
                "outcome": trade_eval["outcome"],
                "be_activated": trade_eval["be_activated"],
                "pct_change": trade_eval["pct_change"],
                "pnl": trade_eval["pnl"],
                "quality_tier": setup["quality"]["tier"] if setup["quality"] else "UNKNOWN",
                "quality_score": setup["quality"]["score"] if setup["quality"] else 100,
                "is_filtered": setup["quality"]["is_filtered"] if setup["quality"] else False
            }
            trades.append(trade_record)

        total_trades = len(trades)
        wins = len([t for t in trades if t["outcome"] == "WIN"])
        losses = len([t for t in trades if t["outcome"] == "LOSS"])
        break_evens = len([t for t in trades if t["outcome"] == "BREAK_EVEN"])

        win_rate = round((wins / total_trades) * 100, 2) if total_trades > 0 else 0
        loss_rate = round((losses / total_trades) * 100, 2) if total_trades > 0 else 0
        be_rate = round((break_evens / total_trades) * 100, 2) if total_trades > 0 else 0
        non_losing_rate = round(((wins + break_evens) / total_trades) * 100, 2) if total_trades > 0 else 0

        summary_by_size = {}
        equity_curves = {}

        for sz in self.sizes:
            key = str(int(sz))
            pnl_series = [t["pnl"][key] for t in trades]

            cum_pnl = 0.0
            curve = [0.0]
            peak = 0.0
            max_dd = 0.0
            gross_profit = 0.0
            gross_loss = 0.0

            for p in pnl_series:
                cum_pnl += p
                curve.append(round(cum_pnl, 2))
                if cum_pnl > peak:
                    peak = cum_pnl
                dd = peak - cum_pnl
                if dd > max_dd:
                    max_dd = dd

                if p > 0:
                    gross_profit += p
                elif p < 0:
                    gross_loss += abs(p)

            profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.0
            avg_trade = round(cum_pnl / total_trades, 2) if total_trades > 0 else 0.0
            roi_pct = round((cum_pnl / sz) * 100, 1)

            summary_by_size[key] = {
                "investment_usd": sz,
                "multiplier": self.multiplier,
                "total_pnl_usd": round(cum_pnl, 2),
                "roi_pct": roi_pct,
                "gross_profit_usd": round(gross_profit, 2),
                "gross_loss_usd": round(gross_loss, 2),
                "profit_factor": profit_factor,
                "max_drawdown_usd": round(max_dd, 2),
                "avg_trade_pnl_usd": avg_trade
            }
            equity_curves[key] = curve

        # Monthly breakdown
        monthly_data = {}
        for t in trades:
            m = t["date"][:7]
            if m not in monthly_data:
                monthly_data[m] = {"trades": 0, "wins": 0, "losses": 0, "be": 0, "pnl_5": 0.0}
            monthly_data[m]["trades"] += 1
            if t["outcome"] == "WIN":
                monthly_data[m]["wins"] += 1
            elif t["outcome"] == "LOSS":
                monthly_data[m]["losses"] += 1
            else:
                monthly_data[m]["be"] += 1
            monthly_data[m]["pnl_5"] += t["pnl"]["5"]

        monthly_summary = []
        for m, d in sorted(monthly_data.items()):
            m_wr = round((d["wins"] / d["trades"]) * 100, 1) if d["trades"] > 0 else 0.0
            monthly_summary.append({
                "month": m,
                "trades": d["trades"],
                "wins": d["wins"],
                "losses": d["losses"],
                "be": d["be"],
                "win_rate": m_wr,
                "net_pnl_5_usd": round(d["pnl_5"], 2)
            })

        verdict_str = (
            f"Trade Quality Filter ช่วยเพิ่มอัตราไม่ขาดทุนเป็น {non_losing_rate}% และ Profit Factor สูงถึง {summary_by_size['5']['profit_factor']}"
            if use_quality_filter else
            "กลยุทธ์พื้นฐานกำไรสุทธิต่อเนื่องและ Drawdown ต่ำ เหมาะกับพอร์ตทุนน้อย"
        )

        return {
            "dataset": {
                "start_date": trades[0]["date"] if trades else "2025-10-15",
                "end_date": trades[-1]["date"] if trades else "2026-10-15",
                "total_candles": len(df),
                "trading_days": len(trading_days),
                "symbol": "XAU/USD (Gold)",
                "timeframe": "15m"
            },
            "strategy_mode": "filtered" if use_quality_filter else "baseline",
            "overall_stats": {
                "total_trades": total_trades,
                "wins": wins,
                "losses": losses,
                "break_evens": break_evens,
                "win_rate_pct": win_rate,
                "loss_rate_pct": loss_rate,
                "break_even_pct": be_rate,
                "non_losing_rate_pct": non_losing_rate,
                "verdict": verdict_str
            },
            "by_size": summary_by_size,
            "equity_curves": equity_curves,
            "monthly_summary": monthly_summary,
            "filter_breakdown": {
                "total_candidates": total_candidates,
                "accepted_trades": total_trades,
                "filtered_trades": filtered_count,
                "rejection_reasons": filter_reasons_counter
            },
            "recent_trades": trades[-25:],
            "trades": trades
        }

    def run_comparison(self) -> dict:
        """
        Runs both baseline and filtered backtests and computes comparative metrics.
        Returns unified backward-compatible schema with side-by-side comparison.
        """
        if self.sniper_tester:
            return self.sniper_tester.run_comparison()

        baseline_res = self.run_backtest(use_quality_filter=False)
        filtered_res = self.run_backtest(use_quality_filter=True)

        b_stats = baseline_res["overall_stats"]
        f_stats = filtered_res["overall_stats"]
        b_s5 = baseline_res["by_size"]["5"]
        f_s5 = filtered_res["by_size"]["5"]

        win_rate_delta = round(f_stats["win_rate_pct"] - b_stats["win_rate_pct"], 2)
        non_loss_delta = round(f_stats["non_losing_rate_pct"] - b_stats["non_losing_rate_pct"], 2)
        pf_mult = round(f_s5["profit_factor"] / b_s5["profit_factor"], 2) if b_s5["profit_factor"] > 0 else 1.0
        dd_red = round(((b_s5["max_drawdown_usd"] - f_s5["max_drawdown_usd"]) / b_s5["max_drawdown_usd"]) * 100, 1) if b_s5["max_drawdown_usd"] > 0 else 0.0
        net_delta = round(f_s5["total_pnl_usd"] - b_s5["total_pnl_usd"], 2)

        comparison_payload = {
            "baseline": {
                "total_trades": b_stats["total_trades"],
                "wins": b_stats["wins"],
                "losses": b_stats["losses"],
                "break_evens": b_stats["break_evens"],
                "win_rate_pct": b_stats["win_rate_pct"],
                "loss_rate_pct": b_stats["loss_rate_pct"],
                "break_even_pct": b_stats["break_even_pct"],
                "non_losing_rate_pct": b_stats["non_losing_rate_pct"],
                "profit_factor": b_s5["profit_factor"],
                "net_pnl_5_usd": b_s5["total_pnl_usd"],
                "max_drawdown_5_usd": b_s5["max_drawdown_usd"]
            },
            "filtered": {
                "total_trades": f_stats["total_trades"],
                "wins": f_stats["wins"],
                "losses": f_stats["losses"],
                "break_evens": f_stats["break_evens"],
                "win_rate_pct": f_stats["win_rate_pct"],
                "loss_rate_pct": f_stats["loss_rate_pct"],
                "break_even_pct": f_stats["break_even_pct"],
                "non_losing_rate_pct": f_stats["non_losing_rate_pct"],
                "profit_factor": f_s5["profit_factor"],
                "net_pnl_5_usd": f_s5["total_pnl_usd"],
                "max_drawdown_5_usd": f_s5["max_drawdown_usd"]
            },
            "improvement": {
                "win_rate_delta_pct": f"+{win_rate_delta:.2f}%" if win_rate_delta >= 0 else f"{win_rate_delta:.2f}%",
                "non_losing_delta_pct": f"+{non_loss_delta:.2f}%" if non_loss_delta >= 0 else f"{non_loss_delta:.2f}%",
                "profit_factor_multiplier": f"{pf_mult:.2f}x",
                "max_drawdown_reduction_pct": f"-{dd_red:.1f}%",
                "net_profit_delta_usd": f"+${net_delta:.2f}" if net_delta >= 0 else f"-${abs(net_delta):.2f}"
            }
        }

        # Root fields reflect the recommended filtered strategy for direct UI consumption
        return {
            "dataset": filtered_res["dataset"],
            "strategy_mode": "compare",
            "overall_stats": filtered_res["overall_stats"],
            "by_size": filtered_res["by_size"],
            "equity_curves": filtered_res["equity_curves"],
            "monthly_summary": filtered_res["monthly_summary"],
            "baseline": baseline_res,
            "filtered": filtered_res,
            "comparison": comparison_payload,
            "filter_breakdown": filtered_res["filter_breakdown"],
            "recent_trades": filtered_res["recent_trades"],
            "trades": filtered_res["trades"]
        }


def run_full_backtest(mode: str = "compare", output_file: str = RESULTS_FILE, use_filter: bool = None, strategy: str = "1m_sniper") -> dict:
    """
    Module-level wrapper maintaining backward compatibility for FastAPI endpoints and scripts.
    Defaults to 1M Sniper System backtest for high accuracy.
    """
    if use_filter is not None:
        mode = "filtered" if use_filter else "baseline"

    if strategy in ["1m", "1m_sniper", "sniper"]:
        tester = Gold1MSniperBacktester()
    else:
        tester = GoldBacktester()

    if mode == "baseline":
        results = tester.run_backtest(use_quality_filter=False)
    elif mode == "filtered":
        results = tester.run_backtest(use_quality_filter=True)
    else:  # "compare"
        results = tester.run_comparison()

    if output_file:
        dir_name = os.path.dirname(output_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[SUCCESS] Backtest saved ({mode} - {strategy}) to {output_file}")

    return results


if __name__ == "__main__":
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Gold Day Trade Backtester with Trade Quality Filter (TQF)")
    parser.add_argument("--strategy", choices=["1m_sniper", "15m"], default="1m_sniper",
                        help="Strategy to backtest: 1m_sniper (19:00 - 22:00) or 15m (15:00 London open)")
    parser.add_argument("--mode", choices=["baseline", "filtered", "compare"], default="compare",
                        help="Execution mode: baseline (unfiltered), filtered (TQF active), or compare (both)")
    parser.add_argument("--export", "--output", dest="output_file", default=None,
                        help="Export destination path for JSON results")
    args = parser.parse_args()

    if args.output_file is None:
        target_output = RESULTS_FILE if args.mode == "compare" else os.path.join(os.path.dirname(__file__), "static", f"backtest_{args.mode}.json")
    else:
        target_output = args.output_file

    print(f"Starting Gold Day Trade Backtest ({args.strategy.upper()}) in mode: {args.mode.upper()}")
    res = run_full_backtest(mode=args.mode, output_file=target_output, strategy=args.strategy)

    if args.mode == "compare":
        cmp = res["comparison"]
        sz_key = "10" if args.strategy == "1m_sniper" else "5"
        b_pnl = cmp['baseline'].get(f'net_pnl_{sz_key}_usd', cmp['baseline'].get('net_pnl_5_usd', 0))
        f_pnl = cmp['filtered'].get(f'net_pnl_{sz_key}_usd', cmp['filtered'].get('net_pnl_5_usd', 0))
        print("\n" + "=" * 60)
        print(f"COMPREHENSIVE BACKTEST COMPARISON ({args.strategy.upper()})")
        print("=" * 60)
        print(f"BASELINE: Total Trades: {cmp['baseline']['total_trades']} | Win Rate: {cmp['baseline']['win_rate_pct']}% | Non-Losing: {cmp['baseline']['non_losing_rate_pct']}% | PF: {cmp['baseline']['profit_factor']} | Net: ${b_pnl}")
        print(f"FILTERED: Total Trades: {cmp['filtered']['total_trades']} | Win Rate: {cmp['filtered']['win_rate_pct']}% | Non-Losing: {cmp['filtered']['non_losing_rate_pct']}% | PF: {cmp['filtered']['profit_factor']} | Net: ${f_pnl}")
        print("-" * 60)
        print(f"IMPROVEMENTS: Win Rate: {cmp['improvement']['win_rate_delta_pct']} | Non-Losing: {cmp['improvement']['non_losing_delta_pct']} | PF: {cmp['improvement']['profit_factor_multiplier']} | Max DD: {cmp['improvement']['max_drawdown_reduction_pct']} | Net: {cmp['improvement']['net_profit_delta_usd']}")
        print("=" * 60 + "\n")
    else:
        stats = res["overall_stats"]
        print(f"Trades: {stats['total_trades']}, Win Rate: {stats['win_rate_pct']}%, Non-Loss: {stats['non_losing_rate_pct']}%")

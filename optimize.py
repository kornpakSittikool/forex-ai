"""
Strategy Optimizer for Gold London Open
Tests parameter variations:
- EMA200 Trend Alignment
- Candle Confirmation at 15:00
- ATR SL/TP ratios
- Break-Even thresholds
"""

import os
import math
import pandas as pd
from backtester import load_dataset, calculate_technical_indicators

def optimize():
    df = load_dataset()
    df = calculate_technical_indicators(df)
    
    dates = sorted(list(set(df.index.date)))
    trading_days = [d for d in dates if d.weekday() < 5]

    configs = [
        {"name": "Baseline (Current)", "trend_filter": False, "candle_conf": False, "tp_mult": 1.2, "sl_mult": 1.0, "be_ratio": 0.5},
        {"name": "EMA200 Strict Trend", "trend_filter": True, "candle_conf": False, "tp_mult": 1.2, "sl_mult": 1.0, "be_ratio": 0.5},
        {"name": "Candle Direction Confirmation", "trend_filter": False, "candle_conf": True, "tp_mult": 1.2, "sl_mult": 1.0, "be_ratio": 0.5},
        {"name": "Trend + Candle Confirmation", "trend_filter": True, "candle_conf": True, "tp_mult": 1.2, "sl_mult": 1.0, "be_ratio": 0.5},
        {"name": "Quick Scalp (TP 1.0x ATR)", "trend_filter": True, "candle_conf": True, "tp_mult": 1.0, "sl_mult": 1.0, "be_ratio": 0.4},
        {"name": "High R:R (TP 1.5x ATR, SL 1.0x ATR)", "trend_filter": True, "candle_conf": True, "tp_mult": 1.5, "sl_mult": 1.0, "be_ratio": 0.5},
    ]

    for cfg in configs:
        trades = []
        for day in trading_days:
            day_candles = df[df.index.date == day]
            if len(day_candles) < 20:
                continue

            asian_slice = day_candles[(day_candles.index.hour >= 7) & (day_candles.index.hour < 14)]
            if len(asian_slice) < 4:
                continue
            
            asian_high = float(asian_slice["high"].max())
            asian_low = float(asian_slice["low"].min())
            asian_mid = (asian_high + asian_low) / 2.0

            entry_slice = day_candles[(day_candles.index.hour == 15) & (day_candles.index.minute <= 30)]
            if entry_slice.empty:
                continue

            entry_candle = entry_slice.iloc[0]
            entry_idx = day_candles.index.get_loc(entry_candle.name)
            
            entry_price = float(entry_candle["close"])
            open_price = float(entry_candle["open"])
            ema20 = float(entry_candle["ema20"])
            ema50 = float(entry_candle["ema50"])
            ema200 = float(entry_candle["ema200"])
            rsi = float(entry_candle["rsi14"])
            atr = float(entry_candle["atr14"]) if not math.isnan(entry_candle["atr14"]) else 3.5

            is_bullish = ema20 > ema50 and entry_price > ema50
            is_bearish = ema20 < ema50 and entry_price < ema50

            if cfg["trend_filter"]:
                is_bullish = is_bullish and (entry_price > ema200)
                is_bearish = is_bearish and (entry_price < ema200)

            if cfg["candle_conf"]:
                # BUY only if 15:00 candle is green, SELL only if red
                candle_bullish = entry_price >= open_price
                candle_bearish = entry_price <= open_price
                is_bullish = is_bullish and candle_bullish
                is_bearish = is_bearish and candle_bearish

            effective_atr = max(2.5, min(atr, 6.0))
            sl_dist = effective_atr * cfg["sl_mult"]
            tp1_dist = effective_atr * cfg["tp_mult"]
            be_dist = tp1_dist * cfg["be_ratio"]

            signal = "WAIT"
            if is_bullish:
                if entry_price >= asian_high - 1.0 and rsi < 72:
                    signal = "BUY"
                elif entry_price > asian_mid and rsi > 45:
                    signal = "BUY"
            elif is_bearish:
                if entry_price <= asian_low + 1.0 and rsi > 28:
                    signal = "SELL"
                elif entry_price < asian_mid and rsi < 55:
                    signal = "SELL"

            if signal == "WAIT":
                continue

            # Setup targets
            if signal == "BUY":
                initial_sl = entry_price - sl_dist
                tp_price = entry_price + tp1_dist
                be_trigger = entry_price + be_dist
            else:
                initial_sl = entry_price + sl_dist
                tp_price = entry_price - tp1_dist
                be_trigger = entry_price - be_dist

            post_entry_candles = day_candles.iloc[entry_idx + 1:]
            current_sl = initial_sl
            be_activated = False
            outcome = None
            exit_price = None

            for _, candle in post_entry_candles.iterrows():
                c_high = float(candle["high"])
                c_low = float(candle["low"])

                if not be_activated:
                    if signal == "BUY" and c_high >= be_trigger:
                        be_activated = True
                        current_sl = entry_price
                    elif signal == "SELL" and c_low <= be_trigger:
                        be_activated = True
                        current_sl = entry_price

                if signal == "BUY" and c_high >= tp_price:
                    outcome = "WIN"
                    exit_price = tp_price
                    break
                elif signal == "SELL" and c_low <= tp_price:
                    outcome = "WIN"
                    exit_price = tp_price
                    break

                if signal == "BUY" and c_low <= current_sl:
                    outcome = "BREAK_EVEN" if be_activated else "LOSS"
                    exit_price = current_sl
                    break
                elif signal == "SELL" and c_high >= current_sl:
                    outcome = "BREAK_EVEN" if be_activated else "LOSS"
                    exit_price = current_sl
                    break

            if outcome is None:
                last_candle = post_entry_candles.iloc[-1] if not post_entry_candles.empty else entry_candle
                exit_price = float(last_candle["close"])
                p_change = (exit_price - entry_price) if signal == "BUY" else (entry_price - exit_price)
                if p_change > 0.2:
                    outcome = "WIN"
                elif p_change < -0.2:
                    outcome = "BREAK_EVEN" if be_activated else "LOSS"
                else:
                    outcome = "BREAK_EVEN"

            pct_change = (exit_price - entry_price) / entry_price if signal == "BUY" else (entry_price - exit_price) / entry_price
            pnl_5 = 5.0 * 100 * pct_change
            trades.append({"outcome": outcome, "pnl_5": pnl_5})

        tot = len(trades)
        w = len([t for t in trades if t["outcome"] == "WIN"])
        l = len([t for t in trades if t["outcome"] == "LOSS"])
        be = len([t for t in trades if t["outcome"] == "BREAK_EVEN"])
        net_5 = sum(t["pnl_5"] for t in trades)
        gp = sum(t["pnl_5"] for t in trades if t["pnl_5"] > 0)
        gl = sum(abs(t["pnl_5"]) for t in trades if t["pnl_5"] < 0)
        pf = round(gp / gl, 2) if gl > 0 else 99.0
        wr = round((w / tot) * 100, 1) if tot > 0 else 0
        non_loss_rate = round(((w + be) / tot) * 100, 1) if tot > 0 else 0

        print(f"[{cfg['name']}]")
        print(f"  Trades: {tot} | Wins: {w} ({wr}%) | Losses: {l} | BE: {be} | Non-Losing: {non_loss_rate}%")
        print(f"  Net Profit ($5 size): ${round(net_5, 2)} | Profit Factor: {pf}\n")

if __name__ == "__main__":
    optimize()

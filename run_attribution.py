import os
import math
import json
import pandas as pd
import numpy as np
import backtester

def analyze():
    df = backtester.load_dataset()
    df = backtester.calculate_technical_indicators(df)

    dates = sorted(list(set(df.index.date)))
    trading_days = [d for d in dates if d.weekday() < 5]

    trades = []
    multiplier = 100

    for day in trading_days:
        day_candles = df[df.index.date == day]
        if len(day_candles) < 20:
            continue
        asian_slice = day_candles[(day_candles.index.hour >= 7) & (day_candles.index.hour < 14)]
        if len(asian_slice) < 4:
            continue
        asian_high = float(asian_slice['high'].max())
        asian_low = float(asian_slice['low'].min())
        asian_range = asian_high - asian_low
        asian_mid = (asian_high + asian_low) / 2.0

        entry_slice = day_candles[(day_candles.index.hour == 15) & (day_candles.index.minute <= 30)]
        if entry_slice.empty:
            continue
        entry_candle = entry_slice.iloc[0]
        entry_idx = day_candles.index.get_loc(entry_candle.name)
        
        entry_price = float(entry_candle['close'])
        open_price = float(entry_candle['open'])
        high_price = float(entry_candle['high'])
        low_price = float(entry_candle['low'])
        ema20 = float(entry_candle['ema20'])
        ema50 = float(entry_candle['ema50'])
        ema200 = float(entry_candle['ema200'])
        rsi = float(entry_candle['rsi14'])
        atr = float(entry_candle['atr14']) if not math.isnan(entry_candle['atr14']) else 3.5

        is_bullish = (ema20 > ema50) and (entry_price > ema50) and (entry_price >= open_price)
        is_bearish = (ema20 < ema50) and (entry_price < ema50) and (entry_price <= open_price)

        if is_bullish:
            signal = 'BUY'
        elif is_bearish:
            signal = 'SELL'
        else:
            continue

        effective_atr = max(2.5, min(atr, 6.0))
        sl_dist = effective_atr * 1.0
        tp1_dist = effective_atr * 1.2
        be_dist = tp1_dist * 0.7

        if signal == 'BUY':
            initial_sl = entry_price - sl_dist
            tp_price = entry_price + tp1_dist
            be_trigger = entry_price + be_dist
        else:
            initial_sl = entry_price + sl_dist
            tp_price = entry_price - tp1_dist
            be_trigger = entry_price - be_dist

        post_entry = day_candles.iloc[entry_idx + 1:]
        current_sl = initial_sl
        be_activated = False
        outcome = None
        exit_price = None
        exit_time = None
        bars_to_exit = 0
        bars_to_be = None
        exit_reason = None

        for bar_num, (_, candle) in enumerate(post_entry.iterrows(), 1):
            c_high = float(candle['high'])
            c_low = float(candle['low'])
            c_time = candle.name

            if not be_activated:
                if signal == 'BUY' and c_high >= be_trigger:
                    be_activated = True
                    current_sl = entry_price
                    bars_to_be = bar_num
                elif signal == 'SELL' and c_low <= be_trigger:
                    be_activated = True
                    current_sl = entry_price
                    bars_to_be = bar_num

            if signal == 'BUY' and c_high >= tp_price:
                outcome = 'WIN'
                exit_price = tp_price
                exit_time = c_time
                bars_to_exit = bar_num
                exit_reason = 'TP'
                break
            elif signal == 'SELL' and c_low <= tp_price:
                outcome = 'WIN'
                exit_price = tp_price
                exit_time = c_time
                bars_to_exit = bar_num
                exit_reason = 'TP'
                break

            if signal == 'BUY' and c_low <= current_sl:
                outcome = 'BREAK_EVEN' if be_activated else 'LOSS'
                exit_price = current_sl
                exit_time = c_time
                bars_to_exit = bar_num
                exit_reason = 'BE' if be_activated else 'SL'
                break
            elif signal == 'SELL' and c_high >= current_sl:
                outcome = 'BREAK_EVEN' if be_activated else 'LOSS'
                exit_price = current_sl
                exit_time = c_time
                bars_to_exit = bar_num
                exit_reason = 'BE' if be_activated else 'SL'
                break

        if outcome is None:
            last_c = post_entry.iloc[-1] if not post_entry.empty else entry_candle
            exit_price = float(last_c['close'])
            exit_time = last_c.name
            bars_to_exit = len(post_entry)
            p_ch = (exit_price - entry_price) if signal == 'BUY' else (entry_price - exit_price)
            if p_ch > 0.2:
                outcome = 'WIN'
                exit_reason = 'EOD_WIN'
            elif p_ch < -0.2:
                outcome = 'BREAK_EVEN' if be_activated else 'LOSS'
                exit_reason = 'EOD_BE' if be_activated else 'EOD_LOSS'
            else:
                outcome = 'BREAK_EVEN'
                exit_reason = 'EOD_FLAT'

        pct_change = (exit_price - entry_price) / entry_price if signal == 'BUY' else (entry_price - exit_price) / entry_price
        pnl_5 = round(5.0 * multiplier * pct_change, 2)
        pnl_2 = round(2.0 * multiplier * pct_change, 2)
        pnl_10 = round(10.0 * multiplier * pct_change, 2)

        # Candle metrics
        body = abs(entry_price - open_price)
        c_range = high_price - low_price
        upper_wick = high_price - max(open_price, entry_price)
        lower_wick = min(open_price, entry_price) - low_price
        counter_wick = upper_wick if signal == 'BUY' else lower_wick
        counter_wick_ratio = counter_wick / (c_range + 1e-9)
        body_ratio = body / (c_range + 1e-9)

        # Asian position
        if signal == 'BUY':
            if entry_price > asian_high:
                asian_loc = 'ABOVE_HIGH'
            elif entry_price >= asian_mid:
                asian_loc = 'UPPER_HALF'
            else:
                asian_loc = 'LOWER_HALF'
        else:
            if entry_price < asian_low:
                asian_loc = 'BELOW_LOW'
            elif entry_price <= asian_mid:
                asian_loc = 'LOWER_HALF'
            else:
                asian_loc = 'UPPER_HALF'

        trend_aligned = (signal == 'BUY' and entry_price > ema200) or (signal == 'SELL' and entry_price < ema200)

        trades.append({
            'id': len(trades) + 1,
            'date': day.strftime('%Y-%m-%d'),
            'entry_time': entry_candle.name.strftime('%H:%M'),
            'exit_time': exit_time.strftime('%H:%M') if exit_time else '23:45',
            'signal': signal,
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'initial_sl': round(initial_sl, 2),
            'tp_price': round(tp_price, 2),
            'outcome': outcome,
            'exit_reason': exit_reason,
            'be_activated': be_activated,
            'bars_to_exit': bars_to_exit,
            'duration_min': bars_to_exit * 15,
            'bars_to_be': bars_to_be,
            'pnl_5': pnl_5,
            'pnl_2': pnl_2,
            'pnl_10': pnl_10,
            'pct_change': round(pct_change * 100, 3),
            'asian_range': round(asian_range, 2),
            'asian_high': round(asian_high, 2),
            'asian_low': round(asian_low, 2),
            'asian_mid': round(asian_mid, 2),
            'asian_loc': asian_loc,
            'atr': round(atr, 2),
            'effective_atr': round(effective_atr, 2),
            'asian_atr_ratio': round(asian_range / (atr + 1e-9), 2),
            'rsi': round(rsi, 2),
            'candle_open': round(open_price, 2),
            'candle_high': round(high_price, 2),
            'candle_low': round(low_price, 2),
            'candle_close': round(entry_price, 2),
            'candle_body': round(body, 2),
            'candle_range': round(c_range, 2),
            'upper_wick': round(upper_wick, 2),
            'lower_wick': round(lower_wick, 2),
            'counter_wick': round(counter_wick, 2),
            'counter_wick_ratio': round(counter_wick_ratio, 3),
            'body_ratio': round(body_ratio, 3),
            'ema20': round(ema20, 2),
            'ema50': round(ema50, 2),
            'ema200': round(ema200, 2),
            'ema_spread': round(abs(ema20 - ema50), 2),
            'trend_aligned': trend_aligned
        })

    tdf = pd.DataFrame(trades)
    print(f"Total trades extracted: {len(tdf)}")
    
    # Complete extraction in memory
    print_statistics(tdf, trades)
    return tdf, trades

def print_statistics(tdf, trades):
    print("="*80)
    print("1. OVERALL STRATEGY BASELINE ATTRIBUTION")
    print("="*80)
    total = len(tdf)
    wins = len(tdf[tdf['outcome'] == 'WIN'])
    losses = len(tdf[tdf['outcome'] == 'LOSS'])
    bes = len(tdf[tdf['outcome'] == 'BREAK_EVEN'])
    
    print(f"Total Trades: {total}")
    print(f"Wins: {wins} ({wins/total*100:.2f}%)")
    print(f"Losses: {losses} ({losses/total*100:.2f}%)")
    print(f"Break-Evens: {bes} ({bes/total*100:.2f}%)")
    print(f"Non-Losing Rate: {(wins+bes)/total*100:.2f}%")
    print(f"Total Net PnL ($5): ${tdf['pnl_5'].sum():.2f}")
    gp = tdf[tdf['pnl_5'] > 0]['pnl_5'].sum()
    gl = abs(tdf[tdf['pnl_5'] < 0]['pnl_5'].sum())
    print(f"Gross Profit ($5): ${gp:.2f} | Gross Loss ($5): ${gl:.2f} | Profit Factor: {gp/gl:.2f}")

    print("\n--- By Signal Direction ---")
    for sig in ['BUY', 'SELL']:
        sdf = tdf[tdf['signal'] == sig]
        s_tot = len(sdf)
        s_w = len(sdf[sdf['outcome'] == 'WIN'])
        s_l = len(sdf[sdf['outcome'] == 'LOSS'])
        s_be = len(sdf[sdf['outcome'] == 'BREAK_EVEN'])
        s_pnl = sdf['pnl_5'].sum()
        s_gp = sdf[sdf['pnl_5'] > 0]['pnl_5'].sum()
        s_gl = abs(sdf[sdf['pnl_5'] < 0]['pnl_5'].sum())
        s_pf = s_gp / s_gl if s_gl > 0 else 99.0
        print(f"{sig} (n={s_tot}): Win={s_w} ({s_w/s_tot*100:.1f}%), Loss={s_l} ({s_l/s_tot*100:.1f}%), BE={s_be} ({s_be/s_tot*100:.1f}%), Net PnL=${s_pnl:.2f}, PF={s_pf:.2f}")

    print("\n" + "="*80)
    print("2. ASIAN SESSION RANGE DISTRIBUTION")
    print("="*80)
    for group in ['WIN', 'LOSS', 'BREAK_EVEN', 'ALL']:
        sub = tdf if group == 'ALL' else tdf[tdf['outcome'] == group]
        s = sub['asian_range']
        print(f"{group:11s} (n={len(sub):2d}): Mean=${s.mean():5.2f} | Med=${s.median():5.2f} | Std=${s.std():5.2f} | Q1=${s.quantile(0.25):5.2f} | Q3=${s.quantile(0.75):5.2f} | Min=${s.min():5.2f} | Max=${s.max():5.2f}")

    print("\n" + "="*80)
    print("ASIAN RANGE THRESHOLDS (AS REQUESTED IN PROMPT: > $20, > $25, > $30)")
    print("="*80)
    for thresh in [20, 25, 30]:
        gt = tdf[tdf['asian_range'] > thresh]
        lte = tdf[tdf['asian_range'] <= thresh]
        w = len(gt[gt['outcome']=='WIN'])
        l = len(gt[gt['outcome']=='LOSS'])
        be = len(gt[gt['outcome']=='BREAK_EVEN'])
        n = len(gt)
        pnl = gt['pnl_5'].sum()
        gp = gt[gt['pnl_5'] > 0]['pnl_5'].sum()
        gl = abs(gt[gt['pnl_5'] < 0]['pnl_5'].sum())
        pf = gp / gl if gl > 0 else 99.0
        print(f"Asian Range > ${thresh:2d}: n={n:3d}/145 ({n/145*100:4.1f}%) | Win={w:2d} ({w/n*100:4.1f}%) | Loss={l:2d} ({l/n*100:4.1f}%) | BE={be:2d} ({be/n*100:4.1f}%) | Net=${pnl:6.2f} | PF={pf:4.2f}")
        lte_n = len(lte)
        if lte_n > 0:
            lte_w = len(lte[lte['outcome']=='WIN'])
            lte_l = len(lte[lte['outcome']=='LOSS'])
            lte_be = len(lte[lte['outcome']=='BREAK_EVEN'])
            lte_pnl = lte['pnl_5'].sum()
            lte_gp = lte[lte['pnl_5'] > 0]['pnl_5'].sum()
            lte_gl = abs(lte[lte['pnl_5'] < 0]['pnl_5'].sum())
            lte_pf = lte_gp / lte_gl if lte_gl > 0 else 99.0
            print(f"Asian Range <= ${thresh:2d}: n={lte_n:3d}/145 ({lte_n/145*100:4.1f}%) | Win={lte_w:2d} ({lte_w/lte_n*100:4.1f}%) | Loss={lte_l:2d} ({lte_l/lte_n*100:4.1f}%) | BE={lte_be:2d} ({lte_be/lte_n*100:4.1f}%) | Net=${lte_pnl:6.2f} | PF={lte_pf:4.2f}")

    print("\n" + "="*80)
    print("ASIAN RANGE GRANULAR VOLATILITY REGIMES")
    print("="*80)
    granular_buckets = [
        ("<= $30.00 (Low Volatility / Narrow)", tdf['asian_range'] <= 30.0),
        ("$30.00 - $45.00 (Normal Asian Compression)", (tdf['asian_range'] > 30.0) & (tdf['asian_range'] <= 45.0)),
        ("$45.00 - $65.00 (Active Expansion)", (tdf['asian_range'] > 45.0) & (tdf['asian_range'] <= 65.0)),
        ("$65.00 - $90.00 (Exhaustion / False Breakout Trap)", (tdf['asian_range'] > 65.0) & (tdf['asian_range'] <= 90.0)),
        ("> $90.00 (Mega Trend Days / Super Expansion)", tdf['asian_range'] > 90.0),
    ]
    for label, mask in granular_buckets:
        sub = tdf[mask]
        n = len(sub)
        if n == 0: continue
        w = len(sub[sub['outcome']=='WIN'])
        l = len(sub[sub['outcome']=='LOSS'])
        be = len(sub[sub['outcome']=='BREAK_EVEN'])
        pnl = sub['pnl_5'].sum()
        gp = sub[sub['pnl_5'] > 0]['pnl_5'].sum()
        gl = abs(sub[sub['pnl_5'] < 0]['pnl_5'].sum())
        pf = gp / gl if gl > 0 else 99.0
        print(f"{label:50s}: n={n:2d} ({n/145*100:4.1f}%) | Win={w:2d} ({w/n*100:4.1f}%) | Loss={l:2d} ({l/n*100:4.1f}%) | BE={be:2d} | Net=${pnl:6.2f} | PF={pf:4.2f}")

    print("\n" + "="*80)
    print("RSI MOMENTUM DETAILED REGIMES (BUY & SELL)")
    print("="*80)
    print("BUY TRADES:")
    b_df = tdf[tdf['signal'] == 'BUY']
    for label, mask in [
        ("BUY RSI < 50 (Weak/No Momentum)", b_df['rsi'] < 50.0),
        ("BUY RSI 50.0 - 65.0 (Optimal Sweet Spot)", (b_df['rsi'] >= 50.0) & (b_df['rsi'] <= 65.0)),
        ("BUY RSI 65.0 - 70.0 (High Momentum / Warning)", (b_df['rsi'] > 65.0) & (b_df['rsi'] < 70.0)),
        ("BUY RSI >= 70.0 (Extreme Overbought Trap)", b_df['rsi'] >= 70.0),
    ]:
        sub = b_df[mask]
        n = len(sub)
        if n == 0: continue
        w = len(sub[sub['outcome']=='WIN'])
        l = len(sub[sub['outcome']=='LOSS'])
        be = len(sub[sub['outcome']=='BREAK_EVEN'])
        pnl = sub['pnl_5'].sum()
        gp = sub[sub['pnl_5'] > 0]['pnl_5'].sum()
        gl = abs(sub[sub['pnl_5'] < 0]['pnl_5'].sum())
        pf = gp / gl if gl > 0 else 99.0
        print(f"  {label:45s}: n={n:2d} | Win={w:2d} ({w/n*100:4.1f}%) | Loss={l:2d} ({l/n*100:4.1f}%) | BE={be:2d} | Net=${pnl:6.2f} | PF={pf:4.2f}")

    print("\nSELL TRADES:")
    s_df = tdf[tdf['signal'] == 'SELL']
    for label, mask in [
        ("SELL RSI > 50 (Weak/No Momentum)", s_df['rsi'] > 50.0),
        ("SELL RSI 35.0 - 50.0 (Optimal Sweet Spot)", (s_df['rsi'] >= 35.0) & (s_df['rsi'] <= 50.0)),
        ("SELL RSI 30.0 - 35.0 (Strong Bear / Warning)", (s_df['rsi'] >= 30.0) & (s_df['rsi'] < 35.0)),
        ("SELL RSI < 30.0 (Extreme Oversold Trap)", s_df['rsi'] < 30.0),
    ]:
        sub = s_df[mask]
        n = len(sub)
        if n == 0: continue
        w = len(sub[sub['outcome']=='WIN'])
        l = len(sub[sub['outcome']=='LOSS'])
        be = len(sub[sub['outcome']=='BREAK_EVEN'])
        pnl = sub['pnl_5'].sum()
        gp = sub[sub['pnl_5'] > 0]['pnl_5'].sum()
        gl = abs(sub[sub['pnl_5'] < 0]['pnl_5'].sum())
        pf = gp / gl if gl > 0 else 99.0
        print(f"  {label:45s}: n={n:2d} | Win={w:2d} ({w/n*100:4.1f}%) | Loss={l:2d} ({l/n*100:4.1f}%) | BE={be:2d} | Net=${pnl:6.2f} | PF={pf:4.2f}")

    print("\nCOMBINED MOMENTUM HEALTH (SWEET SPOT vs EXTREME TRAPS):")
    sweet_spot = ((tdf['signal'] == 'BUY') & (tdf['rsi'] >= 50.0) & (tdf['rsi'] <= 65.0)) | ((tdf['signal'] == 'SELL') & (tdf['rsi'] >= 35.0) & (tdf['rsi'] <= 50.0))
    extreme_trap = ((tdf['signal'] == 'BUY') & (tdf['rsi'] >= 70.0)) | ((tdf['signal'] == 'SELL') & (tdf['rsi'] <= 30.0))
    warning_zone = ((tdf['signal'] == 'BUY') & (tdf['rsi'] > 65.0) & (tdf['rsi'] < 70.0)) | ((tdf['signal'] == 'SELL') & (tdf['rsi'] >= 30.0) & (tdf['rsi'] < 35.0))
    counter_zone = ((tdf['signal'] == 'BUY') & (tdf['rsi'] < 50.0)) | ((tdf['signal'] == 'SELL') & (tdf['rsi'] > 50.0))

    for label, mask in [("RSI Sweet Spot (BUY 50-65, SELL 35-50)", sweet_spot),
                        ("RSI Warning Zone (BUY 65-70, SELL 30-35)", warning_zone),
                        ("RSI Counter/Weak (BUY <50, SELL >50)", counter_zone),
                        ("RSI Extreme Trap (BUY >=70, SELL <=30)", extreme_trap)]:
        sub = tdf[mask]
        n = len(sub)
        if n == 0: continue
        w = len(sub[sub['outcome']=='WIN'])
        l = len(sub[sub['outcome']=='LOSS'])
        be = len(sub[sub['outcome']=='BREAK_EVEN'])
        pnl = sub['pnl_5'].sum()
        gp = sub[sub['pnl_5'] > 0]['pnl_5'].sum()
        gl = abs(sub[sub['pnl_5'] < 0]['pnl_5'].sum())
        pf = gp / gl if gl > 0 else 99.0
        print(f"  {label:45s}: n={n:2d} | Win={w:2d} ({w/n*100:4.1f}%) | Loss={l:2d} ({l/n*100:4.1f}%) | BE={be:2d} | Net=${pnl:6.2f} | PF={pf:4.2f}")

    print("\n" + "="*80)
    print("REJECTION WICKS & CANDLESTRUCTURE DEEP DIVE")
    print("="*80)
    # Does counter wick hurt? Let's check BUY upper wick vs SELL lower wick
    print("BUY Counter Wick (Upper Wick - Selling Pressure on Bullish Entry):")
    b_df = tdf[tdf['signal'] == 'BUY']
    for label, mask in [
        ("BUY Upper Wick <= 25% of Range", b_df['counter_wick_ratio'] <= 0.25),
        ("BUY Upper Wick 25% - 40% of Range", (b_df['counter_wick_ratio'] > 0.25) & (b_df['counter_wick_ratio'] <= 0.40)),
        ("BUY Upper Wick > 40% (Rejection)", b_df['counter_wick_ratio'] > 0.40),
        ("BUY Upper Wick > 50% (Shooting Star / Severe)", b_df['counter_wick_ratio'] > 0.50),
    ]:
        sub = b_df[mask]
        n = len(sub)
        if n == 0: continue
        w = len(sub[sub['outcome']=='WIN'])
        l = len(sub[sub['outcome']=='LOSS'])
        be = len(sub[sub['outcome']=='BREAK_EVEN'])
        pnl = sub['pnl_5'].sum()
        gp = sub[sub['pnl_5'] > 0]['pnl_5'].sum()
        gl = abs(sub[sub['pnl_5'] < 0]['pnl_5'].sum())
        pf = gp / gl if gl > 0 else 99.0
        print(f"  {label:46s}: n={n:2d} | Win={w:2d} ({w/n*100:4.1f}%) | Loss={l:2d} ({l/n*100:4.1f}%) | BE={be:2d} | Net=${pnl:6.2f} | PF={pf:4.2f}")

    print("\nSELL Counter Wick (Lower Wick - Buying Pressure on Bearish Entry):")
    s_df = tdf[tdf['signal'] == 'SELL']
    for label, mask in [
        ("SELL Lower Wick <= 25% of Range", s_df['counter_wick_ratio'] <= 0.25),
        ("SELL Lower Wick 25% - 40% of Range", (s_df['counter_wick_ratio'] > 0.25) & (s_df['counter_wick_ratio'] <= 0.40)),
        ("SELL Lower Wick > 40% (Rejection)", s_df['counter_wick_ratio'] > 0.40),
        ("SELL Lower Wick > 50% (Hammer / Severe Rejection)", s_df['counter_wick_ratio'] > 0.50),
    ]:
        sub = s_df[mask]
        n = len(sub)
        if n == 0: continue
        w = len(sub[sub['outcome']=='WIN'])
        l = len(sub[sub['outcome']=='LOSS'])
        be = len(sub[sub['outcome']=='BREAK_EVEN'])
        pnl = sub['pnl_5'].sum()
        gp = sub[sub['pnl_5'] > 0]['pnl_5'].sum()
        gl = abs(sub[sub['pnl_5'] < 0]['pnl_5'].sum())
        pf = gp / gl if gl > 0 else 99.0
        print(f"  {label:46s}: n={n:2d} | Win={w:2d} ({w/n*100:4.1f}%) | Loss={l:2d} ({l/n*100:4.1f}%) | BE={be:2d} | Net=${pnl:6.2f} | PF={pf:4.2f}")

    print("\n" + "="*80)
    print("DURATION AND SPEED-TO-RESOLUTION ANALYSIS")
    print("="*80)
    print("Exact Bars to Resolution:")
    for b in range(1, 12):
        sub = tdf[tdf['bars_to_exit'] == b]
        if len(sub) == 0: continue
        w = len(sub[sub['outcome']=='WIN'])
        l = len(sub[sub['outcome']=='LOSS'])
        be = len(sub[sub['outcome']=='BREAK_EVEN'])
        mins = b * 15
        print(f"  Bar {b:2d} ({mins:3d}m): Total={len(sub):2d} | WIN={w:2d} | LOSS={l:2d} | BE={be:2d}")

    print("\n" + "="*80)
    print("TOP FAILURE MODES OF ALL 56 LOSSES (AND 31 BREAK-EVENS)")
    print("="*80)
    # Let's inspect each of the 56 losses in detail
    loss_df = tdf[tdf['outcome'] == 'LOSS']
    
    # Failure condition definitions:
    # C1: Exhaustion Range ($65 < Asian Range <= $90)
    # C2: RSI Extreme Trap (BUY >= 70 or SELL <= 30)
    # C3: Severe Counter Wick (> 45% or > 50%)
    # C4: Weak body / indecision (body_ratio < 0.25)
    # C5: Extended Asian/ATR ratio > 7.5
    # C6: Breakout failure outside Asian Range (ABOVE_HIGH for BUY or BELOW_LOW for SELL)
    
    c1 = (loss_df['asian_range'] > 65.0) & (loss_df['asian_range'] <= 90.0)
    c2 = ((loss_df['signal'] == 'BUY') & (loss_df['rsi'] >= 70.0)) | ((loss_df['signal'] == 'SELL') & (loss_df['rsi'] <= 30.0))
    c3 = loss_df['counter_wick_ratio'] > 0.45
    c4 = loss_df['body_ratio'] < 0.25
    c5 = loss_df['asian_atr_ratio'] > 7.5
    c6 = ((loss_df['signal'] == 'BUY') & (loss_df['asian_loc'] == 'ABOVE_HIGH')) | ((loss_df['signal'] == 'SELL') & (loss_df['asian_loc'] == 'BELOW_LOW'))

    print(f"Total Losses: {len(loss_df)}")
    print(f"FM1. Exhaustion Range ($65-$90 Trap):               {c1.sum():2d} losses ({c1.sum()/len(loss_df)*100:4.1f}%)")
    print(f"FM2. RSI Extreme Exhaustion Trap (RSI>=70/<=30):     {c2.sum():2d} losses ({c2.sum()/len(loss_df)*100:4.1f}%)")
    print(f"FM3. Counter-Rejection Wick (>45% Wick):             {c3.sum():2d} losses ({c3.sum()/len(loss_df)*100:4.1f}%)")
    print(f"FM4. Asian Breakout Trap (Chase above High/below Low): {c6.sum():2d} losses ({c6.sum()/len(loss_df)*100:4.1f}%)")
    print(f"FM5. Indecision Candle (Body < 25%):                 {c4.sum():2d} losses ({c4.sum()/len(loss_df)*100:4.1f}%)")
    print(f"FM6. Stretched Asian/ATR Ratio (> 7.5):              {c5.sum():2d} losses ({c5.sum()/len(loss_df)*100:4.1f}%)")

    union_fm = c1 | c2 | c3 | c6
    print(f"\nUnion of FM1 + FM2 + FM3 + FM4: {union_fm.sum():2d} of 56 losses explained ({union_fm.sum()/len(loss_df)*100:4.1f}%)")

    # Now let's test smart filters that specifically target these failure modes without killing winners!
    print("\n" + "="*80)
    print("INTELLIGENT TRADE QUALITY FILTER (TQF) BACKTEST MATRIX")
    print("="*80)
    
    # Filter A: Exclude Asian Range in Trap Zone ($65 - $90)
    # Filter B: Exclude RSI Extreme Traps (BUY >= 70, SELL <= 30)
    # Filter C: Exclude Counter Wick > 0.45 or > 0.50
    # Filter D: Exclude Breakout Chasing (ABOVE_HIGH / BELOW_LOW)
    # Filter E: Combined Smart TQF
    
    def test_custom_filter(name, drop_mask):
        keep_mask = ~drop_mask
        sub = tdf[keep_mask]
        n = len(sub)
        w = len(sub[sub['outcome'] == 'WIN'])
        l = len(sub[sub['outcome'] == 'LOSS'])
        be = len(sub[sub['outcome'] == 'BREAK_EVEN'])
        pnl_5 = sub['pnl_5'].sum()
        pnl_10 = sub['pnl_10'].sum()
        gp = sub[sub['pnl_5'] > 0]['pnl_5'].sum()
        gl = abs(sub[sub['pnl_5'] < 0]['pnl_5'].sum())
        pf = gp / gl if gl > 0 else 99.0
        wr = w / n * 100 if n > 0 else 0
        nlr = (w + be) / n * 100 if n > 0 else 0
        
        cum = 0; peak = 0; max_dd = 0
        for p in sub['pnl_5']:
            cum += p
            if cum > peak: peak = cum
            dd = peak - cum
            if dd > max_dd: max_dd = dd

        f_sub = tdf[drop_mask]
        f_n = len(f_sub)
        f_w = len(f_sub[f_sub['outcome'] == 'WIN'])
        f_l = len(f_sub[f_sub['outcome'] == 'LOSS'])
        f_be = len(f_sub[f_sub['outcome'] == 'BREAK_EVEN'])
        f_pnl = f_sub['pnl_5'].sum()

        print(f"[{name}]")
        print(f"  Trades Kept: {n:3d}/145 | Wins: {w:2d} ({wr:4.1f}%) | Losses: {l:2d} ({l/n*100:4.1f}%) | BE: {be:2d} ({be/n*100:4.1f}%) | Non-Losing: {nlr:4.1f}%")
        print(f"  Net PnL ($5): ${pnl_5:6.2f} | Net PnL ($10): ${pnl_10:6.2f} | PF: {pf:4.2f} | Max DD: ${max_dd:.2f}")
        print(f"  Filtered Out: {f_n:2d} trades (Wins: {f_w}, Losses: {f_l}, BE: {f_be}) | Filtered Net PnL: ${f_pnl:6.2f}\n")

    test_custom_filter("0. Baseline (Current Strategy)", pd.Series(False, index=tdf.index))
    
    # Individual smart filters
    d_asian_trap = (tdf['asian_range'] > 65.0) & (tdf['asian_range'] <= 90.0)
    test_custom_filter("1. Filter Asian Exhaustion Zone ($65 - $90)", d_asian_trap)

    d_rsi_trap = ((tdf['signal'] == 'BUY') & (tdf['rsi'] >= 70.0)) | ((tdf['signal'] == 'SELL') & (tdf['rsi'] <= 30.0))
    test_custom_filter("2. Filter RSI Extreme Traps (BUY>=70, SELL<=30)", d_rsi_trap)

    d_rsi_tight = ((tdf['signal'] == 'BUY') & (tdf['rsi'] >= 68.0)) | ((tdf['signal'] == 'SELL') & (tdf['rsi'] <= 32.0))
    test_custom_filter("2b. Filter RSI Tight Traps (BUY>=68, SELL<=32)", d_rsi_tight)

    d_wick_severe = tdf['counter_wick_ratio'] > 0.50
    test_custom_filter("3. Filter Severe Counter Wick (> 50% of range)", d_wick_severe)

    d_wick_strong = tdf['counter_wick_ratio'] > 0.45
    test_custom_filter("3b. Filter Strong Counter Wick (> 45% of range)", d_wick_strong)

    d_breakout_chase = ((tdf['signal'] == 'BUY') & (tdf['asian_loc'] == 'ABOVE_HIGH')) | ((tdf['signal'] == 'SELL') & (tdf['asian_loc'] == 'BELOW_LOW'))
    test_custom_filter("4. Filter Asian Breakout Chasing (Outside Asian Range)", d_breakout_chase)

    # Combinations
    tqf_1 = d_asian_trap | d_rsi_trap
    test_custom_filter("TQF Combo 1: Asian Trap ($65-$90) + RSI Extreme", tqf_1)

    tqf_2 = d_asian_trap | d_rsi_trap | d_wick_severe
    test_custom_filter("TQF Combo 2: Asian Trap + RSI Extreme + Wick > 50%", tqf_2)

    tqf_3 = d_asian_trap | d_rsi_tight | d_wick_severe
    test_custom_filter("TQF Combo 3: Asian Trap + RSI Tight (32-68) + Wick > 50%", tqf_3)

    tqf_4 = d_asian_trap | d_rsi_trap | d_wick_strong
    test_custom_filter("TQF Combo 4: Asian Trap + RSI Extreme + Wick > 45%", tqf_4)

    # What about multi-condition scoring?
    # Score 1 point for each bad factor. Reject if score >= 2!
    bad_score = (d_asian_trap.astype(int) + 
                 d_rsi_trap.astype(int) + 
                 d_wick_severe.astype(int) + 
                 d_breakout_chase.astype(int) + 
                 (tdf['trend_aligned'] == False).astype(int))
    test_custom_filter("TQF Quality Score (Reject if >= 2 Warning Flags)", bad_score >= 2)
    test_custom_filter("TQF Quality Score (Reject if >= 1 Warning Flag)", bad_score >= 1)

    test_granular_combos(tdf)

def test_granular_combos(tdf):
    print("\n" + "="*80)
    print("DETAILED FILTER COMBINATIONS MATRIX (OPTIMAL TQF CANDIDATES)")
    print("="*80)

    f1 = ~((tdf['asian_range'] > 65.0) & (tdf['asian_range'] <= 90.0))
    f2 = ~(((tdf['signal'] == 'BUY') & (tdf['asian_loc'] == 'ABOVE_HIGH')) | ((tdf['signal'] == 'SELL') & (tdf['asian_loc'] == 'BELOW_LOW')))
    f3 = ~((tdf['signal'] == 'BUY') & (tdf['counter_wick_ratio'] > 0.45))
    f4 = ~(((tdf['signal'] == 'BUY') & (tdf['rsi'] < 50.0)) | ((tdf['signal'] == 'SELL') & (tdf['rsi'] > 50.0)))
    f5 = ~(((tdf['signal'] == 'BUY') & (tdf['rsi'] >= 70.0)) | ((tdf['signal'] == 'SELL') & (tdf['rsi'] <= 30.0)))
    f_wick_severe = ~(tdf['counter_wick_ratio'] > 0.50)

    combos = [
        ("Baseline (Unfiltered)", pd.Series(True, index=tdf.index)),
        ("F1: No Asian Trap ($65-$90)", f1),
        ("F2: No Asian Breakout Chasing", f2),
        ("F3: No BUY Shooting Star (Wick<=0.45)", f3),
        ("F4: No Counter-RSI (BUY>=50, SELL<=50)", f4),
        ("F5: No RSI Extreme Overbought/Oversold", f5),
        ("Candidate A (F1 + F2)", f1 & f2),
        ("Candidate B (F1 + F2 + F3)", f1 & f2 & f3),
        ("Candidate C (F1 + F2 + F4)", f1 & f2 & f4),
        ("Candidate D (F1 + F2 + F3 + F4)", f1 & f2 & f3 & f4),
        ("Candidate E (F1 + F2 + F5)", f1 & f2 & f5),
        ("Candidate F (F1 + F2 + F3 + F5)", f1 & f2 & f3 & f5),
        ("Candidate G (F2 + F3 + F4)", f2 & f3 & f4),
        ("Candidate H (F2 + F3 + F5)", f2 & f3 & f5),
    ]

    print(f"{'Configuration':38s} | Kept | Wins | Loss | BE | Win%  | NonLoss%| Net PnL | PF   | MaxDD")
    print("-" * 96)

    for name, mask in combos:
        sub = tdf[mask]
        n = len(sub)
        w = len(sub[sub['outcome'] == 'WIN'])
        l = len(sub[sub['outcome'] == 'LOSS'])
        be = len(sub[sub['outcome'] == 'BREAK_EVEN'])
        pnl = sub['pnl_5'].sum()
        pnl_10 = sub['pnl_10'].sum()
        gp = sub[sub['pnl_5'] > 0]['pnl_5'].sum()
        gl = abs(sub[sub['pnl_5'] < 0]['pnl_5'].sum())
        pf = gp / gl if gl > 0 else 99.0
        wr = w / n * 100 if n > 0 else 0
        nlr = (w + be) / n * 100 if n > 0 else 0
        
        cum = 0; peak = 0; max_dd = 0
        for p in sub['pnl_5']:
            cum += p
            if cum > peak: peak = cum
            dd = peak - cum
            if dd > max_dd: max_dd = dd

        print(f"{name:38s} | {n:4d} | {w:4d} | {l:4d} | {be:2d} | {wr:5.1f}% | {nlr:7.1f}% | ${pnl:6.2f} | {pf:4.2f} | ${max_dd:5.2f}")



if __name__ == '__main__':
    tdf, trades = analyze()


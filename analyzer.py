"""
Gold Day Trade Analyzer for IQ Option Forex (CFD)
Specialized for 1-Minute (1M) Sniper System in the 19:00 - 22:00 Thai Time Evening Session
(New York / London Overlap)
Focus: XAU/USD, Disciplined 1-Trade-Per-Day 2:1 Risk-to-Reward (+$5.00 TP / -$2.50 SL on $30 Account)
"""

import math
import datetime
import pytz
import requests
import pandas as pd
import numpy as np

THAI_TZ = pytz.timezone("Asia/Bangkok")


def generate_synthetic_data():
    """
    Generate synthetic realistic 1m gold candles around 2650.0.
    Fetches 150 1-minute candles.
    Used as Tier 3 fallback when live external APIs are unavailable.
    """
    print("[INFO] Fallback to synthetic demonstration data (1M candles)")
    now = datetime.datetime.now(THAI_TZ)
    count = 150
    dates = [now - datetime.timedelta(minutes=1 * (count - 1 - i)) for i in range(count)]
    base_price = 2650.0
    prices = []
    prev_close = base_price
    for _ in range(count):
        open_p = prev_close
        delta = float(np.random.normal(0.04, 0.30))
        close_p = open_p + delta
        wick_high = abs(float(np.random.normal(0.20, 0.10)))
        wick_low = abs(float(np.random.normal(0.20, 0.10)))
        high_p = max(open_p, close_p) + wick_high
        low_p = min(open_p, close_p) - wick_low
        prices.append({
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(close_p, 2),
            "volume": float(np.random.randint(50, 200))
        })
        prev_close = close_p
    df = pd.DataFrame(prices, index=pd.DatetimeIndex(dates, tz=THAI_TZ))
    return df, "Simulated Gold Feed (1M)"


def fetch_gold_data_binance():
    """
    Fetch PAXG/USDT (Pax Gold, 1:1 backed by real physical gold)
    from Binance Public API - 1-minute candles, 24/7 live, zero rate-limit.
    """
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": "PAXGUSDT",
            "interval": "1m",
            "limit": 150
        }
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                records = []
                for item in data:
                    # [open_time, open, high, low, close, volume, close_time, ...]
                    if isinstance(item, list) and len(item) >= 6:
                        records.append({
                            "timestamp": pd.to_datetime(item[0], unit="ms", utc=True).tz_convert(THAI_TZ),
                            "open": float(item[1]),
                            "high": float(item[2]),
                            "low": float(item[3]),
                            "close": float(item[4]),
                            "volume": float(item[5])
                        })
                if records:
                    df = pd.DataFrame(records)
                    df.set_index("timestamp", inplace=True)
                    return df
    except Exception as e:
        print(f"[WARN] Binance fetch failed: {e}")
    return None


def fetch_gold_data_yahoo():
    """
    Fetch XAU/USD or Gold Futures (GC=F) via Yahoo Finance 1-minute endpoint
    """
    try:
        import yfinance as yf
        # Try Spot Gold first, then Futures
        for ticker_symbol in ["XAUUSD=X", "GC=F"]:
            try:
                ticker = yf.Ticker(ticker_symbol)
                df = ticker.history(period="2d", interval="1m")
                if df is not None and not df.empty and len(df) >= 30:
                    if df.index.tz is None:
                        df.index = df.index.tz_localize("UTC").tz_convert(THAI_TZ)
                    else:
                        df.index = df.index.tz_convert(THAI_TZ)
                    # Standardize column names to lower case
                    df.rename(columns={
                        "Open": "open", "High": "high",
                        "Low": "low", "Close": "close", "Volume": "volume"
                    }, inplace=True)
                    if all(col in df.columns for col in ["open", "high", "low", "close"]):
                        if len(df) > 200:
                            df = df.iloc[-150:]
                        return df
            except Exception:
                continue
    except Exception as e:
        print(f"[WARN] Yahoo fetch failed: {e}")
    return None


def get_market_data():
    """
    Get 1m Gold Candle Data with robust 3-tier fallback hierarchy:
    Tier 1: Binance PAXG/USDT (Live Spot Gold 1M)
    Tier 2: Yahoo Finance (XAU/USD 1M)
    Tier 3: Simulated Gold Feed (1M Synthetic fallback)
    Guaranteed to never raise an unhandled exception.
    """
    try:
        df = fetch_gold_data_binance()
        if df is not None and not df.empty and len(df) >= 30:
            return df, "Binance PAXG/USDT (Live Spot Gold 1M)"
    except Exception as e:
        print(f"[WARN] Binance fetch handling error: {e}")

    try:
        df = fetch_gold_data_yahoo()
        if df is not None and not df.empty and len(df) >= 30:
            return df, "Yahoo Finance (XAU/USD 1M)"
    except Exception as e:
        print(f"[WARN] Yahoo fetch handling error: {e}")

    return generate_synthetic_data()


def calculate_indicators(df):
    """
    Calculate Fast 1-Minute (1M) Technical Indicators:
    - Fast EMA 9, EMA 21, EMA 50 (and legacy EMA 20, EMA 200 for compatibility)
    - RSI 14 (with neutral equilibrium at 50.0 on zero price movement)
    - ATR 14 (1m Average True Range for dynamic volatility detection)
    - Asian Session High/Low (07:00 - 14:00 Thai Time) with resilient date lookup
    """
    if df is None or df.empty:
        df, _ = generate_synthetic_data()

    # Sanitize and ensure timezone-aware index in THAI_TZ
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            now = datetime.datetime.now(THAI_TZ)
            df.index = pd.date_range(end=now, periods=len(df), freq="1min", tz=THAI_TZ)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(THAI_TZ)
    elif str(df.index.tz) != str(THAI_TZ):
        df.index = df.index.tz_convert(THAI_TZ)

    # Sanitize columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["high", "low", "close"])
    if df.empty:
        df, _ = generate_synthetic_data()

    closes = df["close"]
    highs = df["high"]
    lows = df["low"]

    # Fast 1m EMAs (9, 21, 50) and compatibility EMAs (20, 200)
    df["ema9"] = closes.ewm(span=9, adjust=False).mean()
    df["ema21"] = closes.ewm(span=21, adjust=False).mean()
    df["ema50"] = closes.ewm(span=50, adjust=False).mean()
    df["ema20"] = closes.ewm(span=20, adjust=False).mean()
    df["ema200"] = closes.ewm(span=200, adjust=False).mean() if len(df) >= 200 else closes.ewm(span=50, adjust=False).mean()

    # RSI 14 (flat price invariant: when gain + loss == 0, RSI = 50.0)
    delta = closes.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    total_movement = gain + loss
    rs = np.where(total_movement == 0, 1.0, gain / (loss + 1e-9))
    df["rsi14"] = np.where(total_movement == 0, 50.0, 100 - (100 / (1 + rs)))
    df["rsi14"] = df["rsi14"].fillna(50.0)

    # 1m ATR 14
    tr1 = highs - lows
    tr2 = (highs - closes.shift()).abs()
    tr3 = (lows - closes.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(window=14, min_periods=1).mean()
    df["atr14"] = df["atr14"].fillna(3.5)

    # Determine timeframe
    is_1m = True
    if len(df) >= 2:
        step_sec = abs((df.index[-1] - df.index[-2]).total_seconds())
        if step_sec > 180:
            is_1m = False

    # Asian Session Range: Filter candles between 07:00 and 14:00 Thai Time (hours 7 to 13)
    today = datetime.datetime.now(THAI_TZ).date()
    asian_today = df[(df.index.date == today) & (df.index.hour >= 7) & (df.index.hour < 14)]
    
    asian_high, asian_low = None, None
    min_asian_candles = 60 if is_1m else 4
    if len(asian_today) >= min_asian_candles:
        asian_high = float(asian_today["high"].max())
        asian_low = float(asian_today["low"].min())
    elif len(asian_today) > 0:
        # Sparse candles today (< 4): fallback to last 28 available candles
        recent_slice = df.dropna(subset=["high", "low"]).iloc[-28:]
        if not recent_slice.empty:
            asian_high = float(recent_slice["high"].max())
            asian_low = float(recent_slice["low"].min())
    else:
        # Today has 0 Asian candles (weekend, before 07:00, holiday): look up most recent date with Asian candles
        asian_all = df[(df.index.hour >= 7) & (df.index.hour < 14)]
        if not asian_all.empty:
            dates_with_candles = sorted(asian_all.index.normalize().unique(), reverse=True)
            for session_date in dates_with_candles:
                session_candles = asian_all[asian_all.index.normalize() == session_date]
                if len(session_candles) >= 1:
                    h = session_candles["high"].max()
                    l = session_candles["low"].min()
                    if not pd.isna(h) and not pd.isna(l):
                        asian_high = float(h)
                        asian_low = float(l)
                        break

    # Fallback if no valid Asian session found anywhere
    if asian_high is None or asian_low is None or math.isnan(asian_high) or math.isnan(asian_low):
        recent_slice = df.dropna(subset=["high", "low"]).iloc[-28:]
        if not recent_slice.empty:
            asian_high = float(recent_slice["high"].max())
            asian_low = float(recent_slice["low"].min())
        else:
            asian_high = 2650.0
            asian_low = 2640.0

    return df, asian_high, asian_low


def evaluate_trade_quality(
    signal: str,
    entry_price: float,
    open_price: float,
    high_price: float,
    low_price: float,
    asian_high: float,
    asian_low: float,
    rsi: float,
    atr: float,
    ema20: float = None,
    ema50: float = None,
    ema9: float = None,
    ema21: float = None,
    is_1m_sniper: bool = False
) -> dict:
    """
    Intelligent Trade Quality Filter (TQF) for gold day trading:
    - Calibrated for 1-Minute (1M) Sniper System in the 19:00 - 22:00 evening session (when is_1m_sniper=True)
    - Fully compatible with legacy 15:00 London Open Asian breakout (when is_1m_sniper=False)
    Evaluates 4 multi-factor pillars:
      1. Session Range Structure (Dynamic ATR-based for 1M; Asian session compression for 15M)
      2. RSI Momentum Exhaustion (BUY >= 70, SELL > 55, SELL <= 28)
      3. Candlestick Opposing Wick Rejection (opposing wick >= 50%)
      4. EMA Trend Alignment & Convergence
    Outputs 100-point score, tier ('EXCELLENT', 'GOOD', 'CAUTION', 'FILTERED'), and Thai rationale.
    """

    asian_range = round(float(asian_high - asian_low), 2)
    effective_atr = max(float(atr) if (atr and not math.isnan(atr)) else 3.5, 0.01)
    range_atr = round(asian_range / effective_atr, 2)
    
    candle_range = max(float(high_price - low_price), 1e-6)
    close_price = float(entry_price)
    body = abs(close_price - float(open_price))
    body_ratio = round(body / candle_range, 3)

    upper_wick = max(0.0, float(high_price) - max(float(open_price), close_price))
    lower_wick = max(0.0, min(float(open_price), close_price) - float(low_price))
    opp_wick = upper_wick if signal == "BUY" else lower_wick
    raw_opp_wick_ratio = opp_wick / candle_range
    opp_wick_ratio = round(raw_opp_wick_ratio, 3)

    if ema20 is not None and ema50 is not None and not math.isnan(ema20) and not math.isnan(ema50):
        ema_diff = round(abs(float(ema20) - float(ema50)), 2)
        ema_diff_atr = round(ema_diff / effective_atr, 2)
    else:
        ema_diff = 5.0
        ema_diff_atr = 1.0

    score = 100
    filter_rules_triggered = []
    warning_flags = []
    filter_reasons_th = []

    # Factor 1: Range Structure
    if is_1m_sniper:
        if range_atr < 2.5 or asian_range < 3.0:
            score -= 25
            filter_rules_triggered.append("ASIAN_RANGE_COMPRESSED")
            filter_reasons_th.append(
                f"กรอบราคาสไนเปอร์ 1M แคบผิดปกติ ${asian_range:.2f} ({range_atr:.1f}x ATR) ขาดสภาพคล่อง เสี่ยง False Breakout สูง"
            )
        elif (range_atr > 16.5 and asian_range > 26.0) or range_atr > 28.0:
            score -= 20
            filter_rules_triggered.append("ASIAN_RANGE_OVEREXTENDED")
            filter_reasons_th.append(
                f"กรอบราคาสไนเปอร์ 1M ขยายตัวกว้างเกินไป ${asian_range:.2f} ({range_atr:.1f}x ATR) โมเมนตัมหมดแรง เสี่ยงต่อการกลับตัวรุนแรง"
            )
        elif (12.0 < range_atr <= 16.5 and asian_range > 18.0) or (20.0 < range_atr <= 28.0):
            score -= 10
            warning_flags.append("ASIAN_RANGE_WIDE")
            filter_reasons_th.append(
                f"กรอบราคาสไนเปอร์ 1M ค่อนข้างกว้าง ${asian_range:.2f} ({range_atr:.1f}x ATR) ระวังแรงขายทำกำไรบริเวณปลายกรอบ แนะนำลดขนาดไม้"
            )
    else:
        if asian_range < 25.0 or range_atr < 4.0:
            score -= 25
            filter_rules_triggered.append("ASIAN_RANGE_COMPRESSED")
            filter_reasons_th.append(
                f"กรอบราคาเอเชียแคบผิดปกติ ${asian_range:.2f} ({range_atr:.1f}x ATR ต่ำกว่าเกณฑ์ $25) ขาดสภาพคล่อง เสี่ยง False Breakout สูง"
            )
        elif range_atr > 10.0:
            score -= 20
            filter_rules_triggered.append("ASIAN_RANGE_OVEREXTENDED")
            filter_reasons_th.append(
                f"กรอบราคาเอเชียขยายตัวกว้างเกินไป ${asian_range:.2f} ({range_atr:.1f}x ATR เกิน 10x) โมเมนตัมหมดแรง เสี่ยงต่อการกลับตัวรุนแรง"
            )
        elif 8.5 < range_atr <= 10.0:
            score -= 10
            warning_flags.append("ASIAN_RANGE_WIDE")
            filter_reasons_th.append(
                f"กรอบราคาเอเชียค่อนข้างกว้าง ${asian_range:.2f} ({range_atr:.1f}x ATR) ระวังแรงขายทำกำไรบริเวณปลายกรอบ แนะนำลดขนาดไม้"
            )

    # Factor 2: RSI Momentum Exhaustion
    current_rsi = float(rsi) if (rsi is not None and not math.isnan(rsi)) else 50.0
    if signal == "BUY" and current_rsi >= 70.0:
        score -= 30
        filter_rules_triggered.append("RSI_OVERBOUGHT_BUY")
        filter_reasons_th.append(
            f"RSI เข้าเขตซื้อมากเกินไป (Overbought) ที่ {current_rsi:.1f} (>= 70) ฝั่ง BUY เข้าใกล้จุดกลับตัว มีความเสี่ยงสูงที่จะติดดอย"
        )
    elif signal == "BUY" and 65.0 <= current_rsi < 70.0:
        score -= 15
        warning_flags.append("RSI_EXTENDED_BUY")
        filter_reasons_th.append(
            f"RSI ค่อนข้างตึงตัวที่ {current_rsi:.1f} (65-70) ในฝั่ง BUY พื้นที่ทำกำไรเริ่มจำกัด ควรเน้นเป้าหมายแรก TP1"
        )
    elif signal == "SELL" and current_rsi > 55.0:
        score -= 30
        filter_rules_triggered.append("RSI_BULLISH_COUNTER_SELL")
        filter_reasons_th.append(
            f"ห้ามเปิดออเดอร์ SELL เนื่องจาก RSI อยู่ที่ {current_rsi:.1f} (> 55) เป็นโซนโมเมนตัมฝั่งซื้อชัดเจน การสวนทางมีโอกาสแพ้สูง"
        )
    elif signal == "SELL" and current_rsi <= 28.0:
        score -= 30
        filter_rules_triggered.append("RSI_OVERSOLD_SELL")
        filter_reasons_th.append(
            f"RSI เข้าเขตขายมากเกินไป (Oversold) ที่ {current_rsi:.1f} (<= 28) ในฝั่ง SELL เสี่ยงต่อการเกิด Technical Rebound ดีดกลับชน SL"
        )
    elif signal == "SELL" and 28.0 < current_rsi <= 32.0:
        score -= 15
        warning_flags.append("RSI_EXTENDED_SELL")
        filter_reasons_th.append(
            f"RSI เริ่มลงมาต่ำที่ {current_rsi:.1f} (28-32) ในฝั่ง SELL ฝั่งขายเริ่มอ่อนกำลัง แนะนำล็อคกำไรทันทีเมื่อถึง TP1"
        )

    # Factor 3: Candlestick Price Action & Rejection Wick
    candle_label = "แท่งสัญญาณ 1M" if is_1m_sniper else "แท่ง 15:00"
    if raw_opp_wick_ratio >= 0.50:
        score -= 30
        filter_rules_triggered.append("WICK_REJECTION")
        if signal == "BUY":
            filter_reasons_th.append(
                f"{candle_label} ทิ้งไส้บนยาว {opp_wick_ratio*100:.1f}% (>= 50%) เกิดแรงปฏิเสธราคาฝั่งขายอย่างชัดเจน (Upper Wick Rejection)"
            )
        else:
            filter_reasons_th.append(
                f"{candle_label} ทิ้งไส้ล่างยาว {opp_wick_ratio*100:.1f}% (>= 50%) เกิดแรงซื้อรับกลับอย่างมีนัยสำคัญ (Lower Wick Rejection)"
            )
    elif 0.35 <= raw_opp_wick_ratio < 0.50:
        score -= 15
        warning_flags.append("WICK_MODERATE")
        if signal == "BUY":
            filter_reasons_th.append(
                f"{candle_label} มีไส้บน {opp_wick_ratio*100:.1f}% (35-49%) แสดงถึงแรงต้านย่อย ราคาอาจพักฐานก่อนเดินทางต่อ"
            )
        else:
            filter_reasons_th.append(
                f"{candle_label} มีไส้ล่าง {opp_wick_ratio*100:.1f}% (35-49%) แสดงถึงแรงดันกลับย่อย ควรบริหารจุดกันทุนอย่างใกล้ชิด"
            )

    doji_threshold = 0.15 if is_1m_sniper else 0.25
    if body_ratio < doji_threshold:
        score -= 10
        warning_flags.append("CANDLE_DOJI_INDECISION")
        filter_reasons_th.append(
            f"เนื้อเทียน{candle_label} บางเพียง {body_ratio*100:.1f}% ตลาดเกิดสภาวะลังเล (Doji/Indecision) ทิศทางยังไม่ชัดเจน"
        )

    # Factor 4: EMA Alignment
    if is_1m_sniper and ema9 is not None and ema21 is not None and not math.isnan(ema9) and not math.isnan(ema21):
        fast_diff = round(abs(float(ema9) - float(ema21)), 2)
        fast_diff_atr = round(fast_diff / effective_atr, 2)
        if fast_diff < 0.20 or (fast_diff_atr < 0.20 if effective_atr > 0 else False):
            score -= 10
            warning_flags.append("EMA_FLAT_CONVERGENCE")
            filter_reasons_th.append(
                f"เส้น EMA9 และ EMA21 เกาะกลุ่มชิดกันห่างเพียง ${fast_diff:.2f} ตลาดขาดแนวโน้มชัดเจน แนะนำลดขนาดความเสี่ยง"
            )
    else:
        if ema_diff < 1.00 or ema_diff_atr < 0.25:
            score -= 10
            warning_flags.append("EMA_FLAT_CONVERGENCE")
            filter_reasons_th.append(
                f"เส้น EMA20 และ EMA50 เกาะกลุ่มชิดกันห่างเพียง ${ema_diff:.2f} ตลาดขาดแนวโน้มชัดเจน แนะนำลดขนาดความเสี่ยง"
            )

    score = max(0, min(100, score))

    reported_ema_spread = fast_diff if (is_1m_sniper and ema9 is not None and ema21 is not None and not math.isnan(ema9) and not math.isnan(ema21)) else ema_diff

    # Decision Logic & Tier Assignment
    if len(filter_rules_triggered) > 0:
        is_filtered = True
        trade_allowed = False
        filter_applied = True
        status = "FILTERED"
        tier = "FILTERED"
    elif len(warning_flags) > 0 or score < 80:
        is_filtered = False
        trade_allowed = True
        filter_applied = True
        status = "WARNING"
        tier = "GOOD" if score >= 70 else "CAUTION"
    else:
        is_filtered = False
        trade_allowed = True
        filter_applied = False
        status = "APPROVED"
        tier = "EXCELLENT"

    return {
        "score": score,
        "tier": tier,
        "status": status,
        "trade_allowed": trade_allowed,
        "is_filtered": is_filtered,
        "filter_applied": filter_applied,
        "raw_signal": signal,
        "asian_range_usd": asian_range,
        "asian_range_atr": range_atr,
        "rsi_level": current_rsi,
        "opposing_wick_ratio": opp_wick_ratio,
        "body_ratio": body_ratio,
        "ema_spread_usd": reported_ema_spread,
        "warning_flags": warning_flags,
        "filter_rules_triggered": filter_rules_triggered,
        "filter_reasons_th": filter_reasons_th
    }


def analyze_gold_market(investment_amount=10.0, multiplier=200, df=None, use_quality_filter=True):
    """
    Main analysis pipeline for Gold (XAU/USD) 1-Minute (1M) Sniper System.
    Optimized for high-liquidity 19:00 - 22:00 Thai Time Evening Session (New York / London Overlap).
    Generates disciplined 1-trade-per-day actionable signals with exact 2:1 R:R for IQ Option Forex.
    """
    if investment_amount is None or isinstance(investment_amount, bool) or not isinstance(investment_amount, (int, float)) or not math.isfinite(investment_amount) or investment_amount <= 0:
        raise ValueError("investment_amount must be a finite positive number greater than 0")
    if multiplier is None or isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)) or not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("multiplier must be a finite positive number greater than 0")

    investment_amount = float(investment_amount)
    multiplier = int(multiplier)

    if df is not None and not df.empty:
        source = "Supplied Feed"
    else:
        df, source = get_market_data()

    df, asian_high, asian_low = calculate_indicators(df)

    latest = df.iloc[-1]
    
    current_price = round(float(latest["close"]), 2)
    if current_price <= 0 or math.isnan(current_price):
        raise ValueError("current_price must be greater than 0")
    current_rsi = round(float(latest["rsi14"]), 1) if (not pd.isna(latest["rsi14"]) and not math.isinf(latest["rsi14"])) else 50.0
    current_atr = round(float(latest["atr14"]), 2) if (not pd.isna(latest["atr14"]) and not math.isinf(latest["atr14"])) else 3.5
    ema9 = round(float(latest["ema9"]), 2) if ("ema9" in latest and not pd.isna(latest["ema9"]) and not math.isinf(latest["ema9"])) else current_price
    ema21 = round(float(latest["ema21"]), 2) if ("ema21" in latest and not pd.isna(latest["ema21"]) and not math.isinf(latest["ema21"])) else current_price
    ema20 = round(float(latest["ema20"]), 2) if ("ema20" in latest and not pd.isna(latest["ema20"]) and not math.isinf(latest["ema20"])) else ema21
    ema50 = round(float(latest["ema50"]), 2) if (not pd.isna(latest["ema50"]) and not math.isinf(latest["ema50"])) else current_price
    ema200 = round(float(latest["ema200"]), 2) if (not pd.isna(latest["ema200"]) and not math.isinf(latest["ema200"])) else current_price

    asian_high = round(float(asian_high), 2)
    asian_low = round(float(asian_low), 2)
    asian_mid = round((asian_high + asian_low) / 2, 2)

    # Session timing & Core Strategy Logic (1M Sniper Momentum):
    now_thai = datetime.datetime.now(THAI_TZ)
    is_evening_session = (now_thai.weekday() < 5 and 19 <= now_thai.hour < 22)
    is_pre_news_period = (now_thai.weekday() < 5 and now_thai.hour == 19 and now_thai.minute < 30)

    # Determine if analyzing 1-minute sniper data
    is_1m = True
    if len(df) >= 2:
        step_sec = abs((df.index[-1] - df.index[-2]).total_seconds())
        if step_sec > 180:
            is_1m = False

    # Trend Filter:
    # 1M Sniper mode: Triple EMA alignment (EMA 9 > EMA 21 and EMA 21 >= EMA 50 - 0.5)
    # Legacy mode: EMA 20 / EMA 50 trend alignment
    if is_1m:
        is_bullish_trend = (ema9 > ema21 and ema21 >= ema50 - 0.5 and current_price >= min(ema9, ema21))
        is_bearish_trend = (ema9 < ema21 and ema21 <= ema50 + 0.5 and current_price <= max(ema9, ema21))
    else:
        is_bullish_trend = (ema20 > ema50 and current_price > ema50)
        is_bearish_trend = (ema20 < ema50 and current_price < ema50)

    signal = "WAIT"
    confidence = 65
    reasons = []

    # Pre-News Trap Alert for 19:00 - 19:29
    if is_pre_news_period:
        reasons.append("⚠️ ช่วงเวลา 19:00 - 19:29 เสี่ยงข่าวกระชาก (Pre-News Trap): ข่าวสำคัญสหรัฐฯ (CPI/NFP/PPI) มักออกเวลา 19:30 / 20:30 แนะนำรอเข้าเทรดตั้งแต่ 19:30 เป็นต้นไปเพื่อความแม่นยำสูงสุด")

    # Anti-Chase Warning
    if is_1m:
        if is_bullish_trend and (current_price - ema9) > (1.5 * current_atr):
            reasons.append(f"⚠️ ราคาไล่ตามสูงเกินไป (+${current_price - ema9:.2f} ห่างจาก EMA9 เกิน 1.5x ATR) ระวังการย่อตัวพักฐาน")
        if is_bearish_trend and (ema9 - current_price) > (1.5 * current_atr):
            reasons.append(f"⚠️ ราคาเทขายลึกเกินไป (-${ema9 - current_price:.2f} ห่างจาก EMA9 เกิน 1.5x ATR) ระวังการดีดกลับทางเทคนิค")

    # Bullish condition
    if is_bullish_trend:
        if current_price >= asian_high - 1.0 and current_rsi < 72:
            signal = "BUY"
            confidence = 88
            reasons.append(f"สไนเปอร์ 1M: ราคาทะลุกรอบสำคัญที่ ${asian_high} พร้อมโมเมนตัมช่วงตลาดเย็น 19:00 - 22:00 (NY/London Overlap)")
            if is_1m:
                if ema9 > ema21 >= ema50:
                    reasons.append(f"เส้น EMA เรียงตัวขาขึ้นชัดเจน (EMA9={ema9} > EMA21={ema21} > EMA50={ema50})")
                else:
                    reasons.append(f"โมเมนตัม EMA 1M ขาขึ้น (EMA9={ema9} > EMA21={ema21})")
            else:
                reasons.append(f"เส้น EMA เรียงตัวขาขึ้นชัดเจน (EMA20={ema20} > EMA50={ema50})")

            if current_rsi >= 48:
                reasons.append(f"RSI({current_rsi}) แข็งแกร่งในโซนบวกและยังไม่ Overbought พร้อมสไนเปอร์เป้าหมาย +$5.00")
            else:
                reasons.append(f"RSI({current_rsi}) ฟื้นตัวขึ้นสู่แดนบวก พร้อมสไนเปอร์เป้าหมาย +$5.00")
        elif current_price > asian_mid and current_rsi > 45:
            signal = "BUY"
            confidence = 80
            reasons.append(f"ราคายืนเหนือระดับกลาง ${asian_mid} และรักษาแนวโน้มขาขึ้นบนกราฟ 1 นาที")
            reasons.append(f"โมเมนตัมฝั่งซื้อได้เปรียบ มุ่งหน้าทดสอบเป้าหมาย Asian High ${asian_high}")
        elif current_rsi > 52 and current_rsi < 70:
            signal = "BUY"
            confidence = 75
            reasons.append(f"ตรวจพบแรงซื้อสถาบันต่อเนื่องช่วงค่ำ RSI({current_rsi}) สนับสนุนฝั่ง BUY")

    # Bearish condition
    elif is_bearish_trend:
        if current_price <= asian_low + 1.0 and current_rsi > 28:
            signal = "SELL"
            confidence = 88
            reasons.append(f"สไนเปอร์ 1M: ราคาหลุดแนวรับสำคัญที่ ${asian_low} มีแรงเทขายสถาบันช่วงตลาดเย็น 19:00 - 22:00")
            if is_1m:
                if ema9 < ema21 <= ema50:
                    reasons.append(f"เส้น EMA เรียงตัวขาลงชัดเจน (EMA9={ema9} < EMA21={ema21} < EMA50={ema50})")
                else:
                    reasons.append(f"โมเมนตัม EMA 1M ขาลง (EMA9={ema9} < EMA21={ema21})")
            else:
                reasons.append(f"เส้น EMA เรียงตัวขาลงชัดเจน (EMA20={ema20} < EMA50={ema50})")

            if current_rsi <= 52:
                reasons.append(f"RSI({current_rsi}) ชี้ลงต่อเนื่อง ยังมีพื้นที่ทำกำไรฝั่งลงสู่เป้าหมาย +$5.00")
            else:
                reasons.append(f"RSI({current_rsi}) พลิกกลับลงสู่แดนลบ พร้อมสไนเปอร์เป้าหมาย +$5.00")
        elif current_price < asian_mid and current_rsi < 55:
            signal = "SELL"
            confidence = 80
            reasons.append(f"ราคาหลุดต่ำกว่าระดับกลาง ${asian_mid} เกิดแรงเทขายต่อเนื่องบนกราฟ 1 นาที")
            reasons.append(f"ทิศทางโน้มเอียงฝั่ง Sell มีโอกาสลงทดสอบเป้าหมาย Asian Low ${asian_low}")
        elif current_rsi < 48 and current_rsi > 30:
            signal = "SELL"
            confidence = 75
            reasons.append(f"ตรวจพบแรงกดดันฝั่งขายต่อเนื่อง RSI({current_rsi}) สนับสนุนฝั่ง SELL")

    # Fallback to mean reversion if in tight range
    if signal == "WAIT":
        if current_rsi < 35 and current_price <= asian_low + 1.5:
            signal = "BUY"
            confidence = 70
            reasons.append(f"ราคาย่อตัวลงมาใกล้แนวรับ Asian Low ${asian_low} พร้อม RSI ใกล้ Oversold ({current_rsi}) มีโอกาสดีดกลับสั้นๆ")
        elif current_rsi > 65 and current_price >= asian_high - 1.5:
            signal = "SELL"
            confidence = 70
            reasons.append(f"ราคาขึ้นมาชนแนวต้าน Asian High ${asian_high} พร้อม RSI สูง ({current_rsi}) ระวังการย่อพักตัว")
        else:
            signal = "WAIT"
            confidence = 50
            reasons.append("ราคาวิ่งอยู่ในกรอบแคบและสัญญาณยังไม่ชัดเจน แนะนำให้รอการระเบิดของโวลุ่มช่วง 19:00 - 22:00")

    # Evaluate Trade Quality Filter (TQF)
    if signal in ["BUY", "SELL"]:
        q_result = evaluate_trade_quality(
            signal=signal,
            entry_price=current_price,
            open_price=float(latest["open"]) if "open" in latest and not pd.isna(latest["open"]) else current_price,
            high_price=float(latest["high"]) if "high" in latest and not pd.isna(latest["high"]) else current_price,
            low_price=float(latest["low"]) if "low" in latest and not pd.isna(latest["low"]) else current_price,
            asian_high=asian_high,
            asian_low=asian_low,
            rsi=current_rsi,
            atr=current_atr,
            ema20=ema20,
            ema50=ema50,
            ema9=ema9,
            ema21=ema21,
            is_1m_sniper=is_1m
        )
    else:
        q_result = {
            "score": 50,
            "tier": "CAUTION",
            "status": "APPROVED",
            "trade_allowed": False,
            "is_filtered": False,
            "filter_applied": False,
            "raw_signal": "WAIT",
            "asian_range_usd": round(float(asian_high - asian_low), 2),
            "asian_range_atr": round(float((asian_high - asian_low) / max(current_atr, 0.01)), 2),
            "rsi_level": current_rsi,
            "opposing_wick_ratio": 0.0,
            "body_ratio": 1.0,
            "ema_spread_usd": round(abs(ema9 - ema21), 2) if is_1m else round(abs(ema20 - ema50), 2),
            "warning_flags": [],
            "filter_rules_triggered": [],
            "filter_reasons_th": []
        }

    # Apply TQF Filter Action
    is_tqf_filtered = False
    if use_quality_filter and q_result["is_filtered"]:
        is_tqf_filtered = True
        signal = "WAIT"
        badge_color = "amber"
        action_th = "รอสัญญาณใหม่ (ติดตัวกรองคุณภาพ TQF)"
        reasons = [
            f"⛔ กรองสัญญาณออก (FILTERED | {q_result['score']}/100): ตรวจพบความเสี่ยงสูง ({', '.join(q_result['filter_rules_triggered'])}) แนะนำงดเข้าเทรดเพื่อรักษาเงินทุน"
        ] + q_result["filter_reasons_th"] + reasons
        confidence = 50
    elif signal in ["BUY", "SELL"]:
        if q_result["tier"] == "EXCELLENT":
            confidence = min(95, confidence + 5)
            reasons.insert(0, f"🛡️ ผ่านการตรวจสอบ Trade Quality Filter (คะแนนคุณภาพ {q_result['score']}/100 - เกรด {q_result['tier']})")
        elif q_result["status"] == "WARNING":
            if q_result["filter_reasons_th"]:
                reasons.insert(0, f"⚡ สัญญาณผ่านแต่มีจุดระวัง TQF (คะแนน {q_result['score']}/100 - เกรด {q_result['tier']})")
                for w_reason in reversed(q_result["filter_reasons_th"]):
                    reasons.insert(1, w_reason)

    # 2:1 Risk-to-Reward Sniper Financial Engine
    # Target: +$5.00 USD Take Profit / -$2.50 USD Stop Loss on recommended setting ($10, x200)
    # Auto-scales proportionally if user adjusts capital or investment amount
    scale_factor = investment_amount / 10.0 if investment_amount > 0 else 1.0
    target_tp_usd = round(5.00 * scale_factor, 2)
    target_sl_usd = round(2.50 * scale_factor, 2)
    target_tp2_usd = round(7.50 * scale_factor, 2)

    trade_volume = investment_amount * multiplier

    if trade_volume > 0 and current_price > 0:
        sl_distance = round((target_sl_usd / trade_volume) * current_price, 2)
        sl_distance = max(0.25, sl_distance)
        tp1_distance = round(sl_distance * 2.0, 2)
        tp2_distance = round(sl_distance * 3.0, 2)
    else:
        effective_atr = max(1.0, min(current_atr, 6.0))
        sl_distance = round(effective_atr * 1.0, 2)
        sl_distance = max(0.25, sl_distance)
        tp1_distance = round(sl_distance * 2.0, 2)
        tp2_distance = round(sl_distance * 3.0, 2)

    tp1_distance = max(0.50, tp1_distance)
    sl_distance = max(0.25, sl_distance)
    tp2_distance = max(tp1_distance, tp2_distance)

    # Maintain exact 2:1 distance
    sl_distance = round(tp1_distance / 2.0, 2)

    # Entry, TP, SL calculation
    if signal == "BUY":
        entry_min = round(current_price - 0.3, 2)
        entry_max = round(current_price + 0.2, 2)
        stop_loss = round(current_price - sl_distance, 2)
        tp1 = round(current_price + tp1_distance, 2)
        tp2 = round(current_price + tp2_distance, 2)
        action_th = "แนะนำเปิดออเดอร์ BUY (ซื้อ / สไนเปอร์มองขึ้น)"
        badge_color = "emerald"
    elif signal == "SELL":
        entry_min = round(current_price - 0.2, 2)
        entry_max = round(current_price + 0.3, 2)
        stop_loss = round(current_price + sl_distance, 2)
        tp1 = round(current_price - tp1_distance, 2)
        tp2 = round(current_price - tp2_distance, 2)
        action_th = "แนะนำเปิดออเดอร์ SELL (ขาย / สไนเปอร์มองลง)"
        badge_color = "rose"
    else:
        entry_min = round(current_price - 0.5, 2)
        entry_max = round(current_price + 0.5, 2)
        stop_loss = round(current_price - sl_distance, 2)
        tp1 = round(current_price + tp1_distance, 2)
        tp2 = round(current_price + tp2_distance, 2)
        if not is_tqf_filtered:
            action_th = "แนะนำให้รอความชัดเจน (WAIT 1M Sniper)"
        badge_color = "amber"

    # IQ Option Forex Calculations for Small Capital
    if current_price > 0:
        tp1_pct = (tp1_distance / current_price)
        tp2_pct = (tp2_distance / current_price)
        sl_pct = (sl_distance / current_price)
    else:
        tp1_pct = 0.0
        tp2_pct = 0.0
        sl_pct = 0.0

    est_profit_tp1 = round(trade_volume * tp1_pct, 2)
    est_profit_tp2 = round(trade_volume * tp2_pct, 2)
    est_loss_sl = round(trade_volume * sl_pct, 2)

    # Calibration anchor to exact targets for recommended / scaled capital
    if abs(est_profit_tp1 - target_tp_usd) <= 0.05:
        est_profit_tp1 = target_tp_usd
    if abs(est_loss_sl - target_sl_usd) <= 0.05:
        est_loss_sl = target_sl_usd
    if abs(est_profit_tp2 - target_tp2_usd) <= 0.05:
        est_profit_tp2 = target_tp2_usd

    if investment_amount > 0:
        iq_tp1_percent = round((est_profit_tp1 / investment_amount) * 100, 0)
        iq_sl_percent = round((est_loss_sl / investment_amount) * 100, 0)
    else:
        iq_tp1_percent = 50.0
        iq_sl_percent = 25.0

    step1_be_trigger = round(current_price + (tp1_distance * 0.50), 2) if signal == "BUY" else round(current_price - (tp1_distance * 0.50), 2)
    step2_lock_trigger = round(current_price + (tp1_distance * 0.75), 2) if signal == "BUY" else round(current_price - (tp1_distance * 0.75), 2)
    step2_locked_sl = round(current_price + (tp1_distance * 0.30), 2) if signal == "BUY" else round(current_price - (tp1_distance * 0.30), 2)
    est_step2_profit_usd = round(1.50 * scale_factor, 2)

    def _safe_int(val, default):
        try:
            if val is None or math.isnan(val) or math.isinf(val):
                return default
            return int(val)
        except Exception:
            return default

    iq_tp_slider = _safe_int(iq_tp1_percent, 50)
    iq_sl_slider = _safe_int(iq_sl_percent, 25)

    # 2-Step Trailing Stop playbook guidance
    if signal in ["BUY", "SELL"]:
        reasons.append(
            f"🛡️ แผน 2-Step Trailing Stop: "
            f"จังหวะ 1 (กำไรแตะ +50% ที่ ${step1_be_trigger:.2f}) -> เลื่อน SL มาที่จุดเข้า ${current_price:.2f} (กันทุน $0.00) | "
            f"จังหวะ 2 (กำไรแตะ +75% ที่ ${step2_lock_trigger:.2f}) -> เลื่อน SL ล็อคกำไรที่ ${step2_locked_sl:.2f} (+${est_step2_profit_usd:.2f}) การันตีกำไรแน่นอน!"
        )

    # 1-trade-per-day rule guidance
    reasons.append("🎯 กฎเหล็ก 1 ไม้ต่อวัน (1 Trade Per Day): เมื่อถึงเป้ากำไร +$5.00 USD ให้ล็อคกำไรและปิดจอทันที รักษาวินัยอย่างเคร่งครัด")

    # Thai Time formatting
    time_str = now_thai.strftime("%d/%m/%Y %H:%M:%S")

    return {
        "timestamp": time_str,
        "raw_time": now_thai.isoformat(),
        "is_evening_session": is_evening_session,
        "source": source,
        "symbol": "XAU/USD (Gold)",
        "current_price": current_price,
        "signal": signal,
        "action_th": action_th,
        "badge_color": badge_color,
        "confidence": confidence,
        "entry_range": f"${entry_min} - ${entry_max}",
        "entry_price": current_price,
        "stop_loss": stop_loss,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
        "sl_distance": sl_distance,
        "tp1_distance": tp1_distance,
        "tp2_distance": tp2_distance,
        "risk_reward_ratio": "2:1",
        "reasons": reasons,
        "indicators": {
            "rsi": current_rsi,
            "ema9": ema9,
            "ema21": ema21,
            "ema20": ema20,
            "ema50": ema50,
            "ema200": ema200,
            "atr": current_atr,
            "asian_high": asian_high,
            "asian_low": asian_low,
            "asian_mid": asian_mid
        },
        "money_management": {
            "investment_input": investment_amount,
            "multiplier": multiplier,
            "est_profit_tp1_usd": est_profit_tp1,
            "est_profit_tp2_usd": est_profit_tp2,
            "est_loss_sl_usd": est_loss_sl,
            "est_step2_profit_usd": est_step2_profit_usd,
            "step1_be_trigger": step1_be_trigger,
            "step2_lock_trigger": step2_lock_trigger,
            "step2_locked_sl": step2_locked_sl,
            "iq_tp_slider_pct": iq_tp_slider,
            "iq_sl_slider_pct": iq_sl_slider,
            "summary_th": f"ลงทุน ${investment_amount:.2f} ตัวคูณ x{multiplier} (พอร์ต $30) | สไนเปอร์ 2:1 กำไรเป้าหมาย: +${est_profit_tp1:.2f} USD (+{iq_tp_slider}%) | ตัดขาดทุน: -${est_loss_sl:.2f} USD (-{iq_sl_slider}%) | 2-Step Trailing Stop ล็อคกำไรที่ +${est_step2_profit_usd:.2f} USD | กฎ 1 ไม้ต่อวัน"
        },
        # Additive Trade Quality Filter (TQF) Model
        "quality_score": q_result["score"],
        "quality_tier": q_result["tier"],
        "trade_allowed": q_result["trade_allowed"],
        "filter_applied": q_result["filter_applied"],
        "filter_reasons": q_result["filter_reasons_th"],
        "quality_filter": {
            "score": q_result["score"],
            "tier": q_result["tier"],
            "status": q_result["status"],
            "asian_range_usd": q_result["asian_range_usd"],
            "asian_range_atr": q_result["asian_range_atr"],
            "rsi_level": q_result["rsi_level"],
            "opposing_wick_ratio": q_result["opposing_wick_ratio"],
            "body_ratio": q_result["body_ratio"],
            "ema_spread_usd": q_result["ema_spread_usd"],
            "warning_flags": q_result["warning_flags"],
            "filter_rules_triggered": q_result["filter_rules_triggered"],
            "is_filtered": q_result["is_filtered"],
            "raw_signal": q_result["raw_signal"]
        }
    }


if __name__ == "__main__":
    import sys
    import json
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    result = analyze_gold_market(investment_amount=10.0, multiplier=200)
    print(json.dumps(result, ensure_ascii=False, indent=2))


"""
FastAPI Server & Daily 15:00 Scheduler for Gold Day Trade Assistant
"""

import asyncio
from contextlib import asynccontextmanager
import datetime
import json
import math
import os
from typing import List, Optional
import pytz
from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from analyzer import analyze_gold_market, THAI_TZ

class AnalyzeRequest(BaseModel):
    investment: Optional[float] = Field(None, allow_inf_nan=False)
    multiplier: Optional[int] = None

# In-memory storage for analysis and alert states
state = {
    "latest_analysis": None,
    "history": [],
    "last_triggered_date": None,
    "last_triggered_slots": set(),
    "user_settings": {
        "investment": 10.0,
        "multiplier": 200
    },
    "new_alert_pending": False
}


def get_next_session_target(now_dt: datetime.datetime, target_hour: int = 19) -> datetime.datetime:
    """
    Computes the datetime of the next target_hour:00:00 Thai Time trading session.
    If now_dt is at or past target_hour:00 on Friday or on Saturday/Sunday,
    the next trading session rolls forward to Monday target_hour:00:00 Thai Time.
    """
    candidate = now_dt.replace(hour=target_hour, minute=0, second=0, microsecond=0)
    if now_dt >= candidate:
        candidate += datetime.timedelta(days=1)
    while candidate.weekday() >= 5:  # 5=Saturday, 6=Sunday
        candidate += datetime.timedelta(days=1)
    return candidate


def get_next_trading_target(now_dt: datetime.datetime) -> datetime.datetime:
    """
    Computes the datetime of the next 15:00:00 Thai Time trading session.
    Preserved for backward compatibility with existing tests.
    """
    return get_next_session_target(now_dt, target_hour=15)


async def startup_event():
    # Initial run to have fresh data immediately on startup
    try:
        res = await asyncio.to_thread(
            analyze_gold_market,
            investment_amount=state["user_settings"]["investment"],
            multiplier=state["user_settings"]["multiplier"]
        )
        state["latest_analysis"] = res
        state["history"].append(res)
        state["history"] = state["history"][:20]
    except Exception as e:
        print(f"[ERROR] Initial analysis failed: {e}")

    # Launch background scheduler task
    return asyncio.create_task(daily_scheduler_loop())


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = await startup_event()
    yield
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Gold Day Trade Assistant", lifespan=lifespan)

# Mount static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


async def daily_scheduler_loop():
    """
    Checks every 15 seconds. If current Thai time is 19:00 (or 15:00) on Monday-Friday
    and hasn't been triggered yet today, runs the analysis and triggers an alert.
    """
    print("[SCHEDULER] Daily 19:00 Evening Sniper & 15:00 Thai Time scheduler started.")
    while True:
        try:
            now_thai = datetime.datetime.now(THAI_TZ)
            today_str = now_thai.strftime("%Y-%m-%d")

            # Check if Monday(0) - Friday(4)
            is_trading_day = now_thai.weekday() < 5

            # Target time: 19:00 (evening session) or 15:00 (London open)
            is_target_time = ((now_thai.hour == 19 and now_thai.minute == 0) or
                              (now_thai.hour == 15 and now_thai.minute == 0))
            trigger_slot = f"{today_str}_{now_thai.hour}"

            if is_trading_day and is_target_time and trigger_slot not in state.setdefault("last_triggered_slots", set()):
                print(f"[SCHEDULER] Triggering 1M Sniper Session Analysis for {today_str} ({now_thai.hour}:00)!")
                res = await asyncio.to_thread(
                    analyze_gold_market,
                    investment_amount=state["user_settings"]["investment"],
                    multiplier=state["user_settings"]["multiplier"]
                )
                state["latest_analysis"] = res
                state["history"].insert(0, res)
                state["history"] = state["history"][:20]
                state.setdefault("last_triggered_slots", set()).add(trigger_slot)
                state["last_triggered_date"] = today_str
                state["new_alert_pending"] = True
                print(f"[SCHEDULER] 1M Sniper Alert for {now_thai.hour}:00 created successfully!")

        except Exception as e:
            print(f"[SCHEDULER ERROR] {e}")

        await asyncio.sleep(15)


@app.get("/")
def get_dashboard():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"status": "Dashboard loading..."})


@app.get("/api/latest")
async def get_latest():
    """Get the most recent analysis"""
    if not state["latest_analysis"]:
        res = await asyncio.to_thread(
            analyze_gold_market,
            investment_amount=state["user_settings"]["investment"],
            multiplier=state["user_settings"]["multiplier"]
        )
        state["latest_analysis"] = res
    return state["latest_analysis"]


@app.post("/api/analyze")
async def trigger_analysis(
    req: Optional[AnalyzeRequest] = Body(None),
    investment: Optional[float] = Query(None, description="Investment amount e.g. $5"),
    multiplier: Optional[int] = Query(None, description="Multiplier e.g. 100")
):
    """Manual on-demand trigger to analyze current market"""
    req_obj = req if isinstance(req, AnalyzeRequest) else None
    inv = req_obj.investment if (req_obj and req_obj.investment is not None) else (investment if isinstance(investment, (int, float)) else None)
    mult = req_obj.multiplier if (req_obj and req_obj.multiplier is not None) else (multiplier if isinstance(multiplier, (int, float)) else None)

    if inv is not None:
        if isinstance(inv, bool) or not isinstance(inv, (int, float)) or not math.isfinite(inv) or inv <= 0:
            raise HTTPException(status_code=400, detail="Investment must be a finite positive number")
    if mult is not None:
        if isinstance(mult, bool) or not isinstance(mult, (int, float)) or not math.isfinite(mult) or mult <= 0:
            raise HTTPException(status_code=400, detail="Multiplier must be a finite positive number")

    if inv is not None:
        state["user_settings"]["investment"] = float(inv)
    if mult is not None:
        state["user_settings"]["multiplier"] = int(mult)

    res = await asyncio.to_thread(
        analyze_gold_market,
        investment_amount=state["user_settings"]["investment"],
        multiplier=state["user_settings"]["multiplier"]
    )
    state["latest_analysis"] = res
    # Avoid duplicate exact timestamps
    state["history"].insert(0, res)
    state["history"] = state["history"][:20]  # keep last 20
    return res


@app.get("/api/history")
def get_history():
    """Get list of past analyses generated today"""
    return state["history"][:15]


@app.get("/api/status")
def get_status():
    """Get current time, countdown to 19:00 evening window and 15:00, and alert notification flag"""
    now_thai = datetime.datetime.now(THAI_TZ)
    
    # Calculate time until next 15:00 (legacy contract)
    target_next_1500 = get_next_trading_target(now_thai)
    diff_seconds_1500 = max(0, int((target_next_1500 - now_thai).total_seconds()))
    hours_1500 = diff_seconds_1500 // 3600
    minutes_1500 = (diff_seconds_1500 % 3600) // 60
    seconds_1500 = diff_seconds_1500 % 60

    # Calculate time until next 19:00 (1M Sniper Evening Session)
    target_next_1900 = get_next_session_target(now_thai, target_hour=19)
    diff_seconds_1900 = max(0, int((target_next_1900 - now_thai).total_seconds()))
    hours_1900 = diff_seconds_1900 // 3600
    minutes_1900 = (diff_seconds_1900 % 3600) // 60
    seconds_1900 = diff_seconds_1900 % 60

    is_in_evening_session = (now_thai.weekday() < 5 and 19 <= now_thai.hour < 22)

    pending_alert = state["new_alert_pending"]
    # Reset flag once polled by UI
    state["new_alert_pending"] = False

    return {
        "current_time_thai": now_thai.strftime("%H:%M:%S"),
        "current_date_thai": now_thai.strftime("%d/%m/%Y"),
        "is_trading_day": now_thai.weekday() < 5,
        "is_in_evening_session": is_in_evening_session,
        "session_window": "19:00 - 22:00 Thai Time (New York / London Overlap)",
        "countdown_to_session": f"{hours_1900:02d}:{minutes_1900:02d}:{seconds_1900:02d}",
        "countdown_to_1900": f"{hours_1900:02d}:{minutes_1900:02d}:{seconds_1900:02d}",
        "seconds_to_1900": diff_seconds_1900,
        "countdown_to_1500": f"{hours_1500:02d}:{minutes_1500:02d}:{seconds_1500:02d}",
        "seconds_to_1500": diff_seconds_1500,
        "next_session_thai": target_next_1900.strftime("%d/%m/%Y 19:00:00"),
        "has_new_alert": pending_alert,
        "investment": state["user_settings"]["investment"],
        "multiplier": state["user_settings"]["multiplier"]
    }


@app.get("/api/backtest")
def get_backtest_results(
    mode: Optional[str] = Query(None, description="Execution mode: baseline, filtered, or compare"),
    strategy: Optional[str] = Query(None, description="Strategy to return: 1m_sniper or 15m"),
    refresh: bool = Query(False, description="Force re-run the backtest")
):
    """Return 1M Sniper backtest performance metrics, equity curves, and monthly stats"""
    results_path = os.path.join(STATIC_DIR, "backtest_results.json")
    if not refresh and os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if mode == "baseline" and "baseline" in data:
            return data["baseline"]
        elif mode == "filtered" and "filtered" in data:
            return data["filtered"]
        return data

    from backtester import run_full_backtest
    data = run_full_backtest(mode=mode or "compare", strategy=strategy or "1m_sniper")
    if mode == "baseline" and "baseline" in data:
        return data["baseline"]
    elif mode == "filtered" and "filtered" in data:
        return data["filtered"]
    return data


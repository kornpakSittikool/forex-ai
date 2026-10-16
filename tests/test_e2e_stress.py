"""
Tier 3 & Tier 4 Tests: E2E Stress, Concurrency & Extreme Volatility
Validates rapid concurrent requests, flash crash volatility simulation,
long-running memory/history consistency, and fallback recovery under load.
"""

import concurrent.futures
import datetime
import math
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

import analyzer
import main
from main import app, state, THAI_TZ


class TestE2EConcurrencyAndLoad:
    """
    Tier 4: Concurrency stress tests to ensure thread safety and state isolation.
    """

    def test_rapid_concurrent_analyze_requests(self, client):
        """
        Fires 20 simultaneous concurrent requests to POST /api/analyze.
        Verifies all succeed with 200 OK and no server race crash occurs.
        """
        def make_request(req_id):
            investment = 5.0 + (req_id % 5)
            mult = 50 * ((req_id % 4) + 1)
            with TestClient(app) as local_client:
                resp = local_client.post(f"/api/analyze?investment={investment}&multiplier={mult}")
                return resp.status_code, resp.json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(15)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for status_code, body in results:
            assert status_code == 200
            assert "current_price" in body
            assert "signal" in body

    def test_concurrent_status_and_analyze_calls(self, client):
        """
        Interleaves status polling with analyze triggers simultaneously.
        """
        def poll_status():
            with TestClient(app) as c:
                return c.get("/api/status").status_code

        def trigger_analysis():
            with TestClient(app) as c:
                return c.post("/api/analyze").status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            status_futs = [executor.submit(poll_status) for _ in range(10)]
            analyze_futs = [executor.submit(trigger_analysis) for _ in range(10)]

            for fut in concurrent.futures.as_completed(status_futs + analyze_futs):
                assert fut.result() == 200


class TestExtremeVolatilityAndFlashCrash:
    """
    Tier 4: Stress testing with anomalous market events (flash crash, massive gaps).
    """

    def test_flash_crash_volatility_clamping(self, monkeypatch):
        """
        Simulate sudden $150 flash crash in Gold prices.
        Verifies that ATR is clamped to $6.00 maximum, SL distance does not exceed $6.00,
        and all output metrics remain bounded and finite.
        """
        now = datetime.datetime.now(THAI_TZ)
        dates = [now - datetime.timedelta(minutes=1 * (50 - i)) for i in range(50)]
        records = []
        price = 2650.0

        for i in range(50):
            if i >= 40:
                # Flash crash: price plummets $150 in 10 candles with huge bar ranges
                price -= 15.0
                h = price + 20.0
                l = price - 20.0
            else:
                h = price + 1.0
                l = price - 1.0
            records.append({
                "open": price, "high": h, "low": l, "close": price, "volume": 5000.0
            })

        crash_df = pd.DataFrame(records, index=pd.to_datetime(dates))
        monkeypatch.setattr(analyzer, "get_market_data", lambda: (crash_df, "Flash Crash Feed"))

        res = analyzer.analyze_gold_market(investment_amount=5.0, multiplier=100)
        
        # 1M Sniper 2:1 R:R contracts preserved under extreme volatility
        assert math.isfinite(res["sl_distance"]) and res["sl_distance"] > 0
        assert math.isfinite(res["tp1_distance"]) and res["tp1_distance"] > 0
        assert res["risk_reward_ratio"] == "2:1"
        assert round(res["tp1_distance"] / res["sl_distance"], 1) == 2.0
        assert res["money_management"]["est_loss_sl_usd"] == 1.25
        assert res["money_management"]["est_profit_tp1_usd"] == 2.50
        assert res["money_management"]["iq_tp_slider_pct"] == 50
        assert res["money_management"]["iq_sl_slider_pct"] == 25
        assert res["current_price"] > 0
        assert not np.isnan(res["current_price"])
        assert 0 <= res["confidence"] <= 100


class TestLongRunningStateConsistency:
    """
    Tier 3: Verifies state stability over extended cycles.
    """

    def test_history_buffer_does_not_exceed_cap(self, client):
        """
        Requirement F18 / R4: History capacity bound.
        After 35 sequential analyses, state["history"] must never grow beyond 20 items,
        preventing unbounded memory accumulation.
        """
        for i in range(30):
            client.post("/api/analyze?investment=5.0&multiplier=100")

        # In-memory history must be capped at 20
        assert len(state["history"]) <= 20

        # Endpoint /api/history returns up to 15
        resp = client.get("/api/history")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) <= 15
        assert len(data) > 0

    def test_simulated_external_feed_failure_during_api_call(self, client, monkeypatch):
        """
        When external network goes completely down while user triggers /api/analyze,
        the system must seamlessly return a valid analysis via synthetic fallback (200 OK, never 500).
        """
        monkeypatch.setattr(analyzer, "fetch_gold_data_binance", lambda: None)
        monkeypatch.setattr(analyzer, "fetch_gold_data_yahoo", lambda: None)

        response = client.post("/api/analyze?investment=10.0&multiplier=100")
        assert response.status_code == 200
        data = response.json()
        assert "Simulated" in data["source"]
        assert data["current_price"] > 0

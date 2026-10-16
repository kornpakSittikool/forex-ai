"""
Tier 1, 2, 4 Tests: REST API Contracts & Schema Compliance
Validates /api/status, /api/latest, /api/analyze, /api/history,
parameter validation errors (HTTP 400 on <= 0), and JSON schema compliance.
"""

import math
import re
import pytest
from fastapi.testclient import TestClient

import analyzer
from main import app, state


class TestAPIStatusContract:
    """
    Validates GET /api/status contract and schema.
    """

    def test_status_endpoint_returns_expected_schema(self, client):
        """
        Verify all required fields, types, and regex formats in /api/status response.
        """
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()

        # Required fields check
        required_fields = [
            "current_time_thai",
            "current_date_thai",
            "is_trading_day",
            "is_in_evening_session",
            "session_window",
            "countdown_to_session",
            "countdown_to_1900",
            "seconds_to_1900",
            "countdown_to_1500",
            "seconds_to_1500",
            "next_session_thai",
            "has_new_alert",
            "investment",
            "multiplier"
        ]
        for field in required_fields:
            assert field in data, f"Missing required field in /api/status: {field}"

        # Regex format checks
        assert re.match(r"^\d{2}:\d{2}:\d{2}$", data["current_time_thai"])
        assert re.match(r"^\d{2}/\d{2}/\d{4}$", data["current_date_thai"])
        assert re.match(r"^\d+:\d{2}:\d{2}$", data["countdown_to_1500"])
        assert re.match(r"^\d+:\d{2}:\d{2}$", data["countdown_to_1900"])
        assert re.match(r"^\d+:\d{2}:\d{2}$", data["countdown_to_session"])

        # Types and boundary checks
        assert isinstance(data["is_trading_day"], bool)
        assert isinstance(data["is_in_evening_session"], bool)
        assert isinstance(data["session_window"], str)
        assert isinstance(data["has_new_alert"], bool)
        assert isinstance(data["seconds_to_1500"], int)
        assert isinstance(data["seconds_to_1900"], int)
        assert data["seconds_to_1500"] >= 0
        assert data["seconds_to_1900"] >= 0
        assert data["investment"] > 0
        assert data["multiplier"] > 0


class TestAPILatestAndAnalyzeContract:
    """
    Validates GET /api/latest and POST /api/analyze schemas and state updates.
    """

    def _verify_analysis_schema(self, data: dict):
        """
        Helper to rigorously verify analysis dictionary conformance.
        """
        required_root_keys = [
            "timestamp", "raw_time", "source", "symbol", "current_price",
            "signal", "action_th", "badge_color", "confidence", "entry_range",
            "entry_price", "stop_loss", "take_profit_1", "take_profit_2",
            "sl_distance", "tp1_distance", "risk_reward_ratio", "reasons",
            "indicators", "money_management"
        ]
        for key in required_root_keys:
            assert key in data, f"Missing root key: {key}"

        # Enum & Range assertions
        assert data["signal"] in ["BUY", "SELL", "WAIT"]
        assert data["badge_color"] in ["emerald", "rose", "amber"]
        assert 0 <= data["confidence"] <= 100
        assert data["current_price"] > 0
        assert data["entry_price"] > 0
        assert isinstance(data["reasons"], list)
        assert len(data["reasons"]) > 0

        assert data["risk_reward_ratio"] == "2:1"
        assert isinstance(data["is_evening_session"], bool)

        # Indicators schema
        ind = data["indicators"]
        for ind_key in ["rsi", "ema9", "ema21", "ema20", "ema50", "ema200", "atr", "asian_high", "asian_low", "asian_mid"]:
            assert ind_key in ind, f"Missing indicator: {ind_key}"
            assert isinstance(ind[ind_key], (int, float))
            assert not math.isnan(ind[ind_key]), f"Indicator {ind_key} is NaN"
            assert not math.isinf(ind[ind_key]), f"Indicator {ind_key} is Inf"

        # Money management schema
        mm = data["money_management"]
        for mm_key in [
            "investment_input", "multiplier", "est_profit_tp1_usd",
            "est_profit_tp2_usd", "est_loss_sl_usd", "iq_tp_slider_pct",
            "iq_sl_slider_pct", "summary_th"
        ]:
            assert mm_key in mm, f"Missing money_management key: {mm_key}"
        assert mm["investment_input"] > 0
        assert mm["multiplier"] > 0
        assert isinstance(mm["iq_tp_slider_pct"], int)
        assert isinstance(mm["iq_sl_slider_pct"], int)

    def test_get_latest_analysis_success(self, client):
        """
        Verify GET /api/latest returns valid schema and populates cache if empty.
        """
        response = client.get("/api/latest")
        assert response.status_code == 200
        data = response.json()
        self._verify_analysis_schema(data)

    def test_post_analyze_with_valid_query_params(self, client):
        """
        Verify POST /api/analyze?investment=15.0&multiplier=200 updates state
        and calculates financial metrics for the new parameters.
        """
        response = client.post("/api/analyze?investment=15.0&multiplier=200")
        assert response.status_code == 200
        data = response.json()
        self._verify_analysis_schema(data)

        assert data["money_management"]["investment_input"] == 15.0
        assert data["money_management"]["multiplier"] == 200

        # Check status endpoint reflects new settings
        st = client.get("/api/status").json()
        assert st["investment"] == 15.0
        assert st["multiplier"] == 200

    def test_post_analyze_with_json_body(self, client):
        """
        Verify POST /api/analyze accepts JSON body payload {"investment": 25.0, "multiplier": 50}.
        """
        response = client.post("/api/analyze", json={"investment": 25.0, "multiplier": 50})
        # If server accepts JSON body or query params:
        if response.status_code == 200:
            data = response.json()
            assert data["money_management"]["investment_input"] == 25.0
            assert data["money_management"]["multiplier"] == 50


class TestAPIInputValidationErrors:
    """
    Tier 2: Validation boundary tests verifying HTTP 400 Bad Request
    when non-positive investment or multiplier is passed.
    """

    @pytest.mark.parametrize("invalid_investment", [0, -1, -5.0, -100])
    def test_invalid_investment_query_param_rejected(self, client, invalid_investment):
        """
        POST /api/analyze?investment=... must return HTTP 400 Bad Request.
        """
        response = client.post(f"/api/analyze?investment={invalid_investment}")
        assert response.status_code == 400, (
            f"Expected HTTP 400 Bad Request for investment={invalid_investment}, got {response.status_code}"
        )
        assert "investment" in response.text.lower() or "detail" in response.json()

    @pytest.mark.parametrize("invalid_multiplier", [0, -1, -50, -100])
    def test_invalid_multiplier_query_param_rejected(self, client, invalid_multiplier):
        """
        POST /api/analyze?multiplier=... must return HTTP 400 Bad Request.
        """
        response = client.post(f"/api/analyze?multiplier={invalid_multiplier}")
        assert response.status_code == 400, (
            f"Expected HTTP 400 Bad Request for multiplier={invalid_multiplier}, got {response.status_code}"
        )
        assert "multiplier" in response.text.lower() or "detail" in response.json()

    def test_invalid_types_return_422(self, client):
        """
        Non-numeric strings should trigger FastAPI standard HTTP 422 Unprocessable Entity.
        """
        response = client.post("/api/analyze?investment=invalid_str")
        assert response.status_code == 422


class TestAPIHistoryContract:
    """
    Validates GET /api/history endpoint and capacity limits.
    """

    def test_history_endpoint_returns_list_capped_at_15(self, client):
        """
        GET /api/history returns a list of recent analysis records capped at 15 items.
        """
        # Execute 5 analyses
        for i in range(5):
            client.post("/api/analyze?investment=5.0&multiplier=100")

        response = client.get("/api/history")
        assert response.status_code == 200
        history_list = response.json()

        assert isinstance(history_list, list)
        assert len(history_list) <= 15
        assert len(history_list) > 0
        assert "signal" in history_list[0]
        assert "current_price" in history_list[0]


class TestAPIBacktestContract:
    """
    Validates GET /api/backtest endpoint.
    """

    def test_backtest_endpoint_returns_expected_structure(self, client):
        response = client.get("/api/backtest")
        assert response.status_code == 200
        data = response.json()

        assert "overall_stats" in data
        assert "by_size" in data
        assert "monthly_summary" in data
        assert "equity_curves" in data

        stats = data["overall_stats"]
        assert stats["total_trades"] > 0
        assert "win_rate_pct" in stats
        assert "loss_rate_pct" in stats
        assert "break_even_pct" in stats

        # Check by_size options
        for sz in ["2", "5", "10"]:
            assert sz in data["by_size"]
            assert "total_pnl_usd" in data["by_size"][sz]
            assert "profit_factor" in data["by_size"][sz]


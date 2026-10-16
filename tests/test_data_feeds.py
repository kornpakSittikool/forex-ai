"""
Tier 1, 2, 3 Tests: Data Feed Resilience & Fallback Hierarchy
Validates Binance PAXG/USDT, Yahoo Finance fallback, and Synthetic failover,
including timeout handling, HTTP errors, and malformed payloads.
"""

import pytest
import requests
import pandas as pd
import numpy as np

import analyzer
from analyzer import fetch_gold_data_binance, fetch_gold_data_yahoo, get_market_data, THAI_TZ


class TestDataFeedHierarchy:
    """
    Validates the 3-tier hierarchy:
    Tier 1: Binance PAXG
    Tier 2: Yahoo Finance
    Tier 3: Synthetic Feed
    """

    def test_binance_success_returns_live_spot_gold(self, monkeypatch, mock_binance_klines):
        """
        Tier 1 Primary Feed: Binance returns 100 klines -> produces tz-aware DataFrame.
        """
        class MockResponse:
            status_code = 200
            def json(self):
                return mock_binance_klines

        monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse())

        df, source = get_market_data()
        assert "Binance" in source
        assert len(df) == 100
        assert all(col in df.columns for col in ["open", "high", "low", "close", "volume"])
        assert df.index.tz is not None

    def test_binance_fail_falls_back_to_yahoo(self, monkeypatch):
        """
        When Binance fails (HTTP 500), get_market_data falls back to Yahoo Finance.
        """
        # Binance fails
        monkeypatch.setattr(analyzer, "fetch_gold_data_binance", lambda: None)
        
        # Yahoo succeeds
        now = pd.Timestamp.now(tz=THAI_TZ)
        dates = [now - pd.Timedelta(minutes=15 * (50 - i)) for i in range(50)]
        records = [{
            "open": 2650.0, "high": 2655.0, "low": 2648.0, "close": 2652.0, "volume": 100
        } for _ in range(50)]
        mock_yahoo_df = pd.DataFrame(records, index=pd.to_datetime(dates))
        
        monkeypatch.setattr(analyzer, "fetch_gold_data_yahoo", lambda: mock_yahoo_df)

        df, source = get_market_data()
        assert "Yahoo" in source
        assert len(df) >= 30
        assert not df.empty

    def test_all_external_feeds_fail_falls_back_to_synthetic(self, monkeypatch):
        """
        When both Binance and Yahoo Finance are unreachable,
        get_market_data falls back to Simulated Gold Feed without raising an exception.
        """
        monkeypatch.setattr(analyzer, "fetch_gold_data_binance", lambda: None)
        monkeypatch.setattr(analyzer, "fetch_gold_data_yahoo", lambda: None)

        df, source = get_market_data()
        assert "Simulated" in source
        assert len(df) >= 30
        assert all(col in df.columns for col in ["open", "high", "low", "close", "volume"])
        # Price must be in a realistic gold range (~2600-2700)
        assert 2400.0 <= df.iloc[-1]["close"] <= 2900.0


class TestDataFeedResilienceAndExceptions:
    """
    Tier 2 & 3: Resilience under network faults, timeouts, and malformed responses.
    """

    def test_binance_timeout_handled_gracefully(self, monkeypatch):
        """
        Simulate requests.Timeout (> 8.0s) on Binance call; verify fetch_gold_data_binance
        returns None without crashing.
        """
        def timeout_get(*args, **kwargs):
            raise requests.exceptions.Timeout("Connection timed out after 8 seconds")

        monkeypatch.setattr(requests, "get", timeout_get)
        result = fetch_gold_data_binance()
        assert result is None

    def test_binance_connection_error_handled_gracefully(self, monkeypatch):
        """
        Simulate network disconnect (ConnectionError); verify clean handling.
        """
        def conn_err_get(*args, **kwargs):
            raise requests.exceptions.ConnectionError("Network disconnected")

        monkeypatch.setattr(requests, "get", conn_err_get)
        result = fetch_gold_data_binance()
        assert result is None

    def test_binance_http_error_status_handled(self, monkeypatch):
        """
        Simulate Binance 429 Rate Limit or 503 Service Unavailable.
        """
        class Mock429Response:
            status_code = 429
            def json(self):
                return {"code": -1003, "msg": "Too many requests"}

        monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Mock429Response())
        result = fetch_gold_data_binance()
        assert result is None

    def test_binance_malformed_json_payload(self, monkeypatch):
        """
        Simulate corrupted / unexpected payload (e.g. non-array or missing fields).
        """
        class MockBadResponse:
            status_code = 200
            def json(self):
                return {"error": "Invalid symbol"}

        monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockBadResponse())
        result = fetch_gold_data_binance()
        assert result is None

    def test_binance_insufficient_rows_triggers_fallback(self, monkeypatch):
        """
        If Binance returns only 5 rows (< 30 required threshold),
        get_market_data must not accept it and must fall back.
        """
        # 5 rows returned
        few_klines = [
            [1713168000000 + i * 900000, "2650.0", "2652.0", "2648.0", "2651.0", "10", 1713168899999]
            for i in range(5)
        ]
        class MockShortResponse:
            status_code = 200
            def json(self):
                return few_klines

        monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockShortResponse())
        monkeypatch.setattr(analyzer, "fetch_gold_data_yahoo", lambda: None)

        df, source = get_market_data()
        # Should have fallen back to synthetic since Binance had < 30 rows
        assert "Simulated" in source
        assert len(df) >= 30

    def test_get_market_data_never_raises_exception(self, monkeypatch):
        """
        Robustness invariant: get_market_data() must NEVER raise any exception to callers.
        """
        monkeypatch.setattr(analyzer, "fetch_gold_data_binance", lambda: (_ for _ in ()).throw(RuntimeError("Catastrophic error")))
        monkeypatch.setattr(analyzer, "fetch_gold_data_yahoo", lambda: (_ for _ in ()).throw(RuntimeError("Catastrophic error")))

        try:
            df, source = get_market_data()
            assert df is not None
            assert len(df) >= 30
        except Exception as e:
            pytest.fail(f"get_market_data raised unexpected exception: {e}")

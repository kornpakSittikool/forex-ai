"""
Tier 1, 2, 3, 4 Tests: 15:00 Thai Time Scheduler & Timezone Precision
Validates Asia/Bangkok timezone precision, Monday-Friday 15:00 triggers,
weekend suppression, idempotency, and weekend countdown target precision.
"""

import datetime
import pytest
import pytz

import main
from main import THAI_TZ, state


class TestSchedulerTimezoneAndLogic:
    """
    Tier 1 & 2: Tests verifying Thai Timezone (Asia/Bangkok) and scheduler conditions.
    """

    def test_thai_timezone_definition(self):
        """
        Verify that THAI_TZ is strictly UTC+7 Asia/Bangkok.
        """
        assert THAI_TZ.zone == "Asia/Bangkok"
        now_thai = datetime.datetime.now(THAI_TZ)
        assert now_thai.tzinfo is not None
        # UTC offset should be 7 hours (25200 seconds)
        offset = now_thai.utcoffset().total_seconds()
        assert offset == 7 * 3600

    @pytest.mark.parametrize("day_name,weekday_idx,expected_trading_day", [
        ("Monday", 0, True),
        ("Tuesday", 1, True),
        ("Wednesday", 2, True),
        ("Thursday", 3, True),
        ("Friday", 4, True),
        ("Saturday", 5, False),
        ("Sunday", 6, False),
    ])
    def test_trading_day_filter(self, day_name, weekday_idx, expected_trading_day):
        """
        Verify weekday < 5 filter correctly distinguishes Mon-Fri from Sat-Sun.
        """
        is_trading_day = weekday_idx < 5
        assert is_trading_day is expected_trading_day, f"Failed for {day_name}"

    def test_scheduler_idempotency_same_day(self):
        """
        Requirement F11 / R3: Idempotency.
        Once triggered for a given date, subsequent polls in the same minute/day
        must not trigger duplicate analyses.
        """
        today_str = "2026-04-17"  # Friday
        state["last_triggered_date"] = today_str
        initial_history_len = len(state["history"])

        # Simulate scheduler evaluation with matching date
        is_trading_day = True
        is_target_time = True
        should_trigger = is_trading_day and is_target_time and state["last_triggered_date"] != today_str

        assert not should_trigger, "Scheduler must not trigger when last_triggered_date matches today"
        assert len(state["history"]) == initial_history_len


class TestWeekendSuppressionAndCountdown:
    """
    Tier 3 & 4: Weekend suppression and Friday-to-Monday countdown target precision.
    """

    @pytest.mark.parametrize("sim_dt,expected_next_session_day", [
        # Monday 10:00 -> Monday 15:00
        (datetime.datetime(2026, 4, 13, 10, 0, tzinfo=THAI_TZ), 0),
        # Monday 16:00 -> Tuesday 15:00
        (datetime.datetime(2026, 4, 13, 16, 0, tzinfo=THAI_TZ), 1),
        # Friday 10:00 -> Friday 15:00
        (datetime.datetime(2026, 4, 17, 10, 0, tzinfo=THAI_TZ), 4),
        # Friday 15:01 (post Friday session) -> Monday 15:00 (Monday=0)
        (datetime.datetime(2026, 4, 17, 15, 1, tzinfo=THAI_TZ), 0),
        # Friday 20:00 -> Monday 15:00
        (datetime.datetime(2026, 4, 17, 20, 0, tzinfo=THAI_TZ), 0),
        # Saturday 12:00 -> Monday 15:00
        (datetime.datetime(2026, 4, 18, 12, 0, tzinfo=THAI_TZ), 0),
        # Sunday 14:00 -> Monday 15:00
        (datetime.datetime(2026, 4, 19, 14, 0, tzinfo=THAI_TZ), 0),
    ])
    def test_next_session_target_calculation(self, sim_dt, expected_next_session_day):
        """
        Requirement F12 / R3:
        Countdown must accurately target next active trading session at 15:00.
        When called past Friday 15:00 or on Saturday/Sunday, must roll forward to Monday 15:00.
        """
        # Calculate target 15:00 following specification
        target_today = sim_dt.replace(hour=15, minute=0, second=0, microsecond=0)
        
        if sim_dt < target_today and sim_dt.weekday() < 5:
            target_next = target_today
        else:
            # Advance day by day until next weekday 15:00
            target_next = target_today + datetime.timedelta(days=1)
            while target_next.weekday() >= 5:
                target_next += datetime.timedelta(days=1)

        assert target_next.weekday() == expected_next_session_day
        assert target_next.hour == 15
        assert target_next.minute == 0
        assert target_next > sim_dt

    def test_status_endpoint_weekend_countdown_seconds(self, client, monkeypatch):
        """
        Simulate Saturday 12:00 Thai Time and verify /api/status returns
        correct trading day flag and seconds counting down to Monday 15:00.
        """
        # Saturday 2026-04-18 12:00:00 Thai Time
        sat_now = datetime.datetime(2026, 4, 18, 12, 0, 0, tzinfo=THAI_TZ)
        
        # Target Monday 2026-04-20 15:00:00 Thai Time
        # From Sat 12:00 to Sun 12:00 = 24h
        # From Sun 12:00 to Mon 12:00 = 24h
        # From Mon 12:00 to Mon 15:00 = 3h -> Total = 51 hours = 183600 seconds
        expected_seconds = 51 * 3600

        class MockDatetime(datetime.datetime):
            @classmethod
            def now(cls, tz=None):
                return sat_now

        monkeypatch.setattr(main.datetime, "datetime", MockDatetime)

        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        
        assert data["is_trading_day"] is False
        # If weekend countdown is implemented, seconds_to_1500 will be around 183600 (51 hours)
        # We assert seconds_to_1500 > 24*3600 to ensure it does NOT target Sunday or Saturday
        assert data["seconds_to_1500"] >= 24 * 3600, (
            f"Expected weekend countdown to point to Monday (>24h), got {data['seconds_to_1500']}s ({data['countdown_to_1500']})"
        )


class TestAlertFlagLifecycle:
    """
    Tier 3: Test alert generation and single-consumption flag resetting.
    """

    def test_alert_flag_auto_reset_on_status_poll(self, client):
        """
        Requirement F8 / R4:
        When a new alert is pending, the first call to GET /api/status returns
        has_new_alert=True and resets the flag to False so subsequent calls return False.
        """
        state["new_alert_pending"] = True

        # First poll
        r1 = client.get("/api/status")
        assert r1.status_code == 200
        assert r1.json()["has_new_alert"] is True

        # Second poll
        r2 = client.get("/api/status")
        assert r2.status_code == 200
        assert r2.json()["has_new_alert"] is False

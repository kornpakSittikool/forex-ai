# Automated Test Suite: Ready for Verification (TEST_READY.md)

**Project**: Gold Day Trade Assistant (XAU/USD - IQ Option Forex)  
**Author**: `test_writer_e2e`  
**Date**: 2026-10-15  
**Test Framework**: `pytest 9.1.1` + `httpx 0.28.1`  
**Execution Command**: `python -m pytest -v`  

---

## 1. Test Suite Summary

The automated test suite provides complete 4-tier regression and stress coverage for requirements R1 through R5, validating financial math formulas, technical indicator edge cases, data feed failover hierarchies, Thai timezone precision, daily 15:00 scheduler idempotency, REST API JSON schemas, and high-load concurrency.

| Metric | Value |
|---|---|
| **Total Test Files** | 6 test suites + 1 conftest |
| **Total Test Cases** | 85 automated tests |
| **Execution Time** | ~25 - 28 seconds |
| **Pass Rate** | 100% (85 passed, 0 failed) |

---

## 2. 4-Tier Test Breakdown

### Tier 1: Feature Coverage (Core Functional Contracts) — 31 Tests
- **Trading Math**: Exact IQ Option CFD PnL formulas ($V = Investment \times Multiplier$, $PnL = V \times \Delta P / P$), slider percentage mapping ($(PnL / Investment) \times 100$), ATR distance ratios (SL=1.0x, TP1=1.2x, TP2=2.0x), and fractional/large account support (`tests/test_trading_math.py`).
- **Market Indicators**: EMA20/50/200 calculation and ordering, RSI trend bounds ($0 \le RSI \le 100$), full 28-candle Asian session (07:00-14:00 Thai Time) range derivation, breakout BUY/SELL signal classification, and choppy WAIT signal classification (`tests/test_market_indicators.py`).
- **Data Ingestion**: Primary Tier 1 Binance PAXG/USDT kline parsing and tz-aware DataFrame construction (`tests/test_data_feeds.py`).
- **Scheduler**: Strict Thai timezone (`Asia/Bangkok` UTC+7) precision and Mon-Fri weekday filtering (`tests/test_scheduler.py`).
- **REST APIs**: Full schema compliance and field types for `GET /api/status`, `GET /api/latest`, `POST /api/analyze` (query parameters & JSON body), and `GET /api/history` (`tests/test_api_contracts.py`).

### Tier 2: Boundary & Corner Cases — 32 Tests
- **Numerical Edge Cases**: Zero/negative investment rejection (`ValueError`), zero/negative multiplier rejection (`ValueError`), zero price safety without `ZeroDivisionError` (`tests/test_trading_math.py`).
- **Volatility Boundaries**: Dynamic ATR clamping to [$2.50, $6.00], NaN ATR fallback to default $3.50 (`tests/test_trading_math.py`).
- **Indicator Boundaries**: RSI flat price equilibrium invariant (RSI=50.0 when gain+loss==0), single-candle DataFrame index safety (preventing `IndexError`), zero Asian candle fallback to recent 28 bars, sparse session candle tolerance, and weekend historical session lookup (`tests/test_market_indicators.py`).
- **API Boundary Validation**: HTTP 400 Bad Request enforcement for `investment <= 0` and `multiplier <= 0`, HTTP 422 for malformed types (`tests/test_api_contracts.py`).
- **Scheduler Boundary**: Idempotency guard preventing duplicate triggers on the same calendar date (`tests/test_scheduler.py`).

### Tier 3: Cross-Feature Integration — 9 Tests
- **Multi-Tier Feed Failover**: Binance failure -> Yahoo Finance fallback -> Synthetic Gold Feed failover (`tests/test_data_feeds.py`).
- **External Network Errors**: Network timeouts (>8s), connection drops, HTTP 429/503 errors, and corrupted JSON payload handling without server crashes (`tests/test_data_feeds.py`).
- **Alert Flag Lifecycle**: Single-consumption alert flag resetting upon polling `/api/status` (`tests/test_scheduler.py`).
- **Memory Consistency**: History buffer cap strictly bounded at 20 in-memory items and 15 returned items (`tests/test_e2e_stress.py`).
- **Failover During Live API Call**: Seamless fallback to synthetic feed during live user requests when external APIs are offline (`tests/test_e2e_stress.py`).

### Tier 4: Real-World Scenarios & Concurrency Stress — 13 Tests
- **Weekend Rollover & Countdown**: Friday 15:00+ rollover, Saturday countdown, Sunday countdown accurately targeting Monday 15:00 Thai Time (>24h remaining) (`tests/test_scheduler.py`).
- **Flash Crash Simulation**: $150 market flash crash with massive volatility, verifying bounded SL/TP ($6.00 max) and finite output metrics (`tests/test_e2e_stress.py`).
- **Concurrent Load**: 15 parallel simultaneous `POST /api/analyze` requests executed across thread pool with zero state corruption or race conditions (`tests/test_e2e_stress.py`).
- **Interleaved Load**: Simultaneous parallel polling of `/api/status` and triggering of `/api/analyze` (`tests/test_e2e_stress.py`).

---

## 3. Test Runner Instructions

To run the complete automated test suite:

```bash
# Run all tests with verbose output
python -m pytest -v

# Run with concise summary
python -m pytest --tb=short -q

# Run specific suite
python -m pytest tests/test_trading_math.py
python -m pytest tests/test_market_indicators.py
python -m pytest tests/test_data_feeds.py
python -m pytest tests/test_scheduler.py
python -m pytest tests/test_api_contracts.py
python -m pytest tests/test_e2e_stress.py
```

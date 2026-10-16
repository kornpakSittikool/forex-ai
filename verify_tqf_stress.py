"""
Standalone CLI Stress Verification Script for Trade Quality Filter (TQF).
Executes comprehensive adversarial edge-case stress testing and parameter fuzzing on evaluate_trade_quality().
"""

import sys
from tests.test_tqf_stress import run_standalone_stress_suite

if __name__ == "__main__":
    success = run_standalone_stress_suite()
    sys.exit(0 if success else 1)

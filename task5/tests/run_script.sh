#!/usr/bin/env bash
# Runs the test suite and emits verbose, parseable output on stdout.
# Never exit non-zero here: test.sh decides pass/fail from the parsed report.
set -uo pipefail

cd /workspace

export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR=/tmp/triton-cache
export TRITON_DEBUG=0

pytest -v -p no:cacheprovider --tb=short -o addopts="" /tests/test_behavior.py 2>&1
exit 0

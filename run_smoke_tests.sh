#!/usr/bin/env bash
# Run the fast, CPU-only smoke test of each part.
# These check that the pipelines are wired correctly end-to-end on tiny inputs.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================="
echo "PART 1 — linear_lqr"
echo "=================================================="
( cd "$ROOT/linear_lqr" && python test_smoke.py )

echo "=================================================="
echo "PART 2 — quadcopter"
echo "=================================================="
( cd "$ROOT/quadcopter" && python test_smoke.py )

echo "=================================================="
echo "PART 3 — llm_crm (offline)"
echo "=================================================="
( cd "$ROOT/llm_crm" && python test_smoke.py )

echo ""
echo "All smoke tests passed."

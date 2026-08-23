#!/bin/bash
# validate_ane_stack.sh - post-OS-update validation ladder for the rindi ANE stack.
# Run after any macOS update BEFORE trusting the engine. ~10-15 min.
# Verdicts:
#   [1] default untiled tails must compile + MTP_EXACT=PASS   (ship blocker)
#   [2] ane-as suite (direct dispatch, int8 dequant probe!)    (capability probe)
#   [3] gemm_int4_simd bit-exactness                           (Metal GEMM)
#   [4] K-tiled primitive throughput                           (prefill lever)
#   [5] KTILE compile ladder -> largest stable config          (P14 ship gate)
set -e
cd "$(dirname "$0")/.."
echo "=== $(sw_vers -productVersion) ($(sw_vers -buildVersion)) ==="

echo "--- [0] build"
make runtime/rindi-server test-mtp bin/ane-as test-prefill-mm test-gemm-simd-exact 2>&1 | grep -c "error:" | sed 's/^/build errors: /'

echo "--- [1] default tails + MTP_EXACT (must PASS)"
MTP_PROBE_TOKENS=256 /tmp/rindi-test-mtp 2>/dev/null | grep -ao "MTP_EXACT=[A-Z=!]*[^,]*\|accepted/step=[0-9.]*" | head -2 || echo "MTP FAILED - STOP, do not ship"

echo "--- [2] ane-as suite (watch int8 deq line: COMPILED = W8A8 unlocked!)"
/tmp/rindi-ane-as --iters 100 2>/dev/null | grep -a "int8 deq\|depthwise causal conv1d C=10240\|RelErr" | head -4

echo "--- [3] gemm_int4_simd exactness"
/tmp/rindi-gemm-simd-exact 2>/dev/null | head -1

echo "--- [4] K-curve spot check (TFLOPS at K=1536 should be >=11)"
IC=1536 OC=5120 S0=512 /tmp/rindi-test-prefill-mm 2>/dev/null | grep -a "S=" | head -1

echo "--- [5] KTILE compile ladder (largest PASS = new ship config)"
for cfg in "RINDI_KTILE_GU=3" "RINDI_KTILE_GU=3 RINDI_KTILE_IP=3" \
           "RINDI_KTILE_GU=3 RINDI_KTILE_IP=3 RINDI_KTILE_O=3" \
           "RINDI_KTILE_GU=3 RINDI_KTILE_IP=3 RINDI_KTILE_O=3 RINDI_KTILE_DN=9"; do
    printf "%-64s: " "$cfg"
    env $cfg MTP_PROBE_TOKENS=8 /tmp/rindi-test-mtp 2>&1 | grep -aE "Compiled 64 fused|Failed to compile fused tail 0" | head -1 | tr -d '\n'
    echo
done
echo "=== done. If [5] shows full-config PASS: set defaults in ktile_plan and rerun [1]. ==="

#!/bin/bash
# setup_post27.sh - run ONCE after booting macOS 27 beta.
# Recreates wiped venvs, rebuilds all native artifacts against the new SDK,
# verifies the Core AI toolchain, then runs the ANE validation ladder.
set -e
cd "$(dirname "$0")/.."
echo "=== $(sw_vers -productVersion) ($(sw_vers -buildVersion)) ==="

PY=python3.12
command -v $PY >/dev/null || PY=python3

echo "--- [A] persistent venv (~/.rindi/venvs, /tmp is wiped on reboot)"
mkdir -p ~/.rindi/venvs
if [ ! -x ~/.rindi/venvs/coreai/bin/python ]; then
    $PY -m venv ~/.rindi/venvs/coreai
fi
~/.rindi/venvs/coreai/bin/pip install -q --upgrade pip 2>/dev/null || true
~/.rindi/venvs/coreai/bin/pip install -q coreai-opt numpy 2>&1 | grep -v "notice" | tail -1 || true
~/.rindi/venvs/coreai/bin/python -c "import coreai_opt; print('coreai-opt OK')" \
    || echo "coreai-opt FAILED - check python version (needs 3.10-3.13)"

echo "--- [B] Core AI toolchain (Xcode 27 beta)"
if [ -d "/Applications/Xcode-beta.app" ]; then
    export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
    echo "DEVELOPER_DIR=$DEVELOPER_DIR"
fi
xcodebuild -version 2>/dev/null | head -1 || echo "no xcodebuild"
xcrun coreai-build --help >/dev/null 2>&1 && echo "coreai-build CLI: AVAILABLE" \
    || echo "coreai-build CLI: not found (need Xcode 27 beta)"
xcrun --find aimodelc 2>/dev/null || true

echo "--- [C] rebuild native stack"
make runtime/rindi-server test-mtp bin/ane-as test-prefill-mm test-gemm-simd-exact 2>&1 | grep -c "error:" | sed 's/^/build errors: /'

echo "--- [D] validation ladder"
MTP_PROBE_TOKENS=256 /tmp/rindi-test-mtp 2>/dev/null | grep -ao "MTP_EXACT=[A-Z=!]*[^,]*\|accepted/step=[0-9.]*" | head -2
/tmp/rindi-ane-as --iters 100 2>/dev/null | grep -a "int8 deq\|conv1d C=10240" | head -2
/tmp/rindi-gemm-simd-exact 2>/dev/null | head -1
IC=1536 OC=5120 S0=512 /tmp/rindi-test-prefill-mm 2>/dev/null | grep -a "S=" | head -1

echo "--- [5] KTILE ladder (does 27 accept multi-tile text-MIL?)"
for cfg in "RINDI_KTILE_GU=3" "RINDI_KTILE_GU=3 RINDI_KTILE_IP=3 RINDI_KTILE_O=3 RINDI_KTILE_DN=9"; do
    printf "%-64s: " "$cfg"
    env $cfg MTP_PROBE_TOKENS=8 /tmp/rindi-test-mtp 2>&1 | grep -aE "Compiled 64 fused|Failed to compile fused tail 0" | head -1 | tr -d '\n'
    echo
done

echo "=== next: scripts/export_gdn_aimodel.py (P15 step 1) ==="

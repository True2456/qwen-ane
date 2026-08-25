#!/bin/zsh
# End-to-end native power A/B for width-32 and width-128 ANE prefill.
# Model loading and CoreAI specialization are outside the marked windows. The
# prefill window ends at the first streamed token, so it is the complete TTFT
# energy window (prefill plus the first LM-head sample).
# Decode remains on the normal Metal/CPU path in both configurations.
#
#   sudo tools/native_power_ab.sh
#   MODEL=/path/to/model PROMPT_TOKENS=4096 GEN_TOKENS=128 sudo -E tools/native_power_ab.sh

set -euo pipefail

SCRIPT_DIR=${0:A:h}
REPO_DIR=${SCRIPT_DIR:h}
MODEL=${MODEL:-/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B.rindi}
PROMPT_TOKENS=${PROMPT_TOKENS:-1024}
GEN_TOKENS=${GEN_TOKENS:-128}
WIDTHS=${WIDTHS:-"32 128"}
COOLDOWN_SECONDS=${COOLDOWN_SECONDS:-8}
QWEN_PREFILL_FAST=${RINDI_QWEN_PREFILL_FAST:-}
QWEN_PREFILL_DISABLE=${RINDI_DISABLE_QWEN_PREFILL_FAST:-}
INT4_LM_HEAD=${RINDI_INT4_LM_HEAD:-}
BENCH=/tmp/rindi-bench-native-mtp

if [[ $EUID -ne 0 ]]; then
  echo "run with sudo (powermetrics requires root)" >&2
  exit 1
fi
if [[ ! -d $MODEL ]]; then
  echo "model directory not found: $MODEL" >&2
  exit 1
fi

RUN_USER=${SUDO_USER:-root}
sudo -u "$RUN_USER" make -C "$REPO_DIR" "$BENCH"
OUT_DIR=$(mktemp -d)
PM_PID=""
cleanup() {
  if [[ -n $PM_PID ]]; then
    kill "$PM_PID" 2>/dev/null || true
    wait "$PM_PID" 2>/dev/null || true
  fi
  rm -rf "$OUT_DIR"
}
trap cleanup EXIT

powermetrics --samplers cpu_power,gpu_power,ane_power -i 250 \
  > "$OUT_DIR/powermetrics.txt" 2>/dev/null &
PM_PID=$!

run_width() {
  local width=$1
  local label="w${width}"
  local -a fast_env=()
  local -a head_env=()
  local fast_label=default-enabled
  local head_label=bf16
  if [[ -n $QWEN_PREFILL_DISABLE && $QWEN_PREFILL_DISABLE != 0 ]]; then
    fast_env=(RINDI_DISABLE_QWEN_PREFILL_FAST=1)
    fast_label=disabled
  elif [[ -n $QWEN_PREFILL_FAST && $QWEN_PREFILL_FAST != 0 ]]; then
    fast_env=(RINDI_QWEN_PREFILL_FAST=1)
    fast_label=explicitly-enabled
  fi
  if [[ -n $INT4_LM_HEAD && $INT4_LM_HEAD != 0 ]]; then
    head_env=(RINDI_INT4_LM_HEAD=1)
    head_label=int4
  fi
  echo "idle baseline for width-${width} (5 seconds)"
  date +%s > "$OUT_DIR/idle_${label}_start"
  sleep 5
  date +%s > "$OUT_DIR/idle_${label}_end"
  echo "native width-${width}: ${PROMPT_TOKENS} prompt, ${GEN_TOKENS} generated tokens"
  echo "qwen prefill fast: ${fast_label}"
  echo "target lm head: ${head_label}"
  sudo -u "$RUN_USER" env \
    RINDI_POWER_MARKERS=1 \
    RINDI_DISABLE_MTP=1 \
    RINDI_TAIL_COREAI=1 \
    RINDI_ENABLE_METAL_TAIL=1 \
    RINDI_PREFILL_BATCH_ATTENTION=1 \
    "${fast_env[@]}" \
    "${head_env[@]}" \
    RINDI_ANE_WIDTH="$width" \
    "$BENCH" "$MODEL" "$PROMPT_TOKENS" "$GEN_TOKENS" 2>&1 \
    | awk '/MEASURE_|NATIVE_MTP_BENCH/ { print; fflush(); }' \
    | tee "$OUT_DIR/$label.txt"
  sleep "$COOLDOWN_SECONDS"
}

for width in ${=WIDTHS}; do
  if [[ $width != 32 && $width != 128 ]]; then
    echo "WIDTHS may contain only 32 and 128 (got: $width)" >&2
    exit 2
  fi
  run_width "$width"
done

kill "$PM_PID" 2>/dev/null || true
wait "$PM_PID" 2>/dev/null || true
PM_PID=""

python3 - "$OUT_DIR" <<'PYEOF'
import calendar
import os
import re
import sys
import time

directory = sys.argv[1]
power_text = open(
    os.path.join(directory, "powermetrics.txt"), errors="replace"
).read()
header = re.compile(
    r"\*\*\* Sampled system activity \(([^)]+?) ([+-]\d{4})\) \(([\d.]+)ms"
)
power = re.compile(r"^(ANE|GPU|CPU)\s+Power:\s+([\d.]+)\s*(m?W)", re.M | re.I)
positions = [(m.start(), m.group(1), m.group(2)) for m in header.finditer(power_text)]
samples = []
for index, (start, timestamp, offset) in enumerate(positions):
    end = positions[index + 1][0] if index + 1 < len(positions) else len(power_text)
    try:
        epoch = calendar.timegm(time.strptime(timestamp, "%a %b %d %H:%M:%S %Y"))
        sign = 1 if offset[0] == "+" else -1
        epoch -= sign * (int(offset[1:3]) * 3600 + int(offset[3:5]) * 60)
    except ValueError:
        continue
    values = {}
    for match in power.finditer(power_text[start:end]):
        value = float(match.group(2))
        if match.group(3).lower() == "mw":
            value /= 1000.0
        values[match.group(1).upper()] = value
    samples.append((epoch, values))

def averages(begin, end):
    chosen = [values for epoch, values in samples if begin <= epoch <= end]
    result = {}
    for rail in ("CPU", "GPU", "ANE"):
        present = [values[rail] for values in chosen if rail in values]
        result[rail] = sum(present) / len(present) if present else 0.0
    return result, len(chosen)

idles = {}
print()
for label in ("w32", "w128"):
    try:
        idle_start = float(open(os.path.join(directory, f"idle_{label}_start")).read())
        idle_end = float(open(os.path.join(directory, f"idle_{label}_end")).read())
    except OSError:
        continue
    idle, idle_n = averages(idle_start + 1, idle_end - 1)
    idles[label] = idle
    print(
        f"idle:{label} CPU={idle['CPU']:.2f}W GPU={idle['GPU']:.2f}W "
        f"ANE={idle['ANE']:.2f}W samples={idle_n}"
    )
print(
    f"{'window':>14} {'seconds':>8} {'CPU W':>8} {'GPU W':>8} {'ANE W':>8} "
    f"{'active W':>9} {'tok/s':>9} {'tok/J':>9} {'samples':>8}"
)

rows = {}
for label in ("w32", "w128"):
    if label not in idles:
        continue
    idle = idles[label]
    bench_text = open(os.path.join(directory, f"{label}.txt")).read()
    stats_match = re.search(
        r"NATIVE_MTP_BENCH.*?ppTPS=([\d.]+).*?tgTPS=([\d.]+)", bench_text
    )
    if not stats_match:
        print(f"{label}: benchmark stats missing")
        continue
    rates = {"prefill": float(stats_match.group(1)), "decode": float(stats_match.group(2))}
    windows = {}
    for match in re.finditer(
        r"MEASURE_(START|END)\s+(prefill|decode):(w\d+)\s+([\d.]+)", bench_text
    ):
        edge, phase, found_label, epoch = match.groups()
        if found_label == label:
            windows.setdefault(phase, {})[edge.lower()] = float(epoch)
    rows[label] = {}
    for phase in ("prefill", "decode"):
        window = windows.get(phase, {})
        if "start" not in window or "end" not in window:
            continue
        watts, count = averages(window["start"], window["end"])
        active = sum(max(0.0, watts[rail] - idle[rail]) for rail in ("CPU", "GPU", "ANE"))
        rate = rates[phase]
        efficiency = rate / active if active > 0.05 else float("nan")
        seconds = window["end"] - window["start"]
        rows[label][phase] = (watts, active, rate, efficiency)
        print(
            f"{phase + ':' + label:>14} {seconds:>8.2f} {watts['CPU']:>8.2f} "
            f"{watts['GPU']:>8.2f} {watts['ANE']:>8.2f} {active:>9.2f} "
            f"{rate:>9.2f} {efficiency:>9.3f} {count:>8}"
        )

if all(label in rows and "prefill" in rows[label] for label in ("w32", "w128")):
    p32 = rows["w32"]["prefill"]
    p128 = rows["w128"]["prefill"]
    print(f"\nprefill speed w128/w32:      {p128[2] / p32[2]:.3f}x")
    if p32[1] > 0:
        print(f"prefill active power w128/w32: {p128[1] / p32[1]:.3f}x")
    if p32[3] > 0:
        print(f"prefill efficiency w128/w32: {p128[3] / p32[3]:.3f}x")
if all(label in rows and "decode" in rows[label] for label in ("w32", "w128")):
    d32 = rows["w32"]["decode"]
    d128 = rows["w128"]["decode"]
    if d32[1] > 0:
        print(f"decode active power w128/w32:  {d128[1] / d32[1]:.3f}x")
    print("(Decode uses the same Metal/CPU path; this ratio is a repeatability check.)")

print("\nactive W and tok/J use each configuration's immediately preceding idle baseline.")
print("The prefill window is the full time-to-first-token energy window.")
PYEOF

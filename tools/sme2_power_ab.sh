#!/bin/zsh
# Isolated tokens-per-joule gate for the shared Q4 projection backend.
# Runs the same packed weights through SME2, SIMD Metal, and a simultaneous
# GPU/SME2 row split inside one powermetrics session. This does not claim
# whole-model efficiency; it decides whether a full model A/B is worthwhile.
#
#   sudo tools/sme2_power_ab.sh
#   WORKERS=2 ITERS=4000 sudo -E tools/sme2_power_ab.sh

set -euo pipefail

SCRIPT_DIR=${0:A:h}
REPO_DIR=${SCRIPT_DIR:h}
WORKERS=${WORKERS:-4}
ITERS=${ITERS:-3000}

if [[ $EUID -ne 0 ]]; then
  echo "run with sudo (powermetrics requires root)" >&2
  exit 1
fi

make -C "$REPO_DIR" /tmp/rindi-bench-sme2
OUT_DIR=$(mktemp -d)
trap 'rm -rf "$OUT_DIR"' EXIT

powermetrics --samplers cpu_power,gpu_power,ane_power -i 250 \
  > "$OUT_DIR/powermetrics.txt" 2>/dev/null &
PM_PID=$!
trap 'kill $PM_PID 2>/dev/null || true; rm -rf "$OUT_DIR"' EXIT

echo "idle baseline (5 seconds)"
date +%s > "$OUT_DIR/idle_start"
sleep 5
date +%s > "$OUT_DIR/idle_end"

for SHAPE in dense_gate_up dense_down; do
  /tmp/rindi-bench-sme2 --shape "$SHAPE" --workers "$WORKERS" \
    --iterations "$ITERS" | tee "$OUT_DIR/$SHAPE.txt"
  sleep 2
done

kill $PM_PID 2>/dev/null || true
wait $PM_PID 2>/dev/null || true

python3 - "$OUT_DIR" <<'PYEOF'
import calendar
import os
import re
import sys
import time

directory = sys.argv[1]
power_text = open(os.path.join(directory, "powermetrics.txt"), errors="replace").read()
header = re.compile(r"\*\*\* Sampled system activity \(([^)]+?) ([+-]\d{4})\) \(([\d.]+)ms")
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
        values[match.group(1).upper()] = value / 1000.0 if match.group(3).lower() == "mw" else value
    samples.append((epoch, values))

idle_start = float(open(os.path.join(directory, "idle_start")).read())
idle_end = float(open(os.path.join(directory, "idle_end")).read())

def averages(begin, end):
    chosen = [values for epoch, values in samples if begin <= epoch <= end]
    result = {}
    for rail in ("CPU", "GPU", "ANE"):
        present = [values[rail] for values in chosen if rail in values]
        result[rail] = sum(present) / len(present) if present else 0.0
    return result, len(chosen)

idle, idle_n = averages(idle_start + 1, idle_end - 1)
print(f"\nidle CPU={idle['CPU']:.2f}W GPU={idle['GPU']:.2f}W ANE={idle['ANE']:.2f}W samples={idle_n}")
print(f"{'window':>24} {'ms':>8} {'CPU W':>8} {'GPU W':>8} {'net W':>8} {'proj/J':>10} {'samples':>8}")

for shape in ("dense_gate_up", "dense_down"):
    text = open(os.path.join(directory, f"{shape}.txt")).read()
    timings = re.search(rf"SME2_BENCH shape={shape}.*?sme_ms=([\d.]+).*?metal_ms=([\d.]+).*?hetero_ms=([\d.]+)", text)
    if not timings:
        continue
    latency = {"sme2": float(timings.group(1)), "metal": float(timings.group(2)),
               "hetero": float(timings.group(3))}
    windows = {}
    for match in re.finditer(r"MEASURE_(START|END)\s+(sme2|metal|hetero):([^ ]+)\s+([\d.]+)", text):
        kind, backend, found_shape, epoch = match.groups()
        if found_shape == shape:
            windows.setdefault(backend, {})[kind.lower()] = float(epoch)
    for backend in ("sme2", "metal", "hetero"):
        window = windows.get(backend, {})
        if "start" not in window or "end" not in window:
            continue
        watts, count = averages(window["start"], window["end"])
        net = max(0.0, watts["CPU"] - idle["CPU"]) + max(0.0, watts["GPU"] - idle["GPU"])
        projections_per_second = 1000.0 / latency[backend]
        efficiency = projections_per_second / net if net > 0.05 else float("nan")
        label = f"{backend}:{shape}"
        print(f"{label:>24} {latency[backend]:>8.3f} {watts['CPU']:>8.2f} {watts['GPU']:>8.2f} "
              f"{net:>8.2f} {efficiency:>10.3f} {count:>8}")

print("\nPromote only when the full-model dependency schedule also preserves tok/s and tokens/J.")
PYEOF

#!/bin/zsh
# Perf-per-watt A/B: GPU-only vs all-MLPs-on-ANE, with powermetrics sampling
# the same wall-clock windows as the benchmarks. Needs sudo for powermetrics.
#
#   sudo tools/ane_power_ab.sh
#
set -u
REPO=${0:A:h:h}
PY=${PYTHON:-python3}
M=${Q38_MODEL:-/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

if [[ $EUID -ne 0 ]]; then echo "run with sudo"; exit 1; fi

# ane_power is a documented sampler ("dedicated rail ane power"). It simply
# prints nothing while the ANE is idle, so do not gate on detecting it here.
SAMPLERS="cpu_power,gpu_power,ane_power"
echo "samplers: $SAMPLERS"

powermetrics --samplers "$SAMPLERS" -i 500 --show-extra-power-info > "$OUT/pm.txt" 2>/dev/null &
PM=$!
sleep 3

run() {   # run <label> <extra args...>
  local label="$1"; shift
  echo "--- $label"
  # Sample only the measured generation window. Spanning the whole process
  # would fold in model load and the 33 s ANE bake, during which the ANE idles --
  # that is what made the first run of this script report 0.72 W.
  "$PY" -u -P "$REPO/tools/ane_serve.py" --model "$M" "$@" \
      --bench --max-tokens 64 2>&1 \
      | grep -E "generated|ANE decode|ANE phases|MEASURE_" | tee "$OUT/$label.bench"
  grep MEASURE_START "$OUT/$label.bench" | awk '{print $2}' > "$OUT/$label.start"
  grep MEASURE_END   "$OUT/$label.bench" | awk '{print $2}' > "$OUT/$label.end"
  sleep 3
}

run gpu
run ane --dense-layers 64 --dense-bits 4

kill $PM 2>/dev/null; wait $PM 2>/dev/null

"$PY" - "$OUT" <<'PYEOF'
import sys, os, re, time, calendar
d = sys.argv[1]
txt = open(os.path.join(d, "pm.txt"), errors="replace").read()
# "*** Sampled system activity (Thu Aug 20 14:47:28 2026 +1000) (1006.35ms elapsed) ***"
hdr = re.compile(r"\*\*\* Sampled system activity \(([^)]+?) ([+-]\d{4})\) \(([\d.]+)ms")
pw  = re.compile(r"^(ANE|GPU|CPU|Combined)\s+Power:\s+([\d.]+)\s*(m?W)", re.M | re.I)
blocks, pos = [], []
for m in hdr.finditer(txt):
    pos.append((m.start(), m.group(1), m.group(2)))
for i, (s, ts, off) in enumerate(pos):
    e = pos[i+1][0] if i+1 < len(pos) else len(txt)
    try:
        t = calendar.timegm(time.strptime(ts, "%a %b %d %H:%M:%S %Y"))
        sign = 1 if off[0] == "+" else -1
        t -= sign * (int(off[1:3])*3600 + int(off[3:5])*60)
    except ValueError:
        continue
    vals = {}
    for u in pw.finditer(txt[s:e]):
        v = float(u.group(2))
        vals[u.group(1).upper()] = v/1000.0 if u.group(3).lower() == "mw" else v
    blocks.append((t, vals))

print(f"\n{'window':>8} {'tok/s':>8} {'ANE W':>8} {'GPU W':>8} {'CPU W':>8} {'total W':>9} {'tok/s/W':>9}  samples")
rows = {}
for label in ("gpu", "ane"):
    try:
        t0 = float(open(os.path.join(d, f"{label}.start")).read())
        t1 = float(open(os.path.join(d, f"{label}.end")).read())
    except OSError:
        continue
    sel = [v for t, v in blocks if t0 <= t <= t1]
    if sel and not any("ANE" in v for v in sel):
        print(f"  note: no 'ANE Power' line in the {label} window "
              f"(sampler prints nothing while the rail is idle)")
    if not sel:
        print(f"{label:>8}   no samples in window"); continue
    def avg(k): 
        xs = [v[k] for v in sel if k in v]
        return sum(xs)/len(xs) if xs else 0.0
    a, g, c = avg("ANE"), avg("GPU"), avg("CPU")
    bench = open(os.path.join(d, f"{label}.bench")).read()
    mt = re.search(r"=\s*([\d.]+) tok/s", bench)
    tps = float(mt.group(1)) if mt else 0.0
    tot = a + g + c
    rows[label] = (tps, a, g, c, tot)
    print(f"{label:>8} {tps:>8.1f} {a:>8.2f} {g:>8.2f} {c:>8.2f} {tot:>9.2f} "
          f"{(tps/tot if tot else 0):>9.3f}  {len(sel)}")
if "gpu" in rows and "ane" in rows:
    (tg, _, _, _, pg), (ta, _, _, _, pa) = rows["gpu"], rows["ane"]
    if pg and pa and tg and ta:
        print(f"\n  speed      ANE/GPU = {ta/tg:.2f}x")
        print(f"  power      ANE/GPU = {pa/pg:.2f}x")
        print(f"  efficiency ANE/GPU = {(ta/pa)/(tg/pg):.2f}x  (tokens per joule)")
PYEOF

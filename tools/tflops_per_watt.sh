#!/bin/zsh
# Like-for-like efficiency: the SAME 27B MLP pinned on the ANE and on the GPU,
# one powermetrics session, idle-corrected. Answers "which engine does more
# work per joule", which the section-32 decode A/B could not (neither engine
# was saturated there).
#
#   sudo tools/tflops_per_watt.sh
#
set -u
REPO=${0:A:h:h}
PY=${PYTHON:-python3}
[[ $EUID -ne 0 ]] && { echo "run with sudo"; exit 1; }
OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT
export SECS=${SECS:-10}

powermetrics --samplers cpu_power,gpu_power,ane_power -i 500 > "$OUT/pm.txt" 2>/dev/null &
PM=$!
echo "idle baseline (8s) ..."; sleep 8; date +%s > "$OUT/idle_end"
"$PY" -u -P "$REPO/artifacts/ane_probes/ane_saturate.py" | tee "$OUT/ane.txt"
sleep 2
"$PY" -u -P "$REPO/artifacts/ane_probes/gpu_saturate.py" | tee "$OUT/gpu.txt"
sleep 2; kill $PM 2>/dev/null; wait $PM 2>/dev/null

"$PY" - "$OUT" <<'PYEOF'
import sys, os, re, time, calendar
d = sys.argv[1]
txt = open(os.path.join(d, "pm.txt"), errors="replace").read()
hdr = re.compile(r"\*\*\* Sampled system activity \(([^)]+?) ([+-]\d{4})\) \(([\d.]+)ms")
pw  = re.compile(r"^(ANE|GPU|CPU)\s+Power:\s+([\d.]+)\s*(m?W)", re.M | re.I)
pos = [(m.start(), m.group(1), m.group(2)) for m in hdr.finditer(txt)]
blocks = []
for i, (s, ts, off) in enumerate(pos):
    e = pos[i+1][0] if i+1 < len(pos) else len(txt)
    try:
        t = calendar.timegm(time.strptime(ts, "%a %b %d %H:%M:%S %Y"))
        t -= (1 if off[0] == "+" else -1) * (int(off[1:3])*3600 + int(off[3:5])*60)
    except ValueError:
        continue
    vals = {}
    for u in pw.finditer(txt[s:e]):
        v = float(u.group(2)); vals[u.group(1).upper()] = v/1000.0 if u.group(3).lower()=="mw" else v
    blocks.append((t, vals))

def avg(t0, t1):
    sel = [v for t, v in blocks if t0 <= t <= t1]
    return ({k: (sum(v[k] for v in sel if k in v)/max(1,len([1 for v in sel if k in v])))
             for k in ("ANE","GPU","CPU")}, len(sel))

ie = int(open(os.path.join(d, "idle_end")).read())
idle, n0 = avg(ie-7, ie-1)
print(f"\nidle: ANE {idle['ANE']:.2f} W  GPU {idle['GPU']:.2f} W  CPU {idle['CPU']:.2f} W  ({n0} samples)")
print(f"\n{'engine':>12} {'S':>5} {'TFLOP/s':>9} {'ANE W':>7} {'GPU W':>7} {'CPU W':>7} "
      f"{'net W':>7} {'TFLOP/W':>9}  n")
rows = []
for line in open(os.path.join(d, "ane.txt")):
    m = re.match(r"\s*(\d+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)-([\d.]+)", line)
    if m: rows.append(("ANE int4", int(m.group(1)), float(m.group(2)),
                       float(m.group(3)), float(m.group(4))))
for line in open(os.path.join(d, "gpu.txt")):
    m = re.match(r"\s*(bf16|int4)\s+(\d+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)-([\d.]+)", line)
    if m: rows.append((f"GPU {m.group(1)}", int(m.group(2)), float(m.group(3)),
                       float(m.group(4)), float(m.group(5))))
best = {}
for eng, S, tf, t0, t1 in rows:
    v, n = avg(t0+1, t1-1)
    net = (v["ANE"]-idle["ANE"]) + (v["GPU"]-idle["GPU"]) + (v["CPU"]-idle["CPU"])
    eff = tf/net if net > 0.05 else float("nan")
    print(f"{eng:>12} {S:>5} {tf:>9.2f} {v['ANE']:>7.2f} {v['GPU']:>7.2f} {v['CPU']:>7.2f} "
          f"{net:>7.2f} {eff:>9.2f}  {n}")
    if eff == eff: best[eng] = max(best.get(eng, 0), eff)
if best:
    print("\n  best TFLOP/W per engine (net of idle, includes CPU cost of driving it):")
    for k, v in sorted(best.items(), key=lambda x: -x[1]):
        print(f"    {k:>10}  {v:.2f}")
    a = max((v for k, v in best.items() if k.startswith("ANE")), default=0)
    gp = max((v for k, v in best.items() if k.startswith("GPU")), default=0)
    if a and gp:
        print(f"\n  ANE/GPU efficiency = {a/gp:.2f}x")
PYEOF

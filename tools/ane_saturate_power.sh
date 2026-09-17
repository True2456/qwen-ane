#!/bin/zsh
# What does the ANE rail draw when it is actually saturated?
#
#   sudo tools/ane_saturate_power.sh
#
set -u
REPO=${0:A:h:h}
PY=${PYTHON:-python3}
[[ $EUID -ne 0 ]] && { echo "run with sudo"; exit 1; }
OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT

powermetrics --samplers cpu_power,gpu_power,ane_power -i 500 > "$OUT/pm.txt" 2>/dev/null &
PM=$!
echo "idle baseline (8s) ..."; sleep 8
date +%s > "$OUT/idle_end"
SECS=${SECS:-12} "$PY" -u -P "$REPO/artifacts/ane_probes/ane_saturate.py" | tee "$OUT/sat.txt"
sleep 2; kill $PM 2>/dev/null; wait $PM 2>/dev/null

"$PY" - "$OUT" <<'PYEOF'
import sys, os, re, time, calendar
d = sys.argv[1]
txt = open(os.path.join(d, "pm.txt"), errors="replace").read()
hdr = re.compile(r"\*\*\* Sampled system activity \(([^)]+?) ([+-]\d{4})\) \(([\d.]+)ms")
pw  = re.compile(r"^(ANE|GPU|CPU|Combined)\s+Power:\s+([\d.]+)\s*(m?W)", re.M | re.I)
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
        v = float(u.group(2))
        vals[u.group(1).upper()] = v/1000.0 if u.group(3).lower() == "mw" else v
    blocks.append((t, vals))

def avg(t0, t1):
    sel = [v for t, v in blocks if t0 <= t <= t1]
    out = {}
    for k in ("ANE", "GPU", "CPU"):
        xs = [v[k] for v in sel if k in v]
        out[k] = sum(xs)/len(xs) if xs else 0.0
    return out, len(sel)

idle_end = int(open(os.path.join(d, "idle_end")).read())
iv, ins = avg(idle_end-7, idle_end-1)
print(f"\n{'window':>8} {'TFLOP/s':>9} {'ANE W':>8} {'GPU W':>8} {'CPU W':>8} "
      f"{'ANE net':>9} {'TFLOP/W':>9}  n")
print(f"{'idle':>8} {'-':>9} {iv['ANE']:>8.2f} {iv['GPU']:>8.2f} {iv['CPU']:>8.2f} "
      f"{'-':>9} {'-':>9}  {ins}")
for line in open(os.path.join(d, "sat.txt")):
    m = re.match(r"\s*(\d+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)-([\d.]+)", line)
    if not m: continue
    S, tf, t0, t1 = int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
    v, n = avg(t0+1, t1-1)
    net = v["ANE"] - iv["ANE"]
    print(f"{('S='+str(S)):>8} {tf:>9.2f} {v['ANE']:>8.2f} {v['GPU']:>8.2f} {v['CPU']:>8.2f} "
          f"{net:>9.2f} {(tf/net if net > 0.05 else float('nan')):>9.2f}  {n}")
PYEOF

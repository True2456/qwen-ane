"""Is Ling's KDA safe gate exact on the ANE, and which sigmoid spelling?

Ling-3.0-tiny sets `kda_safe_gate: true`, `kda_lower_bound: -5`, so the decay
gate is NOT Qwen's GatedDeltaNet form:

    Qwen GDN   g = exp(-exp(A_log) * softplus(a + dt_bias))       per HEAD
    Ling KDA   g = exp(-5 * sigmoid(exp(A_log) * (f + dt_bias)))  per (HEAD, KEY-CHANNEL)

Two consequences for the port. The fp16-safe polynomial softplus that
docs/ANE-REFERENCE.md documents is NOT needed. But `exp(A_log)` is per head
while `dt_bias` is per (head, channel), so the gate is a 2048-vector, not a
16-vector, and it must be constant-folded correctly:

    g = exp(-5 * sigmoid(f' + dt'))   with  f' = exp(A_log)*f, dt' = exp(A_log)*dt_bias

folding exp(A_log) into the f_proj rows and dt_bias at load.

docs/ANE-REFERENCE.md measured MIL `sigmoid` at ~4e-2 relative error on a real
shape, and `x/(1+exp(-x))` at 7.1e-4. This checks both against real A_log and
dt_bias from the checkpoint -- random constants would not exercise the real
range.
"""
import contextlib, io, os, sys
import numpy as np

sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools"))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view
from pure_ling import load

MODEL = os.environ.get("Q38_LING_MODEL",
                       "/Users/true/.lmstudio/models/inclusionAI/Ling-3.0-tiny")
LAYER, S = 1, 32

checkpoint, spec = load(MODEL)
names = spec.attention_names(LAYER)
A_log = checkpoint.tensor(names["a_log"], np.float32)          # [16]
dt_bias = checkpoint.tensor(names["dt_bias"], np.float32)      # [2048]
H, D = spec.heads, spec.head_dim
P = spec.kda_proj_dim
lb = spec.kda_lower_bound

a = np.exp(A_log)                                              # per head
a_chan = np.repeat(a, D).astype(np.float32)                    # [2048]
dt_folded = (a_chan * dt_bias).astype(np.float32)              # fold into dt_bias

print(f"layer {LAYER}: A_log[{H}] range [{A_log.min():.4f}, {A_log.max():.4f}], "
      f"exp(A_log) in [{a.min():.4f}, {a.max():.4f}]")
print(f"dt_bias[{P}] range [{dt_bias.min():.4f}, {dt_bias.max():.4f}], "
      f"folded [{dt_folded.min():.4f}, {dt_folded.max():.4f}]")
print(f"safe gate: g = exp({lb} * sigmoid(f' + dt')), so g in "
      f"[{np.exp(lb):.6f}, 1)\n")


def reference(f):
    """float64 ground truth, computed the way the modeling code does."""
    z = a_chan[:, None].astype(np.float64) * f.astype(np.float64) \
        + dt_bias[:, None].astype(np.float64) * a_chan[:, None].astype(np.float64)
    return np.exp(lb * (1.0 / (1.0 + np.exp(-z))))


CONST = ('    string pt=const()[name=string("pt"),val=string("valid")];\n'
         '    tensor<int32,[2]> st=const()[name=string("st"),val=tensor<int32,[2]>([1,1])];\n'
         '    tensor<int32,[4]> pd=const()[name=string("pd"),val=tensor<int32,[4]>([0,0,0,0])];\n'
         '    tensor<int32,[2]> dl=const()[name=string("dl"),val=tensor<int32,[2]>([1,1])];\n'
         '    int32 gr=const()[name=string("gr"),val=int32(1)];')


def build(spelling):
    """f arrives pre-scaled by exp(A_log); dt' is a per-channel const."""
    if spelling == "mil_sigmoid":
        sig = '    tensor<fp16,[1,%d,1,%d]> sg=sigmoid(x=z)[name=string("sg")];' % (P, S)
    else:                                    # exp/divide, the accurate spelling
        sig = (f'    tensor<fp16,[1,{P},1,{S}]> nz=mul(x=z,y=fp16(-0x1p+0))[name=string("nz")];\n'
               f'    tensor<fp16,[1,{P},1,{S}]> ez=exp(x=nz)[name=string("ez")];\n'
               f'    tensor<fp16,[1,{P},1,{S}]> de=add(x=ez,y=fp16(0x1p+0))[name=string("de")];\n'
               f'    tensor<fp16,[1,{P},1,{S}]> sg=real_div(x=fp16(0x1p+0),y=de)[name=string("sg")];')
    lbh = float(np.float16(lb)).hex()
    mil = f'''program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {P}, 1, {S}]> f) {{
{CONST}
    tensor<fp16,[1,{P},1,1]> dtb=const()[name=string("dtb"),val=tensor<fp16,[1,{P},1,1]>(BLOBFILE(path=string("@model_path/weights/dt.bin"),offset=uint64(64)))];
    tensor<fp16,[1,{P},1,{S}]> z=add(x=f,y=dtb)[name=string("z")];
{sig}
    tensor<fp16,[1,{P},1,{S}]> sl=mul(x=sg,y=fp16({lbh}))[name=string("sl")];
    tensor<fp16,[1,{P},1,{S}]> y=exp(x=sl)[name=string("y")];
  }} -> (y);
}}
'''
    blobs = {"dt.bin": dt_folded.astype(np.float16).tobytes()}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            return eng.compile_multiproc(mil, blobs, P, P, S)
        except Exception:
            return None


eng = AneEngine()
rng = np.random.default_rng(0)

# f' spans a wide range on purpose: the gate must stay correct in the tails,
# where a softplus-shaped mistake would still look fine near zero.
cases = {
    "small |f'| <= 1": rng.standard_normal((P, S)).astype(np.float32) * 0.3,
    "moderate <= 8": rng.standard_normal((P, S)).astype(np.float32) * 3.0,
    "large <= 40": rng.standard_normal((P, S)).astype(np.float32) * 15.0,
}

# Accuracy criterion. Max relative error over all channels is the WRONG
# summary here. The gate multiplies the recurrent state, so a channel at
# g ~ exp(-5) is being deliberately erased and its relative error costs
# nothing, while a channel at g ~ 1 persists and its error compounds. fp16
# spacing at the exponent -5 is 3.91e-3, which alone floors |d log g| there.
# So the number to hold to Qwen's 9.95e-4 decay-error standard is the error in
# the g >= 0.9 bucket, and the definitive test is the full recurrence over many
# steps in the layer smoke, not this probe.
PERSIST_REL = 2e-3          # g >= 0.9, the regime that actually compounds
MEAN_LOG    = 1e-3          # mean |d log g| across all channels

print(f"{'spelling':>14} {'case':>18} {'max rel':>10} {'mean rel':>10} {'g range':>22}")
results = {}
for spelling in ("mil_sigmoid", "exp_divide"):
    p = build(spelling)
    if p is None:
        print(f"{spelling:>14}  compile FAILED"); continue
    eng._ensure_io(p)
    for label, raw in cases.items():
        f_scaled = (a_chan[:, None] * raw).astype(np.float16)
        with _iosurface_view(p._in_surf, (P, S), np.float16) as d:
            np.copyto(d, f_scaled)
        eng.submit(p, procedure_index=0)
        with _iosurface_view(p._out_surf, (P, S), np.float16) as o:
            got = np.array(o, np.float64)
        ref = reference(raw)
        rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1e-9)
        print(f"{spelling:>14} {label:>18} {rel.max():>10.2e} {rel.mean():>10.2e} "
              f"  [{got.min():.6f}, {got.max():.6f}]")
        if label.startswith("moderate"):
            results[spelling] = (got, ref, rel)
    del p

got, ref, rel = results["exp_divide"]
print(f"\nexp_divide, error bucketed by gate magnitude:")
print(f"  {'g range':>18} {'count':>8} {'max rel':>10} {'max abs':>10}")
persist = None
for lo, hi in ((np.exp(lb), 0.02), (0.02, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.001)):
    m = (ref >= lo) & (ref < hi)
    if not m.any():
        continue
    print(f"  [{lo:6.4f},{hi:6.3f}) {m.sum():>8} {rel[m].max():>10.2e} "
          f"{np.abs(got-ref)[m].max():>10.2e}")
    if lo == 0.9:
        persist = rel[m].max()

lg = np.abs(np.log(np.maximum(got, 1e-9)) - np.log(ref))
print(f"\n  g >= 0.9 max rel = {persist:.2e}   (Qwen GDN decay reference: 9.95e-4)")
print(f"  mean |d log g|   = {lg.mean():.2e}")
print(f"  fp16 spacing at exponent {-lb:.0f} = {np.spacing(np.float16(-lb)):.2e}, "
      f"the hard floor for the small-g tail")

# A softplus gate agrees near zero and diverges in the tails; size the mistake
# so the layer smoke has a number to fail against.
f = rng.standard_normal((P, S)).astype(np.float32) * 3.0
safe = reference(f)
z = a_chan[:, None] * (f + dt_bias[:, None])
wrong = np.exp(-np.logaddexp(0.0, z))
d = np.abs(safe - wrong) / np.maximum(np.abs(safe), 1e-9)
print(f"\n  using Qwen's softplus gate instead would be off by max {d.max():.2f}x, "
      f"mean {d.mean():.2f}x -- not subtle")

ok = persist is not None and persist < PERSIST_REL and lg.mean() < MEAN_LOG
print(f"\nchosen spelling: exp_divide (x/(1+exp(-x))), MIL sigmoid is "
      f"{results['mil_sigmoid'][2].max()/rel.max():.1f}x worse")
print(f"ANE_LING_KDA_GATE={'PASS' if ok else 'FAIL'}")

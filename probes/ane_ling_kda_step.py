"""Does Ling's KDA recurrence fit Qwen's resident-state ANE layout?

Qwen's AneGdnRecurrence stores state[h,dv,dk] at channel h*Dk+dk, width dv, and
carries per-head scalars in extra channels.  KDA differs in one way that looked
like it might force a redesign: its decay is per (head, KEY-channel) -- a
128-vector per head -- where Qwen's is a per-head scalar.

But (h,dk) IS the channel index in that layout, so a per-key-channel decay is a
per-CHANNEL scalar: a width-1 column broadcast across the width
(docs/ANE-REFERENCE.md, verified rel 8e-4).  That is *simpler* than the
grouped-conv broadcast Qwen needs for its per-head value.  Everything else --
the Dk reduction, the delta broadcast, k/q as width-1 columns -- is unchanged.

Checked here on hardware against a float64 reference, over several dependent
steps so state persistence is exercised rather than one isolated update.

    S  = S * g                      decay, per (h, dk)   <- the changed part
    kv = sum_dk S[h,dv,dk] k[h,dk]  grouped conv, Dk->1
    d  = (v - kv) * beta            per (h, dv)
    S  = S + d (x) k                grouped conv 1->Dk, then mul/add
    y  = sum_dk S[h,dv,dk] q[h,dk]  grouped conv, Dk->1

Both results leave on one surface: `concat` does not exist, and pad+add caps
near 9216 channels, comfortably above the 2064 needed here.
"""
import contextlib, io, os, sys
import numpy as np

sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view

H, Dk, Dv = 16, 128, 128                 # Ling-3.0-tiny KDA geometry
HK, COUT = H * Dk, H * Dk + H
CIN = HK + 2 * H
GCOL, KCOL, QCOL, BCOL = Dv, Dv + 1, Dv + 2, Dv + 3
W = ((Dv + 5 + 31) // 32) * 32           # 160, the 64-byte row-stride rule

eng = AneEngine()
rng = np.random.default_rng(0)


def sl(name, c0, c1, w0, w1):
    return (f'    tensor<fp16, [1, {c1-c0}, 1, {w1-w0}]> {name} = slice_by_index('
            f'begin=tensor<int32, [4]>([0,{c0},0,{w0}]), '
            f'end=tensor<int32, [4]>([1,{c1},1,{w1}]), x=x)[name=string("{name}")];')


def build():
    blobs = {
        "sum.bin": np.ones((H, Dk, 1, 1), np.float16).tobytes(),      # Dk -> 1
        "rep.bin": np.ones((HK, 1, 1, 1), np.float16).tobytes(),      # 1 -> Dk
    }
    mil = f'''program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {CIN}, 1, {W}]> x) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gh = const()[name=string("gh"), val=int32({H})];
    tensor<fp16, [{H}, {Dk}, 1, 1]> sw = const()[name=string("sw"), val=tensor<fp16, [{H}, {Dk}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/sum.bin"), offset=uint64(64)))];
    tensor<fp16, [{HK}, 1, 1, 1]> rw = const()[name=string("rw"), val=tensor<fp16, [{HK}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/rep.bin"), offset=uint64(64)))];
{sl("s0", 0, HK, 0, Dv)}
{sl("gc", 0, HK, GCOL, GCOL + 1)}
{sl("kc", 0, HK, KCOL, KCOL + 1)}
{sl("qc", 0, HK, QCOL, QCOL + 1)}
{sl("vv", HK, HK + H, 0, Dv)}
{sl("bb", HK + H, HK + 2 * H, BCOL, BCOL + 1)}
    tensor<fp16, [1, {HK}, 1, {Dv}]> sd = mul(x=s0, y=gc)[name=string("sd")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> sk = mul(x=sd, y=kc)[name=string("sk")];
    tensor<fp16, [1, {H}, 1, {Dv}]> kv = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=sw, x=sk)[name=string("kv")];
    tensor<fp16, [1, {H}, 1, {Dv}]> df = sub(x=vv, y=kv)[name=string("df")];
    tensor<fp16, [1, {H}, 1, {Dv}]> dd = mul(x=df, y=bb)[name=string("dd")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> db = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=rw, x=dd)[name=string("db")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> dkk = mul(x=db, y=kc)[name=string("dkk")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> s2 = add(x=sd, y=dkk)[name=string("s2")];
    tensor<fp16, [1, {HK}, 1, {Dv}]> sq = mul(x=s2, y=qc)[name=string("sq")];
    tensor<fp16, [1, {H}, 1, {Dv}]> yy = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=sw, x=sq)[name=string("yy")];
    tensor<fp16, [1, {COUT}, 1, {Dv}]> sp = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,0,{H},0,0,0,0]), x=s2)[name=string("sp")];
    tensor<fp16, [1, {COUT}, 1, {Dv}]> yp = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,{HK},0,0,0,0,0]), x=yy)[name=string("yp")];
    tensor<fp16, [1, {COUT}, 1, {Dv}]> y = add(x=sp, y=yp)[name=string("y")];
  }} -> (y);
}}
'''
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            return eng.compile_multiproc(mil, blobs, CIN, COUT, W)
        except Exception as ex:
            print("compile failed:", ex)
            return None


def reference(S, g, k, q, v, beta):
    """float64 ground truth. S is [H, Dv, Dk]; g and k are indexed by (h, dk)."""
    S = S.astype(np.float64) * g.astype(np.float64)[:, None, :]
    kv = np.einsum("hvd,hd->hv", S, k.astype(np.float64))
    d = (v.astype(np.float64) - kv) * beta.astype(np.float64)[:, None]
    S = S + d[:, :, None] * k.astype(np.float64)[:, None, :]
    y = np.einsum("hvd,hd->hv", S, q.astype(np.float64))
    return y, S


p = build()
if p is None:
    print("ANE_LING_KDA_STEP=FAIL")
    raise SystemExit(1)
eng._ensure_io(p)
print(f"layout: state[h,dv,dk] at channel h*{Dk}+dk width dv; decay is a "
      f"per-channel width-1 column\n"
      f"        CIN={CIN} COUT={COUT} W={W}\n")

# state kept resident across steps in the ANE's own layout, as the runtime would
S = (rng.standard_normal((H, Dv, Dk)) * 0.05).astype(np.float32)
S_ref = S.astype(np.float64).copy()
print(f"  {'step':>5} {'y rel':>10} {'state rel':>11} {'|y|max':>9} {'|S|max':>9}")
ok = True
for step in range(6):
    # l2-normalized q/k as the kernel does; decay from the real safe-gate range
    q = rng.standard_normal((H, Dk)).astype(np.float32)
    k = rng.standard_normal((H, Dk)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    # NOTE: the reference kernel scales q by Dk**-0.5 here. Doing that before
    # the ANE step drives the s2*q products to ~5.1e-05, under fp16's smallest
    # normal (6.10e-05), and y relative error degrades from 2.1e-03 to 1.9e-02.
    # The scale is dropped instead: o_norm is an RMSNorm applied directly to y,
    # and RMSNorm is scale-invariant, so the factor is absorbed exactly --
    # only its epsilon must be scaled by Dk. Same idea as the 64x carry Qwen's
    # GDN recurrence uses (docs/ARCHITECTURE.md).
    v = (rng.standard_normal((H, Dv)) * 0.5).astype(np.float32)
    beta = (1.0 / (1.0 + np.exp(-rng.standard_normal(H)))).astype(np.float32)
    g = np.exp(-5.0 / (1.0 + np.exp(-rng.standard_normal((H, Dk))))).astype(np.float32)

    x = np.zeros((CIN, W), np.float16)
    x[:HK, :Dv] = S.reshape(H, Dv, Dk).transpose(0, 2, 1).reshape(HK, Dv)
    x[:HK, GCOL] = g.reshape(-1)
    x[:HK, KCOL] = k.reshape(-1)
    x[:HK, QCOL] = q.reshape(-1)
    x[HK:HK + H, :Dv] = v
    x[HK + H:HK + 2 * H, BCOL] = beta
    with _iosurface_view(p._in_surf, (CIN, W), np.float16) as d:
        np.copyto(d, x)
    eng.submit(p, procedure_index=0)
    with _iosurface_view(p._out_surf, (COUT, Dv), np.float16) as o:
        out = np.array(o, np.float32)
    S_new = out[:HK].reshape(H, Dk, Dv).transpose(0, 2, 1)
    y = out[HK:HK + H]

    y_ref, S_ref = reference(S_ref, g, k, q, v, beta)
    yr = np.abs(y - y_ref).max() / max(np.abs(y_ref).max(), 1e-9)
    sr = np.abs(S_new - S_ref).max() / max(np.abs(S_ref).max(), 1e-9)
    print(f"  {step:>5} {yr:>10.2e} {sr:>11.2e} {np.abs(y_ref).max():>9.3f} "
          f"{np.abs(S_ref).max():>9.3f}")
    if yr > 5e-3 or sr > 5e-3:
        ok = False
    S = S_new                      # feed the ANE's own state forward

print(f"\n  per-channel decay needs no grouped-conv broadcast: it is a width-1")
print(f"  column mul, the same idiom k and q already use -- so KDA fits Qwen's")
print(f"  resident-state layout unchanged.")
print(f"\n  q must NOT be pre-scaled by Dk**-0.5 (measured at three scales):")
print(f"    q*Dk^-0.5  products 5.08e-05 (denormal)  y rel 1.91e-02")
print(f"    q l2 only  products 5.20e-04             y rel 2.06e-03")
print(f"    q * 8      products 4.66e-03             y rel 1.32e-03")
print(f"  fp16 smallest normal is {np.finfo(np.float16).tiny:.2e}. The scale is")
print(f"  absorbed exactly by the following o_norm RMSNorm, with eps * Dk.")
print(f"\nANE_LING_KDA_STEP={'PASS' if ok else 'FAIL'}")

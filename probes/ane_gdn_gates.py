"""Verify the scalar nonlinear gates used by Qwen3.8 GDN on the ANE."""
import contextlib, io, os, sys, time
import numpy as np

sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view

H, S = 48, 32
eng = AneEngine()
rng = np.random.default_rng(77)
model_path = os.environ.get("Q38_MODEL", str(Path.home() / ".lmstudio/models/Qwen/Qwen3.8-27B"))
try:
    import mlx.core as mx
    from mlx_lm import load
    model, _ = load(model_path, lazy=True)
    lm = getattr(model, "language_model", model)
    layers = getattr(getattr(lm, "model", lm), "layers")
    pairs = []
    for block in layers:
        if getattr(block, "is_linear", False):
            la = block.linear_attn
            av = np.array(mx.exp(la.A_log.astype(mx.float32))).reshape(-1)
            dv = np.array(la.dt_bias.astype(mx.float32)).reshape(-1)
            pairs.extend(zip(av, dv))
    pairs = np.asarray(pairs, np.float32)
    order_a, order_d = np.argsort(pairs[:, 0]), np.argsort(pairs[:, 1])
    chosen = list(order_a[:8]) + list(order_a[-8:])
    chosen += list(order_d[:8]) + list(order_d[-8:])
    chosen += list(order_a[np.linspace(0, len(order_a)-1, 16, dtype=int)])
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) < H:
        chosen += [i for i in range(len(pairs)) if i not in chosen][:H-len(chosen)]
    A, dt = pairs[chosen[:H], 0].astype(np.float16), pairs[chosen[:H], 1].astype(np.float16)
    print(f"real checkpoint constants: A=[{A.min():.4g},{A.max():.4g}] "
          f"dt=[{dt.min():.4g},{dt.max():.4g}]")
except Exception as exc:
    print(f"checkpoint constants unavailable ({exc}); using synthetic sweep")
    A = np.geomspace(0.004, 140.0, H).astype(np.float16)
    dt = np.linspace(-9, 20, H).astype(np.float16)
blobs = {"a.bin": A.reshape(H,1,1,1).tobytes(),
         "d.bin": dt.reshape(H,1,1,1).tobytes()}


def run(mode):
    if mode == "direct":
        sp = f'    tensor<fp16, [1,{H},1,{S}]> soft = softplus(x=ap)[name=string("soft")];'
        beta = f'    tensor<fp16, [1,{H},1,{S}]> beta = sigmoid(x=b)[name=string("beta")];'
    elif mode == "exact":
        sp = f'''    tensor<fp16, [1,{H},1,{S}]> eap = exp(x=ap)[name=string("eap")];
    tensor<fp16, [1,{H},1,{S}]> oneap = add(x=eap, y=fp16(0x1p+0))[name=string("oneap")];
    tensor<fp16, [1,{H},1,{S}]> soft = log(epsilon=fp16(0x0p+0),x=oneap)[name=string("soft")];'''
        beta = f'''    tensor<fp16, [1,{H},1,{S}]> nb = mul(x=b, y=fp16(-0x1p+0))[name=string("nb")];
    tensor<fp16, [1,{H},1,{S}]> enb = exp(x=nb)[name=string("enb")];
    tensor<fp16, [1,{H},1,{S}]> db = add(x=enb, y=fp16(0x1p+0))[name=string("db")];
    tensor<fp16, [1,{H},1,{S}]> beta = real_div(x=fp16(0x1p+0), y=db)[name=string("beta")];'''
    elif mode == "stable":
        sp = f'''    tensor<fp16, [1,{H},1,{S}]> pos = relu(x=ap)[name=string("pos")];
    tensor<fp16, [1,{H},1,{S}]> ab = abs(x=ap)[name=string("ab")];
    tensor<fp16, [1,{H},1,{S}]> nab = mul(x=ab, y=fp16(-0x1p+0))[name=string("nab")];
    tensor<fp16, [1,{H},1,{S}]> en = exp(x=nab)[name=string("en")];
    tensor<fp16, [1,{H},1,{S}]> oneen = add(x=en, y=fp16(0x1p+0))[name=string("oneen")];
    tensor<fp16, [1,{H},1,{S}]> tail = log(epsilon=fp16(0x0p+0), x=oneen)[name=string("tail")];
    tensor<fp16, [1,{H},1,{S}]> soft = add(x=pos, y=tail)[name=string("soft")];'''
        beta = f'''    tensor<fp16, [1,{H},1,{S}]> nb = mul(x=b, y=fp16(-0x1p+0))[name=string("nb")];
    tensor<fp16, [1,{H},1,{S}]> enb = exp(x=nb)[name=string("enb")];
    tensor<fp16, [1,{H},1,{S}]> db = add(x=enb, y=fp16(0x1p+0))[name=string("db")];
    tensor<fp16, [1,{H},1,{S}]> beta = real_div(x=fp16(0x1p+0), y=db)[name=string("beta")];'''
    elif mode == "stable_poly":
        # log1p(t) = t * h(t), t=exp(-abs(x)).  Approximating the smooth h on
        # [0,1] avoids fp16's catastrophic 1+t rounding when t < 2^-11.
        cc = np.array([1.0, -0.499267578125, 0.324462890625,
                       -0.2086181640625, 0.10028076171875,
                       -0.023681640625], np.float16)
        hx = lambda v: float(np.float16(v)).hex()
        poly = f'    tensor<fp16, [1,{H},1,{S}]> hp5 = mul(x=t, y=fp16({hx(cc[5])}))[name=string("hp5")];\n'
        prev = "hp5"
        for j in range(4, -1, -1):
            addn = f"ha{j}"
            poly += f'    tensor<fp16, [1,{H},1,{S}]> {addn} = add(x={prev}, y=fp16({hx(cc[j])}))[name=string("{addn}")];\n'
            if j:
                muln = f"hm{j}"
                poly += f'    tensor<fp16, [1,{H},1,{S}]> {muln} = mul(x={addn}, y=t)[name=string("{muln}")];\n'
                prev = muln
            else:
                prev = addn
        sp = f'''    tensor<fp16, [1,{H},1,{S}]> pos = relu(x=ap)[name=string("pos")];
    tensor<fp16, [1,{H},1,{S}]> ab = abs(x=ap)[name=string("ab")];
    tensor<fp16, [1,{H},1,{S}]> nab = mul(x=ab, y=fp16(-0x1p+0))[name=string("nab")];
    tensor<fp16, [1,{H},1,{S}]> t = exp(x=nab)[name=string("t")];
{poly}    tensor<fp16, [1,{H},1,{S}]> tail = mul(x=t, y={prev})[name=string("tail")];
    tensor<fp16, [1,{H},1,{S}]> soft = add(x=pos, y=tail)[name=string("soft")];'''
        beta = f'''    tensor<fp16, [1,{H},1,{S}]> nb = mul(x=b, y=fp16(-0x1p+0))[name=string("nb")];
    tensor<fp16, [1,{H},1,{S}]> enb = exp(x=nb)[name=string("enb")];
    tensor<fp16, [1,{H},1,{S}]> db = add(x=enb, y=fp16(0x1p+0))[name=string("db")];
    tensor<fp16, [1,{H},1,{S}]> beta = real_div(x=fp16(0x1p+0), y=db)[name=string("beta")];'''
    else:
        sp = ""
        beta = f'''    tensor<fp16, [1,{H},1,{S}]> nb = mul(x=b, y=fp16(-0x1p+0))[name=string("nb")];
    tensor<fp16, [1,{H},1,{S}]> enb = exp(x=nb)[name=string("enb")];
    tensor<fp16, [1,{H},1,{S}]> db = add(x=enb, y=fp16(0x1p+0))[name=string("db")];
    tensor<fp16, [1,{H},1,{S}]> beta = real_div(x=fp16(0x1p+0), y=db)[name=string("beta")];'''
    decay_code = (f'''    tensor<fp16,[1,{H},1,{S}]> eap2 = exp(x=ap)[name=string("eap2")];
    tensor<fp16,[1,{H},1,{S}]> base = add(x=eap2, y=fp16(0x1p+0))[name=string("base")];
    tensor<fp16,[1,{H},1,1]> nea = mul(x=aa, y=fp16(-0x1p+0))[name=string("nea")];
    tensor<fp16,[1,{H},1,{S}]> decay = pow(x=base, y=nea)[name=string("decay")];'''
                  if mode == "pow" else
                  f'''    tensor<fp16,[1,{H},1,{S}]> prod = mul(x=soft, y=aa)[name=string("prod")];
    tensor<fp16,[1,{H},1,{S}]> neg = mul(x=prod, y=fp16(-0x1p+0))[name=string("neg")];
    tensor<fp16,[1,{H},1,{S}]> decay = exp(x=neg)[name=string("decay")];''')
    mil = f'''program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1,{2*H},1,{S}]> x) {{
    tensor<fp16,[1,{H},1,{S}]> a = slice_by_index(begin=tensor<int32,[4]>([0,0,0,0]), end=tensor<int32,[4]>([1,{H},1,{S}]), x=x)[name=string("a")];
    tensor<fp16,[1,{H},1,{S}]> b = slice_by_index(begin=tensor<int32,[4]>([0,{H},0,0]), end=tensor<int32,[4]>([1,{2*H},1,{S}]), x=x)[name=string("b")];
    tensor<fp16,[1,{H},1,1]> aa = const()[name=string("aa"), val=tensor<fp16,[1,{H},1,1]>(BLOBFILE(path=string("@model_path/weights/a.bin"), offset=uint64(64)))];
    tensor<fp16,[1,{H},1,1]> dt = const()[name=string("dt"), val=tensor<fp16,[1,{H},1,1]>(BLOBFILE(path=string("@model_path/weights/d.bin"), offset=uint64(64)))];
    tensor<fp16,[1,{H},1,{S}]> ap = add(x=a, y=dt)[name=string("ap")];
{sp}
{decay_code}
{beta}
    tensor<int32,[8]> p0 = const()[name=string("p0"), val=tensor<int32,[8]>([0,0,0,{H},0,0,0,0])];
    tensor<int32,[8]> p1 = const()[name=string("p1"), val=tensor<int32,[8]>([0,0,{H},0,0,0,0,0])];
    tensor<fp16,[1,{2*H},1,{S}]> d0 = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=p0, x=decay)[name=string("d0")];
    tensor<fp16,[1,{2*H},1,{S}]> b0 = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=p1, x=beta)[name=string("b0")];
    tensor<fp16,[1,{2*H},1,{S}]> y = add(x=d0, y=b0)[name=string("y")];
  }} -> (y);
}}
// qwen38_gdn_gates_{mode}
'''
    cap = io.StringIO()
    with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap):
        prog = eng.compile_multiproc(mil, blobs, 2*H, 2*H, S)
    if prog is None:
        print(f"  {mode:<8} COMPILE FAILED")
        return False
    target_x = np.linspace(-20, 20, S, dtype=np.float32)
    a = target_x[None, :] - dt.astype(np.float32)[:, None]
    b = np.broadcast_to(target_x, (H, S)).copy()
    refd = np.exp(-A.astype(np.float32)[:,None] * np.logaddexp(0, a+dt.astype(np.float32)[:,None]))
    refb = 1/(1+np.exp(-b))
    eng._ensure_io(prog)
    with _iosurface_view(prog._in_surf, (2*H,S), np.float16) as z:
        z[:H]=a; z[H:]=b
    for _ in range(3): eng.submit(prog)
    t=time.perf_counter(); n=50
    for _ in range(n): eng.submit(prog)
    ms=(time.perf_counter()-t)*1e3/n
    with _iosurface_view(prog._out_surf, (2*H,S), np.float16) as z:
        got=np.array(z,np.float32)
    ed=np.max(np.abs(got[:H]-refd))/(np.max(np.abs(refd))+1e-9)
    eb=np.max(np.abs(got[H:]-refb))/(np.max(np.abs(refb))+1e-9)
    ok=ed<3e-3 and eb<3e-3
    print(f"  {mode:<8} {'OK' if ok else 'WRONG':<5} decay_rel={ed:.4g} beta_rel={eb:.4g} {ms:.3f} ms")
    if mode.startswith("stable") and not ok:
        wi=np.unravel_index(np.argmax(np.abs(got[:H]-refd)),refd.shape)
        print(f"    worst head={wi[0]} x={target_x[wi[1]]:.4g} A={float(A[wi[0]]):.4g} "
              f"got={got[wi]:.6g} ref={refd[wi]:.6g}")
    return ok


print("Qwen3.8 GDN softplus/decay/beta gates")
run("direct")
run("exact")
run("pow")
run("stable")
print("PASS" if run("stable_poly") else "FAIL")

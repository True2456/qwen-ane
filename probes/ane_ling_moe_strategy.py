"""Which MoE strategy should the Ling-3.0-tiny port use? Measure before building.

docs/ANE-MOE-HANDOFF.md 27 measured Qwen3.6-35B-A3B MoE at ~11x slower than the
GPU: the ANE has no `gather`, so the selected experts must be physically staged
into a weight surface, and dynamic (feature-map) weights cannot be quantized
because constexpr_blockwise_shift_scale needs a const operand.

Ling-3.0-tiny may invert that.  One expert is 3 x [512,1536] = 4.5 MiB bf16 /
1.125 MiB int4, so all 128 experts of a layer are 144 MiB at int4 and all 23 MoE
layers are ~3.3 GB -- small enough to BAKE as constants and simply compute every
expert, masking to the routed 8.  That option did not exist at 35B scale.

  (A) dynamic staging   stage the 8 routed experts per layer per token, fp16
  (B) baked dense       all 128 experts as int4 constants, compute all, mask

Timing uses random weights, which is correct for speed and wrong for accuracy
(docs/ANE-REFERENCE.md).  The router comparison uses the real layer-1 router.
"""
import contextlib, io, json, os, struct, sys, time
import numpy as np

sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, AneDynamicLinear, _iosurface_view

MODEL = os.environ.get("Q38_LING_MODEL",
                       str(Path.home() / ".lmstudio/models/inclusionAI/Ling-3.0-tiny"))
H, M, NE, TOP_K = 1536, 512, 128, 8      # hidden, moe_intermediate, experts, top-k
N_GROUP, TOPK_GROUP, SCALE = 8, 4, 2.5
MOE_LAYERS = 23

eng = AneEngine()
rng = np.random.default_rng(0)


def med(fn, n=15):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    ts.sort(); return ts[len(ts) // 2]


# ------------------------------------------------------------------ strategy A
def strategy_a(S=32):
    """Stage TOP_K experts into two dynamic-weight programs, then dispatch."""
    gu = AneDynamicLinear.compile(H, 2 * TOP_K * M, S)       # [8192, 1536]
    dn = AneDynamicLinear.compile(TOP_K * M, H, S)           # [1536, 4096]
    if gu is None or dn is None:
        return None
    # expert pools, exactly the layout ane_serve.AneMoE keeps: down pre-transposed
    G = rng.standard_normal((NE, M, H)).astype(np.float16) * 0.02
    U = rng.standard_normal((NE, M, H)).astype(np.float16) * 0.02
    D = rng.standard_normal((NE, M, H)).astype(np.float16) * 0.02   # [NE, M, H]
    sel = rng.choice(NE, TOP_K, replace=False)
    x = rng.standard_normal((H,)).astype(np.float16) * 0.02
    MS = TOP_K * M

    def stage():
        with _iosurface_view(gu._w_surf, (2 * MS, H), np.float16) as w:
            for j, e in enumerate(sel):
                w[j*M:(j+1)*M] = G[e]; w[MS+j*M:MS+(j+1)*M] = U[e]
        with _iosurface_view(dn._w_surf, (MS, H), np.float16) as w:
            for j, e in enumerate(sel):
                w[j*M:(j+1)*M] = D[e]

    def dispatch():
        gu.submit(); dn.submit()

    def both():
        stage(); dispatch()

    with _iosurface_view(gu._x_surf, (H, S), np.float16) as d:
        d[:] = 0; d[:, 0] = x
    with _iosurface_view(dn._x_surf, (MS, S), np.float16) as d:
        d[:] = 0
    staged_mb = (2 * MS * H + MS * H) * 2 / 1e6
    return dict(stage=med(stage), dispatch=med(dispatch), total=med(both),
                staged_mb=staged_mb)


# ------------------------------------------------------------------ strategy B
CONV = ('    string pt=const()[name=string("pt"),val=string("valid")];\n'
        '    tensor<int32,[2]> st=const()[name=string("st"),val=tensor<int32,[2]>([1,1])];\n'
        '    tensor<int32,[4]> pd=const()[name=string("pd"),val=tensor<int32,[4]>([0,0,0,0])];\n'
        '    tensor<int32,[2]> dl=const()[name=string("dl"),val=tensor<int32,[2]>([1,1])];\n'
        '    int32 gr=const()[name=string("gr"),val=int32(1)];')


def _q4(W):
    s = np.abs(W).max(axis=1, keepdims=True) / 7.0
    s = np.where(s == 0, 1, s)
    q = np.clip(np.rint(W / s), -8, 7).astype(np.int8)
    n = q.reshape(-1).astype(np.uint8) & 0x0F
    return (n[0::2] | (n[1::2] << 4)).tobytes(), s.astype(np.float16).tobytes()


def _decl(name, out_d, in_d):
    return (f'    tensor<int4, [{out_d}, {in_d}, 1, 1]> {name}q = const()[name=string("{name}q"), '
            f'val=tensor<int4, [{out_d}, {in_d}, 1, 1]>(BLOBFILE('
            f'path=string("@model_path/weights/{name}.bin"), offset=uint64(64)))];\n'
            f'    tensor<fp16, [{out_d}, 1, 1, 1]> {name}s = const()[name=string("{name}s"), '
            f'val=tensor<fp16, [{out_d}, 1, 1, 1]>(BLOBFILE('
            f'path=string("@model_path/weights/{name}s.bin"), offset=uint64(64)))];\n'
            f'    tensor<fp16, [{out_d}, {in_d}, 1, 1]> {name}w = '
            f'constexpr_blockwise_shift_scale(data={name}q, scale={name}s)[name=string("{name}d")];')


def build_conv(out_d, in_d, S, parts=1):
    """One int4 conv, optionally split across `parts` disjoint input channels."""
    W = rng.standard_normal((out_d, in_d)).astype(np.float32) * 0.02
    blobs, decls, terms = {}, [], []
    c = in_d // parts
    for i in range(parts):
        p, s = _q4(W[:, i*c:(i+1)*c])
        blobs[f"w{i}.bin"], blobs[f"w{i}s.bin"] = p, s
        decls.append(_decl(f"w{i}", out_d, c))
        if parts > 1:
            decls.append(
                f'    tensor<int32,[4]> b{i}=const()[name=string("b{i}"),val=tensor<int32,[4]>([0,{i*c},0,0])];\n'
                f'    tensor<int32,[4]> e{i}=const()[name=string("e{i}"),val=tensor<int32,[4]>([1,{(i+1)*c},1,{S}])];\n'
                f'    tensor<fp16,[1,{c},1,{S}]> x{i}=slice_by_index(begin=b{i},end=e{i},x=x)[name=string("s{i}")];')
        src = f"x{i}" if parts > 1 else "x"
        decls.append(f'    tensor<fp16,[1,{out_d},1,{S}]> p{i}=conv(dilations=dl,groups=gr,pad=pd,'
                     f'pad_type=pt,strides=st,weight=w{i}w,x={src})[name=string("c{i}")];')
        terms.append(f"p{i}")
    acc = terms[0]
    for i in range(1, parts):
        nx = f"a{i}"
        decls.append(f'    tensor<fp16,[1,{out_d},1,{S}]> {nx}=add(x={acc},y={terms[i]})[name=string("{nx}")];')
        acc = nx
    decls.append(f'    tensor<fp16,[1,{out_d},1,{S}]> y=identity(x={acc})[name=string("id")];')
    mil = f'''program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {in_d}, 1, {S}]> x) {{
{CONV}
{chr(10).join(decls)}
  }} -> (y);
}}
'''
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            return eng.compile_multiproc(mil, blobs, in_d, out_d, S)
        except Exception:
            return None


def strategy_b(S=32, gu_chunks=4, dn_parts=4):
    """All 128 experts baked at int4. gate|up chunked, down input-split.

    The four gate|up chunks are the same shape, and ANECCompile is content
    addressed, so building all four in one process either deduplicates to one
    resident model or exhausts the budget.  Measure one chunk and multiply --
    they are independent dispatches, which is the same accounting
    ane_decode_budget.py uses.
    """
    gu_rows = 2 * NE * M                      # 131072, above the single-conv limit
    per = gu_rows // gu_chunks
    gu = build_conv(per, H, S)
    if gu is None:
        return None
    eng._ensure_io(gu)
    gu_one = med(lambda: eng.submit(gu, procedure_index=0), n=15)
    del gu
    dn = build_conv(H, NE * M, S, parts=dn_parts)
    if dn is None:
        return None
    eng._ensure_io(dn)
    dn_t = med(lambda: eng.submit(dn, procedure_index=0), n=15)
    mb = (gu_rows * H + H * NE * M) * 0.5 / 1e6
    return dict(gu=gu_one * gu_chunks, gu_one=gu_one, dn=dn_t,
                total=gu_one * gu_chunks + dn_t, blob_mb=mb)


# ------------------------------------------------- (C) router fp32 vs fp16
def _route(x, W, bias, dtype):
    """Exact BailingMoeV3 group-limited top-k. Returns selected expert ids."""
    logits = x.astype(dtype) @ W.astype(dtype).T
    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
    routing = scores + bias
    g = routing.reshape(N_GROUP, -1)
    gs = np.sort(g, axis=-1)[:, -2:].sum(-1)                  # sum of top-2
    live = np.argsort(gs)[-TOPK_GROUP:]
    mask = np.zeros(N_GROUP, bool); mask[live] = True
    masked = np.where(np.repeat(mask, g.shape[1]), routing, -np.inf)
    sel = np.argsort(masked)[-TOP_K:]
    return set(sel.tolist())


def router_check(trials=2000):
    D = MODEL
    wm = json.load(open(os.path.join(D, "model.safetensors.index.json")))["weight_map"]

    def get(name):
        f = os.path.join(D, wm[name])
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            i = hdr[name]; o = i["data_offsets"]
            fh.seek(8 + n + o[0]); raw = fh.read(o[1] - o[0])
        if i["dtype"] == "BF16":
            a = np.frombuffer(raw, np.uint16).astype(np.uint32)
            return (a << 16).view(np.float32).reshape(i["shape"])
        return np.frombuffer(raw, np.float32).reshape(i["shape"])

    W = get("model.layers.1.mlp.gate.weight")
    bias = get("model.layers.1.mlp.gate.expert_bias")
    # Post-RMSNorm hidden states have ~unit RMS scaled by the norm weight; this
    # is a distributional stand-in, not real activations. Milestone 4 is the
    # definitive test with real hidden states.
    nw = get("model.layers.1.post_attention_layernorm.weight")
    agree = same = 0
    for _ in range(trials):
        v = rng.standard_normal(H).astype(np.float32)
        v = v / np.sqrt((v * v).mean()) * nw
        a = _route(v, W, bias, np.float32)
        b = _route(v, W, bias, np.float16)
        agree += len(a & b); same += (a == b)
    return dict(expert_overlap=agree / (trials * TOP_K), exact=same / trials)


# Each case runs in its own process: ANE programs are never unloaded, and the
# ~127 resident-program budget is system-wide, so one process cannot hold the
# dynamic pair and the baked set at the same time.
def main() -> None:
    case = sys.argv[1] if len(sys.argv) > 1 else "all"
    if case == "router":
        r = router_check()
        print(f"  (C) router fp16 vs fp32: {100*r['expert_overlap']:.2f}% of the top-8 "
              f"agree, {100*r['exact']:.1f}% of tokens select an identical set")
        print("      (distributional stand-in for hidden states, not real activations)")
        return
    kind, S = case.split(":")
    S = int(S)
    if kind == "a":
        a = strategy_a(S)
        if a is None:
            print(f"  S={S} (A) dynamic staging : FAILED"); return
        print(f"  S={S} (A) dynamic staging : stage {a['stage']:6.3f} + dispatch "
              f"{a['dispatch']:6.3f} = {a['total']:6.3f} ms/layer   "
              f"({a['staged_mb']:.1f} MB staged)")
        print(f"          -> {a['total']*MOE_LAYERS:7.1f} ms/token over {MOE_LAYERS} layers")
    else:
        b = strategy_b(S)
        if b is None:
            print(f"  S={S} (B) baked dense     : FAILED"); return
        print(f"  S={S} (B) baked dense     : gate|up {b['gu']:6.3f} (4 x {b['gu_one']:.3f}) "
              f"+ down {b['dn']:6.3f} = {b['total']:6.3f} ms/layer   "
              f"({b['blob_mb']:.0f} MB int4)")
        print(f"          -> {b['total']*MOE_LAYERS:7.1f} ms/token at 1 position/step, "
              f"but {b['total']*MOE_LAYERS/S:6.2f} ms/token with all {S} lanes filled")
        print(f"          -> {b['blob_mb']*MOE_LAYERS/1000:.2f} GB of blobs, no staging at all")


if __name__ == "__main__":
    main()

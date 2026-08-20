"""What would a Ling-3.0-tiny decode token cost on the ANE, block by block?

MLX runs this checkpoint at 27.3 tok/s on this machine, which is 36.6 ms per
token. This measures every distinct ANE block at Ling's real shapes and weights
it by how many times it fires per token, so the budget can be compared against
that directly instead of guessed at.

Precision follows docs/LING-PORT.md: routed experts int4, attention fp16,
lm_head int4.

One case per process: ANE programs are never unloaded and the ~127-program
budget is system-wide, so several large blocks cannot be resident at once.

    for c in kda_in kda_conv kda_out mla_a mla_qb mla_absorb mla_unabsorb \\
             mla_dense moe_gu moe_down shared_gu shared_down lm_head; do
        python3 -P probes/ane_ling_budget.py $c
    done
"""
import contextlib, io, os, sys, time
import numpy as np

sys.path.insert(0, os.environ.get("Q38_ANE_ENGINE",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools"))
from ane_peak_real import build, build_split, measure, eng          # noqa: E402
import runtime.q38_ane_engine as E                                  # noqa: E402
from runtime.q38_ane_engine import _iosurface_view                  # noqa: E402

MLX_TOK_S = 27.269                       # measured, bf16 on the GPU
BUDGET_MS = 1000.0 / MLX_TOK_S           # 36.7 ms/token

H, P, HEADS, D = 1536, 2048, 16, 128
KDA, MLA, MOE = 18, 6, 23
LAT, ROPE = 512, 64

# name -> (out, in, bits, parts, count per token, note)
CASES = {
    "kda_in":       (5 * P + HEADS, H, 16, 1, KDA, "q|k|v|f|g|b fused"),
    "kda_out":      (H, P, 16, 1, KDA, "o_proj"),
    "mla_a":        (256 + 576 + HEADS, H, 16, 1, MLA, "q_a|kv_a|g fused"),
    "mla_qb":       (3072, 256, 16, 1, MLA, "q_b"),
    "mla_dense":    (H, P, 16, 1, MLA, "dense"),
    "moe_gu":       (32768, H, 4, 1, MOE * 4, "gate|up, 4 chunks of 128 experts"),
    "moe_down":     (H, 128 * 512, 4, 4, MOE, "down, all 128 experts"),
    "shared_gu":    (1024, H, 16, 1, MOE, "shared expert gate|up"),
    "shared_down":  (H, 512, 16, 1, MOE, "shared expert down"),
    "lm_head":      (52395, H, 4, 1, 3, "vocab 157184 in 3 chunks"),
}


def grouped(out_per_head, in_per_head, S):
    """Block-diagonal conv over the 16 heads: the absorbed MLA projections."""
    cin, cout = HEADS * in_per_head, HEADS * out_per_head
    w = (np.random.default_rng(0).standard_normal((cout, in_per_head)) * 0.02)
    blobs = {"w.bin": w.astype(np.float16).tobytes()}
    mil = f'''program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {cin}, 1, {S}]> x) {{
    string pt=const()[name=string("pt"),val=string("valid")];
    tensor<int32,[2]> st=const()[name=string("st"),val=tensor<int32,[2]>([1,1])];
    tensor<int32,[4]> pd=const()[name=string("pd"),val=tensor<int32,[4]>([0,0,0,0])];
    tensor<int32,[2]> dl=const()[name=string("dl"),val=tensor<int32,[2]>([1,1])];
    int32 gh=const()[name=string("gh"),val=int32({HEADS})];
    tensor<fp16, [{cout}, {in_per_head}, 1, 1]> ww = const()[name=string("ww"), val=tensor<fp16, [{cout}, {in_per_head}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/w.bin"), offset=uint64(64)))];
    tensor<fp16,[1,{cout},1,{S}]> y=conv(dilations=dl,groups=gh,pad=pd,pad_type=pt,strides=st,weight=ww,x=x)[name=string("mm")];
  }} -> (y);
}}
'''
    cap = io.StringIO()
    with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap):
        return eng.compile_multiproc(mil, blobs, cin, cout, S)


def depthwise(channels, k, S):
    """KDA's three K=4 causal convs, concatenated channel-wise into one."""
    w = (np.random.default_rng(0).standard_normal((channels, 1, 1, k)) * 0.1)
    blobs = {"w.bin": w.astype(np.float16).tobytes()}
    mil = f'''program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {channels}, 1, {S + k - 1}]> x) {{
    string pt=const()[name=string("pt"),val=string("valid")];
    tensor<int32,[2]> st=const()[name=string("st"),val=tensor<int32,[2]>([1,1])];
    tensor<int32,[4]> pd=const()[name=string("pd"),val=tensor<int32,[4]>([0,0,0,0])];
    tensor<int32,[2]> dl=const()[name=string("dl"),val=tensor<int32,[2]>([1,1])];
    int32 gc=const()[name=string("gc"),val=int32({channels})];
    tensor<fp16, [{channels}, 1, 1, {k}]> ww = const()[name=string("ww"), val=tensor<fp16, [{channels}, 1, 1, {k}]>(BLOBFILE(path=string("@model_path/weights/w.bin"), offset=uint64(64)))];
    tensor<fp16,[1,{channels},1,{S}]> y=conv(dilations=dl,groups=gc,pad=pd,pad_type=pt,strides=st,weight=ww,x=x)[name=string("dw")];
  }} -> (y);
}}
'''
    cap = io.StringIO()
    with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap):
        return eng.compile_multiproc(mil, blobs, channels, channels, S + k - 1)


def run(case, S):
    if case == "kda_conv":
        p = depthwise(3 * P, 4, S)
        if p is None:
            return None
        eng._ensure_io(p)
        ms = measure(p, 3 * P, 3 * P, S + 3, n=15)[0]
        return ms, KDA, "3 x K=4 depthwise, fused", 3 * P * 4 * 2 / 1e6
    if case in ("mla_absorb", "mla_unabsorb"):
        a, b = (D, LAT) if case == "mla_absorb" else (LAT, D)
        p = grouped(b, a, S)
        if p is None:
            return None
        eng._ensure_io(p)
        ms = measure(p, HEADS * b, HEADS * a, S, n=15)[0]
        return ms, MLA, "grouped conv, 16 heads", HEADS * a * b * 2 / 1e6
    out, inp, bits, parts, count, note = CASES[case]
    p, _ = (build(out, inp, S, bits) if parts == 1
            else build_split(out, inp, S, bits, parts))
    if p is None:
        return None
    ms, _ = measure(p, out, inp, S, n=15)
    return ms, count, note, out * inp * (bits / 8) / 1e6


if __name__ == "__main__":
    case = sys.argv[1]
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    r = run(case, S)
    if r is None:
        print(f"  {case:>13} S={S:<4} FAILED"); raise SystemExit(1)
    ms, count, note, mb = r
    print(f"  {case:>13} {count:>4}x {ms:>8.3f} ms {ms*count:>9.2f} ms/token "
          f"{100*ms*count/BUDGET_MS:>7.1f}% {mb:>8.1f} MB   {note}")

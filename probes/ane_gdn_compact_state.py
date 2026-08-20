"""Progressively compile and verify a stride-compatible GDN recurrence.

The first HK rows of a [C, 128] IOSurface are the recurrent state.  The ANE
writes the next [HK, 128] state directly into the first HK rows of the other
ping-pong input surface, so no state copy or transpose is needed between
tokens.  Small per-token tensors occupy aligned [H, 128] row blocks after the
state and are reshaped to broadcast columns inside the graph.

Run without arguments to compile every cumulative stage, then execute the
complete two-output graph for two dependent steps.
"""
import argparse
import contextlib
import ctypes
import io
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.expanduser("~/AppleLLM/q38_native_engine"))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view


H, DK, DV = 48, 128, 128
HK = H * DK
W = DV
# state, decay, k, q, v, beta; every parameter block is H x W.  Keeping C a
# multiple of 32 also avoids conflating a compiler alignment issue with MIL.
CIN = HK + 5 * H
BLOCK = {
    "decay": HK,
    "k": HK + H,
    "q": HK + 2 * H,
    "v": HK + 3 * H,
    "beta": HK + 4 * H,
}


def sl(name, c0, c1, w0, w1):
    return (
        f'    tensor<fp16, [1, {c1-c0}, 1, {w1-w0}]> {name} = '
        f'slice_by_index(begin=tensor<int32, [4]>([0,{c0},0,{w0}]), '
        f'end=tensor<int32, [4]>([1,{c1},1,{w1}]), x=x)'
        f'[name=string("{name}")];'
    )


def mil_for(stage, input_channels=CIN, input_width=W):
    lines = [
        "program(1.3)",
        E._BUILD_INFO,
        "{",
        f"  func main<ios18>(tensor<fp16, [1, {input_channels}, 1, {input_width}]> x) {{",
    ]
    if stage >= 1:
        lines += [
            '    string pt = const()[name=string("pt"), val=string("valid")];',
            '    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];',
            '    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];',
            '    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];',
            f'    int32 gh = const()[name=string("gh"), val=int32({H})];',
            f'    tensor<fp16, [{HK}, 1, 1, {DK}]> gpack = const()[name=string("gpack"), val=tensor<fp16, [{HK}, 1, 1, {DK}]>(BLOBFILE(path=string("@model_path/weights/gpack.bin"), offset=uint64(64)))];',
        ]
    if stage >= 2:
        lines += [
            f'    tensor<fp16, [{H}, {DK}, 1, 1]> gsum = const()[name=string("gsum"), val=tensor<fp16, [{H}, {DK}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/gsum.bin"), offset=uint64(64)))];',
        ]
    if stage >= 3:
        lines += [
            f'    tensor<fp16, [{HK}, 1, 1, 1]> grep = const()[name=string("grep"), val=tensor<fp16, [{HK}, 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/grep.bin"), offset=uint64(64)))];',
        ]
    lines.append(sl("state", 0, HK, 0, W))
    output = "state"
    if stage == 0:
        lines.append(
            f'    tensor<fp16, [1, {HK}, 1, {DV}]> state_copy = mul(x=state, y=fp16(0x1p-1))[name=string("state_copy")];'
        )
        output = "state_copy"
    if stage >= 1:
        lines += [
            sl("decay_h", BLOCK["decay"], BLOCK["decay"] + H, 0, DK),
            f'    tensor<fp16, [1, {HK}, 1, 1]> decay = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gpack, x=decay_h)[name=string("decay")];',
            f'    tensor<fp16, [1, {HK}, 1, {DV}]> s1 = mul(x=state, y=decay)[name=string("s1")];',
        ]
        output = "s1"
    if stage >= 2:
        lines += [
            sl("k_h", BLOCK["k"], BLOCK["k"] + H, 0, DK),
            f'    tensor<fp16, [1, {HK}, 1, 1]> k = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gpack, x=k_h)[name=string("k")];',
            f'    tensor<fp16, [1, {HK}, 1, {DV}]> sk = mul(x=s1, y=k)[name=string("sk")];',
            f'    tensor<fp16, [1, {H}, 1, {DV}]> memory = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gsum, x=sk)[name=string("memory")];',
        ]
        output = "memory"
    if stage >= 3:
        lines += [
            sl("v", BLOCK["v"], BLOCK["v"] + H, 0, DV),
            sl("beta_h", BLOCK["beta"], BLOCK["beta"] + H, 0, 1),
            f'    tensor<fp16, [1, {H}, 1, {DV}]> delta = sub(x=v, y=memory)[name=string("delta")];',
            f'    tensor<fp16, [1, {H}, 1, {DV}]> db = mul(x=delta, y=beta_h)[name=string("db")];',
            f'    tensor<fp16, [1, {HK}, 1, {DV}]> dup = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=grep, x=db)[name=string("dup")];',
            f'    tensor<fp16, [1, {HK}, 1, {DV}]> ku = mul(x=dup, y=k)[name=string("ku")];',
            f'    tensor<fp16, [1, {HK}, 1, {DV}]> new_state = add(x=s1, y=ku)[name=string("new_state")];',
        ]
        output = "new_state"
    if stage >= 4:
        lines += [
            sl("q_h", BLOCK["q"], BLOCK["q"] + H, 0, DK),
            f'    tensor<fp16, [1, {HK}, 1, 1]> q = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gpack, x=q_h)[name=string("q")];',
            f'    tensor<fp16, [1, {HK}, 1, {DV}]> sq = mul(x=new_state, y=q)[name=string("sq")];',
            f'    tensor<fp16, [1, {H}, 1, {DV}]> y = conv(dilations=dl, groups=gh, pad=pd, pad_type=pt, strides=st, weight=gsum, x=sq)[name=string("y")];',
        ]
        output = "y"
    outputs = "(y, new_state)" if stage >= 5 else f"({output})"
    lines += [f"  }} -> {outputs};", "}", f"// qwen38_gdn_compact_stage_{stage}"]
    return "\n".join(lines) + "\n"


def compile_stage(engine, stage, input_channels=CIN, input_width=W):
    one_hot = np.tile(np.eye(DK, dtype=np.float16)[:, None, None, :], (H, 1, 1, 1))
    blobs = {
        "gpack.bin": one_hot.tobytes(),
        "gsum.bin": np.ones((H, DK, 1, 1), np.float16).tobytes(),
        "grep.bin": np.ones((HK, 1, 1, 1), np.float16).tobytes(),
    }
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        program = engine.compile_multiproc(
            mil_for(stage, input_channels, input_width),
            blobs,
            input_channels,
            H,
            input_width,
        )
    if program is None:
        tail = "\n".join(capture.getvalue().strip().splitlines()[-8:])
        print(f"stage {stage}: COMPILE FAILED\n{tail}")
    else:
        print(f"stage {stage}: compiled")
    return program


def output_channels(program):
    inner = E._msg(program.model, "model") or program.model
    desc = E._desc(E._msg(inner, "description"))
    return [int(c) for c, _, _ in re.findall(
        r'Channels = (\d+);((?:(?!Channels =).)*?)Name = "([^"]*@output)";',
        desc,
        re.S,
    )]


def make_request(program, source, destination, y_surface):
    channels = output_channels(program)
    outputs = [E._wrap_iosurface(y_surface if c == H else destination) for c in channels]
    request_init = ctypes.CFUNCTYPE(*([ctypes.c_void_p] * 12))
    request = request_init(("objc_msgSend", E._objc))(
        E._msg(E._cls("_ANERequest"), "alloc"),
        E._sel("initWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:sharedEvents:transactionHandle:"),
        E._nsarray([E._wrap_iosurface(source)]),
        E._nsarray([E._nsnumber_int(0)]),
        E._nsarray(outputs),
        E._nsarray([E._nsnumber_int(i) for i in range(len(outputs))]),
        None,
        None,
        E._nsnumber_int(0),
        None,
        None,
    )
    if not request:
        raise RuntimeError("request creation failed")
    return request, channels


def write_params(surface, q, k, v, decay, beta):
    # State rows are deliberately untouched: they were written by the prior
    # ANE request. Parameter rows are host-owned and only 60 KiB per step.
    with _iosurface_view(surface, (CIN, W), np.float16) as buf:
        buf[BLOCK["decay"]:BLOCK["decay"] + H] = decay[:, None]
        buf[BLOCK["k"]:BLOCK["k"] + H] = k
        buf[BLOCK["q"]:BLOCK["q"] + H] = q
        buf[BLOCK["v"]:BLOCK["v"] + H] = v
        buf[BLOCK["beta"]:BLOCK["beta"] + H] = beta[:, None]


def reference(state, q, k, v, decay, beta):
    state = state * decay[:, None, None]
    memory = np.sum(state * k[:, None, :], axis=-1)
    delta = (v - memory) * beta[:, None]
    state = state + k[:, None, :] * delta[:, :, None]
    y = np.sum(state * q[:, None, :], axis=-1)
    return y, state


def run_ping_pong(program):
    E._load_iosurface()
    surf0 = E._create_iosurface(E._iosurface_alloc_size(CIN * W))
    surf1 = E._create_iosurface(E._iosurface_alloc_size(CIN * W))
    y_surface = E._create_iosurface(E._iosurface_alloc_size(H * DV))
    if not all((surf0, surf1, y_surface)):
        raise RuntimeError("IOSurface allocation failed")
    req01, chans = make_request(program, surf0, surf1, y_surface)
    req10, _ = make_request(program, surf1, surf0, y_surface)
    print(f"stage 5 outputs (channels): {chans}")

    evaluate_type = ctypes.CFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    evaluate = evaluate_type(("objc_msgSend", E._objc))

    def submit(request):
        error = ctypes.c_void_p(0)
        ok = evaluate(
            program.model,
            E._sel("evaluateWithQoS:options:request:error:"),
            21,
            program._compile_opts,
            request,
            ctypes.byref(error),
        )
        if not ok:
            raise RuntimeError(E._desc(error.value) if error.value else "evaluate failed")

    rng = np.random.default_rng(42)
    initial = rng.normal(0, 0.02, (H, DV, DK)).astype(np.float32)
    with _iosurface_view(surf0, (CIN, W), np.float16) as dst:
        dst[:] = 0
        dst[:HK] = initial.transpose(0, 2, 1).reshape(HK, DV)
    with _iosurface_view(surf1, (CIN, W), np.float16) as dst:
        dst[:] = 0

    ref_state = initial
    last_dst = None
    for step, (source, request, destination) in enumerate(
        ((surf0, req01, surf1), (surf1, req10, surf0))
    ):
        q = rng.normal(0, 0.05, (H, DK)).astype(np.float32)
        k = rng.normal(0, 0.05, (H, DK)).astype(np.float32)
        v = rng.normal(0, 0.05, (H, DV)).astype(np.float32)
        decay = rng.uniform(0.94, 0.999, H).astype(np.float32)
        beta = rng.uniform(0.05, 0.4, H).astype(np.float32)
        write_params(source, q, k, v, decay, beta)
        submit(request)
        ref_y, ref_state = reference(ref_state, q, k, v, decay, beta)
        with _iosurface_view(y_surface, (H, DV), np.float16) as out:
            got_y = np.array(out, np.float32)
        rel_y = np.max(np.abs(got_y - ref_y)) / (np.max(np.abs(ref_y)) + 1e-9)
        print(f"step {step + 1}: y rel={rel_y:.4g}")
        last_dst = destination

    with _iosurface_view(last_dst, (CIN, W), np.float16) as out:
        got_state = np.array(out[:HK], np.float32).reshape(H, DK, DV).transpose(0, 2, 1)
    rel_state = np.max(np.abs(got_state - ref_state)) / (np.max(np.abs(ref_state)) + 1e-9)

    # Parameter writes are outside this timing. State stays on the two ANE
    # surfaces and alternates request-to-request.
    for _ in range(4):
        submit(req01)
        submit(req10)
    iterations = 50
    start = time.perf_counter()
    for _ in range(iterations):
        submit(req01)
        submit(req10)
    elapsed_ms = (time.perf_counter() - start) * 1e3 / (2 * iterations)
    print(f"state after two dependent steps: rel={rel_state:.4g}")
    print(f"pure zero-copy recurrence: {elapsed_ms:.3f} ms/step")
    print("PASS" if rel_state < 3e-3 else "FAIL")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=range(6), help="compile only this cumulative stage")
    parser.add_argument(
        "--scan-channels",
        action="store_true",
        help="compile stage 0 across candidate input-channel counts",
    )
    parser.add_argument(
        "--scan-widths",
        action="store_true",
        help="compile stage 0 across candidate input widths",
    )
    args = parser.parse_args()
    engine = AneEngine()
    if args.scan_channels:
        for channels in (HK, 6176, 6208, 6240, 6256, 6272, 6304, 6336, CIN):
            print(f"input channels {channels}:", end=" ", flush=True)
            compile_stage(engine, 0, channels)
        return
    if args.scan_widths:
        for width in (128, 160, 192, 224, 256):
            print(f"input width {width}:", end=" ", flush=True)
            compile_stage(engine, 0, HK + H, width)
        return
    stages = [args.stage] if args.stage is not None else range(6)
    complete = None
    for stage in stages:
        program = compile_stage(engine, stage)
        if program is None:
            raise SystemExit(1)
        if stage == 5:
            complete = program
    if complete is not None:
        run_ping_pong(complete)


if __name__ == "__main__":
    main()

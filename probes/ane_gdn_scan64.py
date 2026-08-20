"""Qualify Qwen3.8's exact 64-position chunked GDN prefill on the ANE.

Qwen's chunked gated-delta rule avoids the compiler-rejected dense 128x128
transition scan. It constructs a lower-triangular 64x64 token-space matrix,
evaluates its inverse with fixed row updates, then uses attention-shaped
dynamic matmuls. This is the Qwen reference algorithm specialized to one cold
prompt chunk.

Inputs come from the real layer-0 int4 projection and causal convolution. The
probe compares every output and the final state against both a NumPy sequential
oracle and the production resident ANE recurrence. Arbitrary prefix-state
support remains the integration gate for chaining chunks.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import statistics
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.pure_ane import (  # noqa: E402
    AneDriver,
    AneGdnConv,
    AneGdnRecurrence,
    AneNormProjection,
    Checkpoint,
    StandaloneTokenizer,
    assert_standalone,
)


H = 48
D = 128


def _slice(name: str, source: str, c0: int, c1: int,
           shape: str) -> str:
    return (
        f'    tensor<fp16, [{shape}]> {name} = slice_by_index('
        f'begin=tensor<int32, [4]>([0,{c0},0,0]), '
        f'end=tensor<int32, [4]>([1,{c1},1,{D}]), x={source})'
        f'[name=string("{name}")];'
    )


def scan_mil(module, tokens: int) -> tuple[str, int, int]:
    if tokens < 1 or tokens & (tokens - 1):
        raise ValueError("scan token count must be a power of two")
    nh = tokens * H
    channels = 5 * nh
    output_channels = nh + H * D
    lines = [f'''program(1.3)
{module._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {channels}, 1, {D}]> x) {{
    tensor<fp16, [1, 1, {D}, {D}]> ident = const()[name=string("ident"), val=tensor<fp16, [1, 1, {D}, {D}]>(BLOBFILE(path=string("@model_path/weights/identity.bin"), offset=uint64(64)))];
{_slice("qrow", "x", 0, nh, f"1, {nh}, 1, {D}")}
{_slice("krow", "x", nh, 2*nh, f"1, {nh}, 1, {D}")}
{_slice("vrow", "x", 2*nh, 3*nh, f"1, {nh}, 1, {D}")}
{_slice("drow", "x", 3*nh, 4*nh, f"1, {nh}, 1, {D}")}
{_slice("brow", "x", 4*nh, 5*nh, f"1, {nh}, 1, {D}")}
    tensor<fp16, [1, {nh}, 1, 1]> decay = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{nh},1,1]), x=drow)[name=string("decay")];
    tensor<fp16, [1, {nh}, 1, 1]> beta = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{nh},1,1]), x=brow)[name=string("beta")];
    tensor<fp16, [1, {nh}, {D}, 1]> kcol = reshape(shape=tensor<int32, [4]>([1,{nh},{D},1]), x=krow)[name=string("kcol")];
    tensor<fp16, [1, {nh}, {D}, 1]> vcol = reshape(shape=tensor<int32, [4]>([1,{nh},{D},1]), x=vrow)[name=string("vcol")];
    tensor<fp16, [1, {nh}, {D}, 1]> qcol = reshape(shape=tensor<int32, [4]>([1,{nh},{D},1]), x=qrow)[name=string("qcol")];
    tensor<fp16, [1, {nh}, {D}, {D}]> kk = matmul(transpose_x=bool(false), transpose_y=bool(false), x=kcol, y=krow)[name=string("kk")];
    tensor<fp16, [1, {nh}, {D}, {D}]> bkk = mul(x=kk, y=beta)[name=string("bkk")];
    tensor<fp16, [1, {nh}, {D}, {D}]> im = sub(x=ident, y=bkk)[name=string("im")];
    tensor<fp16, [1, {nh}, {D}, {D}]> m0 = mul(x=im, y=decay)[name=string("m0")];
    tensor<fp16, [1, {nh}, {D}, {D}]> vk = matmul(transpose_x=bool(false), transpose_y=bool(false), x=vcol, y=krow)[name=string("vk")];
    tensor<fp16, [1, {nh}, {D}, {D}]> b0 = mul(x=vk, y=beta)[name=string("b0")];''']

    mcur, bcur = "m0", "b0"
    offset = 1
    stage = 0
    while offset < tokens:
        cut = offset * H
        tail = nh - cut
        mn = f"m{stage+1}"
        bn = f"b{stage+1}"
        lines.append(f'''    tensor<fp16, [1, {tail}, {D}, {D}]> ml{stage} = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{tail},{D},{D}]), x={mcur})[name=string("ml{stage}")];
    tensor<fp16, [1, {tail}, {D}, {D}]> mr{stage} = slice_by_index(begin=tensor<int32, [4]>([0,{cut},0,0]), end=tensor<int32, [4]>([1,{nh},{D},{D}]), x={mcur})[name=string("mr{stage}")];
    tensor<fp16, [1, {tail}, {D}, {D}]> bl{stage} = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{tail},{D},{D}]), x={bcur})[name=string("bl{stage}")];
    tensor<fp16, [1, {tail}, {D}, {D}]> br{stage} = slice_by_index(begin=tensor<int32, [4]>([0,{cut},0,0]), end=tensor<int32, [4]>([1,{nh},{D},{D}]), x={bcur})[name=string("br{stage}")];
    tensor<fp16, [1, {cut}, {D}, {D}]> me{stage} = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{cut},{D},{D}]), x={mcur})[name=string("me{stage}")];
    tensor<fp16, [1, {cut}, {D}, {D}]> be{stage} = slice_by_index(begin=tensor<int32, [4]>([0,0,0,0]), end=tensor<int32, [4]>([1,{cut},{D},{D}]), x={bcur})[name=string("be{stage}")];
    tensor<fp16, [1, {tail}, {D}, {D}]> mt{stage} = matmul(transpose_x=bool(false), transpose_y=bool(false), x=ml{stage}, y=mr{stage})[name=string("mt{stage}")];
    tensor<fp16, [1, {tail}, {D}, {D}]> bt0_{stage} = matmul(transpose_x=bool(false), transpose_y=bool(false), x=bl{stage}, y=mr{stage})[name=string("bt0_{stage}")];
    tensor<fp16, [1, {tail}, {D}, {D}]> bt{stage} = add(x=bt0_{stage}, y=br{stage})[name=string("bt{stage}")];
    tensor<fp16, [1, {nh}, {D}, {D}]> mep{stage} = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,0,{tail},0,0,0,0]), x=me{stage})[name=string("mep{stage}")];
    tensor<fp16, [1, {nh}, {D}, {D}]> mtp{stage} = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,{cut},0,0,0,0,0]), x=mt{stage})[name=string("mtp{stage}")];
    tensor<fp16, [1, {nh}, {D}, {D}]> {mn} = add(x=mep{stage}, y=mtp{stage})[name=string("{mn}")];
    tensor<fp16, [1, {nh}, {D}, {D}]> bep{stage} = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,0,{tail},0,0,0,0]), x=be{stage})[name=string("bep{stage}")];
    tensor<fp16, [1, {nh}, {D}, {D}]> btp{stage} = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,{cut},0,0,0,0,0]), x=bt{stage})[name=string("btp{stage}")];
    tensor<fp16, [1, {nh}, {D}, {D}]> {bn} = add(x=bep{stage}, y=btp{stage})[name=string("{bn}")];''')
        mcur, bcur = mn, bn
        stage += 1
        offset *= 2

    lines.append(f'''    tensor<fp16, [1, {nh}, {D}, 1]> y0 = matmul(transpose_x=bool(false), transpose_y=bool(false), x={bcur}, y=qcol)[name=string("y0")];
    tensor<fp16, [1, {nh}, {D}, 1]> y64 = mul(x=y0, y=fp16(0x1p+6))[name=string("y64")];
    tensor<fp16, [1, {nh}, 1, {D}]> yr = transpose(perm=tensor<int32, [4]>([0,1,3,2]), x=y64)[name=string("yr")];
    tensor<fp16, [1, {H}, {D}, {D}]> sf = slice_by_index(begin=tensor<int32, [4]>([0,{nh-H},0,0]), end=tensor<int32, [4]>([1,{nh},{D},{D}]), x={bcur})[name=string("sf")];
    tensor<fp16, [1, {H}, {D}, {D}]> sft = transpose(perm=tensor<int32, [4]>([0,1,3,2]), x=sf)[name=string("sft")];
    tensor<fp16, [1, {H*D}, 1, {D}]> sr = reshape(shape=tensor<int32, [4]>([1,{H*D},1,{D}]), x=sft)[name=string("sr")];
    tensor<fp16, [1, {output_channels}, 1, {D}]> yp = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,0,{H*D},0,0,0,0]), x=yr)[name=string("yp")];
    tensor<fp16, [1, {output_channels}, 1, {D}]> sp = pad(mode=string("constant"), constant_val=fp16(0x0p+0), pad=tensor<int32, [8]>([0,0,{nh},0,0,0,0,0]), x=sr)[name=string("sp")];
    tensor<fp16, [1, {output_channels}, 1, {D}]> y = add(x=yp, y=sp)[name=string("y")];
  }} -> (y);
}}
// qwen38_gdn_affine_scan_h{H}_n{tokens}
''')
    return "\n".join(lines), channels, output_channels


class AneGdnAffineScan:
    def __init__(self, driver: AneDriver, tokens: int):
        self.driver = driver
        self.tokens = tokens
        self.nh = tokens * H
        mil, channels, output_channels = scan_mil(driver.module, tokens)
        blobs = {"identity.bin": np.eye(D, dtype=np.float16).tobytes()}
        capture = io.StringIO()
        started = time.perf_counter()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            self.program = driver.engine.compile_multiproc(
                mil, blobs, channels, output_channels, D
            )
        self.compile_seconds = time.perf_counter() - started
        if self.program is None:
            detail = "\n".join(capture.getvalue().strip().splitlines()[-12:])
            raise RuntimeError(f"ANE GDN scan compile failed:\n{detail}")
        driver.engine._ensure_io(self.program)
        self.channels = channels
        self.output_channels = output_channels

    def load(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
             decay: np.ndarray, beta: np.ndarray) -> None:
        shape = (self.tokens, H, D)
        for name, value in (("q", q), ("k", k), ("v", v)):
            if value.shape != shape:
                raise ValueError(f"invalid {name} shape {value.shape}")
        if decay.shape != (self.tokens, H) or beta.shape != (self.tokens, H):
            raise ValueError("invalid gate shapes")
        nh = self.nh
        with self.driver.view(
            self.program._in_surf, (self.channels, D), np.float16
        ) as dst:
            dst[:] = 0
            dst[:nh] = q.reshape(nh, D)
            dst[nh:2*nh] = k.reshape(nh, D)
            dst[2*nh:3*nh] = v.reshape(nh, D)
            dst[3*nh:4*nh, 0] = decay.reshape(nh)
            dst[4*nh:5*nh, 0] = beta.reshape(nh)

    def run_loaded(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.driver.engine.submit(self.program, procedure_index=0):
            raise RuntimeError("ANE GDN scan evaluation failed")
        nh = self.nh
        with self.driver.view(
            self.program._out_surf, (self.output_channels, D), np.float16
        ) as src:
            y = np.array(src[:nh], np.float32).reshape(self.tokens, H, D)
            state = np.array(src[nh:], np.float32).reshape(H, D, D)
        return y, state


def actual_inputs(driver: AneDriver, checkpoint: Checkpoint,
                  token_ids: list[int], bits: int) -> tuple[np.ndarray, ...]:
    prefix = "model.language_model.layers.0"
    names = [f"{prefix}.linear_attn.{name}.weight" for name in
             ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")]
    head = AneNormProjection(
        driver, checkpoint, f"{prefix}.input_layernorm.weight", names,
        bits=bits, tag="scan64_layer0_head", active_lanes=3
    )
    conv = AneGdnConv(
        driver, checkpoint, f"{prefix}.linear_attn.conv1d.weight"
    )
    raw_q, raw_k, values, avec, bvec = [], [], [], [], []
    for start in range(0, len(token_ids), 3):
        ids = token_ids[start:start+3]
        hidden = np.stack([checkpoint.embedding(t) for t in ids], axis=1)
        projection = head(hidden)
        activated = conv(projection[:10240])
        if projection.ndim == 1:
            projection = projection[:, None]
            activated = activated[:, None]
        for lane in range(len(ids)):
            raw_q.append(activated[:2048, lane].reshape(16, D))
            raw_k.append(activated[2048:4096, lane].reshape(16, D))
            values.append(activated[4096:10240, lane].reshape(H, D))
            bvec.append(projection[16384:16432, lane])
            avec.append(projection[16432:16480, lane])
    return tuple(np.asarray(x, np.float16) for x in
                 (raw_q, raw_k, values, avec, bvec))


def scan_inputs(raw_q: np.ndarray, raw_k: np.ndarray, values: np.ndarray,
                avec: np.ndarray, bvec: np.ndarray,
                a_log: np.ndarray, dt_bias: np.ndarray) -> tuple[np.ndarray, ...]:
    q = raw_q.astype(np.float32)
    k = raw_k.astype(np.float32)
    qn = q / np.sqrt(np.mean(q*q, axis=-1, keepdims=True) + 1e-6) / D
    kn = k / np.sqrt(np.mean(k*k, axis=-1, keepdims=True) + 1e-6) / np.sqrt(D)
    q48 = np.repeat(qn, 3, axis=1).astype(np.float16)
    k48 = np.repeat(kn, 3, axis=1).astype(np.float16)
    decay = np.exp(
        -np.exp(a_log.astype(np.float32))[None, :]
        * np.logaddexp(avec.astype(np.float32)
                       + dt_bias.astype(np.float32)[None, :], 0.0)
    ).astype(np.float16)
    beta = (1.0 / (1.0 + np.exp(-bvec.astype(np.float32)))).astype(np.float16)
    return q48, k48, values.astype(np.float16), decay, beta


def numpy_sequential(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                     decay: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros((H, D, D), np.float32)
    outputs = []
    for position in range(q.shape[0]):
        state *= decay[position, :, None, None].astype(np.float32)
        memory = np.sum(state * k[position, :, None, :].astype(np.float32), axis=-1)
        delta = (v[position].astype(np.float32) - memory) \
                * beta[position, :, None].astype(np.float32)
        state += delta[:, :, None] * k[position, :, None, :].astype(np.float32)
        outputs.append(64.0 * np.sum(
            state * q[position, :, None, :].astype(np.float32), axis=-1
        ))
    return np.asarray(outputs), state


def relative(got: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(got-expected)) /
                 (np.max(np.abs(expected)) + 1e-9))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get(
        "Q38_MODEL", "/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B"))
    parser.add_argument("--engine-path", default=os.environ.get(
        "Q38_ANE_ENGINE", ROOT))
    parser.add_argument("--tokens", type=int, choices=(2,4,8,16,32,64), default=64)
    parser.add_argument("--bits", type=int, choices=(4,8,16), default=4)
    parser.add_argument("--runs", type=int, default=7)
    args = parser.parse_args()

    assert_standalone("GDN scan probe startup")
    checkpoint = Checkpoint(args.model)
    tokenizer = StandaloneTokenizer(args.model)
    seed_text = (
        "Implement a persistent inference server, explain each optimization, "
        "and verify every numerical result before changing the runtime. " * 16
    )
    token_ids = tokenizer.encode(seed_text)[:args.tokens]
    if len(token_ids) != args.tokens:
        raise RuntimeError("probe prompt did not produce enough tokens")
    driver = AneDriver(args.engine_path)
    raw_q, raw_k, values, avec, bvec = actual_inputs(
        driver, checkpoint, token_ids, args.bits
    )
    prefix = "model.language_model.layers.0.linear_attn"
    a_log = checkpoint.tensor(f"{prefix}.A_log", np.float16)
    dt_bias = checkpoint.tensor(f"{prefix}.dt_bias", np.float16)
    q, k, v, decay, beta = scan_inputs(
        raw_q, raw_k, values, avec, bvec, a_log, dt_bias
    )
    expected_y, expected_state = numpy_sequential(q, k, v, decay, beta)

    recurrence = AneGdnRecurrence(driver)
    state = recurrence.new_state()

    def sequential_ane() -> tuple[np.ndarray, np.ndarray]:
        recurrence.restore(state, np.zeros((H*D, D), np.float16))
        ys = []
        for position in range(args.tokens):
            ys.append(recurrence(
                state, raw_q[position], raw_k[position], values[position],
                avec[position], bvec[position], a_log, dt_bias
            ).astype(np.float32))
        return np.asarray(ys), recurrence.materialize(state)

    seq_y, seq_state = sequential_ane()
    scan = AneGdnAffineScan(driver, args.tokens)
    scan.load(q, k, v, decay, beta)
    scan_y, scan_state = scan.run_loaded()

    scan_y_rel = relative(scan_y, expected_y)
    scan_state_rel = relative(scan_state, expected_state)
    seq_y_rel = relative(seq_y, expected_y)
    seq_state_rel = relative(seq_state, expected_state)

    # Warm both paths, then include the production state copies/parameter
    # writes for sequential timing. Scan timing uses an already-loaded input so
    # the large transform graph itself is visible separately from host loading.
    scan.run_loaded()
    sequential_ane()
    scan_ms = []
    seq_ms = []
    for _ in range(args.runs):
        started = time.perf_counter();scan.run_loaded()
        scan_ms.append((time.perf_counter()-started)*1e3)
        started = time.perf_counter();sequential_ane()
        seq_ms.append((time.perf_counter()-started)*1e3)

    scan_median = statistics.median(scan_ms)
    seq_median = statistics.median(seq_ms)
    passed = max(scan_y_rel, scan_state_rel) < 0.04
    faster = scan_median < seq_median
    print(f"GDN_SCAN tokens={args.tokens} heads={H} dim={D} int{args.bits}")
    print(f"compile_seconds={scan.compile_seconds:.3f}")
    print(f"scan_y_relative_error={scan_y_rel:.6g}")
    print(f"scan_state_relative_error={scan_state_rel:.6g}")
    print(f"sequential_y_relative_error={seq_y_rel:.6g}")
    print(f"sequential_state_relative_error={seq_state_rel:.6g}")
    print(f"scan_median_ms={scan_median:.3f}")
    print(f"sequential_median_ms={seq_median:.3f}")
    print(f"speedup={seq_median/scan_median:.3f}x")
    print("GDN_SCAN_NUMERICS=" + ("PASS" if passed else "FAIL"))
    print("GDN_SCAN_INTEGRATION=" + ("GO" if passed and faster else "NO_GO"))
    assert_standalone("GDN scan probe completion")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Resident quantized MoE diagnostic for the ANE attention runner.

GPU execution is explicit and opt-in. No selected expert is materialized as
FP16/FP32; only input/output activations cross the NumPy boundary.
"""
from __future__ import annotations

import time
import os
import numpy as np
import mlx.core as mx

from runtime.expert_bank import MlxSafe, MLX4_DEFAULT


class ResidentMoe:
    _configured = False

    def __init__(self, layer, shared, path=MLX4_DEFAULT):
        # fp16 throughout. fp32 scales and biases were 15 GB of a 76 GB bank,
        # and on a 137 GB machine that margin decides whether the bank stays
        # resident — see docs/W8A8-PROJECTIONS.md. It is also the faster pair.
        self.dtype = getattr(mx, os.environ.get("FLASHNEXT_MOE_DTYPE", "float16"))
        # gather_qmm promotes FP16 + BF16 to FP32, then casts *all* scales
        # and biases before selecting experts (mlx/ops.cpp). With 48 banks
        # this creates ~15 GB/token of conversion output. Materialize the
        # matching dtype once at load, including when callers select FP16.
        scale_dtype = getattr(mx, os.environ.get("FLASHNEXT_MOE_SCALE_DTYPE",
                                               str(self.dtype).split('.')[-1]))
        if not self._configured:
            # MLX defaults to no wired residency. A 69 GB expert bank must
            # survive intervening CPU/ANE execution without being paged out.
            limit = int(os.environ.get("FLASHNEXT_MLX_WIRED_GB", "80")) * 1024**3
            limit = min(limit, mx.device_info()["max_recommended_working_set_size"])
            mx.set_wired_limit(limit)
            # A 256 MB cache made every routed call re-map its buffers:
            # 2.00 ms/layer at 256 MB vs 1.30 ms at 1 GB (/tmp probe).
            cache_mb = int(os.environ.get("FLASHNEXT_MLX_CACHE_MB", "2048"))
            mx.set_cache_limit(cache_mb * 1024**2)
            ResidentMoe._configured = True
        source = MlxSafe(path)
        prefix = f"model.layers.{layer}.mlp.switch_mlp"
        self.projections = []
        try:
            # Copy one tensor at a time and drop its file pages so the 71 GB
            # safetensors mapping does not sit in the UBC next to the 68 GB
            # of Metal buffers. mx.array is lazy; eval before DONTNEED.
            for name in ("gate_proj", "up_proj", "down_proj"):
                p = f"{prefix}.{name}"
                raw_w = source.raw(p + ".weight")
                weight = mx.array(raw_w)
                mx.eval(weight)
                del raw_w
                source.drop_pages(p + ".weight")
                # f32() of an F32 mmap is a view. DONTNEED before the MLX
                # copy can leave zeros in the dequant tables; ppl nll then
                # drifts (1.884456 vs 1.886208) without looking broken.
                scales_np = np.array(source.f32(p + ".scales"), np.float32, copy=True)
                biases_np = np.array(source.f32(p + ".biases"), np.float32, copy=True)
                source.drop_pages(p + ".scales")
                source.drop_pages(p + ".biases")
                scales = mx.array(scales_np, dtype=scale_dtype)
                biases = mx.array(biases_np, dtype=scale_dtype)
                mx.eval(scales, biases)
                del scales_np, biases_np
                self.projections.append((weight, scales, biases))
            source.drop_all_pages()
        finally:
            source.close()
        # Store the shared expert already transposed and contiguous. ``g.T`` in
        # the decode path materialized a fresh 6.5 MB copy per matmul per layer.
        self.shared = tuple(
            mx.contiguous(mx.array(x).astype(self.dtype).T) for x in shared
        )
        mx.eval(*self.shared)
        self.nbytes = sum(a.nbytes for p in self.projections for a in p)
        self.nbytes += sum(a.nbytes for a in self.shared)

    def routed(self, x, ids, scores):
        # Scales/biases already have the activation dtype; no whole-bank casts.
        x = mx.array(np.asarray(x, np.float32)).astype(self.dtype).reshape(1, 1, 1, -1)
        ids = mx.array(np.asarray(ids, np.uint32)).reshape(1, -1)
        scores = mx.array(np.asarray(scores, np.float32)).astype(self.dtype).reshape(1, -1, 1)
        def project(x, p):
            return mx.gather_qmm(x, *p, rhs_indices=ids, transpose=True,
                                 group_size=64, bits=4)
        gate = project(x, self.projections[0])
        up = project(x, self.projections[1])
        y = project(gate * mx.sigmoid(gate) * up, self.projections[2])
        return mx.sum(y.squeeze(-2) * scores, axis=-2)

    # A GPU router (softmax + argpartition + gather in one graph, one eval a
    # layer) was tried and is a net loss: it removed the 6 ms of host routing
    # and added 34 ms to the gather. The MoE eval is latency-bound, and the
    # extra ops in the same graph cost more than the NumPy matmul they replace.

    def routed_multi(self, x_k, ids_k, scores_k, shared_k=None, hyp_k=None, inj_k=None):
        """Routed experts for k tokens in one submit.

        Same expert weights serve every token in the batch, so cost is close to
        flat in k: 0.584 ms/layer at k=1 against 1.138 ms at k=16 (/tmp probe),
        i.e. 28.0 -> 3.4 ms/token across 48 layers. Shapes follow mlx_lm's
        SwitchGLU: expand_dims(x, (-2, -3)) with rhs_indices (1, k, K).
        """
        k = int(np.asarray(x_k).reshape(-1, x_k.shape[-1]).shape[0])
        x = mx.array(x_k, dtype=self.dtype).reshape(1, k, -1)
        x = mx.expand_dims(x, (-2, -3))
        ids = mx.array(ids_k, dtype=mx.uint32).reshape(1, k, -1)
        sc = mx.array(scores_k, dtype=self.dtype).reshape(1, k, -1, 1)

        def project(t, p):
            return mx.gather_qmm(t, *p, rhs_indices=ids, transpose=True,
                                 group_size=64, bits=4)

        gate = project(x, self.projections[0])
        up = project(x, self.projections[1])
        y = project(gate * mx.sigmoid(gate) * up, self.projections[2])
        routed = mx.sum(y.squeeze(-2) * sc, axis=-2)

        if shared_k is not None:
            sh = mx.array(shared_k, dtype=mx.float32).reshape(1, k, -1)
            y_tot = routed.astype(mx.float32) + sh
        else:
            y_tot = routed.astype(mx.float32)

        if hyp_k is not None and inj_k is not None:
            inj = mx.array(inj_k, dtype=mx.float32).reshape(1, k, -1, 1)
            hyp = mx.array(hyp_k, dtype=mx.float32).reshape(1, k, -1)
            injection = mx.reshape(inj * mx.expand_dims(y_tot, -2), (1, k, -1))
            res = hyp + injection
            mx.eval(res)
            return np.array(res).reshape(1, k, -1)

        mx.eval(y_tot)
        return np.array(y_tot).reshape(1, k, -1)

    def apply(self, x, ids, scores):
        t = time.perf_counter()
        y = self.routed(x, ids, scores)
        xx = mx.array(np.asarray(x, np.float32)).astype(self.dtype).reshape(1, -1)
        g, u, d, s = self.shared
        gate = xx @ g
        shared = (gate * mx.sigmoid(gate) * (xx @ u)) @ d
        y = y + shared * mx.sigmoid(xx @ s)
        y = y.astype(mx.float32)
        mx.eval(y)
        result = np.array(y)
        return result, (time.perf_counter() - t) * 1e3

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
        self.dtype = getattr(mx, os.environ.get("FLASHNEXT_MOE_DTYPE", "float32"))
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
            for name in ("gate_proj", "up_proj", "down_proj"):
                p = f"{prefix}.{name}"
                weight = mx.array(source.raw(p + ".weight"))
                scales = mx.array(source.f32(p + ".scales"), dtype=scale_dtype)
                biases = mx.array(source.f32(p + ".biases"), dtype=scale_dtype)
                mx.eval(weight, scales, biases)
                self.projections.append((weight, scales, biases))
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

    def routed_multi(self, x_k, ids_k, scores_k):
        """Routed experts for k tokens in one submit.

        Same expert weights serve every token in the batch, so cost is close to
        flat in k: 0.584 ms/layer at k=1 against 1.138 ms at k=16 (/tmp probe),
        i.e. 28.0 -> 3.4 ms/token across 48 layers. Shapes follow mlx_lm's
        SwitchGLU: expand_dims(x, (-2, -3)) with rhs_indices (1, k, K).
        """
        k = int(np.asarray(x_k).reshape(-1, x_k.shape[-1]).shape[0])
        x = mx.array(np.ascontiguousarray(x_k, np.float32)).astype(self.dtype)
        x = mx.expand_dims(x.reshape(1, k, -1), (-2, -3))
        ids = mx.array(np.ascontiguousarray(ids_k, np.uint32)).reshape(1, k, -1)
        sc = mx.array(np.ascontiguousarray(scores_k, np.float32))
        sc = sc.astype(self.dtype).reshape(1, k, -1, 1)

        def project(t, p):
            return mx.gather_qmm(t, *p, rhs_indices=ids, transpose=True,
                                 group_size=64, bits=4)

        gate = project(x, self.projections[0])
        up = project(x, self.projections[1])
        y = project(gate * mx.sigmoid(gate) * up, self.projections[2])
        y = mx.sum(y.squeeze(-2) * sc, axis=-2)
        y = y.astype(mx.float32)
        mx.eval(y)
        return np.array(y).reshape(1, k, -1)

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

"""Quantized lm_head on the GPU.

The host path reads a 2.54 GB fp32 copy of `lm_head.weight` per token (~13 ms).
The MLX 4-bit checkpoint stores this head 8-bit in 0.68 GB, and
`mx.quantized_matmul` plus the numpy round trip costs 1.65 ms. It is also the
head the MLX reference decode actually uses, so logits move toward that
reference rather than away from it.
"""
from __future__ import annotations

import os

import numpy as np
import mlx.core as mx

from runtime.expert_bank import MlxSafe, MLX4_DEFAULT


class QuantizedHead:
    def __init__(self, path=MLX4_DEFAULT, key: str = "lm_head"):
        source = MlxSafe(os.environ.get("FLASHNEXT_MLX4") or path)
        try:
            raw_w = source.raw(f"{key}.weight")
            self.w = mx.array(raw_w)
            mx.eval(self.w)
            del raw_w
            source.drop_pages(f"{key}.weight")
            scales = np.array(source.f32(f"{key}.scales"), np.float32, copy=True)
            biases = np.array(source.f32(f"{key}.biases"), np.float32, copy=True)
            source.drop_pages(f"{key}.scales")
            source.drop_pages(f"{key}.biases")
            self.scales = mx.array(scales).astype(mx.float16)
            self.biases = mx.array(biases).astype(mx.float16)
            del scales, biases
            mx.eval(self.scales, self.biases)
            source.drop_all_pages()
        finally:
            source.close()
        # uint32 packing: columns * (32 / bits) elements per row.
        scale_cols = int(self.scales.shape[-1])
        for bits in (8, 4, 6, 2):
            if (self.w.shape[-1] * (32 // bits)) % scale_cols == 0:
                group = (self.w.shape[-1] * (32 // bits)) // scale_cols
                if group in (32, 64, 128):
                    self.bits, self.group_size = bits, group
                    break
        else:
            raise ValueError(
                f"cannot infer lm_head packing from {self.w.shape} / {self.scales.shape}")
        self.nbytes = self.w.nbytes + self.scales.nbytes + self.biases.nbytes

    def logits_mx(self, x):
        """Logits for an MLX activation, left on the GPU.

        The drafter chains several passes before anything needs to reach the
        host, so keep the round trip out of the inner loop.
        """
        return mx.quantized_matmul(x.astype(mx.float16).reshape(-1, self.w.shape[0] * 0 + x.shape[-1]),
                                   self.w, self.scales, self.biases, transpose=True,
                                   group_size=self.group_size, bits=self.bits)

    def __call__(self, hidden: np.ndarray) -> np.ndarray:
        x = mx.array(np.ascontiguousarray(hidden, np.float32))
        y = self.logits_mx(x)
        mx.eval(y)
        return np.array(y.astype(mx.float32)).reshape(-1, y.shape[-1]).squeeze()

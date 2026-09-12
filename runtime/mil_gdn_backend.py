"""MIL int8 GDN layer backend — an alternative to the Core AI `pure_step` graph.

Separate module on purpose. The two paths cannot both be resident (the ANE
ceiling is ~80 models and the Core AI configuration already loads 72), so this
is selected instead of `pure_step`, never alongside it. Nothing here edits the
Core AI path.

Measured against the Core AI graph it replaces (`probes/flashnext_mil_layer.py`):

    MIL int8 + shared expert   1.265 ms
    Core AI fp16 + shared      2.020 ms      1.60x

    mixed  rel 0.027   state rel 0.0056   vs MultiTokenStep

Error is bounded in depth (~0.06 across 16 stacked layers, plateauing after
about six — each mixer's grouped RMS renormalizes the stream) and stable in
time across tokens. For calibration the shipping MLX 4-bit build measures
0.102 per tensor.

Six MIL spellings this depends on, each replacing something the compiler
rejects — see docs/W8A8-PROJECTIONS.md:
  * `rsqrt` -> `pow(x, -0.5)`
  * rank-4 consts -> runtime input rows (`param`, `hcn`)
  * `pow` with a const tensor exponent -> a runtime tensor exponent
  * `repeat_interleave` -> `concat(axis=2)` + `reshape`
  * the 4-tap -> a real depthwise `conv`, weights being a rank-4 blob const
  * multi-IO surfaces bind in ALPHABETICAL symbol order
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts"), str(_ROOT / "probes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import runtime.q38_ane_engine as _E
from runtime.q38_ane_engine import AneEngine, _iosurface_view

from export_flashnext_coreai import (  # noqa: E402
    H, I, HC, HC_W, HV, DK, DV, QKV, GDN_Y, IN_O, SEQ_DEFAULT,
)

S = SEQ_DEFAULT


class MilGdnLayer:
    """One GDN layer: mixer, in_proj int8, conv, GDN, out_proj int8, mixer, shared.

    State (recurrent + conv cache) lives in host buffers and is fed each call,
    matching the Core AI graph's contract so decode can swap between them.
    """

    __slots__ = ("layer", "k", "_prog", "_param", "_hcn", "_conv", "_eng",
                 "_conv_out_width", "_conv_surface_current")

    def __init__(self, layer: int, weights, ref_step, k: int = 1,
                 engine: AneEngine | None = None):
        import flashnext_mil_layer as _ML
        self._eng = engine or _ML.eng
        _ML.LAYER[0] = int(layer)
        _ML.K[0] = int(k)
        self.k = int(k)
        built = _ML.build_layer(weights, ref_step)
        if built is None:
            raise RuntimeError(f"MIL layer {layer} failed to compile")
        self.layer = int(layer)
        self._prog, self._param, self._hcn = built
        self._conv_out_width = int(getattr(self._prog, "conv_out_width", 64))
        # These immutable arrays are MIL inputs only because the frontend
        # rejects their constant spelling.  Copy them to their IOSurfaces once.
        for surf, val in zip(self._prog._in_surfs[2:4],
                             (self._param, self._hcn)):
            with _iosurface_view(surf, val.shape, np.float16) as dst:
                np.copyto(dst, val)
        # Only column 0 of the conv cache is read (the three taps are stacked
        # on the channel axis), so the surface is written a column at a time
        # rather than 1.97 MB a layer.
        self._conv = np.zeros((3 * QKV,), np.float16)
        self._conv_surface_current = False
        # The recurrent state lives in its input surface and never visits a
        # host array: it is 1.57 MB a layer, 56 MB a pass across 36 layers, and
        # staging it through NumPy copied that twice.
        with _iosurface_view(self._prog._in_surfs[4], (HV, DV, DK),
                             np.float16) as dst:
            dst[:] = 0

    def reset(self) -> None:
        self._conv[:] = 0
        with _iosurface_view(self._prog._in_surfs[1], (3 * QKV, S),
                             np.float16) as dst:
            dst[:] = 0
        self._conv_surface_current = True
        with _iosurface_view(self._prog._in_surfs[4], (HV, DV, DK),
                             np.float16) as dst:
            dst[:] = 0

    def __call__(self, x_bc1s: np.ndarray, n: int | None = None):
        """x is (1, HC_W, 1, S). Returns (mixed, hyper, inj, shared) over n slots.

        The recurrent state and conv cache advance in place, as `pure_step` does.
        """
        x = np.asarray(x_bc1s, np.float16).reshape(HC_W, S)
        p = self._prog
        with _iosurface_view(p._in_surfs[0], (HC_W, S), np.float16) as dst:
            np.copyto(dst, x)
        if not self._conv_surface_current:
            with _iosurface_view(p._in_surfs[1], (3 * QKV, S), np.float16) as dst:
                dst[:, 0] = self._conv
            self._conv_surface_current = True
        if not self._eng.submit(p, procedure_index=0):
            raise RuntimeError(f"MIL layer {self.layer}: submit failed")
        # Surfaces bind alphabetically: q_state0 .. q_state{k-1}, then
        # u_shared, v_mixed, w_hyper, x_inj, y_conv.
        kk = self.k
        w = kk if n is None else int(n)
        if n is None:
            # Plain decode consumes the whole block; speculation calls commit()
            # itself once it knows how many tokens the backbone confirmed.
            self.commit(kk - 1)
        out = []
        for j, shape in enumerate(((H, S), (H, S), (HC_W, S), (HC, S))):
            with _iosurface_view(p._out_surfs[kk + j], shape, np.float16) as o:
                out.append(np.array(o[:, :w], np.float32).reshape(1, shape[0], 1, w))
        shared, mixed, hyper, inj = out
        return mixed, hyper, inj, shared

    def state_at(self, j: int) -> np.ndarray:
        """The recurrent state having consumed `j + 1` of this pass's tokens."""
        with _iosurface_view(self._prog._out_surfs[int(j)], (HV, DV, DK),
                             np.float16) as o:
            return np.array(o, np.float16)

    def set_state(self, value) -> None:
        """Seed the recurrent state; the surface is the only copy."""
        with _iosurface_view(self._prog._in_surfs[4], (HV, DV, DK),
                             np.float16) as dst:
            np.copyto(dst, np.asarray(value, np.float16).reshape(HV, DV, DK))

    def commit(self, j: int) -> None:
        """Advance the layer's state and conv window to the `j + 1` boundary.

        Speculation needs every prefix, not just the last: the graph emits a
        state per token and the whole conv window, so a partially accepted
        block costs no extra pass.
        """
        j = int(j)
        p = self._prog
        with _iosurface_view(p._out_surfs[j], (HV, DV, DK), np.float16) as o:
            with _iosurface_view(p._in_surfs[4], (HV, DV, DK), np.float16) as d:
                np.copyto(d, o)
        with _iosurface_view(p._out_surfs[self.k + 4],
                             (QKV, self._conv_out_width), np.float16) as o:
            with _iosurface_view(p._in_surfs[1], (3 * QKV, S), np.float16) as d:
                for t in range(3):
                    value = o[:, j + 1 + t]
                    self._conv[t * QKV:(t + 1) * QKV] = value
                    d[t * QKV:(t + 1) * QKV, 0] = value
        self._conv_surface_current = True


def build_layers(layer_indices, loader_fn, step_fn, engine=None):
    """Compile a MIL layer per index. Returns {index: MilGdnLayer}.

    Budget the model count: the ANE runs out of resources near 80 resident
    programs, and each of these is one.
    """
    out = {}
    for li in layer_indices:
        loader, w = loader_fn(li)
        try:
            out[li] = MilGdnLayer(li, w, step_fn(w))
        finally:
            loader.close()
    return out

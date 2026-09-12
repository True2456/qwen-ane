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

import time

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

#: Where a GDN call's wall time goes, for the prefill breakdown.
TIMERS = {"write": 0.0, "submit": 0.0, "take": 0.0}


class MilGdnLayer:
    """One GDN layer: mixer, in_proj int8, conv, GDN, out_proj int8, mixer, shared.

    State (recurrent + conv cache) lives in host buffers and is fed each call,
    matching the Core AI graph's contract so decode can swap between them.
    """

    __slots__ = ("layer", "k", "_prog", "_param", "_hcn", "_conv", "_eng",
                 "_conv_out_width", "_conv_surface_current",
                 "_fut", "_prefix_state", "_prefix_fseq", "_ns",
                 "_proc", "_ob", "_proc_k", "_proc_ns")

    def __init__(self, layer: int, weights, ref_step, k: int = 1,
                 engine: AneEngine | None = None, single_state: bool = False,
                 prefill_k: int = 0):
        import flashnext_mil_layer as _ML
        self._eng = engine or _ML.eng
        _ML.LAYER[0] = int(layer)
        _ML.K[0] = int(k)
        _ML.SINGLE_STATE[0] = bool(single_state)
        self.k = int(k)
        try:
            if prefill_k:
                # Two unrolls in one program. A program carries its own copy
                # of the baked weights, so building them separately costs
                # 94 MB a layer twice; as two procedures they share one.
                built = _ML.build_program_multi(
                    weights, ref_step, [(k, single_state), (prefill_k, True)])
            else:
                built = _ML.build_layer(weights, ref_step)
        finally:
            _ML.SINGLE_STATE[0] = False
        if built is None:
            raise RuntimeError(f"MIL layer {layer} failed to compile")
        self.layer = int(layer)
        self._prog, self._param, self._hcn = built
        # Prefix states exported by the selected procedure. One means only the
        # last slot's state is readable, which is all a prompt chunk needs.
        # `_ob` is where the non-state outputs start and never moves, because
        # every procedure shares one set of surfaces.
        self._proc = 0
        self._proc_k = dict(getattr(self._prog, "proc_states", {}) and
                            {0: int(k), 1: int(prefill_k)} or {0: int(k)})
        self._proc_ns = dict(getattr(self._prog, "proc_states", None)
                             or {0: int(getattr(self._prog, "n_states", self.k))})
        self._ns = self._proc_ns[0]
        self._ob = int(getattr(self._prog, "n_state_surfs", self._ns))
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
        self._fut = None
        self._prefix_state = None
        self._prefix_fseq = None
        # The recurrent state lives in its input surface and never visits a
        # host array: it is 1.57 MB a layer, 56 MB a pass across 36 layers, and
        # staging it through NumPy copied that twice.
        # Two in-flight micro-batches cannot share that surface: either a
        # second slot or a fence before overwrite. The pipeline fences.
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
        _t0 = time.perf_counter()
        x = np.asarray(x_bc1s, np.float16).reshape(HC_W, S)
        p = self._prog
        with _iosurface_view(p._in_surfs[0], (HC_W, S), np.float16) as dst:
            np.copyto(dst, x)
        if not self._conv_surface_current:
            with _iosurface_view(p._in_surfs[1], (3 * QKV, S), np.float16) as dst:
                dst[:, 0] = self._conv
            self._conv_surface_current = True
        _t = time.perf_counter()
        TIMERS["write"] += _t - _t0
        if not self._eng.submit(p, procedure_index=self._proc):
            raise RuntimeError(f"MIL layer {self.layer}: submit failed")
        _t2 = time.perf_counter()
        TIMERS["submit"] += _t2 - _t
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
            with _iosurface_view(p._out_surfs[self._ob + j], shape, np.float16) as o:
                out.append(np.array(o[:, :w], np.float32).reshape(1, shape[0], 1, w))
        shared, mixed, hyper, inj = out
        TIMERS["take"] += time.perf_counter() - _t2
        return mixed, hyper, inj, shared

    def state_at(self, j: int) -> np.ndarray:
        """The recurrent state having consumed `j + 1` of this pass's tokens."""
        with _iosurface_view(self._prog._out_surfs[self._state_slot(j)],
                             (HV, DV, DK),
                             np.float16) as o:
            return np.array(o, np.float16)

    def select(self, proc: int) -> None:
        """Point every accessor at one procedure of a shared program."""
        proc = int(proc)
        if proc not in self._proc_ns:
            raise RuntimeError(f"MIL layer {self.layer}: no procedure {proc}")
        self._proc = proc
        self.k = self._proc_k[proc]
        self._ns = self._proc_ns[proc]

    def _state_slot(self, j: int) -> int:
        """Output index holding the state after slot j.

        A procedure that exports every prefix indexes them directly. One that
        exports only the end of its chunk writes into the last state surface,
        which is where the widest procedure's `commit` already looks, so the
        two need no handover between them.
        """
        if self._ns == self.k:
            return int(j)
        if int(j) != self.k - 1:
            raise RuntimeError(
                f"MIL layer {self.layer}: only the last of {self.k} slots has "
                f"a state in this graph, asked for {j}")
        return self._ob - 1

    def current_state(self) -> np.ndarray:
        """The committed recurrent state, read out of its input surface.

        Handing a prompt from a wide prefill graph to a narrow decode graph is
        a copy of this plus the conv window; nothing else in the layer carries
        across a pass.
        """
        with _iosurface_view(self._prog._in_surfs[4], (HV, DV, DK),
                             np.float16) as src:
            return np.array(src, np.float16)

    def set_state(self, value) -> None:
        """Seed the recurrent state; the surface is the only copy."""
        with _iosurface_view(self._prog._in_surfs[4], (HV, DV, DK),
                             np.float16) as dst:
            np.copyto(dst, np.asarray(value, np.float16).reshape(HV, DV, DK))

    def commit(self, j: int) -> None:
        """Advance the layer's state and conv window to the `j + 1` boundary.

        Speculation needs every prefix, not just the last: the graph emits a
        state per token and the whole conv window, so a partially accepted
        block costs no extra pass. A pipelined two-half forward snapshots
        prefixes on host because the second half overwrites the output
        surfaces; `_block_commit` still calls this with the accepted j.
        """
        j = int(j)
        if self._prefix_state is not None and self._prefix_state[j] is not None:
            self._commit_saved(j)
            return
        p = self._prog
        with _iosurface_view(p._out_surfs[self._state_slot(j)], (HV, DV, DK), np.float16) as o:
            with _iosurface_view(p._in_surfs[4], (HV, DV, DK), np.float16) as d:
                np.copyto(d, o)
        self._apply_conv_from_fseq(self._read_fseq(), j)

    def begin_block(self, n: int) -> None:
        """Reset prefix snapshots for a K-slot verification pass of width n."""
        self._prefix_state = [None] * int(n)
        self._prefix_fseq = []
        self._fut = None

    def _write_x(self, x_bc1s: np.ndarray) -> None:
        x = np.asarray(x_bc1s, np.float16).reshape(HC_W, S)
        p = self._prog
        with _iosurface_view(p._in_surfs[0], (HC_W, S), np.float16) as dst:
            np.copyto(dst, x)
        if not self._conv_surface_current:
            with _iosurface_view(p._in_surfs[1], (3 * QKV, S), np.float16) as dst:
                dst[:, 0] = self._conv
            self._conv_surface_current = True

    def _take(self, w: int):
        p = self._prog
        kk = self.k
        out = []
        for j, shape in enumerate(((H, S), (H, S), (HC_W, S), (HC, S))):
            with _iosurface_view(p._out_surfs[self._ob + j], shape, np.float16) as o:
                out.append(np.array(o[:, :w], np.float32).reshape(1, shape[0], 1, w))
        shared, mixed, hyper, inj = out
        return mixed, hyper, inj, shared

    def _read_fseq(self) -> np.ndarray:
        p = self._prog
        with _iosurface_view(p._out_surfs[self._ob + 4],
                             (QKV, self._conv_out_width), np.float16) as o:
            return np.array(o, np.float16)

    def _apply_conv_from_fseq(self, fseq: np.ndarray, j: int) -> None:
        p = self._prog
        with _iosurface_view(p._in_surfs[1], (3 * QKV, S), np.float16) as d:
            for t in range(3):
                value = fseq[:, j + 1 + t]
                self._conv[t * QKV:(t + 1) * QKV] = value
                d[t * QKV:(t + 1) * QKV, 0] = value
        self._conv_surface_current = True

    def _commit_saved(self, j: int) -> None:
        self.set_state(self._prefix_state[j])
        for off, w, fseq in self._prefix_fseq:
            if off <= j < off + w:
                self._apply_conv_from_fseq(fseq, j - off)
                return
        raise RuntimeError(f"MIL layer {self.layer}: no fseq snapshot for j={j}")

    def snapshot(self, offset: int, n: int) -> None:
        """Keep prefix states so a later half can overwrite the output surfaces."""
        if self._prefix_state is None:
            return
        for j in range(int(n)):
            self._prefix_state[int(offset) + j] = self.state_at(j)
        self._prefix_fseq.append((int(offset), int(n), self._read_fseq()))

    def fence_state(self, n: int) -> None:
        """Copy the last live slot's recurrent state into the input surface.

        The next micro-batch of this layer reads that surface. Must not run
        while an evaluate on this program is in flight.
        """
        if int(n) <= 0:
            return
        p = self._prog
        j = int(n) - 1
        with _iosurface_view(p._out_surfs[self._state_slot(j)], (HV, DV, DK), np.float16) as o:
            with _iosurface_view(p._in_surfs[4], (HV, DV, DK), np.float16) as d:
                np.copyto(d, o)
        self._apply_conv_from_fseq(self._read_fseq(), j)

    def run(self, x_bc1s: np.ndarray, n: int, *, async_: bool = False):
        """Write inputs and evaluate. Async returns a future; caller must finish()."""
        self._write_x(x_bc1s)
        if async_:
            self._fut = self._eng.submit_async(self._prog, procedure_index=self._proc)
            return self._fut
        if not self._eng.submit(self._prog, procedure_index=self._proc):
            raise RuntimeError(f"MIL layer {self.layer}: submit failed")
        self._fut = None
        return None

    def finish(self, n: int, *, offset: int = 0, fence: bool = False):
        """Wait for an in-flight evaluate, take outputs, snapshot, maybe fence."""
        if self._fut is not None:
            ok = self._fut.wait()
            self._fut = None
            if not ok:
                raise RuntimeError(f"MIL layer {self.layer}: async submit failed")
        out = self._take(int(n))
        self.snapshot(int(offset), int(n))
        if fence:
            self.fence_state(int(n))
        return out


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

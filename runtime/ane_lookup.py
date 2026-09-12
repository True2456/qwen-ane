# SPDX-License-Identifier: Apache-2.0
"""ANE gather: one program, table is a persistent IOSurface, only ids change.

CoreML ``predict({"table": ..., "ids": ...})`` copies the whole table every
call. That cannot beat host ``copyto``. Same contract as ``AneDynamicLinear``:
compile once, ``write_table`` pages the table surface, ``gather`` only writes
10 indices and evaluates.

Authoring is CoreML NN ``gather`` → espresso ``gather_nd``. Execution is
``_ANEInMemoryModel`` + ``_ANERequest`` IOSurfaces.

Rank-2 flatten dies at W>16384 (ANEC_IR serialize). Expert slabs are rank-3
``[E, 1280, 2560]`` / ``[E, 2560, 640]`` so W stays 2560/640.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import (
    _ane_input_symbols,
    _as_void_p,
    _cls,
    _copyto_fp16,
    _create_iosurface,
    _desc,
    _indexset_to_nsarray,
    _iosurface_alloc_size,
    _iosurface_view,
    _load_iosurface,
    _nsarray,
    _nsdata,
    _nsdict,
    _nsnumber_int,
    _nsnumber_uint,
    _nsstring,
    _objc_call,
    _wrap_iosurface,
)

BANK = 64
K_PIN = 10
_VOID = ctypes.c_void_p
_QOS = ctypes.c_uint
_ERR = ctypes.POINTER(ctypes.c_void_p)
_BOOL = ctypes.c_bool
_ULONGLONG = ctypes.c_ulonglong


def _compile_espresso_gather(
    vocab: int,
    dim: int,
    k: int,
    *,
    table_shape: tuple[int, ...] | None = None,
    projection: str | bool = False,
):
    """Author gather via CoreML, return espresso.net text + shape dict."""
    import coremltools as ct
    from coremltools.models import datatypes
    from coremltools.models.neural_network import NeuralNetworkBuilder

    table_shape = tuple(table_shape) if table_shape is not None else (vocab, dim)
    inputs = [
            ("table", datatypes.Array(*table_shape)),
            ("ids", datatypes.Array(k)),
        ]
    if projection:
        xs = ((1, table_shape[-1], 1, 32) if projection == "conv" else
              (1, table_shape[-1], 32) if projection == "matmul" else
              (1, 1, table_shape[-1]))
        inputs.append(("x", datatypes.Array(*xs)))
    builder = NeuralNetworkBuilder(
        inputs,
        [("y", None)],
        disable_rank5_shape_mapping=True,
    )
    builder.add_gather(
        name="lookup",
        input_names=["table", "ids"],
        output_name="selected" if projection else "y",
        axis=0,
    )
    if projection == "conv":
        builder.add_reshape_static(name="weight", input_name="selected", output_name="weight",
                                   output_shape=[k * table_shape[1], table_shape[2], 1, 1])
        builder.add_convolution(name="project", kernel_channels=table_shape[2],
                                output_channels=k * table_shape[1], height=1, width=1,
                                stride_height=1, stride_width=1, border_mode="valid", groups=1,
                                W=None, b=None, has_bias=False,
                                input_name=["x", "weight"], output_name="y")
    elif projection == "matmul":
        builder.add_batched_mat_mul(name="project", input_names=["selected", "x"], output_name="y")
    elif projection:
        builder.add_transpose(name="wt", axes=[0, 2, 1], input_name="selected", output_name="wt")
        builder.add_transpose(name="xt", axes=[0, 2, 1], input_name="x", output_name="xt")
        builder.add_multiply_broadcastable(name="multiply", input_names=["wt", "xt"], output_name="products")
        builder.add_reduce_sum(name="project", input_name="products", output_name="y",
                               axes=[1], keepdims=True)
    spec = builder.spec
    spec.description.input[1].type.multiArrayType.dataType = (
        ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32
    )
    ml = ct.models.MLModel(spec, compute_units=ct.ComputeUnit.CPU_AND_NE)
    root = Path(ml.get_compiled_model_path())
    net = (root / "model.espresso.net").read_text()
    shape = json.loads((root / "model.espresso.shape").read_text())
    weights = (root / "model.espresso.weights").read_bytes()
    # Content-addressed ANE ident ignored the live table shape; tag the net.
    try:
        obj = json.loads(net)
        props = obj.setdefault("properties", {})
        props["ane_port_gather"] = f"t{table_shape}_k{k}"
        net = json.dumps(obj)
    except json.JSONDecodeError:
        net = net + f"\n/* ane_port_gather t{table_shape} k{k} */\n"
    return net, shape, weights, ml


def _ane_compile_espresso(net: str, shape: dict, weights: bytes):
    os.environ.setdefault("Q38_ANE_REUSE_COMPILED", "0")
    eng = E.AneEngine()
    if not eng.available:
        raise RuntimeError("AneEngine unavailable")
    net_data = _nsdata(net.encode())
    # Identifier was hashing an empty weights dict (SHA-256 of "") so every
    # gather_nd compiled to the same artifact. Unique dict key per shape.
    tag = hashlib.sha256(net.encode() + json.dumps(shape, sort_keys=True).encode()
                         + weights).digest()
    wd = _nsdict({
        _nsstring("ane_port_gather"): _nsdict({
            _nsstring("data"): _nsdata(tag),
            _nsstring("offset"): _nsnumber_uint(0),
        }),
    })
    descriptor = _objc_call(
        ctypes.c_void_p,
        [_VOID, _VOID, _VOID],
        _cls("_ANEInMemoryModelDescriptor"),
        "modelWithNetworkDescription:weights:optionsPlist:",
        net_data,
        wd,
        None,
    )
    if not descriptor:
        raise RuntimeError("espresso descriptor nil")
    model = _objc_call(
        ctypes.c_void_p,
        [_VOID],
        _cls("_ANEInMemoryModel"),
        "inMemoryModelWithDescriptor:",
        descriptor,
    )
    if not model:
        raise RuntimeError("espresso model nil")
    local = _desc(
        _objc_call(ctypes.c_void_p, (), model, "localModelPath")
    )
    if local and local != "(nil)":
        shutil.rmtree(local, ignore_errors=True)
        os.makedirs(local, exist_ok=True)
        Path(local, "model.espresso.net").write_text(net)
        Path(local, "model.espresso.shape").write_text(json.dumps(shape))
        Path(local, "model.espresso.weights").write_bytes(weights)

    err_ptr = ctypes.c_void_p(0)
    ok = _objc_call(
        _BOOL,
        [_QOS, _VOID, _ERR],
        model,
        "compileWithQoS:options:error:",
        _QOS(21),
        None,
        ctypes.byref(err_ptr),
    )
    if not ok:
        raise RuntimeError(f"espresso compile: {_desc(err_ptr.value)}")
    err_ptr = ctypes.c_void_p(0)
    ok = _objc_call(
        _BOOL,
        [_QOS, _VOID, _ERR],
        model,
        "loadWithQoS:options:error:",
        _QOS(21),
        None,
        ctypes.byref(err_ptr),
    )
    if not ok:
        raise RuntimeError(f"espresso load: {_desc(err_ptr.value)}")
    keep = [eng, net_data, wd, descriptor]
    return eng, model, descriptor, keep


class AneGather:
    """One ANE gather_nd. Table IOSurface stays bound; gather writes ids only."""

    def __init__(
        self,
        vocab: int,
        dim: int = 2560,
        k: int = K_PIN,
        *,
        table_shape: tuple[int, ...] | None = None,
        _projection: str | bool = False,
    ):
        if table_shape is not None:
            table_shape = tuple(int(x) for x in table_shape)
            vocab = table_shape[0]
            row_shape = table_shape[1:]
            dim = int(np.prod(row_shape)) if row_shape else 1
        else:
            row_shape = (int(dim),)
            table_shape = (int(vocab), int(dim))
        self.vocab = int(vocab)
        self.dim = int(dim)
        self.row_shape = row_shape
        self.table_shape = table_shape
        self.y_shape = ((1, int(k) * row_shape[0], 1, 32) if _projection == "conv" else
                        (int(k), row_shape[0], 32) if _projection == "matmul" else
                        (int(k), 1, row_shape[0]) if _projection else (int(k), *row_shape))
        self._projection = _projection
        self.k = int(k)
        net, shape, weights, ml = _compile_espresso_gather(
            self.vocab, self.dim, self.k, table_shape=self.table_shape,
            projection=_projection,
        )
        self._ml = ml  # keep CoreML artifact alive
        self.shape = shape
        self.eng, self.model, self._desc, self._keep = _ane_compile_espresso(
            net, shape, weights
        )
        self.symbols = _ane_input_symbols(self.model)
        _load_iosurface()
        n_table = int(np.prod(self.table_shape))
        n_y = int(np.prod(self.y_shape))
        self._table_surf = _create_iosurface(_iosurface_alloc_size(n_table))
        self._ids_surf = _create_iosurface(max(self.k * 4, 0x10000))
        self._y_surf = _create_iosurface(_iosurface_alloc_size(n_y))
        if not (self._table_surf and self._ids_surf and self._y_surf):
            raise RuntimeError("IOSurface alloc failed")
        self._table_obj = _wrap_iosurface(self._table_surf)
        self._ids_obj = _wrap_iosurface(self._ids_surf)
        self._y_obj = _wrap_iosurface(self._y_surf)
        if _projection:
            self.x_shape = ((1, self.table_shape[-1], 1, 32) if _projection == "conv" else
                            (1, self.table_shape[-1], 32) if _projection == "matmul" else
                            (1, 1, self.table_shape[-1]))
            self._x_surf = _create_iosurface(_iosurface_alloc_size(int(np.prod(self.x_shape))))
            self._x_obj = _wrap_iosurface(self._x_surf)
        if not (self._table_obj and self._ids_obj and self._y_obj):
            raise RuntimeError("IOSurface wrap failed")
        in_objs = self._bind_inputs()
        inner = (
            _objc_call(ctypes.c_void_p, (), self.model, "model") or self.model
        )
        in_idx = _objc_call(
            ctypes.c_void_p,
            [_ULONGLONG],
            inner,
            "inputSymbolIndicesForProcedureIndex:",
            0,
        )
        out_idx = _objc_call(
            ctypes.c_void_p,
            [_ULONGLONG],
            inner,
            "outputSymbolIndicesForProcedureIndex:",
            0,
        )
        in_idx = _indexset_to_nsarray(in_idx) or _nsarray(
            [_nsnumber_int(i) for i in range(len(in_objs))]
        )
        out_idx = _indexset_to_nsarray(out_idx) or _nsarray([_nsnumber_int(0)])
        raw = _objc_call(ctypes.c_void_p, (), _cls("_ANERequest"), "alloc")
        self._request = _objc_call(
            ctypes.c_void_p,
            [_VOID] * 9,
            raw,
            "initWithInputs:inputIndices:outputs:outputIndices:"
            "weightsBuffer:perfStats:procedureIndex:sharedEvents:"
            "transactionHandle:",
            _nsarray(in_objs),
            in_idx,
            _nsarray([self._y_obj]),
            out_idx,
            None,
            None,
            _nsnumber_int(0),
            None,
            None,
        )
        if not self._request:
            raise RuntimeError("ANE request nil")
        self._keep.extend(
            [
                self._table_obj,
                self._ids_obj,
                self._y_obj,
                in_idx,
                out_idx,
                self._request,
            ]
        )
        self._paged = False

    def _bind_inputs(self) -> list:
        by_name = {"table": self._table_obj, "ids": self._ids_obj}
        if self._projection:
            by_name["x"] = self._x_obj
        if self.symbols and all(s in by_name for s in self.symbols):
            return [by_name[s] for s in self.symbols]
        if self._projection:
            raise RuntimeError(f"Cannot bind projection inputs: {self.symbols}")
        return [self._ids_obj, self._table_obj]

    def write_table(self, table: np.ndarray) -> None:
        t = np.ascontiguousarray(table)
        if t.shape == (self.dim, self.vocab) and len(self.table_shape) == 2:
            t = np.ascontiguousarray(t.T)
        if t.shape != self.table_shape:
            raise ValueError(f"table {t.shape} != {self.table_shape}")
        with _iosurface_view(self._table_surf, self.table_shape, np.float16) as dst:
            _copyto_fp16(dst, t)
        self._paged = True

    def gather(self, ids, *, as_float32: bool = True) -> np.ndarray:
        if not self._paged:
            raise RuntimeError("write_table first")
        ids = np.ascontiguousarray(ids, dtype=np.int32).reshape(self.k)
        # ANE IOSurfaces are fp16. int32 36 reads as fp16 ~0 → always row 0.
        with _iosurface_view(self._ids_surf, (self.k,), np.float16) as dst:
            np.copyto(dst, ids.astype(np.float16), casting="unsafe")
        err_ptr = ctypes.c_void_p(0)
        ok = _objc_call(
            _BOOL,
            [_QOS, _VOID, _VOID, _ERR],
            self.model,
            "evaluateWithQoS:options:request:error:",
            _QOS(21),
            None,
            _as_void_p(self._request),
            ctypes.byref(err_ptr),
        )
        if not ok:
            raise RuntimeError(f"ANE gather eval: {_desc(err_ptr.value)}")
        with _iosurface_view(self._y_surf, self.y_shape, np.float16) as src:
            # ascontiguousarray(fp16) can return the mapped array itself.
            # The lock ends here and the next evaluate overwrites that memory.
            return np.array(src, dtype=np.float32 if as_float32 else np.float16,
                            order="C", copy=True)


# Back-compat names used by the earlier probe.
AneLookup = AneGather


class AneGatherMM(AneGather):
    """Experimental live-table gather + matmul, returning only activations."""

    def __init__(self, experts: int, n_out: int, n_in: int, k: int = K_PIN,
                 method: str = "conv"):
        if method not in ("conv", "matmul", "reduce"):
            raise ValueError(method)
        super().__init__(experts, k=k, table_shape=(experts, n_out, n_in),
                         _projection=method)

    def project(self, x, ids):
        x = np.asarray(x, np.float32).reshape(self.table_shape[-1])
        with _iosurface_view(self._x_surf, self.x_shape, np.float16) as dst:
            if self._projection in ("conv", "matmul"):
                dst.reshape(self.table_shape[-1], 32)[:] = x[:, None]
            else:
                _copyto_fp16(dst, x.reshape(self.x_shape))
        y = self.gather(ids)
        if self._projection in ("conv", "matmul"):
            return y.reshape(self.k, self.table_shape[1], 32)[..., 0].copy()
        return y[:, 0, :].copy()


class PagedLookup:
    """Host 512-row table, one AneGather of vocab=64. Prefer AneGather(vocab=512)."""

    def __init__(self, table_oc_v: np.ndarray, k: int = K_PIN, bank: int = BANK):
        table = np.ascontiguousarray(table_oc_v, dtype=np.float32)
        dim, vocab = table.shape
        self.rows = np.ascontiguousarray(table.T)
        self.dim, self.vocab, self.k, self.bank = dim, vocab, k, bank
        self.prog = AneGather(vocab=bank, dim=dim, k=k)
        self._scratch = np.zeros((k, dim), np.float32)
        self._paged_bank = -1

    def gather(self, ids) -> np.ndarray:
        ids = np.asarray(ids, np.int32).reshape(self.k)
        out = self._scratch
        banks = ids // self.bank
        for b in np.unique(banks):
            b = int(b)
            lo = b * self.bank
            if self._paged_bank != b:
                self.prog.write_table(self.rows[lo : lo + self.bank])
                self._paged_bank = b
            hit = banks == b
            local = np.zeros(self.k, np.int32)
            local[hit] = ids[hit] - lo
            rows = self.prog.gather(local)
            out[hit] = rows[hit]
        return out

# SPDX-License-Identifier: Apache-2.0
"""ane_direct_engine.py - Direct _ANEModel / _ANEClient low-level loader and evaluation runtime."""

from __future__ import annotations

import ctypes
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import (
    _cls,
    _desc,
    _msg,
    _nsdata,
    _nsdict,
    _nsnumber_int,
    _nsstring,
    _objc,
    _sel,
)

logger = logging.getLogger(__name__)


@dataclass
class DirectAneProgram:
    """Represents a directly loaded ANE model without intermediate compiler layers."""
    model: ctypes.c_void_p
    model_url: str
    input_dim: int
    output_dim: int
    seq_len: int
    _client: ctypes.c_void_p
    _in_surf: Optional[ctypes.c_void_p] = None
    _out_surf: Optional[ctypes.c_void_p] = None
    _request: Optional[ctypes.c_void_p] = None


class AneDirectEngine:
    """Low-level direct driver for ANE packages and _ANEClient hardware evaluation."""

    def __init__(self):
        self._client_cls = _cls("_ANEClient")
        self.client = _msg(self._client_cls, "sharedConnection")
        if not self.client:
            self.client = _msg(_msg(self._client_cls, "alloc"), "init")
        
        self._ane_model_cls = _cls("_ANEModel")
        self.engine = E.AneEngine()

    def load_compiled_package(
        self,
        package_path: str | Path,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        cache_id: Optional[str] = None,
        qos: int = 21,
    ) -> Optional[DirectAneProgram]:
        """Load a compiled ANE model package directly via _ANEClient."""
        package_path = str(Path(package_path).expanduser().resolve())
        if not os.path.exists(package_path):
            logger.error("Package path does not exist: %s", package_path)
            return None

        NSURL = _cls("NSURL")
        file_url = _msg(NSURL, "fileURLWithPath:", _nsstring(package_path), argtypes=[ctypes.c_void_p])

        raw_model = _msg(self._ane_model_cls, "alloc")
        InitModel = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool
        )
        
        key = f"direct_{input_dim}_{output_dim}_{seq_len}"
        cid = cache_id if cache_id else key
        ane_model = InitModel(("objc_msgSend", _objc))(
            raw_model,
            _sel("initWithModelAtURL:key:identifierSource:cacheURLIdentifier:modelAttributes:standardizeURL:"),
            file_url,
            _nsstring(key),
            1,
            _nsstring(cid),
            _nsdict({}),
            True
        )

        if not ane_model:
            logger.error("Failed to initialize _ANEModel for %s", package_path)
            return None

        # Load model with _ANEClient
        LoadModel = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)
        )
        opts = _nsdict({_nsstring("kANEFKeepModelMemoryWiredKey"): _nsnumber_int(0)})
        err_ptr = ctypes.c_void_p(0)
        
        ok = LoadModel(("objc_msgSend", _objc))(
            self.client,
            _sel("loadModel:options:qos:error:"),
            ane_model,
            opts,
            qos,
            ctypes.byref(err_ptr)
        )

        if not ok:
            logger.error("Direct _ANEClient loadModel failed: %s", _desc(err_ptr.value) if err_ptr.value else "unknown")
            return None

        prog = DirectAneProgram(
            model=ane_model,
            model_url=package_path,
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            _client=self.client,
        )
        return prog

    def unload(self, prog: DirectAneProgram, qos: int = 21) -> bool:
        """Unload model from ANE hardware."""
        if not prog.model:
            return True
        UnloadModel = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)
        )
        opts = _nsdict({})
        err_ptr = ctypes.c_void_p(0)
        ok = UnloadModel(("objc_msgSend", _objc))(
            self.client,
            _sel("unloadModel:options:qos:error:"),
            prog.model,
            opts,
            qos,
            ctypes.byref(err_ptr)
        )
        prog.model = 0
        return bool(ok)

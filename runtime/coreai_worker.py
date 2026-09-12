"""Bound Core AI's inference IOSurface lifetime without restarting the decoder.

Core AI currently exhausts IOSurface allocations after ~3000 evaluations of
the 48 Flash-Next assets. This optional adapter puts inference in a spawned
worker and recycles it after 1536 calls. All state is explicit input/output;
only the caller owns persistent state. Expert banks never enter this worker.

The copies, IPC, specialization, and restarts are part of call latency. This
is a long-run workaround, not a claim of zero-copy ANE/GPU interoperability.
"""
from __future__ import annotations

import atexit
import asyncio
import multiprocessing as mp
import os
import traceback
from pathlib import Path

import numpy as np


def _worker(conn):
    from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions

    async def serve():
        ane = next(k for k in ComputeUnitKind.available_kinds() if str(k) == 'Neural Engine')
        spec = SpecializationOptions.from_preferred_compute_unit_kind(ane)
        models, functions = {}, {}
        while True:
            request = conn.recv()
            if request is None:
                return
            path, function, inputs = request
            try:
                key = (path, function)
                if key not in functions:
                    model = await AIModel.load(path, specialization_options=spec)
                    models[path] = model
                    functions[key] = model.load_function(function)
                result = await functions[key]({k: NDArray(v) for k,v in inputs.items()})
                arrays = {k: np.array(v.numpy(), copy=True) for k,v in result.items()}
                result.clear()
                conn.send((True, arrays))
                del arrays, result, inputs, request
            except Exception:
                conn.send((False, traceback.format_exc()))
    try:
        asyncio.run(serve())
    finally:
        conn.close()


class _Client:
    def __init__(self):
        self.process = self.conn = None
        self.calls = self.restarts = 0
        self.limit = int(os.environ.get('FLASHNEXT_ANE_WORKER_CALLS', '1536'))
        if not 48 <= self.limit <= 1920:
            raise ValueError('FLASHNEXT_ANE_WORKER_CALLS must be between 48 and 1920')
        atexit.register(self.close)

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.conn.send(None)
                self.process.join(timeout=10)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=5)
            self.conn.close()
            self.process.close()
            self.process = self.conn = None

    def call(self, path, function, inputs):
        if self.process is None or self.calls >= self.limit:
            if self.process is not None:
                self.close()
                self.restarts += 1
                print(f'  ANE worker restart {self.restarts} (state held by decoder)', flush=True)
            ctx = mp.get_context('spawn')
            self.conn, child = ctx.Pipe()
            self.process = ctx.Process(target=_worker, args=(child,), daemon=True)
            self.process.start()
            child.close()
            self.calls = 0
        self.conn.send((path, function, inputs))
        try:
            ok, value = self.conn.recv()
        except EOFError as exc:
            raise RuntimeError('ANE worker exited during inference') from exc
        if not ok:
            raise RuntimeError(value)
        self.calls += 1
        return value


_client = None


class WorkerAIModel:
    @classmethod
    async def load(cls, path, specialization_options=None):
        # This adapter intentionally supports ANE-preferred inference only.
        # Specialization happens in the worker on first call and is timed.
        obj = cls()
        obj.path = str(Path(path).resolve())
        return obj

    def load_function(self, name, **kwargs):
        async def call(inputs, state=None):
            from coreai.runtime import NDArray
            global _client
            if state:
                raise NotImplementedError('Worker adapter requires explicit state I/O')
            if _client is None:
                _client = _Client()
            arrays = {k: np.array(v.numpy(), copy=True) for k,v in inputs.items()}
            result = _client.call(self.path, name, arrays)
            return {k: NDArray(v) for k,v in result.items()}
        return call

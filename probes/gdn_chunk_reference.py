"""Independent NumPy implementation of the published gated WY/UT equations.

Used for tests and the isolated ANE probe, not the production execution path.
See https://arxiv.org/html/2412.06464v3#S3.SS3 .
"""
import numpy as np


def chunk(q, k, v, gates, beta, state):
    c = q.shape[-2]
    decay = np.zeros((*gates.shape, c), dtype=q.dtype)
    for i in range(c):
        decay[..., i, i] = 1
        for j in range(i-1, -1, -1):
            decay[..., i, j] = decay[..., i, j+1] * gates[..., j+1]
    gamma = np.cumprod(gates, axis=-1)[...,None]
    a = np.tril((k @ k.swapaxes(-1,-2)) * decay * beta[...,None], -1)
    inv = np.broadcast_to(np.eye(c, dtype=q.dtype), a.shape).copy()
    row, col = np.arange(c)[:,None], np.arange(c)[None,:]
    for stage in range((c-1).bit_length()):
        half = 1 << stage
        cross_mask = ((row//(2*half) == col//(2*half)) &
                      ((row//half)%2 == 1) & ((col//half)%2 == 0))
        inv = inv - (inv @ (a * cross_mask)) @ inv
    delta = inv @ (beta[...,None] * (v - gamma * (k @ state.swapaxes(-1,-2))))
    y = gamma * (q @ state.swapaxes(-1,-2)) + ((q @ k.swapaxes(-1,-2)) * decay) @ delta
    final = gamma[...,-1:, :] * state + delta.swapaxes(-1,-2) @ (k * decay[...,-1,:,None])
    return y, final

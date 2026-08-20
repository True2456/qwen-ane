"""Verify and benchmark the server's ANE-resident GDN recurrence state."""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.expanduser("~/AppleLLM/q38_native_engine"))

import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view
from tools.ane_serve import AneGdnStep


H, DK, DV = 48, 128, 128
rng = np.random.default_rng(7)
step = AneGdnStep(AneEngine(), E, _iosurface_view, H, DK, DV)
marker = object()
initial = rng.normal(0, 0.02, (H, DV, DK)).astype(np.float32)
reference_state = initial.copy()

for index in range(2):
    q = rng.normal(0, 0.05, (H, DK)).astype(np.float32)
    k = rng.normal(0, 0.05, (H, DK)).astype(np.float32)
    v = rng.normal(0, 0.05, (H, DV)).astype(np.float32)
    a = rng.uniform(-12, 8, H).astype(np.float32)
    beta_logits = rng.uniform(-3, 1, H).astype(np.float32)
    A = np.geomspace(0.004, 140, H).astype(np.float32)
    dt_bias = np.linspace(-9, 20, H).astype(np.float32)
    got_y = step.resident(
        marker,
        initial if index == 0 else None,
        q,
        k,
        v,
        a,
        beta_logits,
        A,
        dt_bias,
    )
    decay = np.exp(-A * np.logaddexp(0, a + dt_bias))
    beta = 1 / (1 + np.exp(-beta_logits))
    reference_state *= decay[:, None, None]
    memory = np.sum(reference_state * k[:, None, :], axis=-1)
    delta = (v - memory) * beta[:, None]
    reference_state += k[:, None, :] * delta[:, :, None]
    expected_y = np.sum(reference_state * q[:, None, :], axis=-1)
    y_rel = np.max(np.abs(got_y - expected_y)) / (
        np.max(np.abs(expected_y)) + 1e-9
    )
    print(f"step {index + 1}: y rel={y_rel:.6f}")

_, state_surface, _ = step._slots[id(marker)]
with _iosurface_view(state_surface, (H * DK, DV), np.float16) as state_view:
    got_state = (
        np.array(state_view, np.float32)
        .reshape(H, DK, DV)
        .transpose(0, 2, 1)
    )
state_rel = np.max(np.abs(got_state - reference_state)) / (
    np.max(np.abs(reference_state)) + 1e-9
)

# Stable small inputs prevent an intentionally repeated synthetic recurrence
# from growing while timing. The cache state remains on its IOSurface.
q = rng.normal(0, 0.005, (H, DK)).astype(np.float32)
k = rng.normal(0, 0.005, (H, DK)).astype(np.float32)
v = rng.normal(0, 0.005, (H, DV)).astype(np.float32)
A = np.full(H, 0.01, np.float32)
dt_bias = np.zeros(H, np.float32)
a = np.full(H, -4, np.float32)
beta_logits = np.full(H, -4.6, np.float32)
for _ in range(8):
    step.resident(marker, None, q, k, v, a, beta_logits, A, dt_bias)
iterations = 100
start = time.perf_counter()
for _ in range(iterations):
    step.resident(marker, None, q, k, v, a, beta_logits, A, dt_bias)
elapsed_ms = (time.perf_counter() - start) * 1e3 / iterations

print(f"state after two dependent steps: rel={state_rel:.6f}")
print(f"resident copy+write+dispatch+y-read: {elapsed_ms:.3f} ms/step")
print("PASS" if state_rel < 3e-3 and y_rel < 1e-2 else "FAIL")

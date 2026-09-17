"""Shared 27B private-MIL chunked gated delta rule, without model weights.

The raw Q/K normalization and gate preparation are shared with the exact
unroll. Only the recurrence changes. State IO remains [head, value, key].
The product scan and blocked inverse follow probes/flashnext_mil_chunk.py;
this emitter uses 32-wide hardware tiles for 8/16/32 live tokens. Only
full runtime batches use it; ragged batches and decode remain stepwise.
"""
from __future__ import annotations

import numpy as np

from probes.ane_gdn_scan64 import AneGdnUnrolled, H, D, unrolled_mil


def chunk_masks(c):
    masks = {"lower": np.tril(np.ones((c, c))),
             "strict": np.tril(np.ones((c, c)), -1), "eye": np.eye(c)}
    row, col = np.indices((c, c))
    for stage in range((c - 1).bit_length()):
        half = 1 << stage
        masks[f"block{stage}"] = ((row // (2*half) == col // (2*half)) &
                                  (row % (2*half) >= half) &
                                  (col % (2*half) < half))
    return {name: value.astype(np.float16) for name, value in masks.items()}


def chunk_body(lines, live):
    c = 32
    def op(name, shape, expression):
        lines.append(f'    tensor<fp16, [{",".join(map(str, shape))}]> {name} = '
                     f'{expression}[name=string("{name}")];')
        return name

    def binary(name, kind, x, y, rows, cols):
        return op(name, (1,H,rows,cols), f'{kind}(x={x}, y={y})')

    def mm(name, x, y, rows, cols, tx=False, ty=False):
        return op(name, (1,H,rows,cols), f'matmul(x={x}, y={y}, '
                  f'transpose_x=bool({str(tx).lower()}), '
                  f'transpose_y=bool({str(ty).lower()}))')

    def reshape(name, source, shape):
        return op(name, shape, f'reshape(x={source}, '
                  f'shape=tensor<int32, [4]>([{",".join(map(str,shape))}]))')

    def sl(name, source, begin, end):
        return op(name, tuple(b-a for a,b in zip(begin,end)),
                  f'slice_by_index(x={source}, '
                  f'begin=tensor<int32, [4]>([{",".join(map(str,begin))}]), '
                  f'end=tensor<int32, [4]>([{",".join(map(str,end))}]))')

    def trans(name, source, rows, cols):
        return op(name, (1,H,rows,cols),
                  f'transpose(x={source}, perm=tensor<int32, [4]>([0,1,3,2]))')

    def shift(name, source, offset, cols):
        cut = sl(name+'cut', source, (0,0,0,0), (1,H,c-offset,cols))
        return op(name, (1,H,c,cols), f'pad(x={cut}, '
                  f'pad=tensor<int32, [8]>([0,0,0,0,{offset},0,0,0]), '
                  f'mode=string("constant"), constant_val=fp16(1))')

    for name in chunk_masks(c):
        lines.append(f'    tensor<fp16, [1,1,{c},{c}]> c{name} = const()[name=string("c{name}"), val=tensor<fp16, [1,1,{c},{c}]>(BLOBFILE(path=string("@model_path/weights/{name}.bin"), offset=uint64(64)))];')

    for i in range(live):
        reshape(f'cq{i}', f'qf{i}', (1, H, 1, D))
        reshape(f'ck{i}', f'kf{i}', (1, H, 1, D))
        reshape(f'cdv{i}', f'dr{i}', (1, H, D, 1))
        sl(f'cg{i}', f'cdv{i}', (0, 0, 0, 0), (1, H, 1, 1))

    def cat(name, stem, cols):
        value = op(name+'live', (1,H,live,cols),
                   f'concat(values=({", ".join(stem+str(i) for i in range(live))}), axis=int32(2), interleave=bool(false))')
        if live == c:
            return value
        return op(name, (1,H,c,cols),
                  f'pad(x={value}, pad=tensor<int32, [8]>([0,0,0,0,0,{c-live},0,0]), '
                  f'mode=string("constant"), constant_val=fp16({1 if name == "cg" else 0}))')

    q, k = cat('cq', 'cq', D), cat('ck', 'ck', D)
    v = cat('cv', 'vt', D)
    beta = cat('cb', 'bt', 1)
    gates = cat('cg', 'cg', 1)
    state = reshape('cskey', 'state_u', (1, H, D, D))
    state = trans('cs', state, D, D)
    gm = binary('cgm', 'mul', gates, 'cstrict', c, c)
    upper = op('cupper', (1, 1, c, c), 'sub(x=fp16(1), y=cstrict)')
    decay = binary('cd0', 'add', gm, upper, c, c)
    gamma = gates
    for stage in range((c - 1).bit_length()):
        sh = shift(f'cds{stage}', decay, 1 << stage, c)
        decay = binary(f'cd{stage+1}', 'mul', decay, sh, c, c)
        sh = shift(f'cgs{stage}', gamma, 1 << stage, 1)
        gamma = binary(f'cgp{stage}', 'mul', gamma, sh, c, 1)
    decay = binary('cdecay', 'mul', decay, 'clower', c, c)
    gram = mm('cgram', k, k, c, c, ty=True)
    a = binary('ca0', 'mul', gram, decay, c, c)
    a = binary('ca1', 'mul', a, beta, c, c)
    a = binary('ca', 'mul', a, 'cstrict', c, c)
    for stage in range((c - 1).bit_length()):
        cross = binary(f'ccross{stage}', 'mul', a, f'cblock{stage}', c, c)
        if stage == 0:
            inv = binary('ci0', 'sub', 'ceye', cross, c, c)
        else:
            left = mm(f'cileft{stage}', inv, cross, c, c)
            corr = mm(f'cicorr{stage}', left, inv, c, c)
            inv = binary(f'ci{stage}', 'sub', inv, corr, c, c)
    ks = mm('cks', k, state, c, D, ty=True)
    ks = binary('cksg', 'mul', ks, gamma, c, D)
    rhs = binary('crhs0', 'sub', v, ks, c, D)
    rhs = binary('crhs', 'mul', rhs, beta, c, D)
    delta = mm('cdelta', inv, rhs, c, D)
    qk = mm('cqk', q, k, c, c, ty=True)
    qk = binary('cqkd', 'mul', qk, decay, c, c)
    local = mm('cylocal', qk, delta, c, D)
    initial = mm('cyinitial', q, state, c, D, ty=True)
    initial = binary('cyinitialg', 'mul', initial, gamma, c, D)
    y = binary('cy', 'add', initial, local, c, D)
    end = sl('cend', decay, (0, 0, live - 1, 0), (1, H, live, c))
    end = trans('cendt', end, c, 1)
    kend = binary('ckend', 'mul', k, end, c, D)
    scaled = binary('cupdate_scaled', 'mul', delta, 'fp16(64)', c, D)
    update = mm('cupdate', scaled, kend, D, D, tx=True)
    update = binary('cupdate_unscaled', 'mul', update, 'fp16(0.015625)', D, D)
    gend = sl('cgend', gamma, (0, 0, live - 1, 0), (1, H, live, 1))
    old = binary('cold', 'mul', state, gend, D, D)
    binary('sout', 'add', old, update, D, D)

    rows = []
    for i in range(live):
        one = sl(f'cyone{i}', y, (0, 0, i, 0), (1, H, i + 1, D))
        rows.append(one)
    op('y', (1, live * H, 1, D),
       f'concat(values=({", ".join(rows)}), axis=int32(1), interleave=bool(false))')

    lines.append('  } -> (y, sout);\n}\n// pure27_chunked_gdn\n')


def chunk_mil(module, tokens):
    if tokens not in (8, 16, 32):
        raise ValueError('chunk width must be 8, 16 or 32')
    return unrolled_mil(module, tokens, chunk_body=chunk_body)


def packed_chunk_mil(module, tokens):
    pack = module._BlobPacker()
    arrays = {
        "sum": np.ones((H, D, 1, 1), np.float16),
        "mean": np.full((H, D, 1, 1), 1 / D, np.float16),
        "repeat": np.ones((H * D, 1, 1, 1), np.float16),
        **chunk_masks(32),
    }
    offsets = {name: pack.append(value.tobytes()) + 64
               for name, value in arrays.items()}
    mil, cin, cout = chunk_mil(module, tokens)
    for name, offset in offsets.items():
        mil = mil.replace(f'weights/{name}.bin"), offset=uint64(64)',
                          f'weights/weight.bin"), offset=uint64({offset})')
    return mil, cin, cout, {"weight.bin": pack.getvalue()}


class AneGdnChunked(AneGdnUnrolled):
    def __init__(self, driver, tokens=16):
        mil, cin, cout, blobs = packed_chunk_mil(driver.module, tokens)
        super().__init__(
            driver, tokens,
            mil_factory=lambda m, t: (mil, cin, cout),
            weight_blobs=blobs,
            raw_weight_files=frozenset({"weight.bin"}),
        )


def numpy_chunk(q, k, v, decay, beta, initial_state):
    """Float64 independent oracle; inputs [token, head, dim], unscaled Q."""
    q, k, v = [np.asarray(a, np.float64).transpose(1, 0, 2) for a in (q, k, v)]
    g, b = [np.asarray(a, np.float64).T for a in (decay, beta)]
    state = np.asarray(initial_state, np.float64)
    c = q.shape[1]
    gamma = np.cumprod(g, axis=1)
    dec = np.zeros((q.shape[0], c, c), np.float64)
    for j in range(c):
        dec[:, j, j] = 1
        dec[:, j + 1:, j] = np.cumprod(g[:, j + 1:], axis=1)
    a = (k @ k.swapaxes(-1, -2)) * dec * b[:, :, None]
    a *= np.tril(np.ones((c, c)), -1)
    rhs = b[:, :, None] * (v - gamma[:, :, None] * (k @ state.swapaxes(-1, -2)))
    delta = np.linalg.solve(np.eye(c)[None] + a, rhs)
    y = (gamma[:, :, None] * (q @ state.swapaxes(-1, -2))
         + (q @ k.swapaxes(-1, -2) * dec) @ delta)
    final = (gamma[:, -1, None, None] * state
             + delta.swapaxes(-1, -2) @ (k * dec[:, -1, :, None]))
    return y.transpose(1, 0, 2) * 64, final

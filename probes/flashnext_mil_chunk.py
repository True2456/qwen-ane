"""Chunkwise gated delta rule (Yang et al., arXiv:2412.06464, section 3.3).

Q/K/V use [1, Hv, C, D]. Scalar decays are accumulated by a product scan,
including pairwise decays, without reciprocal prefix products or log(0).
The UT inverse uses the finite geometric series of a nilpotent matrix.
Only the final recurrent state is materialized; decode retains prefix states.

A 32-wide chunk is a different recurrence from eight 4-wide ones, and that
difference is enough to flip MoE experts down a 48-layer stack. When
`MIL_GDN_CHUNK_TILE` divides the live width, each tile is the same 4-wide
(or 8-wide) chunk decode uses, with state threaded between them.
"""
from __future__ import annotations

import os


def gdn_chunk(m, c, offs):
    if c not in (1, 2, 4, 8, 16, 32):
        raise ValueError("chunk live width must be 1, 2, 4, 8, 16 or 32")
    tile = int(os.environ.get("MIL_GDN_CHUNK_TILE", "4") or 0)
    if tile and c > tile:
        if c % tile:
            raise ValueError(f"chunk width {c} is not a multiple of tile {tile}")
        if tile not in (1, 2, 4, 8, 16, 32):
            raise ValueError("chunk tile must be 1, 2, 4, 8, 16 or 32")
        state = "e_state"
        for i, start in enumerate(range(0, c, tile)):
            state = _chunk_body(m, tile, offs, start=start, state_in=state,
                                prefix=f"p{i}")
        return
    _chunk_body(m, c, offs, start=0, state_in="e_state", prefix="")


def _chunk_body(m, live, offs, start, state_in, prefix):
    """One decode-width chunk over slots [start, start+live). Returns state name."""
    q_scale = int(os.environ.get("MIL_GDN_CHUNK_Q_SCALE", "64"))
    update_scale = int(os.environ.get("MIL_GDN_CHUNK_UPDATE_SCALE", "1"))
    if q_scale not in (1, 64) or update_scale not in (1, 64):
        raise ValueError("diagnostic matmul scales must be 1 or 64")
    c = 32
    h, d = m.HV, m.DK
    em = m.emit
    p = prefix

    def op(name, shape, expression):
        name = p + name
        attrs = f'name=string("{name}")'
        if expression.endswith("]"):
            expression = expression[:-1] + ", " + attrs + "]"
        else:
            expression += "[" + attrs + "]"
        em(f'tensor<fp16, [{", ".join(map(str,shape))}]> {name} = {expression};')
        return name

    def binary(name, kind, x, y, rows, cols):
        return op(name, (1, h, rows, cols), f'{kind}(x={x}, y={y})')

    def mm(name, x, y, rows, cols, tx=False, ty=False):
        return op(name, (1, h, rows, cols),
                  f'matmul(x={x}, y={y}, transpose_x=bool({str(tx).lower()}), '
                  f'transpose_y=bool({str(ty).lower()}))')

    def cat(name, vals, rows, cols):
        if len(vals) == 1:
            value = vals[0]
        else:
            value = op(name + 'live', (1, h, live, cols),
                       f'concat(values=({", ".join(vals)}), axis=int32(2), '
                       f'interleave=bool(false))')
        if live == rows:
            return value
        fill = 1 if name == 'cg' else 0
        return op(name, (1, h, rows, cols),
                  f'pad(x={value}, pad=tensor<int32, [8]>([0,0,0,0,0,{rows-live},0,0]), '
                  f'mode=string("constant"), constant_val=fp16({fill}))')

    def shift(name, src, n, rows, cols):
        m.sl4(p + name + 'cut', src, (0, 0, 0, 0),
              (1, h, rows - n, cols), (1, h, rows - n, cols))
        return op(name, (1, h, rows, cols),
                  f'pad(x={p + name}cut, pad=tensor<int32, [8]>([0,0,0,0,{n},0,0,0]), '
                  f'mode=string("constant"), constant_val=fp16(1))')

    slots = list(range(start, start + live))
    for t in slots:
        m.gdn_prepare(t)
        m.sl4(f'cg{t}', f'g{t}dec', (0, 0, 0, 0), (1, h, 1, 1), (1, h, 1, 1))
    q = cat('cq', [f'g{t}qn48' for t in slots], c, d)
    q = binary('cqscale', 'mul', q, f'fp16({q_scale})', c, d)
    k = cat('ck', [f'g{t}kn48' for t in slots], c, d)
    v = cat('cv', [f'g{t}vv' for t in slots], c, d)
    beta = cat('cb', [f'g{t}bet' for t in slots], c, 1)
    gates = cat('cg', [f'cg{t}' for t in slots], c, 1)
    for name in ['lower', 'strict', 'eye']:
        op('c' + name, (1, 1, c, c),
           f'const()[val=tensor<fp16, [1, 1, {c}, {c}]>(BLOBFILE('
           f'path=string("@model_path/weights/weight_scale.bin"), '
           f'offset=uint64({offs["chunk_" + name]})))]')
    gm = binary('cgm', 'mul', gates, p + 'cstrict', c, c)
    upper = op('cupper', (1, 1, c, c), f'sub(x=fp16(1), y={p}cstrict)')
    decay = binary('cd0', 'add', gm, upper, c, c)
    gamma = gates
    for stage in range((c - 1).bit_length()):
        n = 1 << stage
        sh = shift(f'cds{stage}', decay, n, c, c)
        decay = binary(f'cd{stage + 1}', 'mul', decay, sh, c, c)
        sh = shift(f'cgs{stage}', gamma, n, c, 1)
        gamma = binary(f'cgp{stage}', 'mul', gamma, sh, c, 1)
    decay = binary('cdecay', 'mul', decay, p + 'clower', c, c)
    gram = mm('cgram', k, k, c, c, ty=True)
    a = binary('ca0', 'mul', gram, decay, c, c)
    a = binary('ca1', 'mul', a, beta, c, c)
    a = binary('ca', 'mul', a, p + 'cstrict', c, c)
    if os.environ.get("MIL_GDN_CHUNK_INVERSE", "blocked") == "series":
        power = binary('cp0', 'mul', a, 'nho', c, c)
        inv = binary('ci0', 'add', p + 'ceye', power, c, c)
        for stage in range(1, (c - 1).bit_length()):
            neg = binary(f'cpn{stage}', 'mul', power, 'nho', c, c)
            square = mm(f'cps{stage}', power, neg, c, c)
            power = binary(f'cp{stage}', 'mul', square, 'nho', c, c)
            term = mm(f'cit{stage}', power, inv, c, c)
            inv = binary(f'ci{stage}', 'add', inv, term, c, c)
    else:
        for stage in range(5):
            op(f'cblock{stage}', (1, 1, c, c),
               f'const()[val=tensor<fp16, [1, 1, {c}, {c}]>(BLOBFILE('
               f'path=string("@model_path/weights/weight_scale.bin"), '
               f'offset=uint64({offs["chunk_block" + str(stage)]})))]')
            cross = binary(f'ccross{stage}', 'mul', a, p + f'cblock{stage}', c, c)
            if stage == 0:
                inv = binary('ci0', 'sub', p + 'ceye', cross, c, c)
            else:
                left = mm(f'cileft{stage}', inv, cross, c, c)
                correction = mm(f'cicorr{stage}', left, inv, c, c)
                inv = binary(f'ci{stage}', 'sub', inv, correction, c, c)
    ks = mm('cks', k, state_in, c, d, ty=True)
    ks = binary('cksg', 'mul', ks, gamma, c, d)
    rhs = binary('crhs0', 'sub', v, ks, c, d)
    rhs = binary('crhs', 'mul', rhs, beta, c, d)
    delta = mm('cdelta', inv, rhs, c, d)
    qk = mm('cqk', q, k, c, c, ty=True)
    qk = binary('cqkd', 'mul', qk, decay, c, c)
    ylocal = mm('cylocal', qk, delta, c, d)
    yinitial = mm('cyinitial', q, state_in, c, d, ty=True)
    yinitial = binary('cyinitialg', 'mul', yinitial, gamma, c, d)
    y = binary('cyscale', 'add', yinitial, ylocal, c, d)
    y = binary('cy', 'mul', y, f'fp16({1 / q_scale})', c, d)
    m.sl4(p + 'cend', decay, (0, 0, c - 1, 0), (1, h, c, c), (1, h, 1, c))
    end = op('cendt', (1, h, c, 1), f'transpose(x={p}cend, perm=pm)')
    kend = binary('ckend', 'mul', k, end, c, d)
    update_delta = delta
    if update_scale != 1:
        update_delta = binary('cupdate_scaled', 'mul', delta,
                              f'fp16({update_scale})', c, d)
    update = mm('cupdate', update_delta, kend, d, d, tx=True)
    if update_scale != 1:
        update = binary('cupdate_unscaled', 'mul', update,
                        f'fp16({1 / update_scale})', d, d)
    m.sl4(p + 'cgend', gamma, (0, 0, c - 1, 0), (1, h, c, 1), (1, h, 1, 1))
    old = binary('cold', 'mul', state_in, p + 'cgend', d, d)
    state_out = f'q_state{start + live - 1:02d}'
    em(f'tensor<fp16, [1, {h}, {d}, {d}]> {state_out} = '
       f'add(x={old}, y={update})[name=string("{state_out}")];')
    for local, t in enumerate(slots):
        m.sl4(f'g{t}yt', y, (0, 0, local, 0), (1, h, local + 1, d), (1, h, 1, d))
        m.gdn_finish(t)
    return state_out

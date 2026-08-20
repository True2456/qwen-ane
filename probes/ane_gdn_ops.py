"""Which ops the GDN recurrence needs, and whether the ANE has them.

Step (per layer, per token), state [H=48, Dv=128, Dk=128]:
    state *= decay
    kv_mem = (state * k).sum(-1)        reduction over Dk
    delta  = (v - kv_mem) * beta
    state += k * delta                  rank-1 outer product
    y      = (state * q).sum(-1)        reduction over Dk
Everything is elementwise except the two reductions and the outer product.
"""
import os, sys, io, contextlib, numpy as np
sys.path.insert(0, os.path.expanduser("~/AppleLLM/q38_native_engine"))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine
eng = AneEngine()

def build(name, body, C, S, out_c, extra_in=None, blobs=None):
    ins = f"tensor<fp16, [1, {C}, 1, {S}]> x"
    if extra_in:
        ins += f", tensor<fp16, [1, {extra_in[0]}, 1, {extra_in[1]}]> w"
    mil = f"""program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>({ins}) {{
    string pt = const()[name=string("pt"), val=string("valid")];
    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
    int32 gr = const()[name=string("gr"), val=int32(1)];
{body}
  }} -> (y);
}}
"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        prog = eng.compile_multiproc(mil, blobs or {}, C, out_c, S)
    print(f"  {name:44} {'OK' if prog else 'COMPILE FAILED'}")
    return prog

Dk, HV = 128, 6144            # state laid out [1, Dk, 1, H*Dv]
ones = {"on.bin": np.full((1, Dk), 1.0, np.float16).tobytes()}
OND = (f'    tensor<fp16, [1, {Dk}, 1, 1]> onw = const()[name=string("onw"), '
       f'val=tensor<fp16, [1, {Dk}, 1, 1]>(BLOBFILE(path=string("@model_path/weights/on.bin"), offset=uint64(64)))];')

# 1. reduction over Dk with state laid out on the channel axis, wide S
build("reduce over Dk via ones-conv (S=6144)", OND + f'''
    tensor<fp16, [1, 1, 1, {HV}]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=onw, x=x)[name=string("y")];''',
      Dk, HV, 1, blobs=dict(ones))

# 2. elementwise multiply of two DYNAMIC tensors (state * k)
build("mul of two dynamic inputs", f'''
    tensor<fp16, [1, {Dk}, 1, {HV}]> y = mul(x=x, y=w)[name=string("y")];''',
      Dk, HV, Dk, extra_in=(Dk, HV))

# 3. matmul of two dynamic tensors (needed if reductions become matmuls)
build("matmul dynamic x dynamic", f'''
    tensor<fp16, [1, 1, {HV}, {Dk}]> xt = transpose(perm=tensor<int32, [4]>([0,2,3,1]), x=x)[name=string("xt")];
    tensor<fp16, [1, 1, {HV}, {Dk}]> wt = transpose(perm=tensor<int32, [4]>([0,2,3,1]), x=w)[name=string("wt")];
    tensor<fp16, [1, 1, {HV}, {HV}]> y = matmul(transpose_x=bool(false), transpose_y=bool(true), x=xt, y=wt)[name=string("y")];''',
      Dk, HV, HV, extra_in=(Dk, HV))

# 4. transpose alone
build("transpose 4d", f'''
    tensor<fp16, [1, {HV}, 1, {Dk}]> y = transpose(perm=tensor<int32, [4]>([0,3,2,1]), x=x)[name=string("y")];''',
      Dk, HV, HV)

# 5. reduce_sum over the channel axis (the natural spelling)
build("reduce_sum axes=[1]", f'''
    tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([1])];
    tensor<fp16, [1, 1, 1, {HV}]> y = reduce_sum(axes=ax, keep_dims=bool(true), x=x)[name=string("y")];''',
      Dk, HV, 1)

# 6. exp (needed for the decay gate g = -exp(A_log) * softplus(...))
build("exp", f'''
    tensor<fp16, [1, {Dk}, 1, {HV}]> y = exp(x=x)[name=string("y")];''', Dk, HV, Dk)

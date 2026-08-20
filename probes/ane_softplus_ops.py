"""Isolate native and approximate softplus formulations on the ANE."""
import contextlib, io, os, sys, time
import numpy as np

sys.path.insert(0, os.path.expanduser("~/AppleLLM/q38_native_engine"))
import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _iosurface_view

H, S = 48, 32
eng = AneEngine()
rng = np.random.default_rng(123)
x = np.linspace(-20, 20, H*S, dtype=np.float32).reshape(H,S)


def build(name, body, blobs=None, ref=None):
    mil=f'''program(1.3)
{E._BUILD_INFO}
{{
  func main<ios18>(tensor<fp16,[1,{H},1,{S}]> x) {{
{body}
  }} -> (y);
}}
// softplus_isolate_{name}
'''
    cap=io.StringIO()
    with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap):
        p=eng.compile_multiproc(mil,blobs or {},H,H,S)
    if p is None:
        print(f"  {name:<26} COMPILE FAILED")
        return None
    eng._ensure_io(p)
    with _iosurface_view(p._in_surf,(H,S),np.float16) as z: z[:]=x
    for _ in range(3):
        if not eng.submit(p):
            print(f"  {name:<26} SUBMIT FAILED"); return None
    t=time.perf_counter(); n=50
    for _ in range(n): eng.submit(p)
    ms=(time.perf_counter()-t)*1e3/n
    with _iosurface_view(p._out_surf,(H,S),np.float16) as z: got=np.array(z,np.float32)
    if ref is None:
        print(f"  {name:<26} COMPILES {ms:.3f} ms")
    else:
        want=ref(x)
        ae=np.max(np.abs(got-want)); re=ae/(np.max(np.abs(want))+1e-9)
        verdict="OK" if ae < 2e-2 else "WRONG"
        print(f"  {name:<26} {verdict} abs={ae:.4g} rel={re:.4g} "
              f"range=[{got.min():.4g},{got.max():.4g}] {ms:.3f} ms")
    return p


soft=lambda a: np.logaddexp(0,a)
print("isolated MIL operators")
build("softplus", f'    tensor<fp16,[1,{H},1,{S}]> y = softplus(x=x)[name=string("y")];', ref=soft)
build("log explicit epsilon", f'''    tensor<fp16,[1,{H},1,{S}]> ax = abs(x=x)[name=string("ax")];
    tensor<fp16,[1,{H},1,{S}]> po = add(x=ax,y=fp16(0x1p+0))[name=string("po")];
    tensor<fp16,[1,{H},1,{S}]> y = log(epsilon=fp16(0x0p+0),x=po)[name=string("y")];''',
      ref=lambda a:np.log(np.abs(a)+1))
build("pow", f'''    tensor<fp16,[1,{H},1,{S}]> ax = abs(x=x)[name=string("ax")];
    tensor<fp16,[1,{H},1,{S}]> po = add(x=ax,y=fp16(0x1p+0))[name=string("po")];
    tensor<fp16,[1,{H},1,{S}]> y = pow(x=po,y=fp16(0x1p-1))[name=string("y")];''',
      ref=lambda a:np.sqrt(np.abs(a)+1))
ones=np.ones(H,np.float16)
build("softplus_parametric", f'''    tensor<fp16,[{H}]> al = const()[name=string("al"),val=tensor<fp16,[{H}]>(BLOBFILE(path=string("@model_path/weights/a.bin"),offset=uint64(64)))];
    tensor<fp16,[{H}]> be = const()[name=string("be"),val=tensor<fp16,[{H}]>(BLOBFILE(path=string("@model_path/weights/b.bin"),offset=uint64(64)))];
    tensor<fp16,[1,{H},1,{S}]> y = softplus_parametric(alpha=al,beta=be,x=x)[name=string("y")];''',
      {"a.bin":ones.tobytes(),"b.bin":ones.tobytes()},ref=soft)
build("relu",f'    tensor<fp16,[1,{H},1,{S}]> y = relu(x=x)[name=string("y")];',ref=lambda a:np.maximum(a,0))
build("abs",f'    tensor<fp16,[1,{H},1,{S}]> y = abs(x=x)[name=string("y")];',ref=np.abs)
build("clip",f'    tensor<fp16,[1,{H},1,{S}]> y = clip(alpha=fp16(-0x1p+3),beta=fp16(0x1p+3),x=x)[name=string("y")];',ref=lambda a:np.clip(a,-8,8))


# softplus(x) = relu(x) + r(min(abs(x), 8)), where
# r(t)=log(1+exp(-t)).  Degree-12 Chebyshev least-squares polynomial, converted
# to the power basis in z=t/4-1.  Only relu/add/mul are needed.
grid=np.linspace(0,8,200001)
coef=np.polynomial.chebyshev.cheb2poly(np.polynomial.chebyshev.chebfit(
    grid/4-1,np.log1p(np.exp(-grid)),12)).astype(np.float16)
lines=[f'    tensor<fp16,[1,{H},1,{S}]> nx = mul(x=x,y=fp16(-0x1p+0))[name=string("nx")];',
       f'    tensor<fp16,[1,{H},1,{S}]> pos = relu(x=x)[name=string("pos")];',
       f'    tensor<fp16,[1,{H},1,{S}]> neg = relu(x=nx)[name=string("neg")];',
       f'    tensor<fp16,[1,{H},1,{S}]> ab = add(x=pos,y=neg)[name=string("ab")];',
       f'    tensor<fp16,[1,{H},1,{S}]> over = sub(x=ab,y=fp16(0x1p+3))[name=string("over")];',
       f'    tensor<fp16,[1,{H},1,{S}]> excess = relu(x=over)[name=string("excess")];',
       f'    tensor<fp16,[1,{H},1,{S}]> cl = sub(x=ab,y=excess)[name=string("cl")];',
       f'    tensor<fp16,[1,{H},1,{S}]> quarter = mul(x=cl,y=fp16(0x1p-2))[name=string("quarter")];',
       f'    tensor<fp16,[1,{H},1,{S}]> z = sub(x=quarter,y=fp16(0x1p+0))[name=string("z")];']
def hx(v):
    # Python's hexadecimal spelling is accepted by MIL for finite fp16 values.
    return np.float16(v).astype(float).hex()
lines.append(f'    tensor<fp16,[1,{H},1,{S}]> p12 = add(x=z,y=fp16({hx(coef[-1])}))[name=string("p12")];')
# p12 above is intentionally z+c, so start Horner correctly with c*z+c next
# by replacing it with a scalar multiplication on the first loop.
lines[-1]=f'    tensor<fp16,[1,{H},1,{S}]> p12 = mul(x=z,y=fp16({hx(coef[-1])}))[name=string("p12")];'
prev="p12"
for j,c in zip(range(11,-1,-1),coef[-2::-1]):
    nm=f"p{j}"
    lines.append(f'    tensor<fp16,[1,{H},1,{S}]> {nm} = add(x={prev},y=fp16({hx(c)}))[name=string("{nm}")];')
    if j:
        mul=f"m{j}"
        lines.append(f'    tensor<fp16,[1,{H},1,{S}]> {mul} = mul(x={nm},y=z)[name=string("{mul}")];')
        prev=mul
    else:
        prev=nm
lines.append(f'    tensor<fp16,[1,{H},1,{S}]> y = add(x=pos,y={prev})[name=string("y")];')
print("softplus approximation")
build("relu+poly12", "\n".join(lines), ref=soft)

#!/usr/bin/env python3
"""ane_artifact_inspector.py - Inspect compiled ANE binary artifact and directory structure."""

import ctypes
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import runtime.q38_ane_engine as E
from runtime.q38_ane_engine import AneEngine, _BUILD_INFO, _objc, _sel, _desc

eng = AneEngine()

# Compile a simple 32x32 program
C = 32
S = 32
mil = f"""program(1.3)
{_BUILD_INFO}
{{
  func main<ios18>(tensor<fp16, [1, {C}, 1, {S}]> x) {{
    tensor<fp16, [1, {C}, 1, {S}]> y = identity(x=x)[name=string("y")];
  }} -> (y);
}}
"""

prog = eng.compile_multiproc(mil, {}, C, C, S)
if not prog:
    print("Failed to compile program")
    sys.exit(1)

model = prog.model

# Inspect properties
def get_nsstring(sel_name):
    f = _objc.objc_msgSend
    f.restype = ctypes.c_void_p
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    res = f(model, _sel(sel_name))
    if not res:
        return None
    return _desc(res)

print(f"HexStringIdentifier: {get_nsstring('hexStringIdentifier')}")
print(f"LocalModelPath:      {get_nsstring('localModelPath')}")

local_path = get_nsstring('localModelPath')
if local_path and os.path.exists(local_path):
    print(f"\nFiles in {local_path}:")
    for root, dirs, files in os.walk(local_path):
        for f in files:
            p = os.path.join(root, f)
            print(f"  {p} ({os.path.getsize(p)} bytes)")

# Check compiled artifact cache directory
cache_id = get_nsstring('hexStringIdentifier')
print(f"\nSearching /var/folders for compiled ANE artifacts matching {cache_id}...")
for root, dirs, files in os.walk("/var/folders"):
    if cache_id and cache_id in root:
        print(f"  Found cache directory: {root}")
        for f in files:
            p = os.path.join(root, f)
            print(f"    {f} ({os.path.getsize(p)} bytes)")
        break

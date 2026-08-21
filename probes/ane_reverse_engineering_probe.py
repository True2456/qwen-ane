#!/usr/bin/env python3
"""ane_reverse_engineering_probe.py - Dissect AppleNeuralEngine.framework Objective-C internals."""

import ctypes
import os
import sys

_objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
_ane = ctypes.cdll.LoadLibrary("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine")

# Configure objc introspection
_objc.objc_getClassList.restype = ctypes.c_int
_objc.objc_getClassList.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]

_objc.class_getName.restype = ctypes.c_char_p
_objc.class_getName.argtypes = [ctypes.c_void_p]

_objc.class_copyMethodList.restype = ctypes.POINTER(ctypes.c_void_p)
_objc.class_copyMethodList.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]

_objc.class_copyIvarList.restype = ctypes.POINTER(ctypes.c_void_p)
_objc.class_copyIvarList.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]

_objc.method_getName.restype = ctypes.c_void_p
_objc.method_getName.argtypes = [ctypes.c_void_p]

_objc.sel_getName.restype = ctypes.c_char_p
_objc.sel_getName.argtypes = [ctypes.c_void_p]

_objc.ivar_getName.restype = ctypes.c_char_p
_objc.ivar_getName.argtypes = [ctypes.c_void_p]

_objc.ivar_getTypeEncoding.restype = ctypes.c_char_p
_objc.ivar_getTypeEncoding.argtypes = [ctypes.c_void_p]


def get_all_classes():
    num_classes = _objc.objc_getClassList(None, 0)
    buf = (ctypes.c_void_p * num_classes)()
    _objc.objc_getClassList(buf, num_classes)
    ane_classes = []
    for cls in buf:
        name = _objc.class_getName(cls).decode("utf-8")
        if name.startswith("_ANE") or "ANE" in name or "AppleNeuralEngine" in name:
            ane_classes.append((name, cls))
    return sorted(ane_classes, key=lambda x: x[0])


def inspect_class(name, cls):
    print(f"\n==================== Class: {name} ====================")
    # Ivars
    out_count = ctypes.c_uint(0)
    ivars = _objc.class_copyIvarList(cls, ctypes.byref(out_count))
    if ivars and out_count.value > 0:
        print("  --- Instance Variables (ivars): ---")
        for i in range(out_count.value):
            ivar = ivars[i]
            ivar_name = _objc.ivar_getName(ivar).decode("utf-8")
            ivar_type = _objc.ivar_getTypeEncoding(ivar).decode("utf-8")
            print(f"    {ivar_name}: {ivar_type}")
    
    # Methods
    out_count = ctypes.c_uint(0)
    methods = _objc.class_copyMethodList(cls, ctypes.byref(out_count))
    if methods and out_count.value > 0:
        print("  --- Methods: ---")
        method_names = []
        for i in range(out_count.value):
            m = methods[i]
            sel = _objc.method_getName(m)
            sel_name = _objc.sel_getName(sel).decode("utf-8")
            method_names.append(sel_name)
        for mname in sorted(method_names):
            print(f"    - {mname}")


def main():
    print("Enumerating AppleNeuralEngine.framework classes and methods...")
    classes = get_all_classes()
    print(f"Found {len(classes)} ANE-related classes.")
    
    # Target high-interest classes
    targets = [
        "_ANEInMemoryModel",
        "_ANEInMemoryModelDescriptor",
        "_ANEModel",
        "_ANEClient",
        "_ANEDeviceController",
        "_ANEDeviceInfo",
        "_ANERequest",
        "_ANECompiler",
        "_ANECloneHelper",
        "_ANEProgramForEvaluation",
    ]
    
    for name, cls in classes:
        if name in targets:
            inspect_class(name, cls)


if __name__ == "__main__":
    main()

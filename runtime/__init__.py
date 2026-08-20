"""
`runtime/` Python package marker.

`q38_ane_engine` is vendored from the q38_native_engine repository
(runtime/q38_ane_engine.py, upstream commit 9122ce1). It is the private
`AppleNeuralEngine.framework` driver: ctypes/objc bindings, MIL program
compilation through `ANECCompile`, `_ANEInMemoryModel` loading, and IOSurface
allocation and binding. It depends only on the standard library and numpy.

Set `Q38_ANE_ENGINE` to another checkout to override this copy.
"""

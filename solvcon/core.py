# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
General mesh data definition and manipulation in one, two, and
three-dimensional space.
"""


# Use flake8 http://flake8.pycqa.org/en/latest/user/error-codes.html

import os

from . import pylibmgr

# buffer directory symbols
list_of_buffer = [
    'ConcreteBuffer',
    'BufferExpander',
    'SimpleArray',
    'SimpleArrayBool',
    'SimpleArrayInt8',
    'SimpleArrayInt16',
    'SimpleArrayInt32',
    'SimpleArrayInt64',
    'SimpleArrayUint8',
    'SimpleArrayUint16',
    'SimpleArrayUint32',
    'SimpleArrayUint64',
    'SimpleArrayFloat32',
    'SimpleArrayFloat64',
    'SimpleArrayComplex64',
    'SimpleArrayComplex128',
    'SimpleCollectorBool',
    'SimpleCollectorInt8',
    'SimpleCollectorInt16',
    'SimpleCollectorInt32',
    'SimpleCollectorInt64',
    'SimpleCollectorUint8',
    'SimpleCollectorUint16',
    'SimpleCollectorUint32',
    'SimpleCollectorUint64',
    'SimpleCollectorFloat32',
    'SimpleCollectorFloat64',
]

# inout directory symbols
list_of_inout = [
    'Gmsh',
    'Plot3d',
]

# math directory symbols
list_of_math = [
    'complex64',
    'complex128',
]

# mesh directory symbols
list_of_mesh = [
    'StaticGrid1d',
    'StaticGrid2d',
    'StaticGrid3d',
    'StaticMesh',
]

# multidim directory symbols
list_of_multidim = [
    'EulerCore',
    'GradientElement',
]

# python directory symbols
list_of_python = [
    'CommandLineInfo',
    'ProcessInfo',
    'HAS_PILOT',
]

# toggle directory symbols
list_of_toggle = [
    'WrapperProfilerStatus',
    'wrapper_profiler_status',
    'StopWatch',
    'stop_watch',
    'CallProfiler',
    'call_profiler',
    'CallProfilerProbe',
    'HierarchicalToggleAccess',
    'Toggle',
    'ToggleCategory',
    'ToggleSubscription',
    'METAL_BUILT',
    'metal_running',
]

# transform directory symbols
list_of_transform = [
    'FourierTransform',
]

# linalg directory symbols
list_of_linalg = [
    'llt_factorization',
    'llt_solve',
    'lu_factorization',
    'lu_solve',
    'lu_inv',
    'lu_det',
    'LuFactorizationFloat32',
    'LuFactorizationFloat64',
    'LuFactorizationComplex64',
    'LuFactorizationComplex128',
    'EigenSystem',
    'EigenSystemFloat32',
    'EigenSystemFloat64',
    'EigenSystemComplex64',
    'EigenSystemComplex128',
    'KalmanStateInfoFp32',
    'KalmanStateInfoFp64',
    'KalmanStateInfoComplex64',
    'KalmanStateInfoComplex128',
    'KalmanFilterFp32',
    'KalmanFilterFp64',
    'KalmanFilterComplex64',
    'KalmanFilterComplex128',
]

# universe directory symbols
list_of_universe = [
    'calc_bernstein_polynomial',
    'interpolate_bernstein',
    'BoundBox3dFp32',
    'BoundBox3dFp64',
    'Point3dFp32',
    'Point3dFp64',
    'Segment3dFp32',
    'Segment3dFp64',
    'Triangle3dFp32',
    'Triangle3dFp64',
    'Bezier3dFp32',
    'Bezier3dFp64',
    'PointPadFp32',
    'PointPadFp64',
    'SegmentPadFp32',
    'SegmentPadFp64',
    'TrianglePadFp32',
    'TrianglePadFp64',
    'CurvePadFp32',
    'CurvePadFp64',
    'WorldFp32',
    'WorldFp64',
    'PolygonPadFp32',
    'PolygonPadFp64',
    'Polygon3dFp32',
    'Polygon3dFp64',
    'TrapezoidPadFp32',
    'TrapezoidPadFp64',
    'TrapezoidalDecomposerFp32',
    'TrapezoidalDecomposerFp64',
    'ViewTransform2dFp64',
]

# OASIS directory symbols
list_of_oasis = [
    'OasisDevice',
    'OasisRecordRect',
    'OasisRecordPoly',
]

__all__ = (  # noqa: F822
    list_of_buffer +
    list_of_inout +
    list_of_math +
    list_of_mesh +
    list_of_multidim +
    list_of_python +
    list_of_toggle +
    list_of_transform +
    list_of_linalg +
    list_of_universe +
    list_of_oasis
)


# A hidden loophole to impolementation; it should only be used for testing
# during development.
try:
    import _solvcon as _impl  # noqa: F401
except ImportError:
    from . import _solvcon as _impl  # noqa: F401


def _load(symbol_list):
    for name in symbol_list:
        globals()[name] = getattr(_impl, name)


_load(list_of_buffer)
_load(list_of_inout)
_load(list_of_math)
_load(list_of_mesh)
_load(list_of_multidim)
_load(list_of_python)
_load(list_of_toggle)
_load(list_of_transform)
_load(list_of_linalg)
_load(list_of_universe)
_load(list_of_oasis)

# Walk through the thirdparty folder and register all library
# into a dictionary.
pylibmgr.search_library_root(os.getcwd(), 'thirdparty')

del _load

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

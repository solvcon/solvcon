# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Acceptance checks that the vendor BLAS/LAPACK backend computes correctly.

SimpleArray.matmul_blas() calls CBLAS ?gemm and EigenSystem calls LAPACK
?geev.  These cross-check both against NumPy's own LAPACK rather than against
another solvcon path, so a miscompiled or ABI-mismatched vendor library (a bad
OpenBLAS or MKL link) surfaces as wrong numbers instead of passing
consistently-wrong values.
"""

import unittest

import numpy as np

import solvcon as sc


# name -> (SimpleArray class, numpy dtype, tolerance)
_DTYPES = {
    "float32": (sc.SimpleArrayFloat32, np.float32, 1e-4),
    "float64": (sc.SimpleArrayFloat64, np.float64, 1e-12),
    "complex64": (sc.SimpleArrayComplex64, np.complex64, 1e-4),
    "complex128": (sc.SimpleArrayComplex128, np.complex128, 1e-12),
}


class MatmulBlasTC(unittest.TestCase):
    """SimpleArray.matmul_blas() (CBLAS ?gemm) must match NumPy."""

    def _operands(self, dtype):
        rng = np.random.default_rng(20260714)
        a = rng.standard_normal((5, 4))
        b = rng.standard_normal((4, 6))
        if np.issubdtype(dtype, np.complexfloating):
            a = a + 1j * rng.standard_normal((5, 4))
            b = b + 1j * rng.standard_normal((4, 6))
        return (np.ascontiguousarray(a, dtype=dtype),
                np.ascontiguousarray(b, dtype=dtype))

    def test_matches_numpy(self):
        for name, (arr_cls, dtype, tol) in _DTYPES.items():
            with self.subTest(dtype=name):
                a_np, b_np = self._operands(dtype)
                got = arr_cls(array=a_np).matmul_blas(
                    arr_cls(array=b_np)).ndarray
                want = a_np @ b_np
                self.assertEqual(got.shape, want.shape)
                np.testing.assert_allclose(got, want, rtol=tol, atol=tol)


@unittest.skipIf(sc.EigenSystem is None,
                 "sc.EigenSystem is not built (no vendor LAPACK)")
class EigenSystemLapackTC(unittest.TestCase):
    """EigenSystem (LAPACK ?geev) must match NumPy's eigensolver."""

    # Nonsymmetric matrix with spectrum {2, +i, -i}: exercises a real
    # eigenvalue and a complex-conjugate pair in one shot.
    _A = np.array([[0.0, -1.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 0.0, 2.0]])

    def _solve(self, a_np):
        solver = sc.EigenSystem(sc.SimpleArray(a_np))
        solver.run()
        w = np.asarray(solver.wr) + 1j * np.asarray(solver.wi)
        return solver, w

    def test_eigenvalues_match_numpy(self):
        for name, (_, dtype, tol) in _DTYPES.items():
            with self.subTest(dtype=name):
                a_np = np.ascontiguousarray(self._A, dtype=dtype)
                _, w = self._solve(a_np)
                np.testing.assert_allclose(
                    np.sort_complex(w),
                    np.sort_complex(np.linalg.eigvals(a_np)),
                    rtol=tol, atol=tol)

    def test_complex_eigenvectors_reconstruct(self):
        # For complex element types vr is a plain complex matrix, so the
        # defining relation A @ vr == vr @ diag(w) can be checked directly.
        # Real types pack a conjugate pair across two columns of vr, so that
        # relation does not hold column-wise and is left to the eigenvalue
        # check above.
        for name in ("complex64", "complex128"):
            _, dtype, tol = _DTYPES[name]
            with self.subTest(dtype=name):
                a_np = np.ascontiguousarray(self._A, dtype=dtype)
                solver, w = self._solve(a_np)
                vr = np.asarray(solver.vr)
                np.testing.assert_allclose(
                    a_np @ vr, vr @ np.diag(w), rtol=tol, atol=tol)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

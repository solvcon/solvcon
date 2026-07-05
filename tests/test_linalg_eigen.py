import unittest

import numpy as np

import solvcon as sc


@unittest.skipIf(sc.EigenSystem is None,
                 "sc.EigenSystem is not built (no vendor LAPACK)")
class TestEigenSystemPlexTC(unittest.TestCase):
    """Verify the type-erased EigenSystem surrogate (C++ EigenSystemPlex).

    sc.EigenSystem accepts a SimpleArrayPlex and dispatches on its runtime
    element type to the matching typed EigenSystem<T>.  Eigenvalues are
    exposed as real/imaginary parts via wr/wi for every element type.
    """

    REAL = ("float32", "float64")
    COMPLEX = ("complex64", "complex128")
    ALL = REAL + COMPLEX

    def _typed(self, arr, dtype):
        # The matching typed (non-plex) solver, used as the oracle.
        table = {
            "float32": (sc.EigenSystemFloat32, sc.SimpleArrayFloat32),
            "float64": (sc.EigenSystemFloat64, sc.SimpleArrayFloat64),
            "complex64": (sc.EigenSystemComplex64, sc.SimpleArrayComplex64),
            "complex128": (sc.EigenSystemComplex128, sc.SimpleArrayComplex128),
        }
        eig_cls, arr_cls = table[dtype]
        return eig_cls(arr_cls(array=arr))

    def test_dispatch_matches_typed_construction(self):
        # For every supported dtype, the plex path must reproduce the typed
        # EigenSystem<T> path bit-for-bit: same dispatch, same GEEV call.
        A_np = np.array([
            [2.0, 1.0, 0.5],
            [0.0, 3.0, -1.0],
            [0.0, 0.0, 5.0],
        ])
        for dtype in self.ALL:
            with self.subTest(dtype=dtype):
                arr = np.ascontiguousarray(A_np, dtype=dtype)
                solver = sc.EigenSystem(sc.SimpleArray(arr))
                self.assertFalse(solver.done)
                solver.run()
                self.assertTrue(solver.done)
                typed = self._typed(arr, dtype)
                typed.run()
                np.testing.assert_array_equal(
                    np.array(solver.wr), np.array(typed.wr))
                np.testing.assert_array_equal(
                    np.array(solver.wi), np.array(typed.wi))
                np.testing.assert_array_equal(
                    np.array(solver.vl), np.array(typed.vl))
                np.testing.assert_array_equal(
                    np.array(solver.vr), np.array(typed.vr))

    def test_rejects_unsupported_dtype(self):
        # Integer element types have no GEEV; they must raise ValueError
        # (std::invalid_argument) rather than dispatch.
        for dtype in ("int32", "int64", "uint8"):
            with self.subTest(dtype=dtype):
                A = sc.SimpleArray(np.eye(2, dtype=dtype))
                with self.assertRaisesRegex(
                        ValueError, r"data type must be"):
                    sc.EigenSystem(A)

    def test_non_square_rejected(self):
        # A supported dtype dispatches, then hits the square-2D guard inside
        # the typed EigenSystem constructor.
        A = sc.SimpleArray(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        with self.assertRaisesRegex(
                ValueError, r"must be a square 2D SimpleArray"):
            sc.EigenSystem(A)

    def test_forwards_do_vl_do_vr_flags(self):
        # do_vl/do_vr pass through to the dispatched solver unchanged.
        A_np = np.array([[2.0, 0.0], [0.0, 3.0]])
        solver = sc.EigenSystem(sc.SimpleArray(A_np))
        self.assertTrue(solver.do_vl)
        self.assertTrue(solver.do_vr)

        solver = sc.EigenSystem(sc.SimpleArray(A_np), do_vr=False)
        solver.run()
        self.assertTrue(solver.do_vl)
        self.assertFalse(solver.do_vr)
        with self.assertRaisesRegex(
                RuntimeError, r"right eigenvectors were not computed"):
            solver.vr  # noqa: B018

    def test_matrix_property_survives_input_gc(self):
        # keep_alive<1, 2>() must keep the plex (and the typed array it owns)
        # alive after the Python-side plex reference is dropped.
        A_np = np.array([[3.0, 1.0], [0.0, 2.0]])
        plex = sc.SimpleArray(A_np)
        solver = sc.EigenSystem(plex)
        del plex
        np.testing.assert_array_equal(np.array(solver.matrix), A_np)


class EigenSystemTB:
    """Verify EigenSystem<T> against *GEEV reference outputs.

    Subclasses bind a concrete element type by setting ``array_cls``,
    ``eig_cls``, ``np_dtype``, ``is_complex`` and the comparison tolerances.
    Real solvers (SGEEV/DGEEV) report eigenvalues as a real/imaginary split
    (wr/wi) and pack a complex-conjugate eigenvector pair into two consecutive
    real columns; complex solvers (CGEEV/ZGEEV) report a single complex w and
    store eigenvectors directly as complex columns.  The helpers below
    normalize both layouts to complex arrays so each test body is
    dtype-agnostic.
    """

    array_cls = None
    eig_cls = None
    np_dtype = None
    is_complex = False
    rtol = 1e-10
    atol = 1e-12

    @classmethod
    def get_complex_type(cls):
        _ = {'float32': 'complex64', 'float64': 'complex128'}
        name = np.dtype(cls.np_dtype).name
        return _.get(name, name)

    def _array(self, A_np):
        return self.array_cls(
            array=np.ascontiguousarray(A_np, dtype=self.np_dtype))

    def _solve(self, A_np, **kwargs):
        solver = self.eig_cls(self._array(A_np), **kwargs)
        self.assertFalse(solver.done)
        solver.run()
        self.assertTrue(solver.done)
        return solver

    def _eigvals(self, solver):
        # Eigenvalues as a complex ndarray: wr/wi for every element type.
        wr = np.asarray(solver.wr, dtype=self.np_dtype)
        wi = np.asarray(solver.wi, dtype=self.np_dtype)
        return wr + 1j * wi

    def _columns_to_complex(self, mat, w):
        # Normalize an eigenvector matrix to complex columns.  For complex
        # element types the columns are already complex; for real types *GEEV
        # packs a conjugate pair (wi[j] > 0, wi[j+1] < 0) as
        # v_j = M[:, j] + i M[:, j+1] and v_{j+1} = M[:, j] - i M[:, j+1].
        mat = np.asarray(mat)
        if self.is_complex:
            return mat.astype(self.np_dtype)
        wi = np.imag(w)
        n = mat.shape[0]
        out = np.zeros((n, n), dtype=self.get_complex_type())
        j = 0
        while j < n:
            if wi[j] == 0.0:
                out[:, j] = mat[:, j]
                j += 1
            else:
                out[:, j] = mat[:, j] + 1j * mat[:, j + 1]
                out[:, j + 1] = mat[:, j] - 1j * mat[:, j + 1]
                j += 2
        return out

    def _assert_eigvals(self, got, expected):
        # Compare eigenvalues as a multiset: order is unspecified and a sort
        # key on the real/imag parts is fragile near ties (e.g. +/- i, where
        # rounding noise in the zero real part flips the order).  Greedily
        # match each expected value to its nearest computed one instead.
        got = list(np.asarray(got, dtype=self.get_complex_type()).ravel())
        expected = np.asarray(expected, dtype=self.get_complex_type()).ravel()
        self.assertEqual(len(got), expected.size)
        for e in expected:
            diffs = np.abs(np.asarray(got) - e)
            k = int(np.argmin(diffs))
            tol = self.atol + self.rtol * abs(e)
            self.assertLessEqual(
                diffs[k], tol,
                msg=f"no computed eigenvalue near {e} (closest off by "
                    f"{diffs[k]:.3e})")
            got.pop(k)

    def test_diagonal_eigenvalues_and_basis_eigenvectors(self):
        # diag(1, 2, 3, 4): eigenvalues are the diagonal; eigenvectors are
        # standard basis vectors up to permutation/phase.
        A_np = np.diag([1.0, 2.0, 3.0, 4.0])
        solver = self._solve(A_np)
        w = self._eigvals(solver)
        self._assert_eigvals(w, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(np.imag(w), np.zeros(4),
                                   atol=10 * self.atol)
        vr = self._columns_to_complex(solver.vr, w)
        # Each column is a (phased) standard basis vector: one unit-magnitude
        # entry and the rest zero.
        for j in range(4):
            col = vr[:, j]
            np.testing.assert_allclose(np.abs(col).sum(), 1.0,
                                       rtol=self.rtol, atol=10 * self.atol)
            self.assertEqual(np.count_nonzero(np.abs(col) > 1e-4), 1)

    def test_symmetric_known_spectrum_reconstructs_eigenpairs(self):
        # The fixture Q below is frozen (not generated at runtime) so the test
        # is independent of the NumPy RNG / QR implementation.  To regenerate,
        # run:
        #   rng = np.random.default_rng(seed=20260509)
        #   Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
        # and paste Q.tolist() back in.
        n = 5
        lam = np.array([-2.0, -0.5, 1.0, 3.5, 7.0], dtype='float64')
        Q = np.array([
            [-0.024712873175908756, 0.6640676133514902,
             0.5915632257468302, 0.4455230309449579,
             -0.09982814051493692],
            [-0.28147871622325943, -0.5692008616441233,
             0.5886397812806473, 0.1575982019764191,
             0.4748116742926241],
            [-0.9139918340187755, 0.2754225371185904,
             -0.14764410500783806, -0.2527961990356955,
             0.05528700935747367],
            [0.24639362994675912, 0.23505747176988095,
             0.4022089819816873, -0.825230080044468,
             0.2031290572268447],
            [0.15513901083013232, 0.32235848835350017,
             -0.34638895898623234, 0.17821737394071255,
             0.8486873093331871],
        ])
        # Check the fixture is orthogonal first.
        np.testing.assert_allclose(Q @ Q.T, np.eye(n), atol=1e-14)
        A_np = Q @ np.diag(lam) @ Q.T
        solver = self._solve(A_np)
        w = self._eigvals(solver)
        self._assert_eigvals(w, lam)
        np.testing.assert_allclose(np.imag(w), np.zeros(n),
                                   atol=10 * self.atol)
        # Per-column eigenpair: A v_j = lambda_j v_j.
        Ac = A_np.astype(complex)
        vr = self._columns_to_complex(solver.vr, w)
        for j in range(n):
            np.testing.assert_allclose(Ac @ vr[:, j], w[j] * vr[:, j],
                                       rtol=self.rtol, atol=10 * self.atol)

    def test_2x2_rotation_complex_conjugate_pair(self):
        # 2x2 90-degree rotation matrix has eigenvalues +/- i.  For real
        # element types this exercises *GEEV's packing of a complex conjugate
        # eigenvector pair into consecutive real columns; for complex types the
        # eigenvectors come back complex directly.
        A_np = np.array([[0.0, -1.0], [1.0, 0.0]], dtype='float64')
        solver = self._solve(A_np)
        w = self._eigvals(solver)
        self._assert_eigvals(w, [1j, -1j])
        Ac = A_np.astype(complex)
        vr = self._columns_to_complex(solver.vr, w)
        for j in range(2):
            np.testing.assert_allclose(Ac @ vr[:, j], w[j] * vr[:, j],
                                       rtol=self.rtol, atol=10 * self.atol)

    def test_left_eigenvectors_satisfy_left_equation(self):
        # Left eigenvectors u_j satisfy A^H u_j = conj(lambda_j) u_j.  Use a
        # non-symmetric upper-triangular matrix whose spectrum is its diagonal.
        A_np = np.array([
            [2.0, 1.0, 0.5],
            [0.0, 3.0, -1.0],
            [0.0, 0.0, 5.0],
        ], dtype='float64')
        solver = self._solve(A_np)
        w = self._eigvals(solver)
        self._assert_eigvals(w, [2.0, 3.0, 5.0])
        Ah = A_np.astype(complex).conj().T
        vl = self._columns_to_complex(solver.vl, w)
        for j in range(3):
            np.testing.assert_allclose(Ah @ vl[:, j],
                                       np.conj(w[j]) * vl[:, j],
                                       rtol=self.rtol, atol=10 * self.atol)

    def test_rejects_non_square_and_non_2d_inputs(self):
        # Non-square 2D, 1D, and 3D inputs must all raise the same shape error
        # from the EigenSystem constructor.
        for bad in (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                    np.array([1.0, 2.0, 3.0]),
                    np.zeros((2, 2, 2))):
            A = self._array(bad)
            with self.assertRaisesRegex(
                    ValueError, r"must be a square 2D SimpleArray"):
                self.eig_cls(A)

    def test_accessors_before_run_are_inert(self):
        A = self._array(np.diag([1.0, 2.0]))
        # Construct but do not call run().
        solver = self.eig_cls(A)
        # Done is False until run() fills them.
        self.assertFalse(solver.done)
        # Before run() the result buffers are not initialized, so assert the
        # accessor's shape.
        self.assertEqual(self._eigvals(solver).shape, (2,))

    def test_matrix_property_survives_input_gc(self):
        # Drop the Python-side reference to A; solver.matrix must still equal
        # the original, confirming keep_alive<1, 2>() keeps m_matrix valid.
        A_np = np.ascontiguousarray(np.array([[3.0, 1.0], [0.0, 2.0]]),
                                    dtype=self.np_dtype)
        A = self.array_cls(array=A_np)
        solver = self.eig_cls(A)
        del A
        np.testing.assert_array_equal(solver.matrix.ndarray, A_np)

    def test_default_constructor_flags_both_true(self):
        # Guard against an accidental default-flag flip.
        solver = self.eig_cls(self._array(np.diag([1.0, 2.0])))
        self.assertTrue(solver.do_vl)
        self.assertTrue(solver.do_vr)

    def test_skip_vr_does_not_compute_right_eigenvectors(self):
        # do_vr=False must keep eigenvalues/vl intact and reject both vr
        # accessors.
        solver = self._solve(np.array([[2.0, 0.0], [0.0, 3.0]]), do_vr=False)
        self.assertTrue(solver.do_vl)
        self.assertFalse(solver.do_vr)
        self._assert_eigvals(self._eigvals(solver), [2.0, 3.0])
        self.assertEqual(np.array(solver.vl).shape, (2, 2))
        self.assertEqual(np.array(solver.get_vl()).shape, (2, 2))
        with self.assertRaisesRegex(
                RuntimeError, r"right eigenvectors were not computed"):
            solver.vr  # noqa: B018
        with self.assertRaisesRegex(
                RuntimeError, r"right eigenvectors were not computed"):
            solver.get_vr()

    def test_skip_vl_does_not_compute_left_eigenvectors(self):
        # Mirror of test_skip_vr_*; checks A v = lambda v stays exact.
        A_np = np.array([[2.0, 1.0], [0.0, 3.0]], dtype='float64')
        solver = self._solve(A_np, do_vl=False)
        self.assertFalse(solver.do_vl)
        self.assertTrue(solver.do_vr)
        w = self._eigvals(solver)
        Ac = A_np.astype(complex)
        vr = self._columns_to_complex(solver.vr, w)
        for j in range(2):
            np.testing.assert_allclose(Ac @ vr[:, j], w[j] * vr[:, j],
                                       rtol=self.rtol, atol=10 * self.atol)
        with self.assertRaisesRegex(
                RuntimeError, r"left eigenvectors were not computed"):
            solver.vl  # noqa: B018
        with self.assertRaisesRegex(
                RuntimeError, r"left eigenvectors were not computed"):
            solver.get_vl()

    def test_skip_both_computes_only_eigenvalues(self):
        # Exercises the eigenvalue-only workspace path (jobvl=jobvr='N').
        solver = self._solve(np.diag([1.0, 5.0, 9.0]),
                             do_vl=False, do_vr=False)
        self._assert_eigvals(self._eigvals(solver), [1.0, 5.0, 9.0])
        with self.assertRaises(RuntimeError):
            solver.vl  # noqa: B018
        with self.assertRaises(RuntimeError):
            solver.vr  # noqa: B018
        with self.assertRaises(RuntimeError):
            solver.get_vl()
        with self.assertRaises(RuntimeError):
            solver.get_vr()

    def test_get_methods_match_property_when_computed(self):
        # Property and get_v* must alias the same matrix; suppress flag is a
        # no-op when the matrix exists.
        solver = self._solve(np.diag([2.0, 4.0]))
        np.testing.assert_array_equal(np.array(solver.vl),
                                      np.array(solver.get_vl()))
        np.testing.assert_array_equal(np.array(solver.vr),
                                      np.array(solver.get_vr()))
        np.testing.assert_array_equal(
            np.array(solver.vl),
            np.array(solver.get_vl(suppress_exception=True)))
        np.testing.assert_array_equal(
            np.array(solver.vr),
            np.array(solver.get_vr(suppress_exception=True)))

    def test_get_methods_default_argument_raises(self):
        # Only suppress_exception=True silences the exception.
        solver = self._solve(np.diag([1.0, 2.0]), do_vl=False, do_vr=False)
        with self.assertRaises(RuntimeError):
            solver.get_vl()
        with self.assertRaises(RuntimeError):
            solver.get_vr()
        with self.assertRaises(RuntimeError):
            solver.get_vl(suppress_exception=False)
        with self.assertRaises(RuntimeError):
            solver.get_vr(suppress_exception=False)

    def test_suppress_exception_returns_empty_when_not_computed(self):
        # suppress_exception=True returns an empty placeholder, not data.
        solver = self._solve(np.diag([1.0, 2.0]), do_vl=False, do_vr=False)
        empty_vl = np.array(solver.get_vl(suppress_exception=True))
        empty_vr = np.array(solver.get_vr(suppress_exception=True))
        self.assertEqual(empty_vl.size, 0)
        self.assertEqual(empty_vr.size, 0)


@unittest.skipIf(sc.EigenSystemFloat32 is None,
                 "EigenSystem is not built (no vendor LAPACK)")
class TestLinalgEigenSystemFloat32TC(EigenSystemTB, unittest.TestCase):
    array_cls = sc.SimpleArrayFloat32
    eig_cls = sc.EigenSystemFloat32
    np_dtype = np.float32
    is_complex = False
    rtol = 1e-4
    atol = 1e-5


@unittest.skipIf(sc.EigenSystemFloat64 is None,
                 "EigenSystem is not built (no vendor LAPACK)")
class TestLinalgEigenSystemFloat64TC(EigenSystemTB, unittest.TestCase):
    array_cls = sc.SimpleArrayFloat64
    eig_cls = sc.EigenSystemFloat64
    np_dtype = np.float64
    is_complex = False
    rtol = 1e-10
    atol = 1e-12


@unittest.skipIf(sc.EigenSystemComplex64 is None,
                 "EigenSystem is not built (no vendor LAPACK)")
class TestLinalgEigenSystemComplex64TC(EigenSystemTB, unittest.TestCase):
    array_cls = sc.SimpleArrayComplex64
    eig_cls = sc.EigenSystemComplex64
    np_dtype = np.complex64
    is_complex = True
    rtol = 1e-4
    atol = 1e-5


@unittest.skipIf(sc.EigenSystemComplex128 is None,
                 "EigenSystem is not built (no vendor LAPACK)")
class TestLinalgEigenSystemComplex128TC(EigenSystemTB, unittest.TestCase):
    array_cls = sc.SimpleArrayComplex128
    eig_cls = sc.EigenSystemComplex128
    np_dtype = np.complex128
    is_complex = True
    rtol = 1e-10
    atol = 1e-12

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

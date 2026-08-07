# Copyright (c) 2025, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest
import itertools

import numpy as np

import solvcon as sc


class MatrixTestBase(sc.testing.TestBase):
    """Base class for matrix operations"""

    def test_eye_method(self):
        """Test eye method creates correct identity matrices"""
        # Test cases: different sizes
        test_sizes = [1, 2, 3, 4, 5, 10]

        for size in test_sizes:
            with self.subTest(size=size):
                # Create identity matrix using our eye method
                identity = self.SimpleArray.eye(size)

                # Create expected identity matrix using NumPy
                expected = np.eye(size, dtype=self.dtype)

                # Check shape
                self.assertEqual(list(identity.shape), [size, size])

                # Check array values
                np.testing.assert_array_almost_equal(identity.ndarray,
                                                     expected)

                # Verify diagonal and off-diagonal elements explicitly
                # using product
                for i, j in itertools.product(range(size), repeat=2):
                    if i == j:
                        self.assertEqual(identity[i, j], 1.0,
                                         f"Diagonal element ({i},{j}) "
                                         f"should be 1.0")
                    else:
                        self.assertEqual(identity[i, j], 0.0,
                                         f"Off-diagonal element ({i},{j}) "
                                         f"should be 0.0")


class MatrixFloat32TC(MatrixTestBase, unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.SimpleArray = sc.SimpleArrayFloat32


class MatrixFloat64TC(MatrixTestBase, unittest.TestCase):
    def setUp(self):
        self.dtype = np.float64
        self.SimpleArray = sc.SimpleArrayFloat64


class MatmulTestBase(sc.testing.TestBase):
    """Tests for SimpleArray matrix multiplication roles."""

    @staticmethod
    def make_strided_view(data, axis, step):
        storage_shape = list(data.shape)
        storage_shape[axis] *= abs(step)
        storage = np.empty(storage_shape, dtype=data.dtype.name)
        selection = [slice(None)] * data.ndim
        selection[axis] = slice(None, None, step)
        view = storage[tuple(selection)]
        view[...] = data
        return view

    @classmethod
    def make_batch_stride_cases(cls, data):
        cases = [('c_contiguous', data)]
        for axis in range(data.ndim - 2):
            cases.append((
                f'negative_batch_axis_{axis}',
                cls.make_strided_view(data, axis, -1),
            ))
            cases.append((
                f'step_two_batch_axis_{axis}',
                cls.make_strided_view(data, axis, 2),
            ))
        return tuple(cases)

    @classmethod
    def make_matrix_stride_cases(cls, data, axis):
        transposed = data.swapaxes(-1, -2)
        f_contiguous = np.ascontiguousarray(
            transposed, dtype=data.dtype.name).swapaxes(-1, -2)
        storage_shape = list(data.shape)
        storage_shape[-1] += 2
        storage = np.empty(storage_shape, dtype=data.dtype.name)
        selection = [slice(None)] * data.ndim
        selection[-1] = slice(0, data.shape[-1])
        padded = storage[tuple(selection)]
        padded[...] = data

        return (
            ('c_contiguous', data),
            ('f_contiguous', f_contiguous),
            ('padded', padded),
            ('negative_stride',
             cls.make_strided_view(data, axis, -1)),
            ('step_two', cls.make_strided_view(data, axis, 2)),
        )

    def assert_matmul(self, lhs, rhs, expected):
        result = lhs.matmul(rhs)

        self.assertEqual(list(result.shape), list(expected.shape))
        np.testing.assert_array_almost_equal(result.ndarray, expected)
        return result

    def assert_matmul_fast(self, lhs, rhs, expected, matmul_result):
        fast_result = lhs.matmul_fast(rhs)

        self.assertEqual(list(fast_result.shape), list(expected.shape))
        np.testing.assert_array_almost_equal(fast_result.ndarray, expected)
        np.testing.assert_array_almost_equal(fast_result.ndarray,
                                             matmul_result.ndarray)
        return fast_result

    def assert_matmul_blas(self, lhs, rhs, expected, matmul_result):
        blas_result = lhs.matmul_blas(rhs)

        self.assertEqual(list(blas_result.shape), list(expected.shape))
        np.testing.assert_array_almost_equal(blas_result.ndarray, expected)
        np.testing.assert_array_almost_equal(blas_result.ndarray,
                                             matmul_result.ndarray)

    def assert_matmul_planned(self, lhs, rhs, expected):
        result = lhs.matmul_planned(rhs)

        self.assertEqual(list(result.shape), list(expected.shape))
        tol = 64 * np.finfo(result.ndarray.real.dtype).eps
        np.testing.assert_allclose(
            result.ndarray, expected, rtol=tol, atol=tol)
        return result

    def test_square(self):
        """Test basic square matrix multiplication"""
        # Create 2x2 matrices
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.dtype)
        b_data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=self.dtype)

        a = self.SimpleArray(array=a_data)
        b = self.SimpleArray(array=b_data)

        # Expected result: [[19, 22], [43, 50]]
        expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=self.dtype)

        # Test matrix multiplication
        result = self.assert_matmul(a, b, expected)
        self.assert_matmul_fast(a, b, expected, result)
        self.assert_matmul_blas(a, b, expected, result)
        self.assert_matmul_planned(a, b, expected)

    def test_rectangular(self):
        """Test rectangular matrix multiplication"""
        # Create 2x3 and 3x2 matrices
        a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                          dtype=self.dtype)
        b_data = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
                          dtype=self.dtype)

        a = self.SimpleArray(array=a_data)
        b = self.SimpleArray(array=b_data)

        # Expected result: [[58, 64], [139, 154]]
        expected = np.array([[58.0, 64.0], [139.0, 154.0]],
                            dtype=self.dtype)

        result = self.assert_matmul(a, b, expected)
        self.assert_matmul_fast(a, b, expected, result)
        self.assert_matmul_blas(a, b, expected, result)
        self.assert_matmul_planned(a, b, expected)

    def test_identity(self):
        """Test multiplication with identity matrix"""
        # 3x3 matrix
        a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]], dtype=self.dtype)

        a = self.SimpleArray(array=a_data)
        identity = self.SimpleArray.eye(3)

        result = self.assert_matmul(a, identity, a_data)
        self.assert_matmul_fast(a, identity, a_data, result)
        self.assert_matmul_blas(a, identity, a_data, result)
        self.assert_matmul_planned(a, identity, a_data)

    def test_zero(self):
        """Test multiplication with zero matrix"""
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.dtype)
        zero_data = np.zeros((2, 2), dtype=self.dtype)

        a = self.SimpleArray(array=a_data)
        zero = self.SimpleArray(array=zero_data)

        result = self.assert_matmul(a, zero, zero_data)
        self.assert_matmul_fast(a, zero, zero_data, result)
        self.assert_matmul_blas(a, zero, zero_data, result)
        self.assert_matmul_planned(a, zero, zero_data)

    def test_dimension_mismatch_error(self):
        """Test error handling for incompatible dimensions"""

        a_data = np.array([[1.0, 2.0], [3.0, 4.0]],
                          dtype=self.dtype)  # 2x2
        b_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]], dtype=self.dtype)  # 3x3

        a = self.SimpleArray(array=a_data)
        b = self.SimpleArray(array=b_data)

        # Should raise error: 2x2 cannot multiply with 3x3
        # (incompatible inner dimensions: 2 != 3)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: this=\(2,2\) other="
            r"\(3,3\)"
        ):
            a.matmul(b)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: this=\(2,2\) other="
            r"\(3,3\)"
        ):
            a.matmul_fast(b)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: this=\(2,2\) other="
            r"\(3,3\)"
        ):
            a.matmul_blas(b)
        with self.assertRaisesRegex(
            ValueError,
            r"SimpleArray::matmul_planned\(\): shape mismatch: "
            r"this=\(2,2\) other=\(3,3\)"
        ):
            a.matmul_planned(b)

    def test_non_positive_fast_tile_error(self):
        cases = itertools.product(
            ('matmul_fast', 'imatmul_fast'),
            ('tile_x', 'tile_y', 'tile_z'),
            (0, -1),
        )
        for method, tile_name, tile_size in cases:
            with self.subTest(
                    method=method, tile_name=tile_name,
                    tile_size=tile_size):
                data = np.ones((2,), dtype=np.dtype(self.dtype).name)
                lhs = self.SimpleArray(array=data)
                rhs = self.SimpleArray(array=data)
                tiles = {'tile_x': 1, 'tile_y': 1, 'tile_z': 1}
                tiles[tile_name] = tile_size
                message = (
                    r"SimpleArray::matmul_fast\(\): tile sizes must "
                    rf"be positive: tile_x={tiles['tile_x']} "
                    rf"tile_y={tiles['tile_y']} "
                    rf"tile_z={tiles['tile_z']}"
                )
                with self.assertRaisesRegex(IndexError, message):
                    getattr(lhs, method)(rhs, **tiles)

    def test_compare_with_numpy(self):
        """Compare results with NumPy using fixed test data"""

        # Test case 1: (2x3) x (3x4)
        a_data_1 = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ], dtype=self.dtype)
        b_data_1 = np.array([
            [7.0, 8.0, 9.0, 10.0],
            [11.0, 12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0]
        ], dtype=self.dtype)
        expected_1 = np.array([
            [74.0, 80.0, 86.0, 92.0],
            [173.0, 188.0, 203.0, 218.0]
        ], dtype=self.dtype)

        # Test case 2: (4x6) x (6x3)
        a_data_2 = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0, 22.0, 23.0, 24.0]
        ], dtype=self.dtype)
        b_data_2 = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0]
        ], dtype=self.dtype)
        expected_2 = np.array([
            [231.0, 252.0, 273.0],
            [537.0, 594.0, 651.0],
            [843.0, 936.0, 1029.0],
            [1149.0, 1278.0, 1407.0]
        ], dtype=self.dtype)

        # Test case 3: (3x3) x (3x3)
        a_data_3 = np.array([
            [2.0, 1.0, 3.0],
            [1.0, 4.0, 2.0],
            [3.0, 2.0, 1.0]
        ], dtype=self.dtype)
        b_data_3 = np.array([
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 3.0],
            [1.0, 3.0, 2.0]
        ], dtype=self.dtype)
        expected_3 = np.array([
            [7.0, 14.0, 11.0],
            [11.0, 12.0, 17.0],
            [8.0, 11.0, 11.0]
        ], dtype=self.dtype)

        # Test case 4: (4x6) x (6)
        a_data_4 = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0, 22.0, 23.0, 24.0]
        ], dtype=self.dtype)
        b_data_4 = np.array([1., 2., 3., 4., 5., 6], dtype=self.dtype)
        expected_4 = np.array([91., 217., 343., 469.], dtype=self.dtype)

        # Test case 5: (6) x (6 x 4)
        a_data_5 = np.array([1., 2., 3., 4., 5., 6], dtype=self.dtype)
        b_data_5 = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
            [17.0, 18.0, 19.0, 20.0],
            [21.0, 22.0, 23.0, 24.0]
        ], dtype=self.dtype)
        expected_5 = np.array([301., 322., 343., 364.], dtype=self.dtype)

        # Test case 6: (3) x (3)
        a_data_6 = np.array([1., 2., 3.], dtype=self.dtype)
        b_data_6 = np.array([4., 5., 6.], dtype=self.dtype)
        expected_6 = np.array([32.], dtype=self.dtype)

        test_cases = [
            (a_data_1, b_data_1, expected_1, "2x3 x 3x4"),
            (a_data_2, b_data_2, expected_2, "4x6 x 6x3"),
            (a_data_3, b_data_3, expected_3, "3x3 x 3x3"),
            (a_data_4, b_data_4, expected_4, "4x6 x 6"),
            (a_data_5, b_data_5, expected_5, "6 x 6x3"),
            (a_data_6, b_data_6, expected_6, "3 x 3x3")
        ]

        for a_data, b_data, expected, description in test_cases:
            with self.subTest(description=description):
                a = self.SimpleArray(array=a_data)
                b = self.SimpleArray(array=b_data)

                # Verify with NumPy
                np_result = np.matmul(a_data, b_data)
                np.testing.assert_array_almost_equal(expected, np_result)

                # Compare our result with expected
                result = self.assert_matmul(a, b, expected)
                self.assert_matmul_fast(a, b, expected, result)
                self.assert_matmul_blas(a, b, expected, result)
                self.assert_matmul_planned(a, b, expected)

    def test_matrix_strides(self):
        """Matrix layouts work across each GEMM route."""
        dtype = np.dtype(self.dtype).name
        for m, k, n in ((3, 4, 2), (9, 9, 9), (16, 17, 18)):
            lhs_data = np.arange(m * k, dtype=dtype).reshape(m, k)
            rhs_data = np.arange(k * n, dtype=dtype).reshape(k, n)
            lhs_cases = self.make_matrix_stride_cases(lhs_data, 1)
            rhs_cases = self.make_matrix_stride_cases(rhs_data, 0)
            cases = itertools.product(lhs_cases, rhs_cases)
            for (lhs_case, case_lhs), (rhs_case, case_rhs) in cases:
                lhs = self.SimpleArray(array=case_lhs)
                rhs = self.SimpleArray(array=case_rhs)
                expected = np.matmul(case_lhs, case_rhs)

                with self.subTest(
                        shape=(m, k, n), lhs=lhs_case, rhs=rhs_case):
                    self.assert_matmul_planned(lhs, rhs, expected)

    def test_batch_strides(self):
        """Batch axes support negative and step-two strides."""
        dtype = np.dtype(self.dtype).name
        shapes = (
            ((3, 3, 4), (3, 4, 2)),
            ((2, 3, 3, 4), (2, 3, 4, 2)),
        )
        for lhs_shape, rhs_shape in shapes:
            lhs_data = np.arange(
                np.prod(lhs_shape), dtype=dtype).reshape(lhs_shape)
            rhs_data = np.arange(
                np.prod(rhs_shape), dtype=dtype).reshape(rhs_shape)
            lhs_cases = self.make_batch_stride_cases(lhs_data)
            rhs_cases = self.make_batch_stride_cases(rhs_data)
            cases = itertools.product(lhs_cases, rhs_cases)
            for (lhs_case, case_lhs), (rhs_case, case_rhs) in cases:
                lhs = self.SimpleArray(array=case_lhs)
                rhs = self.SimpleArray(array=case_rhs)
                expected = np.matmul(case_lhs, case_rhs)

                with self.subTest(
                        shape=lhs_shape, lhs=lhs_case, rhs=rhs_case):
                    self.assert_matmul_planned(lhs, rhs, expected)

    def test_broadcast_batch_strides(self):
        """Broadcast batch strides across generic and direct BLAS routes."""
        dtype = np.dtype(self.dtype).name
        for m, k, n in ((3, 4, 2), (17, 18, 19)):
            lhs_data = np.arange(2 * m * k, dtype=dtype).reshape(2, 1, m, k)
            rhs_data = np.arange(5 * k * n, dtype=dtype).reshape(1, 5, k, n)
            cases = itertools.product(
                self.make_batch_stride_cases(lhs_data),
                self.make_batch_stride_cases(rhs_data),
            )
            for (lhs_case, case_lhs), (rhs_case, case_rhs) in cases:
                lhs = self.SimpleArray(array=case_lhs)
                rhs = self.SimpleArray(array=case_rhs)
                expected = np.matmul(case_lhs, case_rhs)

                with self.subTest(
                        shape=(m, k, n),
                        lhs=lhs_case, rhs=rhs_case):
                    self.assert_matmul_planned(lhs, rhs, expected)

    def test_broadcast_matrix_strides(self):
        """Broadcast matrix strides work across each GEMM route."""
        dtype = np.dtype(self.dtype).name
        fixed_shapes = tuple((side, side, side) for side in range(8, 16))
        for m, k, n in ((3, 4, 2), *fixed_shapes, (17, 18, 19)):
            lhs_data = np.arange(2 * m * k, dtype=dtype).reshape(
                2, 1, m, k)
            rhs_data = np.arange(5 * k * n, dtype=dtype).reshape(
                1, 5, k, n)
            lhs_cases = self.make_matrix_stride_cases(lhs_data, 3) + ((
                'negative_batch_and_matrix',
                self.make_strided_view(
                    self.make_strided_view(lhs_data, 0, -1), 3, -1),
            ),)
            rhs_cases = self.make_matrix_stride_cases(rhs_data, 2) + ((
                'negative_batch_and_matrix',
                self.make_strided_view(
                    self.make_strided_view(rhs_data, 1, -1), 2, -1),
            ),)
            cases = itertools.product(lhs_cases, rhs_cases)
            for (lhs_case, case_lhs), (rhs_case, case_rhs) in cases:
                lhs = self.SimpleArray(array=case_lhs)
                rhs = self.SimpleArray(array=case_rhs)
                expected = np.matmul(case_lhs, case_rhs)

                with self.subTest(
                        shape=(m, k, n), lhs=lhs_case, rhs=rhs_case):
                    self.assert_matmul_planned(lhs, rhs, expected)

    def test_vector_strides(self):
        """Vector roles preserve signed vector, matrix, and batch strides."""
        dtype = np.dtype(self.dtype).name
        vector_data = np.arange(4, dtype=dtype)
        vector_cases = (
            ('contiguous', vector_data),
            ('negative_stride',
             self.make_strided_view(vector_data, 0, -1)),
            ('step_two', self.make_strided_view(vector_data, 0, 2)),
        )

        gevm_data = np.arange(24, dtype=dtype).reshape(2, 1, 4, 3)
        gevm_cases = (
            self.make_matrix_stride_cases(gevm_data, 2) +
            self.make_batch_stride_cases(gevm_data)[1:]
        )
        gemv_data = np.arange(24, dtype=dtype).reshape(2, 1, 3, 4)
        gemv_cases = (
            self.make_matrix_stride_cases(gemv_data, 3) +
            self.make_batch_stride_cases(gemv_data)[1:]
        )
        role_cases = (
            ('gevm', vector_cases, gevm_cases),
            ('gemv', gemv_cases, vector_cases),
        )
        for role, lhs_cases, rhs_cases in role_cases:
            cases = itertools.product(lhs_cases, rhs_cases)
            for (lhs_case, lhs_data), (rhs_case, rhs_data) in cases:
                lhs = self.SimpleArray(array=lhs_data)
                rhs = self.SimpleArray(array=rhs_data)
                expected = np.matmul(lhs_data, rhs_data)

                with self.subTest(
                        role=role, lhs=lhs_case, rhs=rhs_case):
                    self.assert_matmul_planned(lhs, rhs, expected)

        lhs_matrix = np.arange(64 * 512, dtype=dtype).reshape(64, 512)
        rhs_matrix = np.arange(512 * 64, dtype=dtype).reshape(512, 64)
        lhs_vector = np.empty(1024, dtype=dtype)[::2]
        rhs_vector = np.empty(1024, dtype=dtype)[::2]
        lhs_vector[...] = lhs_matrix[0]
        rhs_vector[...] = rhs_matrix[:, 0]
        contiguous_rhs = np.ascontiguousarray(rhs_matrix[:, 0], dtype=dtype)
        cases = (
            (lhs_matrix[0], contiguous_rhs),
            (lhs_matrix[0, ::-1], contiguous_rhs[::-1]),
            (lhs_matrix[0], rhs_matrix),
            (lhs_matrix, contiguous_rhs),
            (lhs_vector, np.asfortranarray(rhs_matrix, dtype=dtype)),
            (np.asfortranarray(lhs_matrix, dtype=dtype), rhs_vector),
        )
        for lhs_data, rhs_data in cases:
            lhs = self.SimpleArray(array=lhs_data)
            rhs = self.SimpleArray(array=rhs_data)
            with self.subTest(lhs=lhs_data.strides,
                              rhs=rhs_data.strides):
                expected = np.atleast_1d(np.matmul(lhs_data, rhs_data))
                self.assert_matmul_planned(
                    lhs, rhs, expected)

    def test_large_batched_vector_strides(self):
        """Batched vector roles preserve layouts across tuned sizes."""
        dtype = np.dtype(self.dtype).name
        for side, batch_size in ((24, 2), (24, 8), (32, 4)):
            vector_data = np.arange(side, dtype=dtype)
            vector_cases = (
                ('contiguous', vector_data),
                ('negative_stride',
                 self.make_strided_view(vector_data, 0, -1)),
                ('step_two', self.make_strided_view(vector_data, 0, 2)),
            )
            matrix_data = np.arange(
                batch_size * side * side, dtype=dtype).reshape(
                    batch_size, side, side)
            role_cases = (
                ('gevm', vector_cases,
                 self.make_matrix_stride_cases(matrix_data, -2)[:2]),
                ('gemv', self.make_matrix_stride_cases(
                    matrix_data, -1)[:2], vector_cases),
            )
            for role, lhs_cases, rhs_cases in role_cases:
                for lhs_case, rhs_case in itertools.product(
                        lhs_cases, rhs_cases):
                    lhs_name, lhs_data = lhs_case
                    rhs_name, rhs_data = rhs_case
                    lhs = self.SimpleArray(array=lhs_data)
                    rhs = self.SimpleArray(array=rhs_data)
                    expected = np.matmul(lhs_data, rhs_data)

                    with self.subTest(
                            role=role, side=side, batch=batch_size,
                            lhs=lhs_name, rhs=rhs_name):
                        self.assert_matmul_planned(lhs, rhs, expected)

    def test_batch_axes_align_right(self):
        """Leading batch axes align from the right like NumPy matmul."""
        dtype = np.dtype(self.dtype).name
        shapes = (
            ((3, 4), (2, 5, 4, 2)),
            ((2, 5, 3, 4), (4, 2)),
            ((1, 3, 4), (2, 5, 4, 2)),
            ((2, 1, 3, 4), (5, 4, 2)),
        )
        for lhs_shape, rhs_shape in shapes:
            lhs_data = np.arange(
                np.prod(lhs_shape), dtype=dtype).reshape(lhs_shape)
            rhs_data = np.arange(
                np.prod(rhs_shape), dtype=dtype).reshape(rhs_shape)
            lhs = self.SimpleArray(array=lhs_data)
            rhs = self.SimpleArray(array=rhs_data)

            with self.subTest(lhs=lhs_shape, rhs=rhs_shape):
                self.assert_matmul_planned(
                    lhs, rhs, np.matmul(lhs_data, rhs_data))

    def test_broadcast_batch_mismatch(self):
        """Incompatible leading batch axes report both operand shapes."""
        dtype = np.dtype(self.dtype).name
        lhs = self.SimpleArray(
            array=np.zeros((2, 3, 4), dtype=dtype))
        rhs = self.SimpleArray(
            array=np.zeros((3, 4, 2), dtype=dtype))
        with self.assertRaisesRegex(
            ValueError,
            r"SimpleArray::matmul_planned\(\): batch shape mismatch: "
            r"this=\(2,3,4\) other=\(3,4,2\)"
        ):
            lhs.matmul_planned(rhs)

    def test_empty_dimensions(self):
        """Matmul preserves empty output and inner axes."""
        dtype = np.dtype(self.dtype).name
        shapes = (((0,), (0,)),
                  ((3, 0), (0,)),
                  ((0,), (0, 2)),
                  ((2, 3, 0), (0,)),
                  ((0,), (2, 3, 0, 4)),
                  ((0, 4), (4, 2)),
                  ((3, 0), (0, 2)),
                  ((3, 4), (4, 0)),
                  ((0, 3, 4), (0, 4, 2)),
                  ((0, 1, 3, 4), (1, 5, 4, 2)),
                  ((1, 3, 4), (0, 4, 2)))
        for lhs_shape, rhs_shape in shapes:
            lhs_data = np.zeros(lhs_shape, dtype=dtype)
            rhs_data = np.zeros(rhs_shape, dtype=dtype)
            expected = np.atleast_1d(np.matmul(lhs_data, rhs_data))
            lhs = self.SimpleArray(array=lhs_data)
            rhs = self.SimpleArray(array=rhs_data)

            with self.subTest(lhs=lhs_shape, rhs=rhs_shape):
                self.assert_matmul_planned(lhs, rhs, expected)

    def test_matmul_with_strided_vectors(self):
        """Matmul supports every pairing of contiguous and strided vectors."""
        dtype = np.dtype(self.dtype).name
        vectors = {
            'contiguous': np.arange(8, dtype=dtype),
            'negative': np.arange(8, dtype=dtype)[::-1],
            'step_two': np.arange(16, dtype=dtype)[::2],
        }
        methods = ('matmul', 'matmul_fast', 'matmul_blas', 'matmul_planned')

        cases = itertools.product(vectors, vectors, methods)
        for lhs_case, rhs_case, method in cases:
            lhs_data = vectors[lhs_case]
            rhs_data = vectors[rhs_case]
            lhs = self.SimpleArray(array=lhs_data)
            rhs = self.SimpleArray(array=rhs_data)
            expected = np.matmul(lhs_data, rhs_data)

            with self.subTest(
                    lhs=lhs_case, rhs=rhs_case, method=method):
                result = getattr(lhs, method)(rhs)
                self.assertEqual((1,), result.shape)
                np.testing.assert_allclose(result.ndarray[0], expected)

    def test_wrong_shape_error(self):
        """Test error handling for wrong shapes"""

        a_3d_data = np.array([[[1.0, 2.0], [3.0, 4.0]],
                              [[5.0, 6.0], [7.0, 8.0]]], dtype=self.dtype)
        b_3d_data = np.array([[[1.0, 0.0], [0.0, 1.0]],
                              [[2.0, 0.0], [0.0, 2.0]]], dtype=self.dtype)
        a_3d = self.SimpleArray(array=a_3d_data)
        b_3d = self.SimpleArray(array=b_3d_data)

        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): unsupported dimensions: "
            r"this=\(2,2,2\) other=\(2,2,2\)\. SimpleArray must be 1D or 2D."
        ):
            a_3d.matmul(b_3d)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): unsupported dimensions: "
            r"this=\(2,2,2\) other=\(2,2,2\)\. SimpleArray must be 1D or 2D."
        ):
            a_3d.matmul_fast(b_3d)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): unsupported dimensions: "
            r"this=\(2,2,2\) other=\(2,2,2\)\. SimpleArray must be 1D or 2D."
        ):
            a_3d.matmul_blas(b_3d)

        a = np.zeros((3, 3), dtype=self.dtype)
        b = np.zeros((2, 3), dtype=self.dtype)
        a = self.SimpleArray(array=a)
        b = self.SimpleArray(array=b)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: "
            r"this=\(3,3\) other=\(2,3\)"
        ):
            a.matmul(b)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: "
            r"this=\(3,3\) other=\(2,3\)"
        ):
            a.matmul_fast(b)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: "
            r"this=\(3,3\) other=\(2,3\)"
        ):
            a.matmul_blas(b)

        a = np.zeros((3, 3), dtype=self.dtype)
        b = np.zeros((2), dtype=self.dtype)
        a = self.SimpleArray(array=a)
        b = self.SimpleArray(array=b)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: "
            r"this=\(3,3\) other=\(2\)"
        ):
            a.matmul(b)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: "
            r"this=\(3,3\) other=\(2\)"
        ):
            a.matmul_fast(b)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: "
            r"this=\(3,3\) other=\(2\)"
        ):
            a.matmul_blas(b)

        a = np.zeros((2), dtype=self.dtype)
        b = np.zeros((3, 3), dtype=self.dtype)
        a = self.SimpleArray(array=a)
        b = self.SimpleArray(array=b)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: "
            r"this=\(2\) other=\(3,3\)"
        ):
            a.matmul(b)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: "
            r"this=\(2\) other=\(3,3\)"
        ):
            a.matmul_fast(b)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: "
            r"this=\(2\) other=\(3,3\)"
        ):
            a.matmul_blas(b)

        a = np.zeros((2), dtype=self.dtype)
        b = np.zeros((3), dtype=self.dtype)
        a = self.SimpleArray(array=a)
        b = self.SimpleArray(array=b)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: "
            r"this=\(2\) other=\(3\)"
        ):
            a.matmul(b)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: "
            r"this=\(2\) other=\(3\)"
        ):
            a.matmul_fast(b)
        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::matmul\(\): shape mismatch: "
            r"this=\(2\) other=\(3\)"
        ):
            a.matmul_blas(b)

    def test_matmul_operator(self):
        """Test @ operator for matrix multiplication"""
        # Create 2x2 matrices
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.dtype)
        b_data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=self.dtype)

        a = self.SimpleArray(array=a_data)
        b = self.SimpleArray(array=b_data)

        # Expected result: [[19, 22], [43, 50]]
        expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=self.dtype)

        # Test @ operator
        result = a @ b

        self.assertEqual(list(result.shape), [2, 2])
        np.testing.assert_array_almost_equal(result.ndarray, expected)

    def test_imatmul_method(self):
        """Test imatmul() method for in-place matrix multiplication"""
        # Create 2x2 matrices
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.dtype)
        b_data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=self.dtype)

        a = self.SimpleArray(array=a_data)
        a_fast = self.SimpleArray(array=a_data)
        a_blas = self.SimpleArray(array=a_data)
        b = self.SimpleArray(array=b_data)

        # Expected result: [[19, 22], [43, 50]]
        expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=self.dtype)

        # Test imatmul() method
        a.imatmul(b)
        # Test imatmul_blas() method
        a_blas.imatmul_blas(b)
        # Test imatmul_fast() method
        a_fast.imatmul_fast(b)

        # Verify the result
        self.assertEqual(list(a.shape), [2, 2])
        np.testing.assert_array_almost_equal(a.ndarray, expected)
        self.assertEqual(list(a_blas.shape), [2, 2])
        np.testing.assert_array_almost_equal(a_blas.ndarray, expected)
        self.assertEqual(list(a_fast.shape), [2, 2])
        np.testing.assert_array_almost_equal(a_fast.ndarray, expected)
        np.testing.assert_array_almost_equal(a_blas.ndarray, a.ndarray)
        np.testing.assert_array_almost_equal(a_fast.ndarray, a.ndarray)

    def test_imatmul_operator(self):
        """Test @= operator for in-place matrix multiplication"""
        # Create 2x2 matrices
        a_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.dtype)
        b_data = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=self.dtype)

        a = self.SimpleArray(array=a_data)
        b = self.SimpleArray(array=b_data)

        # Expected result: [[19, 22], [43, 50]]
        expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=self.dtype)

        # Test @= operator
        a @= b

        self.assertEqual(list(a.shape), [2, 2])
        np.testing.assert_array_almost_equal(a.ndarray, expected)


class MatrixPowerTestBase(sc.testing.TestBase):
    """Tests for matrix power A^n with non-negative integer n"""

    def assert_pow(self, mat, mat_data, n):
        result = mat.pow(n)
        expected = np.linalg.matrix_power(mat_data, n)

        self.assertEqual(list(result.shape), list(expected.shape))
        np.testing.assert_array_almost_equal(result.ndarray, expected)
        return result

    def test_zero_exponent(self):
        """A^0 is the identity matrix"""
        mat_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.dtype)
        mat = self.SimpleArray(array=mat_data)

        result = self.assert_pow(mat, mat_data, 0)
        np.testing.assert_array_almost_equal(
            result.ndarray, np.eye(2, dtype=self.dtype))

    def test_one_exponent(self):
        """A^1 reproduces the original matrix"""
        mat_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.dtype)
        mat = self.SimpleArray(array=mat_data)

        result = self.assert_pow(mat, mat_data, 1)
        np.testing.assert_array_almost_equal(result.ndarray, mat_data)

    def test_small_exponents(self):
        """A^n matches numpy.linalg.matrix_power for small n"""
        mat_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.dtype)
        mat = self.SimpleArray(array=mat_data)

        for n in range(0, 8):
            with self.subTest(n=n):
                self.assert_pow(mat, mat_data, n)

    def test_identity_power(self):
        """The identity matrix is invariant under any power"""
        identity = self.SimpleArray.eye(4)
        identity_data = np.eye(4, dtype=self.dtype)

        for n in (0, 1, 5, 10):
            with self.subTest(n=n):
                self.assert_pow(identity, identity_data, n)

    def test_matrix_dim_to_5(self):
        """A^n matches numpy across several square matrices and exponents"""
        fixtures = [
            [[-3]],
            [[2, 1], [0, 0]],
            [[3, -3, 1], [-2, -3, 0], [3, 2, 2]],
            [[2, 2, 0, -3, 2],
             [0, 0, -1, -2, 3],
             [2, 1, -1, 2, 0],
             [0, 0, -2, -3, 0],
             [3, -3, 3, 2, -2]],
        ]
        exponents = [0, 1, 2, 3, 6]

        for fixture in fixtures:
            mat_data = np.array(fixture, dtype=self.dtype)
            mat = self.SimpleArray(array=mat_data)
            for n in exponents:
                with self.subTest(size=mat_data.shape[0], n=n):
                    self.assert_pow(mat, mat_data, n)

    def test_negative_exponent_error(self):
        """A negative exponent is rejected"""
        mat_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.dtype)
        mat = self.SimpleArray(array=mat_data)

        with self.assertRaisesRegex(
                ValueError,
                r"SimpleArray::pow\(\): exponent must be non-negative, "
                r"but got -1"):
            mat.pow(-1)

    def test_non_square_error(self):
        """A non-square matrix cannot be raised to a power"""
        mat_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                            dtype=self.dtype)
        mat = self.SimpleArray(array=mat_data)

        with self.assertRaisesRegex(
                RuntimeError,
                r"SimpleArray::pow\(\): operation requires square "
                r"SimpleArray, but got 2x3 shape"):
            mat.pow(2)

    def test_non_2d_error(self):
        """A non-2D SimpleArray cannot be raised to a power"""
        # 1D, 3D, and 4D arrays must all be rejected, and the error must
        # report the offending dimensionality.
        shapes = [(3,), (2, 2, 2), (2, 2, 2, 2)]

        for shape in shapes:
            ndim = len(shape)
            with self.subTest(ndim=ndim):
                mat = self.SimpleArray(
                    array=np.ones(shape, dtype=self.dtype))
                with self.assertRaisesRegex(
                        RuntimeError,
                        r"SimpleArray::pow\(\): operation requires 2D "
                        r"SimpleArray, but got %dD SimpleArray" % ndim):
                    mat.pow(2)


class MatrixPowerFloat32TC(MatrixPowerTestBase, unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.SimpleArray = sc.SimpleArrayFloat32


class MatrixPowerFloat64TC(MatrixPowerTestBase, unittest.TestCase):
    def setUp(self):
        self.dtype = np.float64
        self.SimpleArray = sc.SimpleArrayFloat64


class MatmulFloat32TC(MatmulTestBase, unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.SimpleArray = sc.SimpleArrayFloat32


class MatmulFloat64TC(MatmulTestBase, unittest.TestCase):
    def setUp(self):
        self.dtype = np.float64
        self.SimpleArray = sc.SimpleArrayFloat64


class MatmulComplex64TC(MatmulTestBase, unittest.TestCase):
    def setUp(self):
        self.dtype = np.complex64
        self.SimpleArray = sc.SimpleArrayComplex64


class MatmulComplex128TC(MatmulTestBase, unittest.TestCase):
    def setUp(self):
        self.dtype = np.complex128
        self.SimpleArray = sc.SimpleArrayComplex128

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

from solvcon import benchmark


def make_matmul_request(**updates):
    data = {
        'operation': 'matmul',
        'lhs': {'shape': [2, 3], 'strides': [3, 1]},
        'rhs': {'shape': [3, 2], 'strides': [2, 1]},
        'dtype': 'float64',
        'sampling': {'warmups': 2, 'repetitions': 5, 'rounds': 2},
        'kernels': ['naive', 'blas_gemm'],
    }
    data.update(updates)
    return benchmark.matmul.MatmulRequest.from_dict(data)


class OperandSpecTC(unittest.TestCase):
    def test_round_trip(self):
        data = {'shape': [2, 3], 'strides': [-5, 0]}
        operand = benchmark.request.OperandSpec.from_dict(data)

        self.assertEqual(operand.shape, (2, 3))
        self.assertEqual(operand.strides, (-5, 0))
        self.assertEqual(operand.to_dict(), data)

    def test_invalid(self):
        cases = (
            ({'shape': [], 'strides': []}, 'at least one'),
            ({'shape': [2, 3], 'strides': [1]}, 'same length'),
            ({'shape': [2, -3], 'strides': [3, 1]}, 'at least 0'),
            ({'shape': [2, True], 'strides': [3, 1]}, 'integer'),
            ({'shape': [2, 3]}, 'missing fields'),
        )
        for data, message in cases:
            with self.subTest(data=data):
                with self.assertRaisesRegex(
                        benchmark.request.RequestError, message):
                    benchmark.request.OperandSpec.from_dict(data)


class SamplingTC(unittest.TestCase):
    def test_no_upper_limit(self):
        data = {
            'warmups': 1_000_000,
            'repetitions': 2_000_000,
            'rounds': 3_000_000,
        }
        sampling = benchmark.request.Sampling.from_dict(data)

        self.assertEqual(sampling.to_dict(), data)

    def test_invalid(self):
        cases = (
            ({'warmups': -1, 'repetitions': 1, 'rounds': 1},
             'warmups'),
            ({'warmups': 0, 'repetitions': 0, 'rounds': 1},
             'repetitions'),
            ({'warmups': 0, 'repetitions': 1, 'rounds': False},
             'integer'),
            ({'warmups': 0, 'repetitions': 1}, 'missing fields'),
        )
        for data, message in cases:
            with self.subTest(data=data):
                with self.assertRaisesRegex(
                        benchmark.request.RequestError, message):
                    benchmark.request.Sampling.from_dict(data)


class MatmulRequestTC(unittest.TestCase):
    def test_round_trip(self):
        request = make_matmul_request(
            lhs={'shape': [2, 3], 'strides': [-5, 1]},
            rhs={'shape': [3, 2], 'strides': [0, 1]},
            sampling={'warmups': 0, 'repetitions': 7, 'rounds': 3},
            kernels=['naive', 'blas_gemm', 'winograd'],
        )

        rebuilt = benchmark.matmul.MatmulRequest.from_dict(
            request.to_dict())

        self.assertEqual(rebuilt, request)

    def test_output_shape(self):
        cases = (
            ({'shape': [3], 'strides': [-1]},
             {'shape': [3], 'strides': [0]}, (1,)),
            ({'shape': [3], 'strides': [1]},
             {'shape': [4, 3, 2], 'strides': [6, 2, 1]},
             (4, 2)),
            ({'shape': [2, 1, 3, 4], 'strides': [12, 0, 4, 1]},
             {'shape': [1, 5, 4, 6], 'strides': [0, 24, 6, 1]},
             (2, 5, 3, 6)),
        )
        for lhs, rhs, output_shape in cases:
            with self.subTest(lhs=lhs['shape'], rhs=rhs['shape']):
                self.assertEqual(
                    make_matmul_request(lhs=lhs, rhs=rhs).output_shape,
                    output_shape,
                )

    def test_invalid_shape(self):
        cases = (
            ({'shape': [2, 3], 'strides': [3, 1]},
             {'shape': [4, 2], 'strides': [2, 1]}, 'contraction'),
            ({'shape': [2, 3, 4], 'strides': [12, 4, 1]},
             {'shape': [5, 4, 6], 'strides': [24, 6, 1]}, 'batch'),
        )
        for lhs, rhs, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                        benchmark.request.RequestError, message):
                    make_matmul_request(lhs=lhs, rhs=rhs)

    def test_invalid_dtype_and_kernel(self):
        cases = (
            ({'dtype': 'int64'}, 'dtype'),
            ({'dtype': []}, 'dtype'),
            ({'kernels': []}, 'must not be empty'),
            ({'kernels': [1]}, 'non-empty strings'),
            ({'kernels': ['naive', 'naive']}, 'duplicates'),
            ({'kernels': ['auto']}, 'unsupported kernels'),
        )
        for updates, message in cases:
            with self.subTest(updates=updates):
                with self.assertRaisesRegex(
                        benchmark.request.RequestError, message):
                    make_matmul_request(**updates)

    def test_invalid_fields(self):
        data = make_matmul_request().to_dict()
        cases = []
        for field in ('operation', 'lhs', 'sampling'):
            missing = data.copy()
            del missing[field]
            cases.append((missing, 'missing fields'))
        unknown = data.copy()
        unknown['mode'] = 'preview'
        cases.append((unknown, 'unknown fields'))
        wrong_operation = data.copy()
        wrong_operation['operation'] = 'convolution'
        cases.append((wrong_operation, 'operation'))

        for request, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                        benchmark.request.RequestError, message):
                    benchmark.matmul.MatmulRequest.from_dict(request)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

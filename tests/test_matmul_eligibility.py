# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Keep metadata queries consistent with forced kernel execution."""

import unittest
import unittest.mock

import numpy as np

import solvcon as sc
from solvcon.benchmark import collector, matmul, spec


def make_inputs(lhs, rhs, dtype='float64', stride=1):
    return matmul.MatmulInputs(
        spec.OperandSpec(lhs, (stride,) * len(lhs)),
        spec.OperandSpec(rhs, (stride,) * len(rhs)), dtype)


class MatmulEligibilityTC(unittest.TestCase):
    def test_matches_execution(self):
        shapes = (
            ((4,), (4,)), ((4,), (4, 2)), ((2, 4), (4,)),
            ((2, 4), (4, 6)), ((3, 4), (4, 2)),
            ((4,), (2, 4, 2)), ((2, 2, 4), (4,)),
            ((1, 2, 4), (2, 4, 2)), ((2, 4), (1, 4, 2)),
            ((0,), (0,)), ((2, 0), (0, 2)), ((0, 2, 4), (4, 2)))
        for dtype in matmul.MATMUL_DTYPES:
            for lhs, rhs in shapes:
                for stride in (-1, 0, 1):
                    inputs = make_inputs(lhs, rhs, dtype, stride)
                    reasons = inputs.kernel_eligibility()
                    request = matmul.MatmulSpec(
                        inputs.lhs, inputs.rhs, dtype,
                        spec.Sampling(0, 1, 1), matmul.MATMUL_KERNELS)
                    execute = collector._make_executor(request)
                    for name, reason in reasons.items():
                        with self.subTest(dtype=dtype, lhs=lhs, rhs=rhs,
                                          stride=stride, kernel=name):
                            if reason is None:
                                execute(name)
                            else:
                                with self.assertRaises(
                                        sc.MatmulKernelUnavailable) as ctx:
                                    execute(name)
                                self.assertIn(reason, str(ctx.exception))

    def test_shape_reasons(self):
        reasons = make_inputs((2, 4), (4, 6)).kernel_eligibility()
        self.assertIsNone(reasons['naive'])
        if reasons['blas_gemm'] is not None:
            self.assertIn('BLAS backend', reasons['blas_gemm'])
            return
        self.assertIsNone(reasons['winograd'])
        self.assertIn('vector @ vector', reasons['blas_dot'])
        self.assertIn('even', make_inputs(
            (3, 4), (4, 2)).kernel_eligibility()['winograd'])
        self.assertIn('unbatched', make_inputs(
            (1, 2, 4), (4, 2)).kernel_eligibility()['winograd'])
        self.assertIn('positive', make_inputs(
            (2, 0), (0, 2)).kernel_eligibility()['blas_gemm'])

    def test_unsupported_dtype(self):
        array_type = sc.SimpleArray.typed_class('int32')
        reasons = array_type.matmul_kernel_eligibility(
            (2, 2), (2, 1), (2, 2), (2, 1))
        operand = array_type(array=np.ones((2, 2), dtype='int32'))
        self.assertIsNone(reasons.pop('naive'))
        for name, reason in reasons.items():
            with self.subTest(kernel=name):
                self.assertIn('dtype', reason)
                with self.assertRaises(sc.MatmulKernelUnavailable):
                    operand.matmul(operand, kernel=name)

    def test_metadata_only(self):
        inputs = make_inputs((2**32, 2), (2, 2), stride=0)
        with unittest.mock.patch.object(
                collector, '_make_operand', side_effect=AssertionError), \
                unittest.mock.patch.object(
                    np, 'empty', side_effect=AssertionError):
            reasons = inputs.kernel_eligibility()
        self.assertEqual(tuple(reasons), matmul.MATMUL_KERNELS)
        self.assertIsNone(reasons['naive'])

    def test_invalid_layout(self):
        query = sc.SimpleArrayFloat64.matmul_kernel_eligibility
        for shape, strides in (((), ()), ((2, 2), (1,)),
                               ((-2, 2), (2, 1)),
                               ((2**62, 4), (0, 0))):
            with self.subTest(shape=shape, strides=strides):
                with self.assertRaises(ValueError):
                    query(shape, strides, (2, 2), (2, 1))
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            query((2, 3), (3, 1), (2, 2), (2, 1))
        with self.assertRaisesRegex(ValueError, 'output shape'):
            query((2**32, 1, 1), (0, 0, 0),
                  (1, 1, 2**32), (0, 0, 0))


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

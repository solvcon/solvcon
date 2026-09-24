# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import collections
import copy
import io
import itertools
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

import numpy as np

import solvcon as sc
from solvcon import benchmark, system
from solvcon.benchmark import collector
from solvcon.benchmark import matmul
from solvcon.benchmark import results
from solvcon.benchmark import spec as benchmark_spec
from solvcon.benchmark import worker


def make_matmul_spec(**updates):
    data = {
        'operation': 'matmul',
        'lhs': {'shape': [2, 3], 'strides': [3, 1]},
        'rhs': {'shape': [3, 2], 'strides': [2, 1]},
        'dtype': 'float64',
        'sampling': {'warmups': 2, 'repetitions': 5, 'rounds': 2},
        'kernels': ['naive', 'blas_gemm'],
    }
    data.update(updates)
    return benchmark.matmul.MatmulSpec.from_dict(data)


class OperandSpecTC(unittest.TestCase):
    def test_round_trip(self):
        data = {'shape': [2, 3], 'strides': [-5, 0]}
        operand = benchmark.spec.OperandSpec.from_dict(data)

        self.assertEqual(operand.shape, (2, 3))
        self.assertEqual(operand.strides, (-5, 0))
        self.assertEqual(operand.to_dict(), data)

    def test_invalid(self):
        cases = (
            ({'shape': [], 'strides': []}, 'at least one'),
            ({'shape': [2, 3], 'strides': [1]}, 'same length'),
            ({'shape': [2, -3], 'strides': [3, 1]}, 'at least 0'),
            ({'shape': [2, True], 'strides': [3, 1]}, 'integer'),
            ({'shape': [sys.maxsize + 1], 'strides': [0]}, 'at most'),
            ({'shape': [1], 'strides': [sys.maxsize + 1]}, 'at most'),
            ({'shape': [1], 'strides': [-sys.maxsize - 2]}, 'at least'),
            ({'shape': [2, 3]}, 'missing fields'),
        )
        for data, message in cases:
            with self.subTest(data=data):
                with self.assertRaisesRegex(
                        benchmark.spec.SpecError, message):
                    benchmark.spec.OperandSpec.from_dict(data)


class SamplingTC(unittest.TestCase):
    def test_no_upper_limit(self):
        data = {
            'warmups': 1_000_000,
            'repetitions': 2_000_000,
            'rounds': 3_000_000,
        }
        sampling = benchmark.spec.Sampling.from_dict(data)

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
                        benchmark.spec.SpecError, message):
                    benchmark.spec.Sampling.from_dict(data)


def make_inputs(lhs, rhs, dtype='float64', stride=1):
    return matmul.MatmulInputs(
        benchmark_spec.OperandSpec(lhs, (stride,) * len(lhs)),
        benchmark_spec.OperandSpec(rhs, (stride,) * len(rhs)), dtype)


class MatmulInputsTC(unittest.TestCase):
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
                        benchmark_spec.Sampling(0, 1, 1),
                        matmul.MATMUL_KERNELS)
                    execute = request.make_executor()
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

    def test_metadata_only(self):
        inputs = make_inputs((2**32, 2), (2, 2), stride=0)
        with unittest.mock.patch.object(
                matmul, '_make_operand', side_effect=AssertionError), \
                unittest.mock.patch.object(
                    np, 'empty', side_effect=AssertionError):
            reasons = inputs.kernel_eligibility()
        self.assertEqual(tuple(reasons), matmul.MATMUL_KERNELS)
        self.assertIsNone(reasons['naive'])


class MatmulSpecTC(unittest.TestCase):
    def test_round_trip(self):
        spec = make_matmul_spec(
            lhs={'shape': [2, 3], 'strides': [-5, 1]},
            rhs={'shape': [3, 2], 'strides': [0, 1]},
            sampling={'warmups': 0, 'repetitions': 7, 'rounds': 3},
            kernels=['naive', 'blas_gemm', 'winograd'],
        )

        rebuilt = benchmark.matmul.MatmulSpec.from_dict(spec.to_dict())

        self.assertEqual(rebuilt, spec)

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
                    make_matmul_spec(lhs=lhs, rhs=rhs).output_shape,
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
                        benchmark.spec.SpecError, message):
                    make_matmul_spec(lhs=lhs, rhs=rhs)

    def test_invalid_layout(self):
        cases = (
            ({'shape': [1, 3],
              'strides': [sys.maxsize // 8 + 1, 1]}, 'byte stride'),
            ({'shape': [3, 3],
              'strides': [sys.maxsize // 8, 1]}, 'byte offset'),
            ({'shape': [2, 3],
              'strides': [sys.maxsize // 8, 0]}, 'byte span'),
            ({'shape': [sys.maxsize // 8 + 1, 3],
              'strides': [0, 0]}, 'logical byte size'),
            ({'shape': [sys.maxsize // 8 + 1, 0, 3],
              'strides': [0, 0, 0]}, 'logical byte size'),
        )
        for lhs, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                        benchmark.spec.SpecError, message):
                    make_matmul_spec(lhs=lhs)

    def test_invalid_output_layout(self):
        side = math.isqrt(sys.maxsize // 8) + 1
        lhs = {
            'shape': [side, 1, 1, 1],
            'strides': [0, 0, 0, 0],
        }
        rhs = {
            'shape': [1, side, 1, 1],
            'strides': [0, 0, 0, 0],
        }

        with self.assertRaisesRegex(
                benchmark.spec.SpecError,
                'output logical byte size'):
            make_matmul_spec(lhs=lhs, rhs=rhs)

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
                        benchmark.spec.SpecError, message):
                    make_matmul_spec(**updates)

    def test_invalid_fields(self):
        data = make_matmul_spec().to_dict()
        cases = []
        for field in ('operation', 'lhs', 'sampling'):
            missing = data.copy()
            del missing[field]
            cases.append((missing, 'missing fields'))
        unknown = data.copy()
        unknown['mode'] = 'preview'
        cases.append((unknown, 'unknown fields'))
        non_string = data.copy()
        non_string[0] = None
        non_string[None] = None
        cases.append((non_string, 'field names'))
        wrong_operation = data.copy()
        wrong_operation['operation'] = 'convolution'
        cases.append((wrong_operation, 'operation'))

        for spec, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                        benchmark.spec.SpecError, message):
                    benchmark.matmul.MatmulSpec.from_dict(spec)


def make_spec(**updates):
    data = {
        'operation': 'matmul',
        'lhs': {'shape': [2, 3], 'strides': [3, 1]},
        'rhs': {'shape': [3, 2], 'strides': [2, 1]},
        'dtype': 'float64',
        'sampling': {'warmups': 1, 'repetitions': 2, 'rounds': 6},
        'kernels': ['naive', 'blas_gemm'],
    }
    data.update(updates)
    return matmul.MatmulSpec.from_dict(data)


class StepClock:
    def __init__(self, step=100, events=None):
        self.value = -step
        self.step = step
        self.events = events

    def __call__(self):
        if self.events is not None:
            self.events.append('clock')
        self.value += self.step
        return self.value


class FakeExecutor:
    unavailable_error = sc.MatmulKernelUnavailable

    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def __call__(self, name):
        self.calls.append(name)
        output = self.outputs[name]
        if isinstance(output, Exception):
            raise output
        if isinstance(output, np.ndarray):
            return output.copy()
        return np.array(output, dtype='float64')


class OperandConstructionTC(unittest.TestCase):
    def test_quantizes_shared_component_stream(self):
        operand = benchmark_spec.OperandSpec.from_dict({
            'shape': [2, 3], 'strides': [3, 1],
        })
        expected = 2 * np.random.default_rng(7).random(
            12, dtype='float64') - 1
        with unittest.mock.patch.object(
                matmul, '_CHUNK_SIZE', 4):
            arrays = {
                dtype: matmul._make_operand(operand, dtype, 7)
                for dtype in ('float32', 'float64',
                              'complex64', 'complex128')
            }
        for array in arrays.values():
            components = array.view(array.real.dtype.name).ravel()
            np.testing.assert_array_equal(
                components,
                expected[:components.size].astype(components.dtype.name))

    def test_preserves_negative_zero_and_empty_strides(self):
        cases = (
            ({'shape': [2, 3], 'strides': [-4, 0]}, (-4, 0)),
            ({'shape': [0, 3], 'strides': [99, -2]}, (99, -2)),
        )
        for data, strides in cases:
            with self.subTest(shape=data['shape']):
                operand = benchmark_spec.OperandSpec.from_dict(data)
                array = matmul._make_operand(operand, 'complex64', 0)

                self.assertEqual(array.shape, tuple(data['shape']))
                self.assertEqual(
                    tuple(value // array.itemsize
                          for value in array.strides),
                    strides,
                )
                self.assertEqual(array.dtype.name, 'complex64')
                self.assertTrue(np.all(np.isfinite(array)))

        operand = benchmark_spec.OperandSpec.from_dict(cases[0][0])
        array = matmul._make_operand(operand, 'complex64', 0)
        np.testing.assert_array_equal(array[:, 0], array[:, 1])
        np.testing.assert_array_equal(array[:, 1], array[:, 2])
        self.assertNotEqual(array[0, 0], array[1, 0])


class MatmulComparisonTC(unittest.TestCase):
    def test_measures_each_result(self):
        expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype='float64')
        execute = FakeExecutor({
            'naive': expected,
            'blas_gemm': expected + 1,
            'winograd': sc.MatmulKernelUnavailable('not eligible'),
            'numpy': expected,
        })
        spec = make_spec(kernels=['naive', 'blas_gemm', 'winograd'])

        comparison = collector._compare(spec, execute)

        self.assertEqual(comparison, {
            'naive': results.KernelResult(
                'naive', 'measured', max_abs_diff=0.0, relative_diff=0.0),
            'blas_gemm': results.KernelResult(
                'blas_gemm', 'measured', max_abs_diff=1.0, relative_diff=0.25),
            'winograd': results.KernelResult(
                'winograd', 'ineligible', reason='not eligible'),
            'numpy': results.KernelResult(
                'numpy', 'measured', max_abs_diff=0.0, relative_diff=0.0),
        })
        self.assertEqual(
            execute.calls,
            ['numpy', 'naive', 'blas_gemm', 'winograd'],
        )

    def test_rejects_invalid_outputs(self):
        expected = np.ones((2, 2), dtype='float64')
        nonfinite = expected.copy()
        nonfinite[0, 0] = np.nan
        cases = (
            (np.ones((1, 4), dtype='float64'), 'shape mismatch'),
            (expected.astype('float32'), 'dtype mismatch'),
            (nonfinite, 'non-finite values'),
        )
        for output, reason in cases:
            with self.subTest(reason=reason):
                result = collector._compare_result(
                    FakeExecutor({'naive': output}),
                    'naive', expected)
                self.assertEqual(result.status, 'invalid')
                self.assertEqual(result.reason, reason)
                self.assertIsNone(result.max_abs_diff)
                self.assertIsNone(result.relative_diff)

    def test_rejects_invalid_numpy_reference(self):
        cases = (
            (np.ones((1, 4), dtype='float64'), 'shape does not match'),
            (np.ones((2, 2), dtype='float32'), 'dtype does not match'),
        )
        for reference, reason in cases:
            with self.subTest(reason=reason):
                execute = FakeExecutor({'numpy': reference})
                with self.assertRaisesRegex(RuntimeError, reason):
                    collector._compare(make_spec(), execute)
                self.assertEqual(execute.calls, ['numpy'])

    def test_marks_nonfinite_numpy_reference_invalid(self):
        reference = np.ones((2, 2), dtype='float64')
        reference[0, 0] = np.nan
        execute = FakeExecutor({'numpy': reference})

        comparison = collector._compare(make_spec(kernels=['naive']), execute)

        self.assertEqual(comparison, {
            name: results.KernelResult(
                name, 'invalid', reason='non-finite NumPy reference')
            for name in ('naive', 'numpy')
        })
        self.assertEqual(execute.calls, ['numpy'])

    def test_internal_failures_propagate(self):
        expected = np.ones((2, 2), dtype='float64')
        for failure in (ValueError('native bug'), RuntimeError('native bug'),
                        MemoryError('out of memory')):
            with self.subTest(failure=type(failure).__name__):
                execute = FakeExecutor({
                    'naive': failure,
                    'blas_gemm': expected,
                    'numpy': expected,
                })
                with self.assertRaises(type(failure)):
                    collector._compare(make_spec(), execute)

    @unittest.mock.patch.object(collector, '_CHUNK_SIZE', 1)
    def test_reports_zero_complex_and_empty_differences(self):
        cases = (
            (np.zeros(1, dtype='float64'),
             np.zeros(1, dtype='float64'), (0.0, 0.0)),
            (np.ones(1, dtype='float64'),
             np.zeros(1, dtype='float64'), (1.0, None)),
            (np.array([3 + 8j, 1 + 2j], dtype='complex64'),
             np.array([3 + 4j, 1 + 1j], dtype='complex64'), (4.0, 0.8)),
            (np.empty(0, dtype='float64'),
             np.empty(0, dtype='float64'), (None, None)),
        )
        for result, reference, expected in cases:
            with self.subTest(result=result, reference=reference):
                self.assertEqual(
                    collector._difference_metrics(result, reference),
                    expected,
                )

    def test_dispatches_real_executor_for_every_dtype(self):
        for dtype in ('float32', 'float64', 'complex64', 'complex128'):
            with self.subTest(dtype=dtype):
                spec = make_spec(dtype=dtype, kernels=['naive'])
                execute = spec.make_executor()
                self.assertEqual(execute._native_lhs.ndarray.dtype.name, dtype)
                self.assertEqual(execute._native_rhs.ndarray.dtype.name, dtype)
                self.assertTrue(execute._native_lhs.is_from_python)
                self.assertTrue(execute._native_rhs.is_from_python)
                self.assertTrue(np.shares_memory(
                    execute._lhs_array, execute._native_lhs.ndarray))
                self.assertTrue(np.shares_memory(
                    execute._rhs_array, execute._native_rhs.ndarray))
                comparison = collector._compare(spec, execute)
                self.assertEqual(
                    [item.status for item in comparison.values()],
                    ['measured', 'measured'],
                )
                for item in comparison.values():
                    self.assertIsInstance(item.max_abs_diff, float)
                    self.assertIsInstance(item.relative_diff, float)

    def test_checks_operand_roles_and_broadcast_batches(self):
        cases = (
            ({'shape': [3], 'strides': [1]},
             {'shape': [3], 'strides': [1]}),
            ({'shape': [3], 'strides': [1]},
             {'shape': [3, 2], 'strides': [2, 1]}),
            ({'shape': [2, 3], 'strides': [3, 1]},
             {'shape': [3], 'strides': [1]}),
            ({'shape': [2, 1, 2, 3], 'strides': [6, 6, 3, 1]},
             {'shape': [1, 4, 3, 2], 'strides': [24, 6, 2, 1]}),
        )
        for lhs, rhs in cases:
            with self.subTest(lhs=lhs['shape'], rhs=rhs['shape']):
                spec = make_spec(lhs=lhs, rhs=rhs, kernels=['naive'])
                comparison = collector._compare(
                    spec, spec.make_executor())
                self.assertEqual(
                    [item.status for item in comparison.values()],
                    ['measured', 'measured'],
                )

    def test_checks_exact_strided_and_empty_operands(self):
        operands = (
            {'shape': [2, 3], 'strides': [-3, 1]},
            {'shape': [2, 3], 'strides': [0, 1]},
            {'shape': [2, 3], 'strides': [1, 1]},
            {'shape': [0, 3], 'strides': [99, -2]},
        )
        for lhs in operands:
            with self.subTest(lhs=lhs):
                spec = make_spec(lhs=lhs, kernels=['naive'])
                execute = spec.make_executor()
                self.assertEqual(execute._native_lhs.stride,
                                 tuple(lhs['strides']))
                comparison = collector._compare(spec, execute)
                self.assertEqual(
                    [item.status for item in comparison.values()],
                    ['measured', 'measured'],
                )
                if lhs['shape'][0] == 0:
                    for item in comparison.values():
                        self.assertIsNone(item.max_abs_diff)
                        self.assertIsNone(item.relative_diff)

    def test_marks_real_ineligible_kernel(self):
        spec = make_spec(kernels=['naive', 'winograd'])

        comparison = collector._compare(spec, spec.make_executor())

        self.assertEqual(
            [item.status for item in comparison.values()],
            ['measured', 'ineligible', 'measured'],
        )


class MatmulTimingTC(unittest.TestCase):
    def test_collects_balanced_timings(self):
        expected = [[1.0, 2.0], [3.0, 4.0]]
        execute = FakeExecutor({
            'naive': expected,
            'blas_gemm': [[1.0, 2.0], [3.0, 5.0]],
            'winograd': expected,
            'numpy': expected,
        })
        spec = make_spec(
            kernels=['naive', 'blas_gemm', 'winograd'],
            sampling={'warmups': 1, 'repetitions': 2, 'rounds': 4})

        with unittest.mock.patch.object(
            matmul.MatmulSpec, 'make_executor', return_value=execute,
        ) as prepare:
            comparison = collector._collect(spec, StepClock())
        prepare.assert_called_once_with()

        names = ('naive', 'blas_gemm', 'winograd', 'numpy')
        round_orders = comparison.round_orders
        expected_rows = collector._williams_rows(names)
        self.assertEqual(round_orders, [list(row) for row in expected_rows])
        by_name = {result.name: result for result in comparison.results}
        for name in names:
            self.assertEqual(by_name[name].status, 'measured')
            self.assertEqual(by_name[name].round_elapsed_ns, [100] * 4)
        self.assertEqual(by_name['blas_gemm'].max_abs_diff, 1.0)

        comparison_calls = ['numpy', 'naive', 'blas_gemm', 'winograd']
        warmup_calls = ['numpy', 'naive', 'winograd', 'blas_gemm']
        self.assertEqual(
            execute.calls[:8], comparison_calls + warmup_calls)
        repetitions = spec.sampling.repetitions
        timed_calls = []
        for order in round_orders:
            for name in order:
                timed_calls.extend([name] * repetitions)
        self.assertEqual(execute.calls[8:], timed_calls)

    def test_balances_williams_rows(self):
        max_candidates = len(matmul.MATMUL_KERNELS) + 1
        for count in range(2, max_candidates + 1):
            names = tuple(range(count))
            rows = collector._williams_rows(names)
            repeated_rows = rows * 2
            for row in rows:
                self.assertEqual(set(row), set(names))
            for length in range(1, len(repeated_rows) + 1):
                prefix = repeated_rows[:length]
                for position in range(count):
                    counts = collections.Counter(
                        row[position] for row in prefix)
                    values = [counts[name] for name in names]
                    self.assertLessEqual(max(values) - min(values), 1)

            transitions = collections.Counter()
            for row in rows:
                transitions.update(zip(row, row[1:]))
            directed_pairs = set(itertools.permutations(names, 2))
            expected = 2 if count % 2 else 1
            self.assertEqual(set(transitions), directed_pairs)
            self.assertEqual(set(transitions.values()), {expected})
            if count % 2:
                boundaries = zip(rows, rows[1:] + rows[:1])
                for previous, following in boundaries:
                    self.assertEqual(previous[-1], following[0])

    def test_times_zero_and_one_candidates(self):
        sampling = make_spec(
            sampling={'warmups': 1, 'repetitions': 2, 'rounds': 2},
        ).sampling
        execute = FakeExecutor({'only': [1.0]})

        progress = []
        orders, elapsed = collector._time_candidates(
            execute, (), sampling, StepClock(),
            lambda *args: progress.append(args),
        )
        self.assertEqual(progress, [])
        self.assertEqual(orders, [[], []])
        self.assertEqual(elapsed, {})
        self.assertEqual(execute.calls, [])

        orders, elapsed = collector._time_candidates(
            execute, ('only',), sampling, StepClock(),
            lambda *args: progress.append(args),
        )
        self.assertEqual(progress, [
            ('warmup', 'only', 0, 3),
            ('timing', 'only', 1, 3),
            ('timing', 'only', 2, 3),
            ('timing', 'only', 3, 3),
        ])
        self.assertEqual(orders, [['only'], ['only']])
        self.assertEqual(elapsed, {'only': [100, 100]})
        self.assertEqual(execute.calls, ['only'] * 5)

    def test_skips_unmeasured_kernels(self):
        expected = [[1.0, 2.0], [3.0, 4.0]]
        execute = FakeExecutor({
            'naive': expected,
            'blas_gemm': [[1.0, 2.0, 3.0, 4.0]],
            'winograd': sc.MatmulKernelUnavailable('not eligible'),
            'numpy': expected,
        })
        spec = make_spec(
            kernels=['naive', 'blas_gemm', 'winograd'],
            sampling={'warmups': 1, 'repetitions': 2, 'rounds': 1},
        )

        progress = []
        with unittest.mock.patch.object(
            matmul.MatmulSpec, 'make_executor', return_value=execute,
        ):
            comparison = collector._collect(
                spec, StepClock(), lambda *args: progress.append(args))
        samples = [
            event for event in progress
            if event[0] in ('warmup', 'timing')
        ]
        _, names, completed, totals = zip(*samples)
        self.assertEqual(completed, (0, 1, 2, 3, 4))
        self.assertEqual(set(totals), {4})
        self.assertEqual(set(names), {'naive', 'numpy'})

        by_name = {result.name: result for result in comparison.results}
        self.assertEqual(by_name['blas_gemm'].status, 'invalid')
        self.assertEqual(by_name['blas_gemm'].round_elapsed_ns, [])
        self.assertEqual(by_name['winograd'].status, 'ineligible')
        self.assertEqual(by_name['winograd'].round_elapsed_ns, [])
        self.assertEqual(comparison.round_orders, [['naive', 'numpy']])
        self.assertEqual(collections.Counter(execute.calls), {
            'numpy': 4,
            'naive': 4,
            'blas_gemm': 1,
            'winograd': 1,
        })

    def test_propagates_execution_failure(self):
        expected = np.ones((2, 2), dtype='float64')
        stages = (
            ('warmup', 1, 'numpy', 4),
            ('timing', 0, 'naive', 2),
        )
        for stage, warmups, kernel, total in stages:
            with self.subTest(stage=stage):
                outcomes = (expected, expected, RuntimeError('native bug'))
                execute = unittest.mock.Mock(side_effect=outcomes)
                sampling = {'warmups': warmups, 'repetitions': 1, 'rounds': 1}
                spec = make_spec(kernels=['naive'], sampling=sampling)
                progress = []
                with (
                    unittest.mock.patch.object(
                        matmul.MatmulSpec, 'make_executor',
                        return_value=execute),
                    self.assertRaisesRegex(RuntimeError, 'native bug'),
                ):
                    collector._collect(
                        spec, StepClock(), lambda *args: progress.append(args))
                self.assertEqual(progress[-1], (stage, kernel, 0, total))

    def test_collect_returns_result_model(self):
        spec = make_spec(
            kernels=['naive'],
            sampling={'warmups': 0, 'repetitions': 1, 'rounds': 2},
        )

        result = benchmark.collector.collect(spec)
        self.assertIsInstance(result, results.RunResult)

        self.assertEqual(result.spec, spec)
        self.assertEqual(len(result.round_orders), 2)
        for entry in result.results:
            self.assertEqual(entry.status, 'measured')
            elapsed = entry.round_elapsed_ns
            self.assertEqual(len(elapsed), 2)
            self.assertTrue(all(isinstance(value, int) for value in elapsed))

    def test_collect_requires_spec_interface(self):
        with self.assertRaisesRegex(TypeError, 'BenchmarkSpec'):
            benchmark.collector.collect(make_spec().to_dict())


def make_comparison():
    return {
        'spec': {
            'operation': 'matmul',
            'lhs': {'shape': [2, 3], 'strides': [3, 1]},
            'rhs': {'shape': [3, 2], 'strides': [2, 1]},
            'dtype': 'float64',
            'sampling': {'warmups': 0, 'repetitions': 1, 'rounds': 2},
            'kernels': ['naive', 'winograd'],
        },
        'round_orders': [['naive', 'numpy'], ['numpy', 'naive']],
        'results': [
            {
                'name': 'naive', 'status': 'measured', 'reason': None,
                'max_abs_diff': 0.0, 'relative_diff': 0.0,
                'round_elapsed_ns': [10, 11],
            },
            {
                'name': 'winograd', 'status': 'ineligible',
                'reason': 'unsupported shape', 'max_abs_diff': None,
                'relative_diff': None, 'round_elapsed_ns': [],
            },
            {
                'name': 'numpy', 'status': 'measured', 'reason': None,
                'max_abs_diff': 0.0, 'relative_diff': 0.0,
                'round_elapsed_ns': [12, 13],
            },
        ],
    }


class TimingStatsTC(unittest.TestCase):
    def test_per_call_percentiles(self):
        cases = (([1200, 400, 800], 4, (200.0, 290.0)),
                 ([10, 30], 4, (5.0, 7.25)),
                 ([80], 5, (16.0, 16.0)),
                 ([0, 0], 4, (0.0, 0.0)),
                 ([], 4, (None, None)))
        for elapsed_ns, repetitions, expected in cases:
            with self.subTest(elapsed_ns=elapsed_ns):
                original = elapsed_ns.copy()
                timing = results.TimingStats.from_rounds(
                    elapsed_ns, repetitions)
                self.assertEqual((timing.median, timing.p95), expected)
                self.assertEqual(elapsed_ns, original)


class ResultsTC(unittest.TestCase):
    def test_round_trip(self):
        document = make_comparison()
        result = results.RunResult.from_dict(document)
        with tempfile.TemporaryDirectory() as dirname:
            path = pathlib.Path(dirname) / 'nested' / 'result.json'
            written = results.write_artifact(result, path)
            loaded = results.load_artifact(path)

        self.assertEqual(written, path)
        self.assertEqual(loaded.to_dict(), document)

    def test_model_owns_result_data(self):
        document = make_comparison()
        result = results.RunResult.from_dict(document)
        document['results'][0]['round_elapsed_ns'][0] = 99
        document['round_orders'][0].reverse()
        self.assertEqual(result.to_dict(), make_comparison())

        exported = result.to_dict()
        exported['results'][0]['round_elapsed_ns'][0] = 99
        exported['round_orders'][0].reverse()
        self.assertEqual(result.to_dict(), make_comparison())

    def test_timing_stats_uses_saved_sampling(self):
        document = make_comparison()
        document['spec']['sampling']['repetitions'] = 2
        result = results.RunResult.from_dict(document)

        self.assertEqual(result.timing_stats(), {
            'naive': results.TimingStats(5.25, 5.475),
            'winograd': results.TimingStats(),
            'numpy': results.TimingStats(6.25, 6.475),
        })

    def test_rejects_incomplete_model_before_writing(self):
        result = results.RunResult.from_dict(make_comparison())
        with tempfile.TemporaryDirectory() as dirname:
            path = pathlib.Path(dirname) / 'result.json'
            results.write_artifact(result, path)
            original = path.read_bytes()
            result.results[0].round_elapsed_ns.pop()
            with self.assertRaisesRegex(results.ArtifactError, 'per round'):
                results.write_artifact(result, path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_rejects_inconsistent_artifact_data(self):
        mutations = (
            lambda item: item.__setitem__('extra', None),
            lambda item: item['results'].reverse(),
            lambda item: item['results'][0]['round_elapsed_ns'].pop(),
            lambda item: item['round_orders'][0].remove('numpy'),
            lambda item: item['results'][1].__setitem__(
                'max_abs_diff', 0.0),
            lambda item: item['results'][1].__setitem__('reason', ''),
            lambda item: item['results'][0].__setitem__(
                'relative_diff', float('nan')),
            lambda item: item['results'][0].__setitem__(
                'max_abs_diff', 0),
            lambda item: item['round_orders'][0].__setitem__(0, []),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                document = make_comparison()
                mutate(document)
                with self.assertRaises(results.ArtifactError):
                    results.RunResult.from_dict(document)

    def test_rejects_measured_kernel_when_numpy_is_invalid(self):
        document = make_comparison()
        for result in document['results'][1:]:
            result.update(
                status='invalid', reason='non-finite NumPy reference',
                max_abs_diff=None, relative_diff=None,
                round_elapsed_ns=[])
        for order in document['round_orders']:
            order.remove('numpy')

        with self.assertRaisesRegex(results.ArtifactError, 'NumPy'):
            results.RunResult.from_dict(document)

    def test_failed_replace_preserves_existing_artifact(self):
        original = make_comparison()
        replacement = copy.deepcopy(original)
        replacement['results'][0]['round_elapsed_ns'][0] = 99
        original = results.RunResult.from_dict(original)
        replacement = results.RunResult.from_dict(replacement)
        with tempfile.TemporaryDirectory() as dirname:
            path = pathlib.Path(dirname) / 'result.json'
            results.write_artifact(original, path)
            with unittest.mock.patch.object(
                    results.os, 'replace', side_effect=OSError('failed')):
                with self.assertRaisesRegex(OSError, 'failed'):
                    results.write_artifact(replacement, path)

            self.assertEqual(results.load_artifact(path), original)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_load_validates(self):
        document = make_comparison()
        document['results'][0]['relative_diff'] = float('nan')
        with tempfile.TemporaryDirectory() as dirname:
            path = pathlib.Path(dirname) / 'result.json'
            path.write_text(json.dumps(document), encoding='ascii')

            with self.assertRaisesRegex(results.ArtifactError, 'finite'):
                results.load_artifact(path)


def make_request(output_path):
    return {
        'spec': {
            'operation': 'matmul',
            'lhs': {'shape': [2, 2], 'strides': [2, 1]},
            'rhs': {'shape': [2, 2], 'strides': [2, 1]},
            'dtype': 'float64',
            'sampling': {'warmups': 0, 'repetitions': 1, 'rounds': 1},
            'kernels': ['naive'],
        },
        'output_path': os.fspath(output_path),
    }


class BenchmarkWorkerTC(unittest.TestCase):
    def test_reports_invalid_requests(self):
        valid = make_request('artifact.json')
        cases = (
            ('', 'missing'),
            ('{', 'valid JSON'),
            (json.dumps({**valid, 'extra': None}), 'unknown fields'),
            (json.dumps({**valid, 'output_path': ''}), 'non-empty'),
        )
        for payload, message in cases:
            with self.subTest(message=message):
                stdout = io.StringIO()
                with unittest.mock.patch.object(
                        worker.collector, 'collect') as collect:
                    return_code = worker.run(
                        io.StringIO(payload), stdout)

                self.assertEqual(return_code, 1)
                event = json.loads(stdout.getvalue())
                self.assertEqual(event['type'], 'error')
                self.assertIn(message, event['message'])
                collect.assert_not_called()

    def test_reports_collection_failure(self):
        request = make_request('unused.json')
        stdin = io.StringIO(json.dumps(request))
        stdout = io.StringIO()
        with unittest.mock.patch.object(
            worker.collector, 'collect',
            side_effect=RuntimeError('native failure'),
        ):
            return_code = worker.run(stdin, stdout)

        events = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(return_code, 1)
        self.assertEqual(events, [
            {
                'type': 'progress', 'phase': 'preparing', 'kernel': None,
                'completed': None, 'total': None,
            },
            {
                'type': 'error', 'error_type': 'RuntimeError',
                'message': 'native failure',
            },
        ])

    def test_process_writes_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'artifact.json'
            request = make_request(path)
            process = subprocess.run(
                system.python_command('-m', 'solvcon.benchmark.worker'),
                input=json.dumps(request) + '\n',
                capture_output=True, text=True, check=False, timeout=30,
            )

            self.assertEqual(
                process.returncode, 0, process.stderr or process.stdout,
            )
            events = [json.loads(line)
                      for line in process.stdout.splitlines()]
            rows = (
                ('preparing', None, None, None),
                ('comparison', 'numpy', None, None),
                ('comparison', 'naive', None, None),
                ('timing', 'naive', 0, 2),
                ('timing', 'numpy', 1, 2),
                ('timing', 'numpy', 2, 2),
                ('finishing', None, None, None),
            )
            expected = [
                {
                    'type': 'progress', 'phase': phase, 'kernel': kernel,
                    'completed': completed, 'total': total,
                }
                for phase, kernel, completed, total in rows
            ]
            self.assertEqual(events[:-1], expected)
            self.assertEqual(events[-1], {
                'type': 'result', 'artifact_path': str(path),
            })
            document = results.load_artifact(path).to_dict()
            self.assertEqual(document['spec'], request['spec'])


class BenchmarkProgressTC(unittest.TestCase):
    def test_progress_stays_outside_timed_blocks(self):
        sampling = make_matmul_spec(
            sampling={'warmups': 1, 'repetitions': 2, 'rounds': 1},
        ).sampling
        events = []
        clock = StepClock(events=events)

        orders, elapsed = benchmark.collector._time_candidates(
            lambda name: events.append(name), ('naive',), sampling, clock,
            lambda *args: events.append(args),
        )

        self.assertEqual(events, [
            ('warmup', 'naive', 0, 2),
            'naive',
            ('timing', 'naive', 1, 2),
            'clock', 'naive', 'naive', 'clock',
            ('timing', 'naive', 2, 2),
        ])
        self.assertEqual(orders, [['naive']])
        self.assertEqual(elapsed, {'naive': [100]})

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

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
import time
import unittest
import unittest.mock

import numpy as np

import solvcon as sc
from solvcon import benchmark, system
from solvcon.benchmark import artifact
from solvcon.benchmark import collector
from solvcon.benchmark import matmul
from solvcon.benchmark import spec as benchmark_spec
from solvcon.benchmark import worker


try:
    from PySide6 import QtCore, QtTest, QtWidgets
except ImportError:
    QtWidgets = None
else:
    from solvcon.pilot import _benchmark


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
    def __init__(self, step=100):
        self.value = -step
        self.step = step

    def __call__(self):
        self.value += self.step
        return self.value


class FakeExecutor:
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
                collector, '_CHUNK_SIZE', 4):
            arrays = {
                dtype: collector._make_operand(operand, dtype, 7)
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
                array = collector._make_operand(operand, 'complex64', 0)

                self.assertEqual(array.shape, tuple(data['shape']))
                self.assertEqual(
                    tuple(value // array.itemsize
                          for value in array.strides),
                    strides,
                )
                self.assertEqual(array.dtype.name, 'complex64')
                self.assertTrue(np.all(np.isfinite(array)))

        operand = benchmark_spec.OperandSpec.from_dict(cases[0][0])
        array = collector._make_operand(operand, 'complex64', 0)
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
            'naive': {
                'status': 'measured',
                'reason': None,
                'max_abs_diff': 0.0,
                'relative_diff': 0.0,
            },
            'blas_gemm': {
                'status': 'measured',
                'reason': None,
                'max_abs_diff': 1.0,
                'relative_diff': 0.25,
            },
            'winograd': {
                'status': 'ineligible',
                'reason': 'not eligible',
                'max_abs_diff': None,
                'relative_diff': None,
            },
            'numpy': {
                'status': 'measured',
                'reason': None,
                'max_abs_diff': 0.0,
                'relative_diff': 0.0,
            },
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
                self.assertEqual(result['status'], 'invalid')
                self.assertEqual(result['reason'], reason)
                self.assertIsNone(result['max_abs_diff'])
                self.assertIsNone(result['relative_diff'])

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

        expected = {
            'status': 'invalid',
            'reason': 'non-finite NumPy reference',
            'max_abs_diff': None,
            'relative_diff': None,
        }
        self.assertEqual(comparison, {
            'naive': expected,
            'numpy': expected,
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
                execute = collector._make_executor(spec)
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
                    [item['status'] for item in comparison.values()],
                    ['measured', 'measured'],
                )
                for item in comparison.values():
                    self.assertIsInstance(item['max_abs_diff'], float)
                    self.assertIsInstance(item['relative_diff'], float)

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
                    spec, collector._make_executor(spec))
                self.assertEqual(
                    [item['status'] for item in comparison.values()],
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
                execute = collector._make_executor(spec)
                self.assertEqual(execute._native_lhs.stride,
                                 tuple(lhs['strides']))
                comparison = collector._compare(spec, execute)
                self.assertEqual(
                    [item['status'] for item in comparison.values()],
                    ['measured', 'measured'],
                )
                if lhs['shape'][0] == 0:
                    for item in comparison.values():
                        self.assertIsNone(item['max_abs_diff'])
                        self.assertIsNone(item['relative_diff'])

    def test_marks_real_ineligible_kernel(self):
        spec = make_spec(kernels=['naive', 'winograd'])

        comparison = collector._compare(spec, collector._make_executor(spec))

        self.assertEqual(
            [item['status'] for item in comparison.values()],
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

        comparison = collector._collect(spec, execute, StepClock())

        names = ('naive', 'blas_gemm', 'winograd', 'numpy')
        round_orders = comparison['round_orders']
        expected_rows = collector._williams_rows(names)
        self.assertEqual(round_orders, [list(row) for row in expected_rows])
        by_name = {result['name']: result for result in comparison['results']}
        for name in names:
            self.assertEqual(by_name[name]['status'], 'measured')
            self.assertEqual(
                by_name[name]['round_elapsed_ns'], [100] * 4)
        self.assertEqual(by_name['blas_gemm']['max_abs_diff'], 1.0)

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
            sampling={
                'warmups': 1, 'repetitions': 2, 'rounds': 2,
            }).sampling
        execute = FakeExecutor({'only': [1.0]})

        orders, elapsed = collector._time_candidates(
            execute, (), sampling, StepClock())
        self.assertEqual(orders, [[], []])
        self.assertEqual(elapsed, {})
        self.assertEqual(execute.calls, [])

        orders, elapsed = collector._time_candidates(
            execute, ('only',), sampling, StepClock())
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

        comparison = collector._collect(spec, execute, StepClock())

        by_name = {result['name']: result for result in comparison['results']}
        self.assertEqual(by_name['blas_gemm']['status'], 'invalid')
        self.assertEqual(by_name['blas_gemm']['round_elapsed_ns'], [])
        self.assertEqual(by_name['winograd']['status'], 'ineligible')
        self.assertEqual(by_name['winograd']['round_elapsed_ns'], [])
        self.assertEqual(
            comparison['round_orders'], [['naive', 'numpy']])
        self.assertEqual(collections.Counter(execute.calls), {
            'numpy': 4,
            'naive': 4,
            'blas_gemm': 1,
            'winograd': 1,
        })

    def test_propagates_execution_failure(self):
        expected = np.ones((2, 2), dtype='float64')
        stages = (
            ('warmup', {'warmups': 1, 'repetitions': 1, 'rounds': 1}),
            ('timing', {'warmups': 0, 'repetitions': 1, 'rounds': 1}),
        )
        for stage, sampling in stages:
            with self.subTest(stage=stage):
                execute = unittest.mock.Mock(side_effect=(
                    expected, expected, RuntimeError('native bug')))
                spec = make_spec(kernels=['naive'], sampling=sampling)
                with self.assertRaisesRegex(RuntimeError, 'native bug'):
                    collector._collect(spec, execute, StepClock())

    def test_collect_returns_json_data(self):
        spec = make_spec(
            kernels=['naive'],
            sampling={'warmups': 0, 'repetitions': 1, 'rounds': 2},
        )

        comparison = benchmark.collector.collect(spec)

        self.assertEqual(comparison['spec'], spec.to_dict())
        self.assertEqual(len(comparison['round_orders']), 2)
        for result in comparison['results']:
            self.assertEqual(result['status'], 'measured')
            elapsed = result['round_elapsed_ns']
            self.assertEqual(len(elapsed), 2)
            self.assertTrue(all(isinstance(value, int) for value in elapsed))
        json.dumps(comparison)

    def test_collect_requires_matmul_spec(self):
        with self.assertRaisesRegex(TypeError, 'spec must be a MatmulSpec'):
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


class ArtifactTC(unittest.TestCase):
    def test_round_trip(self):
        document = make_comparison()
        with tempfile.TemporaryDirectory() as dirname:
            path = pathlib.Path(dirname) / 'nested' / 'result.json'
            written = artifact.write_artifact(document, path)
            loaded = artifact.load_artifact(path)

        self.assertEqual(written, path)
        self.assertEqual(loaded, document)

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
                with self.assertRaises(artifact.ArtifactError):
                    artifact.validate_artifact(document)

    def test_rejects_measured_kernel_when_numpy_is_invalid(self):
        document = make_comparison()
        for result in document['results'][1:]:
            result.update(
                status='invalid', reason='non-finite NumPy reference',
                max_abs_diff=None, relative_diff=None,
                round_elapsed_ns=[])
        for order in document['round_orders']:
            order.remove('numpy')

        with self.assertRaisesRegex(artifact.ArtifactError, 'NumPy'):
            artifact.validate_artifact(document)

    def test_failed_replace_preserves_existing_artifact(self):
        original = make_comparison()
        replacement = copy.deepcopy(original)
        replacement['results'][0]['round_elapsed_ns'][0] = 99
        with tempfile.TemporaryDirectory() as dirname:
            path = pathlib.Path(dirname) / 'result.json'
            artifact.write_artifact(original, path)
            with unittest.mock.patch.object(
                    artifact.os, 'replace', side_effect=OSError('failed')):
                with self.assertRaisesRegex(OSError, 'failed'):
                    artifact.write_artifact(replacement, path)

            self.assertEqual(artifact.load_artifact(path), original)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_load_validates(self):
        document = make_comparison()
        document['results'][0]['relative_diff'] = float('nan')
        with tempfile.TemporaryDirectory() as dirname:
            path = pathlib.Path(dirname) / 'result.json'
            path.write_text(json.dumps(document), encoding='ascii')

            with self.assertRaisesRegex(artifact.ArtifactError, 'finite'):
                artifact.load_artifact(path)


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
        stdout = io.StringIO()
        with unittest.mock.patch.object(
                worker.collector, 'collect',
                side_effect=RuntimeError('native failure')):
            return_code = worker.run(io.StringIO(
                json.dumps(make_request('unused.json'))), stdout)

        event = json.loads(stdout.getvalue())
        self.assertEqual(return_code, 1)
        self.assertEqual(event['error_type'], 'RuntimeError')
        self.assertEqual(event['message'], 'native failure')

    def test_process_writes_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'artifact.json'
            request = make_request(path)
            process = subprocess.run(
                system.python_command('-m', 'solvcon.benchmark.worker'),
                input=json.dumps(request) + '\n',
                capture_output=True, text=True, check=False)

            self.assertEqual(
                process.returncode, 0, process.stderr or process.stdout)
            events = [json.loads(line)
                      for line in process.stdout.splitlines()]
            self.assertEqual(events[:-1], [
                {'type': 'progress', 'phase': phase, 'kernel': kernel}
                for phase, kernel in (('comparison', 'numpy'),
                                      ('comparison', 'naive'),
                                      ('timing', 'naive'), ('timing', 'numpy'))
            ])
            self.assertEqual(events[-1], {
                'type': 'result',
                'artifact_path': str(path),
            })
            document = artifact.load_artifact(path)
            self.assertEqual(document['spec'], request['spec'])


class BenchmarkProgressTC(unittest.TestCase):
    def test_progress_stays_outside_timed_blocks(self):
        sampling = make_matmul_spec(sampling={
            'warmups': 1, 'repetitions': 2, 'rounds': 1,
        }).sampling
        events = []

        def clock():
            events.append('clock')
            return len(events)

        orders, elapsed = benchmark.collector._time_candidates(
            lambda name: events.append(name), ('naive',), sampling, clock,
            lambda phase, name: events.append((phase, name)))

        self.assertEqual(events, [
            ('warmup', 'naive'), 'naive', ('timing', 'naive'),
            'clock', 'naive', 'naive', 'clock',
        ])
        self.assertEqual(orders, [['naive']])
        self.assertEqual(elapsed, {'naive': [3]})


@unittest.skipIf(QtWidgets is None, 'PySide6 is not installed')
class BenchmarkControlTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = (QtWidgets.QApplication.instance()
                   or QtWidgets.QApplication([]))

    def setUp(self):
        self.control = _benchmark.BenchmarkControl()
        self.directory = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.directory.name) / 'result.json'
        self.spec = make_matmul_spec(
            kernels=['naive'],
            sampling={'warmups': 1, 'repetitions': 2, 'rounds': 2})
        self.events = []
        self.control.completed.connect(
            lambda path: self.events.append(('result', path)))
        self.control.failed.connect(
            lambda message: self.events.append(('error', message)))
        self.control.stopped.connect(
            lambda: self.events.append(('stopped', None)))

    def tearDown(self):
        self.control.stop()
        self.wait_for(lambda: not self.control.running)
        self.control.close()
        self.control.deleteLater()
        self.directory.cleanup()

    def wait_for(self, predicate):
        deadline = time.monotonic() + 15
        while not predicate() and time.monotonic() < deadline:
            QtTest.QTest.qWait(10)
        self.assertTrue(predicate(), self.control.status.text())

    def start_script(self, script):
        script = 'import sys\nsys.stdin.readline()\n' + script
        command = system.python_command('-c', script)
        with unittest.mock.patch.object(
                system, 'python_command', return_value=command):
            self.control.start(self.spec, self.path)

    def assert_finished(self, kind):
        self.wait_for(lambda: not self.control.running)
        self.assertEqual([event[0] for event in self.events], [kind])
        self.assertEqual(self.control._process.state(),
                         QtCore.QProcess.ProcessState.NotRunning)
        self.assertFalse(self.control.stop_button.isEnabled())
        self.assertFalse(self.control._timer.isActive())

    def test_collect_and_repeat(self):
        for _ in range(2):
            self.events.clear()
            self.control.start(self.spec, self.path)
            with self.assertRaisesRegex(RuntimeError, 'already running'):
                self.control.start(self.spec, self.path)
            self.assert_finished('result')
            self.assertEqual(self.events[0][1], str(self.path))
            self.assertEqual(artifact.load_artifact(self.path)['spec'],
                             self.spec.to_dict())

    def test_progress_stop_and_recover(self):
        event = json.dumps({'type': 'progress', 'phase': 'timing',
                            'kernel': 'naive'})
        self.start_script(
            'import sys, time\n'
            f'sys.stdout.write({event[:12]!r})\n'
            'sys.stdout.flush()\n'
            'time.sleep(0.15)\n'
            f'print({event[12:]!r}, flush=True)\n'
            'time.sleep(60)\n')
        self.wait_for(lambda: self.control.status.text() == 'Timing: naive')
        self.wait_for(
            lambda: self.control.elapsed.text() != 'Elapsed: 0.0 s')
        self.assertEqual(self.events, [])
        self.control.stop_button.click()
        self.assert_finished('stopped')
        self.events.clear()
        self.control.start(self.spec, self.path)
        self.assert_finished('result')

    def assert_error_recovery(self, cases):
        for script, message in cases:
            with self.subTest(message=message):
                self.events.clear()
                self.start_script(script)
                self.assert_finished('error')
                self.assertIn(message, self.events[0][1])
        self.events.clear()
        self.control.start(self.spec, self.path)
        self.assert_finished('result')

    def test_protocol_error_recovery(self):
        self.assert_error_recovery((
            ("print('not json', flush=True)\nimport time\ntime.sleep(60)",
             'protocol'),
            ("print('{}', flush=True)", 'unknown'),
            ('print(\'{"type":"error","message":"bad spec"}\', '
             'flush=True)\nimport time\ntime.sleep(60)', 'bad spec'),
        ))

    def test_exit_error_recovery(self):
        self.assert_error_recovery((
            ("print('{}', end='', flush=True)", 'incomplete'),
            ('pass', 'without a result'),
            ("import sys\nsys.stderr.write('native crash')\nsys.exit(3)",
             'native crash'),
            ('print(\'{"type":"result","artifact_path":"fake"}\', '
             'flush=True)\nimport sys\nsys.exit(3)', 'code 3'),
        ))

    def test_crash(self):
        self.start_script('import time\ntime.sleep(60)')
        self.wait_for(lambda: self.control._process.state() ==
                      QtCore.QProcess.ProcessState.Running)
        self.control._process.kill()
        self.assert_finished('error')

    def test_failed_start_recovers_from_signal(self):
        process_errors = []
        self.control._process.errorOccurred.connect(process_errors.append)

        def restart(message):
            # Reentering QProcess inside its error signal can block Qt.
            if not process_errors:
                return
            self.control.failed.disconnect(restart)
            self.events.clear()
            self.control.start(self.spec, self.path)

        self.control.failed.connect(restart)
        with unittest.mock.patch.object(system, 'python_command',
                                        return_value=['/missing/worker']):
            self.control.start(self.spec, self.path)
        self.assert_finished('result')

    def test_stop_during_start_and_close(self):
        for action in (self.control.stop, self.control.close):
            self.events.clear()
            self.start_script('import time\ntime.sleep(60)')
            action()
            self.assert_finished('stopped')


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

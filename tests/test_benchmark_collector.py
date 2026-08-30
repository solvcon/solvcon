# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import collections
import itertools
import json
import unittest
import unittest.mock

import numpy as np

import solvcon as sc
import solvcon.benchmark as benchmark
from solvcon.benchmark import collector
from solvcon.benchmark import matmul
from solvcon.benchmark import spec as benchmark_spec


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


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import copy
import io
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

from solvcon import benchmark
from solvcon.benchmark import artifact
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

    @unittest.skipIf(
        '_solvcon' in sys.builtin_module_names,
        'the pilot carries _solvcon in the executable, so the interpreter '
        'this test spawns cannot import it')
    def test_process_writes_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'artifact.json'
            request = make_request(path)
            python = shutil.which('python3')
            self.assertIsNotNone(python)
            process = subprocess.run(
                [python, '-m', 'solvcon.benchmark.worker'],
                input=json.dumps(request) + '\n',
                capture_output=True, text=True, check=False)

            self.assertEqual(
                process.returncode, 0, process.stderr or process.stdout)
            self.assertEqual(json.loads(process.stdout), {
                'type': 'result',
                'artifact_path': str(path),
            })
            document = artifact.load_artifact(path)
            self.assertEqual(document['spec'], request['spec'])


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

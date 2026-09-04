# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Validate and atomically store benchmark artifacts."""

import json
import math
import os
import pathlib
import tempfile

from . import matmul
from . import spec as benchmark_spec


class ArtifactError(ValueError):
    """Report an invalid benchmark artifact."""


def _require_exact_fields(value, label, expected_fields):
    if not isinstance(value, dict):
        raise ArtifactError(f'{label} must be an object')
    if any(not isinstance(field, str) for field in value):
        raise ArtifactError(f'{label} field names must be strings')
    actual = set(value)
    expected = set(expected_fields)
    missing = sorted(expected - actual)
    if missing:
        raise ArtifactError(f'{label} is missing fields: {missing}')
    unknown = sorted(actual - expected)
    if unknown:
        raise ArtifactError(f'{label} has unknown fields: {unknown}')


def _require_list(value, label):
    if not isinstance(value, list):
        raise ArtifactError(f'{label} must be an array')


def _require_nonnegative_int(value, label):
    is_integer = isinstance(value, int) and not isinstance(value, bool)
    if not is_integer or value < 0:
        raise ArtifactError(f'{label} must be a non-negative integer')


def _require_diff(value, label, allow_none=False):
    if value is None and allow_none:
        return
    if not isinstance(value, float):
        raise ArtifactError(f'{label} must be a floating-point number')
    if not math.isfinite(value) or value < 0:
        raise ArtifactError(f'{label} must be finite and non-negative')


def _validate_result(result, label, rounds, output_empty):
    expected_fields = (
        'name', 'status', 'reason', 'max_abs_diff', 'relative_diff',
        'round_elapsed_ns')
    _require_exact_fields(result, label, expected_fields)
    if not isinstance(result['name'], str) or not result['name']:
        raise ArtifactError(f'{label}.name must be a non-empty string')
    status = result['status']
    if status not in ('measured', 'invalid', 'ineligible'):
        raise ArtifactError(f'{label}.status is unsupported')

    elapsed = result['round_elapsed_ns']
    elapsed_label = f'{label}.round_elapsed_ns'
    _require_list(elapsed, elapsed_label)
    for index, value in enumerate(elapsed):
        value_label = f'{elapsed_label}[{index}]'
        _require_nonnegative_int(value, value_label)

    max_abs_diff = result['max_abs_diff']
    relative_diff = result['relative_diff']
    has_differences = max_abs_diff is not None or relative_diff is not None
    if status == 'measured':
        if result['reason'] is not None:
            raise ArtifactError(f'{label}.reason must be null when measured')
        _require_diff(max_abs_diff, f'{label}.max_abs_diff', output_empty)
        relative_label = f'{label}.relative_diff'
        _require_diff(relative_diff, relative_label, allow_none=True)
        if output_empty and has_differences:
            raise ArtifactError(f'{label} differences must be null when empty')
        if len(elapsed) != rounds:
            raise ArtifactError(f'{label} must contain one time per round')
    else:
        if not isinstance(result['reason'], str) or not result['reason']:
            raise ArtifactError(f'{label}.reason must explain its status')
        if has_differences or elapsed:
            raise ArtifactError(f'{label} must not contain measurements')


def validate_artifact(artifact):
    """Validate one benchmark artifact."""
    expected_fields = ('spec', 'round_orders', 'results')
    _require_exact_fields(artifact, 'artifact', expected_fields)
    try:
        parsed_spec = matmul.MatmulSpec.from_dict(artifact['spec'])
    except benchmark_spec.SpecError as exc:
        raise ArtifactError(f'invalid artifact spec: {exc}') from exc
    if artifact['spec'] != parsed_spec.to_dict():
        raise ArtifactError('artifact spec must use canonical JSON data')

    expected_names = parsed_spec.kernels + ('numpy',)
    results = artifact['results']
    _require_list(results, 'artifact.results')
    rounds = parsed_spec.sampling.rounds
    output_empty = 0 in parsed_spec.output_shape
    for index, result in enumerate(results):
        result_label = f'artifact.results[{index}]'
        _validate_result(result, result_label, rounds, output_empty)
    names = tuple(result['name'] for result in results)
    if names != expected_names:
        raise ArtifactError('artifact results do not match spec kernels')
    numpy_status = results[-1]['status']
    if numpy_status == 'ineligible':
        raise ArtifactError('the NumPy result cannot be ineligible')
    if numpy_status == 'invalid':
        statuses = {result['status'] for result in results}
        if statuses != {'invalid'}:
            raise ArtifactError(
                'all results must be invalid when NumPy is invalid')

    measured_names = {result['name'] for result in results
                      if result['status'] == 'measured'}

    round_orders = artifact['round_orders']
    _require_list(round_orders, 'artifact.round_orders')
    if len(round_orders) != rounds:
        raise ArtifactError('artifact must contain every timing round')
    for index, order in enumerate(round_orders):
        order_label = f'artifact.round_orders[{index}]'
        _require_list(order, order_label)
        names_are_valid = all(isinstance(name, str) and name for name in order)
        if not names_are_valid:
            raise ArtifactError(f'{order_label} has an invalid name')
        same_size = len(order) == len(measured_names)
        same_names = set(order) == measured_names
        if not same_size or not same_names:
            raise ArtifactError(f'{order_label} must order measured results')
    return artifact


def write_artifact(artifact, path):
    """Atomically replace a path with one validated artifact."""
    validate_artifact(artifact)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf8', dir=path.parent,
                delete=False) as stream:
            temporary_path = pathlib.Path(stream.name)
            json.dump(artifact, stream, indent=2, sort_keys=True,
                      allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def load_artifact(path):
    """Load and validate one artifact."""
    with pathlib.Path(path).open(encoding='utf8') as stream:
        artifact = json.load(stream)
    return validate_artifact(artifact)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

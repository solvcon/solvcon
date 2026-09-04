# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Collect one exact matmul comparison without storing it."""

import time

import numpy as np

import solvcon as sc
from . import matmul


_CHUNK_SIZE = 1 << 20


def _storage_bounds(operand):
    """Return inclusive element offsets, using (0, 0) for empty layouts."""
    if any(extent == 0 for extent in operand.shape):
        return 0, 0
    minimum = 0
    maximum = 0
    for extent, stride in zip(operand.shape, operand.strides):
        displacement = (extent - 1) * stride
        minimum += min(0, displacement)
        maximum += max(0, displacement)
    return minimum, maximum


def _fill_random_components(storage, seed):
    components = storage.view(storage.real.dtype.name)
    generator = np.random.default_rng(seed)
    values = np.empty(
        min(components.size, _CHUNK_SIZE), dtype='float64')
    for start in range(0, components.size, _CHUNK_SIZE):
        stop = min(start + _CHUNK_SIZE, components.size)
        chunk = values[:stop - start]
        generator.random(chunk.shape, dtype='float64', out=chunk)
        chunk *= 2
        chunk -= 1
        components[start:stop] = chunk


def _make_operand(operand, dtype, seed):
    dtype = np.dtype(dtype)
    minimum, maximum = _storage_bounds(operand)
    storage = np.empty(maximum - minimum + 1, dtype=dtype.name)
    _fill_random_components(storage, seed)

    return np.ndarray(
        shape=operand.shape,
        dtype=dtype.name,
        buffer=storage,
        offset=-minimum * dtype.itemsize,
        strides=tuple(stride * dtype.itemsize for stride in operand.strides),
    )


class _MatmulExecutor:
    def __init__(self, spec, lhs, rhs):
        array_type = sc.SimpleArray.typed_class(spec.dtype)
        self._lhs_array = lhs
        self._rhs_array = rhs
        self._native_lhs = array_type(array=lhs)
        self._native_rhs = array_type(array=rhs)

    def __call__(self, name):
        if name == 'numpy':
            return np.matmul(self._lhs_array, self._rhs_array)
        return self._native_lhs.matmul(self._native_rhs, kernel=name)


def _make_executor(spec):
    lhs = _make_operand(spec.lhs, spec.dtype, 0)
    rhs = _make_operand(spec.rhs, spec.dtype, 1)
    return _MatmulExecutor(spec, lhs, rhs)


def _difference_metrics(result, reference):
    if result.size == 0:
        return None, None
    max_abs_diff = 0.0
    reference_scale = 0.0
    chunks = np.nditer(
        (result, reference),
        flags=('external_loop', 'buffered'),
        op_flags=(('readonly',), ('readonly',)),
        order='C', buffersize=_CHUNK_SIZE,
    )
    for result_chunk, reference_chunk in chunks:
        max_abs_diff = max(
            max_abs_diff,
            float(np.max(np.abs(result_chunk - reference_chunk))))
        reference_scale = max(
            reference_scale, float(np.max(np.abs(reference_chunk))))
    if reference_scale:
        return max_abs_diff, max_abs_diff / reference_scale
    return max_abs_diff, 0.0 if max_abs_diff == 0 else None


def _compare_result(execute, name, reference):
    try:
        result = np.atleast_1d(execute(name))
    except sc.MatmulKernelUnavailable as exc:
        return {
            'status': 'ineligible',
            'reason': str(exc),
            'max_abs_diff': None,
            'relative_diff': None,
        }
    if result.shape != reference.shape:
        reason = 'shape mismatch'
    elif result.dtype != reference.dtype:
        reason = 'dtype mismatch'
    elif not np.all(np.isfinite(result)):
        reason = 'non-finite values'
    else:
        max_abs_diff, relative_diff = _difference_metrics(result, reference)
        return {
            'status': 'measured',
            'reason': None,
            'max_abs_diff': max_abs_diff,
            'relative_diff': relative_diff,
        }
    return {
        'status': 'invalid',
        'reason': reason,
        'max_abs_diff': None,
        'relative_diff': None,
    }


def _compare(spec, execute):
    """Compare every requested kernel with the NumPy reference."""
    reference = np.atleast_1d(execute('numpy'))
    if reference.shape != spec.output_shape:
        raise RuntimeError('NumPy reference shape does not match spec')
    if reference.dtype.name != spec.dtype:
        raise RuntimeError('NumPy reference dtype does not match spec')
    if not np.all(np.isfinite(reference)):
        return {
            name: {
                'status': 'invalid',
                'reason': 'non-finite NumPy reference',
                'max_abs_diff': None,
                'relative_diff': None,
            }
            for name in spec.kernels + ('numpy',)
        }
    comparison = {
        name: _compare_result(execute, name, reference)
        for name in spec.kernels
    }
    comparison['numpy'] = {
        'status': 'measured',
        'reason': None,
        'max_abs_diff': 0.0 if reference.size else None,
        'relative_diff': 0.0 if reference.size else None,
    }
    return comparison


def _williams_rows(names):
    """Build a Williams schedule for benchmark candidates.

    Each row contains every candidate once. For four candidates:

        A B D C
        B C A D
        C D B A
        D A C B

    Across any prefix of rows, occurrence counts at each position differ by
    at most one. A complete design balances immediate predecessors within
    rows. This reduces bias from cache, thermal, or frequency state left by
    the previous candidate. For two or more candidates, even counts use one
    row per candidate; odd counts require both orientations and twice as many
    rows.

    See https://doi.org/10.1071/CH9490149.
    """

    count = len(names)
    if count < 2:
        return (tuple(names),)

    pattern = []
    for position in range(count):
        if position % 2:
            candidate_index = (position + 1) // 2
        else:
            candidate_index = count - position // 2
        pattern.append(candidate_index % count)

    base_rows = []
    for offset in range(count):
        row = tuple((index + offset) % count for index in pattern)
        base_rows.append(row)

    if count % 2:
        # Both orientations balance predecessors for odd counts. This row
        # traversal also makes adjacent rows meet on the same candidate.
        forward_step = (count + 1) // 2
        reverse_step = count - forward_step
        rows = []
        for offset in range(count):
            row_index = (offset * forward_step) % count
            rows.append(base_rows[row_index])
        for offset in range(1, count + 1):
            row_index = (offset * reverse_step) % count
            rows.append(tuple(reversed(base_rows[row_index])))
    else:
        rows = base_rows

    named_rows = []
    for row in rows:
        named_rows.append(tuple(names[index] for index in row))
    return tuple(named_rows)


def _time_candidates(execute, names, sampling, clock):
    """Time one repetition block per candidate in each scheduled round.

    A round selects one Williams row. Each candidate gets one clock interval
    containing all repetitions, so scheduling stays outside the sample.
    """

    rows = _williams_rows(names)
    # Warmups use the rows immediately before the first timed row, keeping
    # both phases on one cyclic schedule.
    for warmup_index in range(-sampling.warmups, 0):
        for name in rows[warmup_index % len(rows)]:
            execute(name)

    elapsed_by_name = {name: [] for name in names}
    round_orders = []
    for round_index in range(sampling.rounds):
        row = rows[round_index % len(rows)]
        round_orders.append(list(row))
        for name in row:
            start = clock()
            for _ in range(sampling.repetitions):
                execute(name)
            elapsed_by_name[name].append(int(clock() - start))
    return round_orders, elapsed_by_name


def _collect(spec, execute, clock):
    names = spec.kernels + ('numpy',)
    comparison = _compare(spec, execute)
    measured_names = tuple(
        name for name in names if comparison[name]['status'] == 'measured')
    round_orders, elapsed_by_name = _time_candidates(
        execute, measured_names, spec.sampling, clock)
    results = []
    for name in names:
        result = {'name': name, **comparison[name]}
        result['round_elapsed_ns'] = elapsed_by_name.get(name, [])
        results.append(result)
    return {
        'spec': spec.to_dict(),
        'round_orders': round_orders,
        'results': results,
    }


def collect(spec):
    """Collect one exact matmul comparison from a validated specification."""

    if not isinstance(spec, matmul.MatmulSpec):
        raise TypeError('spec must be a MatmulSpec')
    return _collect(
        spec, _make_executor(spec), time.perf_counter_ns)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Collect one exact kernel comparison without storing it."""

import time

import numpy as np

from . import operation
from . import results


_CHUNK_SIZE = 1 << 20


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
    except execute.unavailable_error as exc:
        return results.KernelResult(name, 'ineligible', reason=str(exc))
    if result.shape != reference.shape:
        reason = 'shape mismatch'
    elif result.dtype != reference.dtype:
        reason = 'dtype mismatch'
    elif not np.all(np.isfinite(result)):
        reason = 'non-finite values'
    else:
        max_abs_diff, relative_diff = _difference_metrics(result, reference)
        return results.KernelResult(
            name, 'measured', max_abs_diff=max_abs_diff,
            relative_diff=relative_diff)
    return results.KernelResult(name, 'invalid', reason=reason)


def _ignore_progress(phase, name, completed=None, total=None):
    pass


def _compare(spec, execute, progress=_ignore_progress):
    """Compare every requested kernel with the NumPy reference."""
    progress('comparison', 'numpy')
    reference = np.atleast_1d(execute('numpy'))
    if reference.shape != spec.output_shape:
        raise RuntimeError('NumPy reference shape does not match spec')
    if reference.dtype.name != spec.dtype:
        raise RuntimeError('NumPy reference dtype does not match spec')
    if not np.all(np.isfinite(reference)):
        return {
            name: results.KernelResult(
                name, 'invalid', reason='non-finite NumPy reference')
            for name in spec.kernels + ('numpy',)
        }
    comparison = {}
    for name in spec.kernels:
        progress('comparison', name)
        comparison[name] = _compare_result(execute, name, reference)
    difference = 0.0 if reference.size else None
    comparison['numpy'] = results.KernelResult(
        'numpy', 'measured', max_abs_diff=difference, relative_diff=difference)
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


def _time_candidates(execute, names, sampling, clock,
                     progress=_ignore_progress):
    """Time one repetition block per candidate in each scheduled round.

    A round selects one Williams row. Each candidate gets one clock interval
    containing all repetitions, so scheduling stays outside the sample.
    """

    rows = _williams_rows(names)
    completed = 0
    total = len(names) * (sampling.warmups + sampling.rounds)
    # Warmups use the rows immediately before the first timed row, keeping
    # both phases on one cyclic schedule.
    for warmup_index in range(-sampling.warmups, 0):
        for name in rows[warmup_index % len(rows)]:
            progress('warmup', name, completed, total)
            execute(name)
            completed += 1

    elapsed_by_name = {name: [] for name in names}
    round_orders = []
    for round_index in range(sampling.rounds):
        row = rows[round_index % len(rows)]
        round_orders.append(list(row))
        for name in row:
            progress('timing', name, completed, total)
            start = clock()
            for _ in range(sampling.repetitions):
                execute(name)
            elapsed_by_name[name].append(int(clock() - start))
            completed += 1
    if names:
        progress('timing', name, completed, total)
    return round_orders, elapsed_by_name


def _collect(spec, clock, progress=_ignore_progress):
    execute = spec.make_executor()
    names = spec.kernels + ('numpy',)
    comparison = _compare(spec, execute, progress)
    measured_names = tuple(
        name for name in names if comparison[name].status == 'measured')
    round_orders, elapsed_by_name = _time_candidates(
        execute, measured_names, spec.sampling, clock, progress)
    for name, result in comparison.items():
        result.round_elapsed_ns = elapsed_by_name.get(name, [])
    return results.RunResult(spec, round_orders, list(comparison.values()))


def collect(spec, *, progress=_ignore_progress):
    """Collect a comparison, reporting progress outside timed blocks.

    Call progress with (phase, kernel), plus completed and total work units
    during sampling. Each warmup call and timed repetition block is one unit.

    :return: A complete :class:`~solvcon.benchmark.results.RunResult`.
    """

    if not isinstance(spec, operation.BenchmarkSpec):
        raise TypeError('spec must implement BenchmarkSpec')
    return _collect(spec, time.perf_counter_ns, progress)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

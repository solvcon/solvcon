# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Prepare exact operands and compare matmul kernel results."""

import numpy as np

import solvcon as sc


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


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

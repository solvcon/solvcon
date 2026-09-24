# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Define exact Matmul inputs and execution on shared operand storage."""

import dataclasses

import numpy as np

import solvcon as sc

from . import spec


_CHUNK_SIZE = 1 << 20

MATMUL_DTYPE_SIZES = {
    'float32': 4,
    'float64': 8,
    'complex64': 8,
    'complex128': 16,
}
MATMUL_DTYPES = tuple(MATMUL_DTYPE_SIZES)
MATMUL_KERNELS = (
    'naive',
    'blas_dot',
    'blas_gevm',
    'blas_gemv',
    'blas_gemm',
    'winograd',
)


def _broadcast_shape(lhs, rhs):
    result = []
    for offset in range(1, max(len(lhs), len(rhs)) + 1):
        left = lhs[-offset] if offset <= len(lhs) else 1
        right = rhs[-offset] if offset <= len(rhs) else 1
        if left == right:
            extent = left
        elif left == 1:
            extent = right
        elif right == 1:
            extent = left
        else:
            raise spec.SpecError(
                'matmul batch dimensions do not match')
        result.append(extent)
    return tuple(reversed(result))


@dataclasses.dataclass(frozen=True)
class MatmulInputs:
    """Validate matmul inputs independently of sampling and kernel choices."""

    lhs: spec.OperandSpec
    rhs: spec.OperandSpec
    dtype: str

    def __post_init__(self):
        if not isinstance(self.dtype, str) or self.dtype not in MATMUL_DTYPES:
            raise spec.SpecError(f'unsupported dtype: {self.dtype!r}')
        if not isinstance(self.lhs, spec.OperandSpec):
            raise spec.SpecError('lhs must be an OperandSpec')
        if not isinstance(self.rhs, spec.OperandSpec):
            raise spec.SpecError('rhs must be an OperandSpec')
        itemsize = MATMUL_DTYPE_SIZES[self.dtype]
        spec._validate_byte_layout(self.lhs, 'lhs', itemsize)
        spec._validate_byte_layout(self.rhs, 'rhs', itemsize)
        self._validate_shape()
        spec._validate_logical_shape(
            self.output_shape, 'output', itemsize)

    def _validate_shape(self):
        rhs_inner_axis = -1 if len(self.rhs.shape) == 1 else -2
        if self.lhs.shape[-1] != self.rhs.shape[rhs_inner_axis]:
            raise spec.SpecError(
                'matmul contraction dimensions do not match')
        _broadcast_shape(self.lhs.shape[:-2], self.rhs.shape[:-2])

    @property
    def output_shape(self):
        lhs_vector = len(self.lhs.shape) == 1
        rhs_vector = len(self.rhs.shape) == 1
        if lhs_vector and rhs_vector:
            return (1,)
        batch = _broadcast_shape(self.lhs.shape[:-2], self.rhs.shape[:-2])
        rows = () if lhs_vector else (self.lhs.shape[-2],)
        columns = () if rhs_vector else (self.rhs.shape[-1],)
        return batch + rows + columns

    def kernel_eligibility(self):
        """Map kernels to rejection reasons, or None when eligible.

        Only metadata is examined; no operand or result storage is allocated.
        """
        array_type = sc.SimpleArray.typed_class(self.dtype)
        return array_type.matmul_kernel_eligibility(
            self.lhs.shape, self.lhs.strides, self.rhs.shape, self.rhs.strides)


@dataclasses.dataclass(frozen=True)
class MatmulSpec(MatmulInputs):
    """Describe one exact comparison of matmul kernels."""

    OPERATION = 'matmul'

    sampling: spec.Sampling
    kernels: tuple

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.sampling, spec.Sampling):
            raise spec.SpecError('sampling must be a Sampling')
        if not isinstance(self.kernels, (list, tuple)):
            raise spec.SpecError('kernels must be an array')
        kernels = tuple(self.kernels)
        if not kernels:
            raise spec.SpecError('kernels must not be empty')
        if any(not isinstance(kernel, str) or not kernel
               for kernel in kernels):
            raise spec.SpecError('kernels must contain non-empty strings')
        if len(kernels) != len(set(kernels)):
            raise spec.SpecError('kernels must not contain duplicates')
        unknown = sorted(set(kernels) - set(MATMUL_KERNELS))
        if unknown:
            raise spec.SpecError(f'unsupported kernels: {unknown}')
        object.__setattr__(self, 'kernels', kernels)

    @classmethod
    def from_dict(cls, data):
        fields = (
            'operation', 'lhs', 'rhs', 'dtype', 'sampling', 'kernels',
        )
        spec._require_fields(data, 'spec', fields)
        operation = data['operation']
        if operation != cls.OPERATION:
            raise spec.SpecError(
                f'unsupported operation: {operation!r}')
        return cls(
            lhs=spec.OperandSpec.from_dict(data['lhs']),
            rhs=spec.OperandSpec.from_dict(data['rhs']),
            dtype=data['dtype'],
            sampling=spec.Sampling.from_dict(data['sampling']),
            kernels=data['kernels'],
        )

    def to_dict(self):
        return {
            'operation': self.OPERATION,
            'lhs': self.lhs.to_dict(),
            'rhs': self.rhs.to_dict(),
            'dtype': self.dtype,
            'sampling': self.sampling.to_dict(),
            'kernels': list(self.kernels),
        }

    def make_executor(self):
        """Allocate deterministic operands outside measured calls."""
        lhs = _make_operand(self.lhs, self.dtype, 0)
        rhs = _make_operand(self.rhs, self.dtype, 1)
        return MatmulExecutor(self, lhs, rhs)


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


class MatmulExecutor:
    """Execute one kernel on shared NumPy and native operand storage."""

    unavailable_error = sc.MatmulKernelUnavailable

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


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

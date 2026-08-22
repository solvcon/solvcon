# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Validate exact matmul benchmark specifications without allocating arrays."""

import dataclasses

from . import spec


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


def _batch_shape(shape):
    return () if len(shape) == 1 else shape[:-2]


@dataclasses.dataclass(frozen=True)
class MatmulSpec:
    """Describe one exact comparison of matmul kernels."""

    OPERATION = 'matmul'

    lhs: spec.OperandSpec
    rhs: spec.OperandSpec
    dtype: str
    sampling: spec.Sampling
    kernels: tuple

    def __post_init__(self):
        if not isinstance(self.dtype, str) or self.dtype not in MATMUL_DTYPES:
            raise spec.SpecError(f'unsupported dtype: {self.dtype!r}')
        if not isinstance(self.lhs, spec.OperandSpec):
            raise spec.SpecError('lhs must be an OperandSpec')
        if not isinstance(self.rhs, spec.OperandSpec):
            raise spec.SpecError('rhs must be an OperandSpec')
        if not isinstance(self.sampling, spec.Sampling):
            raise spec.SpecError('sampling must be a Sampling')
        itemsize = MATMUL_DTYPE_SIZES[self.dtype]
        spec._validate_byte_layout(self.lhs, 'lhs', itemsize)
        spec._validate_byte_layout(self.rhs, 'rhs', itemsize)
        if not isinstance(self.kernels, (list, tuple)):
            raise spec.SpecError('kernels must be an array')
        kernels = tuple(self.kernels)
        if not kernels:
            raise spec.SpecError('kernels must not be empty')
        if any(not isinstance(kernel, str) or not kernel
               for kernel in kernels):
            raise spec.SpecError(
                'kernels must contain non-empty strings')
        if len(kernels) != len(set(kernels)):
            raise spec.SpecError(
                'kernels must not contain duplicates')
        unknown = sorted(set(kernels) - set(MATMUL_KERNELS))
        if unknown:
            raise spec.SpecError(f'unsupported kernels: {unknown}')
        object.__setattr__(self, 'kernels', kernels)
        self._validate_shape()
        spec._validate_logical_shape(
            self.output_shape, 'output', itemsize)

    def _validate_shape(self):
        rhs_inner_axis = -1 if len(self.rhs.shape) == 1 else -2
        if self.lhs.shape[-1] != self.rhs.shape[rhs_inner_axis]:
            raise spec.SpecError(
                'matmul contraction dimensions do not match')
        _broadcast_shape(
            _batch_shape(self.lhs.shape), _batch_shape(self.rhs.shape))

    @property
    def output_shape(self):
        lhs_vector = len(self.lhs.shape) == 1
        rhs_vector = len(self.rhs.shape) == 1
        if lhs_vector and rhs_vector:
            return (1,)
        batch = _broadcast_shape(
            _batch_shape(self.lhs.shape), _batch_shape(self.rhs.shape))
        rows = () if lhs_vector else (self.lhs.shape[-2],)
        columns = () if rhs_vector else (self.rhs.shape[-1],)
        return batch + rows + columns

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


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

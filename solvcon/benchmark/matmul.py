# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Validate exact matmul benchmark requests without allocating arrays."""

import dataclasses

from . import request


MATMUL_DTYPES = ('float32', 'float64', 'complex64', 'complex128')
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
            raise request.RequestError(
                'matmul batch dimensions do not match')
        result.append(extent)
    return tuple(reversed(result))


def _batch_shape(shape):
    return () if len(shape) == 1 else shape[:-2]


@dataclasses.dataclass(frozen=True)
class MatmulRequest:
    """Describe one exact comparison of matmul kernels."""

    OPERATION = 'matmul'

    lhs: request.OperandSpec
    rhs: request.OperandSpec
    dtype: str
    sampling: request.Sampling
    kernels: tuple

    def __post_init__(self):
        if not isinstance(self.dtype, str) or self.dtype not in MATMUL_DTYPES:
            raise request.RequestError(f'unsupported dtype: {self.dtype!r}')
        if not isinstance(self.lhs, request.OperandSpec):
            raise request.RequestError('lhs must be an OperandSpec')
        if not isinstance(self.rhs, request.OperandSpec):
            raise request.RequestError('rhs must be an OperandSpec')
        if not isinstance(self.sampling, request.Sampling):
            raise request.RequestError('sampling must be a Sampling')
        if not isinstance(self.kernels, (list, tuple)):
            raise request.RequestError('kernels must be an array')
        kernels = tuple(self.kernels)
        if not kernels:
            raise request.RequestError('kernels must not be empty')
        if any(not isinstance(kernel, str) or not kernel
               for kernel in kernels):
            raise request.RequestError(
                'kernels must contain non-empty strings')
        if len(kernels) != len(set(kernels)):
            raise request.RequestError(
                'kernels must not contain duplicates')
        unknown = sorted(set(kernels) - set(MATMUL_KERNELS))
        if unknown:
            raise request.RequestError(f'unsupported kernels: {unknown}')
        object.__setattr__(self, 'kernels', kernels)
        self._validate_shape()

    def _validate_shape(self):
        rhs_inner_axis = -1 if len(self.rhs.shape) == 1 else -2
        if self.lhs.shape[-1] != self.rhs.shape[rhs_inner_axis]:
            raise request.RequestError(
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
        request._require_fields(data, 'request', fields)
        operation = data['operation']
        if operation != cls.OPERATION:
            raise request.RequestError(
                f'unsupported operation: {operation!r}')
        return cls(
            lhs=request.OperandSpec.from_dict(data['lhs']),
            rhs=request.OperandSpec.from_dict(data['rhs']),
            dtype=data['dtype'],
            sampling=request.Sampling.from_dict(data['sampling']),
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

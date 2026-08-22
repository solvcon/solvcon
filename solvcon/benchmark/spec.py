# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Types shared by operation benchmark specifications."""

import dataclasses
import sys


class SpecError(ValueError):
    """Report an invalid benchmark specification."""


def _require_fields(data, name, fields):
    if not isinstance(data, dict):
        raise SpecError(f'{name} must be an object')
    if any(not isinstance(field, str) for field in data):
        raise SpecError(f'{name} field names must be strings')
    missing = sorted(set(fields) - set(data))
    if missing:
        raise SpecError(f'{name} is missing fields: {missing}')
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise SpecError(f'{name} has unknown fields: {unknown}')


def _require_int(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(f'{name} must be an integer')
    if minimum is not None and value < minimum:
        raise SpecError(f'{name} must be at least {minimum}')
    if maximum is not None and value > maximum:
        raise SpecError(f'{name} must be at most {maximum}')
    return value


def _int_tuple(value, name, minimum=None, maximum=None):
    if not isinstance(value, (list, tuple)):
        raise SpecError(f'{name} must be an array')
    return tuple(
        _require_int(item, f'{name}[{index}]', minimum, maximum)
        for index, item in enumerate(value)
    )


def _validate_logical_shape(shape, name, itemsize):
    byte_size = itemsize
    for extent in shape:
        byte_size *= max(extent, 1)
        if byte_size > sys.maxsize:
            raise SpecError(
                f'{name} logical byte size exceeds {sys.maxsize}')


def _validate_byte_layout(operand, name, itemsize):
    _validate_logical_shape(operand.shape, name, itemsize)
    minimum_stride = (-sys.maxsize - 1) // itemsize
    maximum_stride = sys.maxsize // itemsize
    for index, stride in enumerate(operand.strides):
        _require_int(
            stride, f'{name} byte stride[{index}]',
            minimum_stride, maximum_stride)

    if any(extent == 0 for extent in operand.shape):
        return
    minimum_offset = 0
    maximum_offset = 0
    for extent, stride in zip(operand.shape, operand.strides):
        displacement = (extent - 1) * stride
        minimum_offset += min(0, displacement)
        maximum_offset += max(0, displacement)
    _require_int(
        minimum_offset * itemsize, f'{name} minimum byte offset',
        -sys.maxsize, sys.maxsize)
    _require_int(
        maximum_offset * itemsize, f'{name} maximum byte offset',
        -sys.maxsize, sys.maxsize)
    byte_span = (maximum_offset - minimum_offset + 1) * itemsize
    _require_int(
        byte_span, f'{name} byte span', 1, sys.maxsize)


@dataclasses.dataclass(frozen=True)
class OperandSpec:
    """Describe one operand with full shape and element strides."""

    shape: tuple
    strides: tuple

    def __post_init__(self):
        shape = _int_tuple(self.shape, 'shape', 0, sys.maxsize)
        strides = _int_tuple(
            self.strides, 'strides', -sys.maxsize - 1, sys.maxsize)
        if not shape:
            raise SpecError('an operand must have at least one axis')
        if len(shape) != len(strides):
            raise SpecError('shape and strides must have the same length')
        object.__setattr__(self, 'shape', shape)
        object.__setattr__(self, 'strides', strides)

    @classmethod
    def from_dict(cls, data):
        _require_fields(data, 'operand', ('shape', 'strides'))
        return cls(shape=data['shape'], strides=data['strides'])

    def to_dict(self):
        return {
            'shape': list(self.shape),
            'strides': list(self.strides),
        }


@dataclasses.dataclass(frozen=True)
class Sampling:
    """Define exact setup and measurement call counts."""

    warmups: int
    repetitions: int
    rounds: int

    def __post_init__(self):
        _require_int(self.warmups, 'sampling.warmups', 0)
        _require_int(self.repetitions, 'sampling.repetitions', 1)
        _require_int(self.rounds, 'sampling.rounds', 1)

    @classmethod
    def from_dict(cls, data):
        fields = ('warmups', 'repetitions', 'rounds')
        _require_fields(data, 'sampling', fields)
        return cls(**{field: data[field] for field in fields})

    def to_dict(self):
        return dataclasses.asdict(self)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

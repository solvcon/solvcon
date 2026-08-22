# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Types shared by operation benchmark requests."""

import dataclasses


class RequestError(ValueError):
    """Report an invalid benchmark request."""


def _require_fields(data, name, fields):
    if not isinstance(data, dict):
        raise RequestError(f'{name} must be an object')
    missing = sorted(set(fields) - set(data))
    if missing:
        raise RequestError(f'{name} is missing fields: {missing}')
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise RequestError(f'{name} has unknown fields: {unknown}')


def _require_int(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError(f'{name} must be an integer')
    if minimum is not None and value < minimum:
        raise RequestError(f'{name} must be at least {minimum}')
    return value


def _int_tuple(value, name, minimum=None):
    if not isinstance(value, (list, tuple)):
        raise RequestError(f'{name} must be an array')
    return tuple(
        _require_int(item, f'{name}[{index}]', minimum)
        for index, item in enumerate(value)
    )


@dataclasses.dataclass(frozen=True)
class OperandSpec:
    """Describe one operand with full shape and element strides."""

    shape: tuple
    strides: tuple

    def __post_init__(self):
        shape = _int_tuple(self.shape, 'shape', 0)
        strides = _int_tuple(self.strides, 'strides')
        if not shape:
            raise RequestError('an operand must have at least one axis')
        if len(shape) != len(strides):
            raise RequestError('shape and strides must have the same length')
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

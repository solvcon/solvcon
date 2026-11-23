# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The colors of the x-t diagram."""

import numpy as np

__all__ = [
    'colormap',
]

# Viridis at nine stops; perceptually even and readable on both grounds.
_VIRIDIS = np.array([[0x44, 0x01, 0x54], [0x48, 0x28, 0x78],
                     [0x3e, 0x4a, 0x89], [0x31, 0x68, 0x8e],
                     [0x26, 0x82, 0x8e], [0x1f, 0x9e, 0x89],
                     [0x35, 0xb7, 0x79], [0x6e, 0xce, 0x58],
                     [0xfd, 0xe7, 0x25]], dtype='float64') / 255.0


def colormap(fractions):
    """Return the viridis colors, rows of red, green, and blue in [0, 1],
    at ``fractions`` in [0, 1]; NaN takes the low end, and an infinity
    the end it points to."""
    fractions = np.nan_to_num(np.asarray(fractions, dtype='float64'),
                              nan=0.0, posinf=1.0, neginf=0.0)
    stops = np.linspace(0.0, 1.0, len(_VIRIDIS), dtype='float64')
    fractions = np.clip(fractions, 0.0, 1.0)
    return np.stack([np.interp(fractions, stops, _VIRIDIS[:, it])
                     for it in range(3)], axis=-1)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

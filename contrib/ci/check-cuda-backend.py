#!/usr/bin/env python3
# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Make sure the build under test carries the real CUDA FFT backend.

`FourierTransform.cuda_available()` returns False on a host without a GPU
in the real backend and in the stub alike, so the two look the same to the
test suite: every CUDA test skips either way, and a stub build keeps the
CUDA lane green while proving nothing. Tell them apart by the failure
message, which only the real backend blames on the missing device.

Run this from the repository root after the extension is built. It exits
0 when the build is the real backend, or when a GPU is present and the
backend works.
"""

import sys

import solvcon as sc

DEVICE_MESSAGE = "no usable CUDA device"


def main():
    if sc.FourierTransform.cuda_available():
        print("CUDA device present; the backend is the real one.")
        return 0

    inp = sc.SimpleArrayComplex128(8, sc.complex128())
    out = sc.SimpleArrayComplex128(8, sc.complex128())
    try:
        sc.FourierTransform.fft(inp, out, backend=sc.FourierBackend.cuda)
    except RuntimeError as err:
        if DEVICE_MESSAGE not in str(err):
            print("CUDA fft raised %r; expected %r. The build compiled the "
                  "stub instead of the CUDA backend."
                  % (str(err), DEVICE_MESSAGE), file=sys.stderr)
            return 1
    else:
        print("CUDA fft did not raise without a device.", file=sys.stderr)
        return 1

    print("No CUDA device; the backend is the real one and reports it.")
    return 0


if __name__ == '__main__':
    sys.exit(main())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

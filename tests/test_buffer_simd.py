# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import platform
import unittest

import numpy as np

import solvcon


class SimdDispatchTC(unittest.TestCase):
    # Without this guard, missing NEON dispatch on aarch64 would silently route
    # every SIMD operation to the scalar path -- correctness tests would still
    # pass and the regression would be invisible.
    def test_neon_active_on_aarch64(self):
        # _simd_feature is intentionally private: the underlying detector only
        # reflects the dispatched backend on aarch64 today, so it is reached
        # through the C++ module rather than the public solvcon namespace.
        feature = solvcon.core._impl._simd_feature()
        if platform.machine() in ("arm64", "aarch64"):
            self.assertEqual(feature, "NEON")
        else:
            self.skipTest("_simd_feature() = " + feature)


class SimdTransformBinaryTC(unittest.TestCase):
    # Each n targets a distinct SIMD code path (int32: 4 lanes per block):
    #   n=1,3  -- below one lane width: pure scalar path, no vector block
    #   n=4    -- exactly one block: no scalar tail
    #   n=5    -- one block + 1-element tail
    #   n=8    -- two blocks: no scalar tail
    #   n=17   -- four blocks + 1-element tail: multi-block with remainder
    # n=0 is omitted because SimpleArray does not accept zero-length shapes.
    def test_add_int32_covers_all_shapes(self):
        for n in (1, 3, 4, 5, 8, 17):
            a_vals = np.arange(n, dtype=np.int32)
            b_vals = np.array([2 * i + 1 for i in range(n)], dtype=np.int32)
            a = solvcon.SimpleArrayInt32(array=a_vals)
            b = solvcon.SimpleArrayInt32(array=b_vals)
            out = a.add_simd(b)
            for i in range(n):
                self.assertEqual(
                    out[i], int(a_vals[i]) + int(b_vals[i]),
                    msg="n=%d i=%d" % (n, i))

    def test_sub_mul_div_float(self):
        # one NEON float lane (4) + 3-element tail
        n = 7
        a_vals = np.array([float(i + 10) for i in range(n)], dtype=np.float32)
        b_vals = np.array([float(i + 1) for i in range(n)], dtype=np.float32)
        a = solvcon.SimpleArrayFloat32(array=a_vals)
        b = solvcon.SimpleArrayFloat32(array=b_vals)

        sub_out = a.sub_simd(b)
        mul_out = a.mul_simd(b)
        div_out = a.div_simd(b)
        for i in range(n):
            self.assertAlmostEqual(sub_out[i], a_vals[i] - b_vals[i], places=6)
            self.assertAlmostEqual(mul_out[i], a_vals[i] * b_vals[i], places=6)
            self.assertAlmostEqual(div_out[i], a_vals[i] / b_vals[i], places=6)

    # vmulq has no int64 overload; SFINAE in the NEON path must route int64
    # multiply to the scalar generic implementation
    def test_int64_mul_falls_back_to_generic(self):
        a = solvcon.SimpleArrayInt64(
            array=np.array([1, 2, 3, 4, 5], dtype=np.int64))
        b = solvcon.SimpleArrayInt64(
            array=np.array([10, 20, 30, 40, 50], dtype=np.int64))
        out = a.mul_simd(b)
        expected = [10, 40, 90, 160, 250]
        for i, want in enumerate(expected):
            self.assertEqual(out[i], want)


class SimdTransformInplaceBinaryTC(unittest.TestCase):
    # The in-place SIMD wrappers (iadd_simd, isub_simd, imul_simd,
    # idiv_simd) must return the receiver itself, not a copy, so that
    # the fluent (chained) API mutates in place across every call.
    def test_inplace_simd_returns_self(self):
        a = solvcon.SimpleArrayFloat32(
            array=np.array([4.0, 6.0, 8.0], dtype='float32'))
        b = solvcon.SimpleArrayFloat32(
            array=np.array([2.0, 2.0, 2.0], dtype='float32'))
        self.assertIs(a.iadd_simd(b), a)

        a = solvcon.SimpleArrayFloat32(
            array=np.array([4.0, 6.0, 8.0], dtype='float32'))
        b = solvcon.SimpleArrayFloat32(
            array=np.array([2.0, 2.0, 2.0], dtype='float32'))
        self.assertIs(a.isub_simd(b), a)

        a = solvcon.SimpleArrayFloat32(
            array=np.array([4.0, 6.0, 8.0], dtype='float32'))
        b = solvcon.SimpleArrayFloat32(
            array=np.array([2.0, 2.0, 2.0], dtype='float32'))
        self.assertIs(a.imul_simd(b), a)

        a = solvcon.SimpleArrayFloat32(
            array=np.array([4.0, 6.0, 8.0], dtype='float32'))
        b = solvcon.SimpleArrayFloat32(
            array=np.array([2.0, 2.0, 2.0], dtype='float32'))
        self.assertIs(a.idiv_simd(b), a)

    def test_inplace_simd_fluent_chaining(self):
        n = 17  # four NEON blocks plus a 1-element scalar tail
        a_vals = np.array(
            [float(i + 4) for i in range(n)], dtype='float32')
        b_vals = np.array(
            [float(i + 1) for i in range(n)], dtype='float32')
        a = solvcon.SimpleArrayFloat32(array=a_vals.copy())
        b = solvcon.SimpleArrayFloat32(array=b_vals.copy())

        chained = a.iadd_simd(b).isub_simd(b).imul_simd(b).idiv_simd(b)

        expected = a_vals.copy()
        expected = expected + b_vals
        expected = expected - b_vals
        expected = expected * b_vals
        expected = expected / b_vals

        self.assertIs(chained, a)
        for i in range(n):
            self.assertAlmostEqual(a[i], expected[i], places=5)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

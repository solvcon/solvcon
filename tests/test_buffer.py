# Copyright (c) 2020, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import numpy as np

import solvcon


class ConcreteBufferBasicTC(unittest.TestCase):

    def test_ConcreteBuffer(self):

        buf = solvcon.ConcreteBuffer(10)
        ndarr = np.array(buf, copy=False)

        # initialization
        for it in range(len(buf)):
            buf[it] = it

        self.assertEqual(10, buf.nbytes)
        self.assertEqual(10, len(buf))

        self.assertEqual(np.int8, ndarr.dtype)

        with self.assertRaisesRegex(
                IndexError,
                "ConcreteBuffer: index 10 is out of bounds with size 10"
        ):
            buf[10]
        with self.assertRaisesRegex(
                IndexError, "index 10 is out of bounds for axis 0 with size 10"
        ):
            ndarr[10]

        buf2 = buf.clone()
        for it in range(len(buf)):
            buf[it] = 100 + it
        self.assertEqual(list(buf), ndarr.tolist())
        self.assertEqual(list(range(100, 110)), ndarr.tolist())
        self.assertEqual(list(range(10)), list(buf2))

        ndarr2 = buf2.ndarray
        buf2[5] = 19
        self.assertEqual(19, ndarr2[5])

    def test_ConcreteBuffer_from_ndarray(self):

        buf = solvcon.ConcreteBuffer(24)
        self.assertFalse(buf.is_from_python)

        ndarr = np.arange(24, dtype='float64').reshape((2, 3, 4))

        buf = solvcon.ConcreteBuffer(array=ndarr)
        self.assertEqual(ndarr.nbytes, buf.nbytes)
        self.assertTrue(buf.is_from_python)

        # The data buffer is shared.
        self.assertFalse((ndarr == 0).all())
        buf.ndarray.fill(0)
        self.assertTrue((ndarr == 0).all())


class ConcreteBufferAlignmentTC(unittest.TestCase):
    """
    Test memory alignment support for ConcreteBuffer.
    Valid alignment values are: 0 (default, no alignment), 16, 32, or 64 bytes.
    """

    def test_default_alignment(self):
        buffer = solvcon.ConcreteBuffer(256)  # default alignment 0
        self.assertEqual(0, buffer.alignment)
        self.assertEqual(256, buffer.nbytes)

    def test_alignment_creation(self):
        buffer = solvcon.ConcreteBuffer(256, 16)
        self.assertEqual(16, buffer.alignment)
        self.assertEqual(256, buffer.nbytes)

        buffer = solvcon.ConcreteBuffer(512, 32)
        self.assertEqual(32, buffer.alignment)
        self.assertEqual(512, buffer.nbytes)

        buffer = solvcon.ConcreteBuffer(1024, 64)
        self.assertEqual(64, buffer.alignment)
        self.assertEqual(1024, buffer.nbytes)

    def test_alignment_validation_valid_values(self):
        solvcon.ConcreteBuffer(256, 16)
        solvcon.ConcreteBuffer(512, 32)
        solvcon.ConcreteBuffer(1024, 64)

    def test_alignment_validation_invalid_values(self):
        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::ConcreteBuffer: alignment must be 0, 16, 32, or 64, but got 17"  # noqa E501
        ):
            solvcon.ConcreteBuffer(256, 17)

        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::ConcreteBuffer: alignment must be 0, 16, 32, or 64, but got 100"  # noqa E501
        ):
            solvcon.ConcreteBuffer(256, 100)

        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::ConcreteBuffer: alignment must be 0, 16, 32, or 64, but got 128"  # noqa E501
        ):
            solvcon.ConcreteBuffer(1024, 128)

    def test_alignment_size_validation(self):
        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::allocate: size .* must be a multiple of alignment 64"  # noqa E501
        ):
            solvcon.ConcreteBuffer(5, 64)

        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::allocate: size .* must be a multiple of alignment 16"  # noqa E501
        ):
            solvcon.ConcreteBuffer(100, 16)

        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::allocate: size .* must be a multiple of alignment 32"  # noqa E501
        ):
            solvcon.ConcreteBuffer(200, 32)

    def test_alignment_with_zero_size(self):
        buffer = solvcon.ConcreteBuffer(0, 0)
        self.assertEqual(0, buffer.alignment)
        self.assertEqual(0, buffer.nbytes)

        buffer = solvcon.ConcreteBuffer(0, 16)
        self.assertEqual(16, buffer.alignment)
        self.assertEqual(0, buffer.nbytes)

    def test_alignment_with_ndarray(self):
        buffer = solvcon.ConcreteBuffer(256, 16)
        self.assertEqual(16, buffer.alignment)

        ndarr = buffer.ndarray
        self.assertEqual(256, ndarr.nbytes)

        buffer[0] = 42
        self.assertEqual(42, ndarr[0])

    def test_alignment_with_clone(self):
        buffer = solvcon.ConcreteBuffer(256, 32)
        for it in range(len(buffer)):
            buffer[it] = it % 128

        cloned = buffer.clone()
        self.assertEqual(32, cloned.alignment)
        self.assertEqual(256, cloned.nbytes)
        self.assertEqual(list(buffer), list(cloned))

        buffer[0] = 99
        self.assertEqual(99, buffer[0])
        self.assertEqual(0, cloned[0])


class BufferExpanderBasicTC(unittest.TestCase):

    def test_BufferExpander(self):
        ep = solvcon.BufferExpander(10)
        self.assertEqual(10, ep.capacity)
        self.assertEqual(10, len(ep))

        ep = solvcon.BufferExpander()
        self.assertEqual(0, ep.capacity)
        self.assertEqual(0, len(ep))

        with self.assertRaisesRegex(
                IndexError,
                "BufferExpander: index 0 is out of bounds with size 0"
        ):
            ep[0]

        ep.reserve(10)
        self.assertEqual(10, ep.capacity)
        self.assertEqual(0, len(ep))  # size unchanged

        with self.assertRaisesRegex(
                IndexError,
                "BufferExpander: index 0 is out of bounds with size 0"
        ):
            ep[0]

        ep.expand(10)
        self.assertEqual(10, ep.capacity)
        self.assertEqual(10, len(ep))  # size changed

        ep[9]  # should not raise an exception
        with self.assertRaisesRegex(
                IndexError,
                "BufferExpander: index 10 is out of bounds with size 10"
        ):
            ep[10]

        # initialize
        for it in range(len(ep)):
            ep[it] = it

        self.assertFalse(ep.is_concrete)
        cbuf = ep.as_concrete()
        self.assertTrue(ep.is_concrete)
        self.assertEqual(10, len(cbuf))
        self.assertEqual(list(range(10)), list(cbuf))

        # prove cbuf and gbuf share memory
        for it in range(len(cbuf)):
            cbuf[it] = it + 10
        self.assertEqual(list(i + 10 for i in range(10)), list(ep))

    def test_BufferExpanderFromConcreteBuffer(self):
        buf = solvcon.ConcreteBuffer(10)
        for it in range(len(buf)):
            buf[it] = it

        ep = solvcon.BufferExpander(buf)
        self.assertEqual(10, ep.capacity)
        self.assertEqual(10, len(ep))
        for it in range(len(ep)):
            ep[it] = it + 100

        for it in range(len(buf)):
            self.assertEqual(buf[it] + 100, ep[it])


class SimpleArrayBasicTC(unittest.TestCase):

    def test_SimpleArray(self):

        sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
        ndarr = np.array(sarr, copy=False)

        self.assertEqual(2 * 3 * 4 * 8, sarr.nbytes)
        self.assertEqual((2, 3, 4), sarr.shape)
        self.assertEqual((12, 4, 1), sarr.stride)  # number of skip elements

        np.ravel(ndarr)[:] = np.arange(24)  # initialize contents

        self.assertEqual(24, len(sarr))  # number of elements
        # Multi-dimensional interface.
        v = 0
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    self.assertEqual(v, sarr[i, j, k])
                    v += 1
        v = 100
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    sarr[i, j, k] = v
                    v += 1

        with self.assertRaisesRegex(
                RuntimeError,
                "SimpleArray: shape byte count 184 differs from buffer 192"
        ):
            sarr.reshape(23)

        sarr2 = sarr.reshape(24)
        self.assertEqual((24,), sarr2.shape)
        self.assertEqual(np.arange(100, 124).tolist(),
                         [sarr2[i] for i in range(24)])
        self.assertEqual(np.arange(100, 124).tolist(), list(sarr2))

        ndarr2 = sarr2.ndarray
        for i in range(24):
            sarr2[i] = 200 + i
        self.assertEqual(np.arange(200, 224).tolist(), ndarr2.tolist())

        self.assertEqual((1, 24), sarr.reshape((1, 24)).shape)
        self.assertEqual((12, 2), sarr.reshape((12, 2)).shape)
        self.assertEqual((2, 2, 2, 3), sarr.reshape((2, 2, 2, 3)).shape)

    def test_SimpleArray_from_numpy_negative_stride(self):
        ndarr = np.arange(2 * 3, dtype='float64').reshape((2, 3))
        view = ndarr[::-1, ::-1]
        sarr = solvcon.SimpleArrayFloat64(array=view)

        self.assertEqual((2, 3), sarr.shape)
        self.assertEqual((-3, -1), sarr.stride)
        np.testing.assert_array_equal(view, sarr.ndarray)
        self.assertEqual(np.sum(view), sarr.sum())

        rhs = np.arange(2 * 3, dtype='float64').reshape((2, 3)) + 100
        sarr[:, :] = rhs
        np.testing.assert_array_equal(rhs, view)
        np.testing.assert_array_equal(rhs, sarr.ndarray)

        row_major = sarr.to_row_major()
        self.assertTrue(row_major.is_c_contiguous)
        np.testing.assert_array_equal(rhs, row_major.ndarray)

        sarr.ndarray[0, 0] = 100
        self.assertEqual(100, ndarr[1, 2])

    def test_SimpleArray_clone(self):
        sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
        sarr.fill(2.0)
        sarr_ref = sarr
        sarr_clone = sarr.clone()

        self.assertTrue(sarr_ref is sarr)
        np.testing.assert_equal(sarr_ref.ndarray[...], sarr.ndarray[...])

        self.assertFalse(sarr_clone is sarr)
        np.testing.assert_equal(sarr_clone.ndarray[...], sarr.ndarray[...])

        sarr[0, 0, 3] = 3.0
        self.assertEqual(sarr_ref[0, 0, 3], 3.0)
        self.assertEqual(sarr_clone[0, 0, 3], 2.0)

    def test_SimpleArray_transpose(self):
        def check_equal(sarr, ndarr):
            self.assertEqual(sarr.shape, ndarr.shape)
            shape = sarr.shape
            for idx1 in range(shape[0]):
                for idx2 in range(shape[1]):
                    for idx3 in range(shape[2]):
                        for idx4 in range(shape[3]):
                            ndnum = ndarr[idx1, idx2, idx3, idx4]
                            sarrnum = sarr[idx1, idx2, idx3, idx4]
                            self.assertEqual(ndnum, sarrnum)

        ndarr = np.arange(2 * 3 * 4 * 5, dtype='float64')
        ndarr = ndarr.reshape((2, 3, 4, 5))
        ndarrT = ndarr.transpose()

        sarr = solvcon.SimpleArrayFloat64(array=ndarr)
        sarr2 = sarr.transpose()
        check_equal(sarr, ndarrT)
        check_equal(sarr2, ndarrT)
        self.assertEqual(memoryview(sarr), memoryview(sarr2))

        sarr = solvcon.SimpleArrayFloat64(array=ndarr)
        sarr2 = sarr.transpose(inplace=False)
        check_equal(sarr, ndarr)
        check_equal(sarr2, ndarrT)
        self.assertNotEqual(memoryview(sarr), memoryview(sarr2))

        sarr = solvcon.SimpleArrayFloat64(array=ndarr)
        sarr2 = sarr.T
        check_equal(sarr, ndarr)
        check_equal(sarr2, ndarrT)
        self.assertNotEqual(memoryview(sarr), memoryview(sarr2))

        ndarrT = ndarr.transpose(0, 3, 2, 1)

        sarr = solvcon.SimpleArrayFloat64(array=ndarr)
        sarr2 = sarr.transpose((0, 3, 2, 1))
        check_equal(sarr, ndarrT)
        check_equal(sarr2, ndarrT)
        self.assertEqual(memoryview(sarr), memoryview(sarr2))

        sarr = solvcon.SimpleArrayFloat64(array=ndarr)
        sarr2 = sarr.transpose((0, 3, 2, 1), inplace=False)
        check_equal(sarr, ndarr)
        check_equal(sarr2, ndarrT)
        self.assertNotEqual(memoryview(sarr), memoryview(sarr2))

    def test_SimpleArray_transpose_copy(self):
        # Physical (deep-copy) transpose variants.  Cover 1D, 2D, 3D, 4D;
        # square / non-square; C-contiguous / F-contiguous input.

        def make_sarr(ndarr):
            return solvcon.SimpleArrayFloat64(array=ndarr.copy())

        def values_equal(sarr, ref):
            np.testing.assert_array_equal(np.array(sarr, copy=False), ref)

        shapes = [
            (7,),
            (3, 5),
            (4, 4),
            (2, 3, 4),
            (2, 3, 4, 5),
        ]

        for shape in shapes:
            ndarr = np.arange(
                int(np.prod(shape)), dtype="float64").reshape(shape)

            # transpose_copy(): equivalent to numpy a.T then a fresh C-contig.
            sarr = make_sarr(ndarr)
            self.assertTrue(sarr.is_c_contiguous)
            tc = sarr.transpose_copy()
            self.assertEqual(tc.shape, ndarr.T.shape)
            self.assertTrue(tc.is_c_contiguous)
            values_equal(tc, np.ascontiguousarray(ndarr.T))
            # Source unchanged.
            values_equal(sarr, ndarr)

            # to_row_major(): same shape and values, C-contiguous.
            rm = sarr.to_row_major()
            self.assertEqual(rm.shape, ndarr.shape)
            self.assertTrue(rm.is_c_contiguous)
            values_equal(rm, ndarr)

            # to_column_major(): same shape and values, F-contiguous.
            cm = sarr.to_column_major()
            self.assertEqual(cm.shape, ndarr.shape)
            self.assertTrue(cm.is_f_contiguous)
            values_equal(cm, ndarr)

            # transpose(copy=True): in-place physical transpose.
            mutated = make_sarr(ndarr)
            mutated.transpose(copy=True)
            self.assertEqual(mutated.shape, ndarr.T.shape)
            self.assertTrue(mutated.is_c_contiguous)
            values_equal(mutated, np.ascontiguousarray(ndarr.T))

            # Round trip: transpose_copy twice gives the original.
            roundtrip = sarr.transpose_copy().transpose_copy()
            self.assertEqual(roundtrip.shape, ndarr.shape)
            values_equal(roundtrip, ndarr)

            # Mutating the source after the copy does not affect the copy.
            tc_before = np.array(tc, copy=True)
            np.array(sarr, copy=False).flat[0] = -42.0
            values_equal(tc, tc_before)

        # F-contiguous source: take a transpose view to get non-C-contig
        # storage, then verify the copy variants still produce correct output.
        for shape in [(3, 5), (2, 3, 4), (2, 3, 4, 5)]:
            ndarr = np.arange(
                int(np.prod(shape)), dtype="float64").reshape(shape)

            sarr = make_sarr(ndarr)
            # transpose(inplace=False) returns a view that swaps shape/stride;
            # the resulting array is F-contiguous w.r.t. its new shape.
            fview = sarr.transpose(inplace=False)
            self.assertEqual(fview.shape, ndarr.T.shape)
            self.assertTrue(fview.is_f_contiguous)
            self.assertFalse(fview.is_c_contiguous)

            # transpose_copy on an F-contig array should yield a C-contig
            # buffer whose contents are the (further) transpose, i.e. ndarr.
            tc = fview.transpose_copy()
            self.assertEqual(tc.shape, ndarr.shape)
            self.assertTrue(tc.is_c_contiguous)
            values_equal(tc, ndarr)

            # to_row_major on an F-contig array reorders bytes.
            rm = fview.to_row_major()
            self.assertEqual(rm.shape, ndarr.T.shape)
            self.assertTrue(rm.is_c_contiguous)
            values_equal(rm, ndarr.T)

            # to_column_major on an F-contig array is essentially a clone.
            cm = fview.to_column_major()
            self.assertEqual(cm.shape, ndarr.T.shape)
            self.assertTrue(cm.is_f_contiguous)
            values_equal(cm, ndarr.T)

    def test_SimpleArray_ghost_1d(self):

        sarr = solvcon.SimpleArrayFloat64(4 * 3 * 2)
        ndarr = np.array(sarr, copy=False)
        ndarr[:] = np.arange(24)  # initialize contents

        sarr.nghost = 10

        self.assertTrue(sarr.has_ghost)
        self.assertEqual(10, sarr.nghost)
        self.assertEqual(14, sarr.nbody)

        v = 0
        for i in range(-10, 14):
            self.assertEqual(v, sarr[i])
            v += 1

        self.assertEqual(23, sarr[-11])
        self.assertEqual(10, sarr[-24])
        self.assertEqual(0, sarr[-34])

    def test_SimpleArray_ghost_1d_out_of_range(self):
        sarr = solvcon.SimpleArrayFloat64(4 * 3 * 2)
        sarr.nghost = 10

        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray: index -35 < -nghost - shape\[0\]: -34"
        ):
            sarr[-35]

        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray: index 14 >= 14 \(shape\[0\]: 24 - nghost: 10\)"
        ):
            sarr[14]

        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray: index -35 < -nghost - shape\[0\]: -34"
        ):
            sarr[-35] = 1

        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray: index 14 >= 14 \(shape\[0\]: 24 - nghost: 10\)"
        ):
            sarr[14] = 1

    def test_SimpleArray_ghost_1d_shifted_negative_index(self):
        sarr = solvcon.SimpleArrayInt8(8)
        sarr.ndarray[:] = np.arange(8, dtype='int8')
        sarr.nghost = 3

        self.assertEqual(5, sarr.nbody)
        self.assertEqual(7, sarr[-4])
        self.assertEqual(0, sarr[-3])
        self.assertEqual(0, sarr[-11])
        sarr[-4] = 70
        sarr[-3] = 30
        self.assertEqual(70, sarr.ndarray[7])
        self.assertEqual(30, sarr.ndarray[0])

    def test_SimpleArray_ghost_1d_strided(self):
        base = np.arange(12, dtype='float64')
        strided = base[::2]
        sarr = solvcon.SimpleArrayFloat64(array=strided)
        sarr.nghost = 2

        self.assertEqual(2, sarr.stride[0])
        self.assertEqual(0, sarr[-2])
        self.assertEqual(10, sarr[-3])
        sarr[-3] = 200
        self.assertEqual(200, base[10])

    def test_SimpleArray_ghost_invalid_setting(self):
        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray: cannot set nghost 11 > shape\(0\) 10"
        ):
            sarr = solvcon.SimpleArrayInt8(10)
            sarr.nghost = 11

        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray: cannot set nghost 1 > 0 to an empty array"
        ):
            sarr = solvcon.SimpleArrayInt8(())
            sarr.nghost = 1

    def test_SimpleArray_ghost_md(self):

        sarr = solvcon.SimpleArrayFloat64((4, 3, 2))
        ndarr = np.array(sarr, copy=False)
        np.ravel(ndarr)[:] = np.arange(24)  # initialize contents

        sarr.nghost = 1

        self.assertTrue(sarr.has_ghost)
        self.assertEqual(1, sarr.nghost)
        self.assertEqual(3, sarr.nbody)

        v = 0
        for i in range(-1, 3):
            for j in range(3):
                for k in range(2):
                    self.assertEqual(v, sarr[i, j, k])
                    v += 1

        self.assertEqual(18, sarr[-2, 0, 0])
        self.assertEqual(0, sarr[-5, 0, 0])
        self.assertEqual(10, sarr[0, -1, 0])
        self.assertEqual(11, sarr[0, -1, -1])
        sarr[-2, 0, 0] = 180
        sarr[0, -1, -1] = 110
        self.assertEqual(180, ndarr[3, 0, 0])
        self.assertEqual(110, ndarr[1, 2, 1])

    def test_SimpleArray_ghost_md_out_of_range(self):
        sarr = solvcon.SimpleArrayFloat64((4, 3, 2))
        sarr.nghost = 1

        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray::normalize_index\(\): empty index"
        ):
            invalid_empty_idx = ()
            sarr[invalid_empty_idx]

        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray: dim 0 in \[-6, 0, 0\] "
                r"< -nghost - shape\[0\]: -5"
        ):
            sarr[-6, 0, 0]

        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray: dim 0 in \[3, 0, 0\] >= nbody: 3 "
                r"\(shape\[0\]: 4 - nghost: 1\)"
        ):
            sarr[3, 0, 0]

        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray: dim 1 in \[0, -4, 0\] < 0"
        ):
            sarr[0, -4, 0]

        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray: dim 2 in \[0, 2, 2\] >= shape\[2\]: 2"
        ):
            sarr[0, 2, 2]

        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray::normalize_index\(\): empty index"
        ):
            invalid_empty_idx = ()
            sarr[invalid_empty_idx] = 1

        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray: dim 0 in \[-6, 0, 0\] "
                r"< -nghost - shape\[0\]: -5"
        ):
            sarr[-6, 0, 0] = 1

        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray: dim 0 in \[3, 0, 0\] >= nbody: 3 "
                r"\(shape\[0\]: 4 - nghost: 1\)"
        ):
            sarr[3, 0, 0] = 1

        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray: dim 1 in \[0, -4, 0\] < 0"
        ):
            sarr[0, -4, 0] = 1

        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray: dim 2 in \[0, 2, 2\] >= shape\[2\]: 2"
        ):
            sarr[0, 2, 2] = 1

    def test_SimpleArray_negative_index_1d(self):
        sarr = solvcon.SimpleArrayFloat64(4 * 3 * 2)
        ndarr = np.array(sarr, copy=False)
        ndarr[:] = np.arange(24)  # initialize contents

        self.assertFalse(sarr.has_ghost)
        self.assertEqual(0, sarr.nghost)
        self.assertEqual(24, sarr.nbody)

        v = 0
        for i in range(24):
            self.assertEqual(v, sarr[i])
            v += 1

        self.assertEqual(23, sarr[-1])
        self.assertEqual(0, sarr[-24])

    def test_SimpleArray_negative_index_1d_strided(self):
        base = np.arange(12, dtype='float64')
        strided = base[::2]
        sarr = solvcon.SimpleArrayFloat64(array=strided)

        self.assertEqual(2, sarr.stride[0])
        self.assertEqual(4, sarr[2])
        self.assertEqual(10, sarr[-1])
        sarr[1] = 100
        self.assertEqual(100, base[2])

    def test_SimpleArray_negative_index_md(self):
        sarr = solvcon.SimpleArrayFloat64((4, 3, 2))
        ndarr = np.array(sarr, copy=False)
        np.ravel(ndarr)[:] = np.arange(24)  # initialize contents

        self.assertFalse(sarr.has_ghost)
        self.assertEqual(0, sarr.nghost)
        self.assertEqual(4, sarr.nbody)

        self.assertEqual(23, sarr[-1, -1, -1])
        self.assertEqual(0, sarr[-4, -3, -2])
        sarr[-1, -1, -1] = 230
        self.assertEqual(230, ndarr[3, 2, 1])

    def test_SimpleArray_scalar_index_rejects_md(self):
        sarr = solvcon.SimpleArrayFloat64((4, 3, 2))

        with self.assertRaisesRegex(
                IndexError,
                r"SimpleArray::normalize_index\(\): cannot use scalar index "
                r"for 3-dimensional array"
        ):
            sarr[-1]

    def test_SimpleArray_types(self):

        def _check(sarr, nbytes, dtype, get=True):
            self.assertEqual(nbytes, sarr.nbytes)
            self.assertEqual(dtype, sarr.ndarray.dtype)
            if get:
                dtype = getattr(np, dtype)
                self.assertEqual(dtype, sarr.ndarray.dtype)

        # Boolean.
        _check(solvcon.SimpleArrayBool((2, 3, 4)), 24, 'bool', get=False)
        _check(solvcon.SimpleArrayBool((2, 3, 4)), 24, 'bool_')

        # Integer types.
        _check(solvcon.SimpleArrayInt8((2, 3)), 6, 'int8')
        _check(solvcon.SimpleArrayUint8((2, 3)), 6, 'uint8')
        _check(solvcon.SimpleArrayInt16((3, 5)), 30, 'int16')
        _check(solvcon.SimpleArrayUint16((3, 5)), 30, 'uint16')
        _check(solvcon.SimpleArrayInt32(7), 28, 'int32')
        _check(solvcon.SimpleArrayUint32(7), 28, 'uint32')
        _check(solvcon.SimpleArrayInt64((2, 3, 4)), 2 * 3 * 4 * 8, 'int64')
        _check(solvcon.SimpleArrayUint64((2, 3, 4)), 2 * 3 * 4 * 8, 'uint64')

        # Real-number types.
        _check(solvcon.SimpleArrayFloat32((2, 3, 4, 5)), 2 * 3 * 4 * 5 * 4,
               'float32')
        _check(solvcon.SimpleArrayFloat64((2, 3, 4, 5)), 2 * 3 * 4 * 5 * 8,
               'float64')

    def test_SimpleArray_from_ndarray(self):

        ndarr = np.arange(24, dtype='float64').reshape((2, 3, 4))

        with self.assertRaisesRegex(RuntimeError, r"dtype mismatch"):
            solvcon.SimpleArrayInt8(array=ndarr)
        with self.assertRaisesRegex(RuntimeError, r"dtype mismatch"):
            solvcon.SimpleArrayUint64(array=ndarr)
        with self.assertRaisesRegex(RuntimeError, r"dtype mismatch"):
            solvcon.SimpleArrayFloat32(array=ndarr)

        sarr_from_py = solvcon.SimpleArrayFloat64(array=ndarr)
        self.assertTrue(sarr_from_py.is_from_python)

        sarr_from_cpp = solvcon.SimpleArrayFloat64(shape=(2, 3, 4))
        self.assertFalse(sarr_from_cpp.is_from_python)

        shape = (2, 3, 5, 7)
        np_sarr = np.empty(shape, dtype='float64')
        py_sarr = solvcon.SimpleArrayFloat64(array=np_sarr)
        self.assertTupleEqual(shape, py_sarr.shape)
        self.assertEqual(np_sarr.nbytes, py_sarr.nbytes)
        self.assertEqual(np_sarr.size, py_sarr.size)

        shape = (5, 5, 5, 5)
        np_sarr = np.empty(shape, dtype='float64')
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    for x in range(5):
                        np_sarr[i, j, k, x] = i * 1000 + j * 100 + k * 10 + x

        py_sarr = solvcon.SimpleArrayFloat64(array=np_sarr)
        self.assertTupleEqual(shape, py_sarr.shape)
        self.assertEqual(np_sarr.nbytes, py_sarr.nbytes)
        self.assertEqual(np_sarr.size, py_sarr.size)

    def test_SimpleArray_from_ndarray_content(self):

        ndarr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        sarr = solvcon.SimpleArrayFloat64(array=ndarr)
        # Populate using ndarray interface.
        sarr.ndarray.fill(1)
        self.assertTrue((ndarr == 1).all())
        # Set value using setitem interface.
        sarr[0, 0, 0] = 10
        self.assertFalse((ndarr == 1).all())
        # Repopulate using ndarray interface.
        sarr.ndarray.fill(100)
        self.assertTrue((ndarr == 100).all())

        ndarr = np.zeros((2, 3, 4), dtype='complex128')
        sarr = solvcon.SimpleArrayComplex128(array=ndarr)
        # Reassign the numpy array to ensure data is not shared.
        ndarr = np.zeros((2, 3, 4), dtype='complex128')

        for i in range(2):
            for j in range(3):
                for k in range(4):
                    ndarr[i, j, k] = complex(i + j, k)
                    sarr[i, j, k] = ndarr[i, j, k]
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    complex_ndarr = complex(ndarr[i, j, k])
                    complex_sarr = complex(sarr[i, j, k])
                    self.assertEqual(complex_ndarr, complex_sarr)

    def test_SimpleArray_from_ndarray_slice(self):
        ndarr = np.arange(1000, dtype='float64').reshape((10, 10, 10))
        parr = ndarr[1:7:3, 6:2:-1, 3:9]
        sarr = solvcon.SimpleArrayFloat64(array=ndarr[1:7:3, 6:2:-1, 3:9])

        for i in range(2):
            for j in range(4):
                for k in range(6):
                    self.assertEqual(parr[i, j, k], sarr[i, j, k])
        self.assertEqual(parr.nbytes, sarr.nbytes)
        self.assertEqual(parr.size, sarr.size)

    def test_SimpleArray_from_ndarray_transpose(self):
        ndarr = np.arange(350, dtype='float64').reshape((5, 7, 10))
        # The following array is F contiguous.
        parr = ndarr[2:4].T
        sarr = solvcon.SimpleArrayFloat64(array=ndarr[2:4].T)

        for i in range(10):
            for j in range(7):
                for k in range(2):
                    self.assertEqual(parr[i, j, k], sarr[i, j, k])

    def test_SimpleArray_broadcast_ellipsis_shape(self):
        sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
        ndarr = np.arange(2 * 3 * 4, dtype='float64').reshape((2, 3, 4))
        sarr[...] = ndarr[...]
        v = 0
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    self.assertEqual(v, sarr[i, j, k])
                    v += 1

        ndarr = np.arange(2 * 3, dtype='float64').reshape((2, 3))
        with self.assertRaisesRegex(
                RuntimeError,
                r"Broadcast input array from shape\(2, 3\) "
                r"into shape\(2, 3, 4\)"
        ):
            sarr[...] = ndarr[...]

        ndarr = np.arange(2 * 4 * 3, dtype='float64').reshape((2, 4, 3))
        with self.assertRaisesRegex(
                RuntimeError,
                r"Broadcast input array from shape\(2, 4, 3\) "
                r"into shape\(2, 3, 4\)"
        ):
            sarr[...] = ndarr[...]

    def test_SimpleArray_broadcast_ellipsis_ghost_1d(self):

        N = 13
        G = 3

        sarr = solvcon.SimpleArrayFloat64(N)
        ndarr = np.arange(N, dtype='float64')
        sarr.nghost = G

        sarr[...] = ndarr[...]

        v = 0
        for i in range(-G, N - G):
            self.assertEqual(v, sarr[i])
            v += 1

    def test_SimpleArray_broadcast_ellipsis_ghost_md(self):
        N = 5
        G = 2

        sarr = solvcon.SimpleArrayFloat64((5, 3, 4))
        ndarr = np.arange(5 * 3 * 4, dtype='float64').reshape((5, 3, 4))
        sarr.nghost = G

        sarr[...] = ndarr[...]

        v = 0
        for i in range(-G, N - G):
            for j in range(3):
                for k in range(4):
                    self.assertEqual(v, sarr[i, j, k])
                    v += 1

    def test_SimpleArray_broadcast_ellipsis_stride(self):

        sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
        ndarr = np.arange(
            (4 * 2) * (3 * 3) * (2 * 4), dtype='float64').reshape(
            (4 * 2, 3 * 3, 2 * 4))

        stride_arr = ndarr[::4, ::3, ::2]

        # point to the same data
        self.assertEqual(ndarr.__array_interface__['data'],
                         stride_arr.__array_interface__['data'])

        sarr[...] = stride_arr[...]

        for i in range(2):
            for j in range(3):
                for k in range(4):
                    self.assertEqual(stride_arr[i, j, k], sarr[i, j, k])

    def test_SimpleArray_broadcast_ellipsis_dtype(self):
        sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
        ndarr = np.arange(2 * 3 * 4, dtype='int32').reshape((2, 3, 4))
        sarr[...] = ndarr[...]
        v = 0
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    self.assertEqual(v, sarr[i, j, k])
                    v += 1

        sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
        ndarr = np.arange(2 * 3 * 4, dtype='uint8').reshape((2, 3, 4))
        sarr[...] = ndarr[...]
        v = 0
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    self.assertEqual(v, sarr[i, j, k])
                    v += 1

        sarr = solvcon.SimpleArrayFloat64((2, 3, 4))
        ndarr = np.arange(2 * 3 * 4, dtype='float32').reshape((2, 3, 4))
        sarr[...] = ndarr[...]
        v = 0
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    self.assertEqual(v, sarr[i, j, k])
                    v += 1

    def test_SimpleArray_broadcast_slice_basic(self):
        ndarr_input = np.arange(
            1 * 2 * 3 * 4 * 5, dtype='float64').reshape((1, 2, 3, 4, 5))
        ndarr_input += 999  # just a magic number

        def init(sarr):
            v = 0
            for x1 in range(sarr.shape[0]):
                for x2 in range(sarr.shape[1]):
                    for x3 in range(sarr.shape[2]):
                        for x4 in range(sarr.shape[3]):
                            for x5 in range(sarr.shape[4]):
                                sarr[x1, x2, x3, x4, x5] = v
                                v += 1

        def check(sarr, ndarr):
            for x1 in range(sarr.shape[0]):
                for x2 in range(sarr.shape[1]):
                    for x3 in range(sarr.shape[2]):
                        for x4 in range(sarr.shape[3]):
                            for x5 in range(sarr.shape[4]):
                                self.assertEqual(
                                    ndarr[x1, x2, x3, x4, x5],
                                    sarr[x1, x2, x3, x4, x5])

        sarr = solvcon.SimpleArrayFloat64((2, 2, 3, 4, 5))
        ndarr = np.arange(2 * 2 * 3 * 4 * 5, dtype='float64').reshape(
            (2, 2, 3, 4, 5))
        init(sarr)
        sarr[::2, ...] = ndarr_input[...]
        ndarr[::2, ...] = ndarr_input[...]
        check(sarr, ndarr)

        sarr = solvcon.SimpleArrayFloat64((2, 4, 3, 4, 5))
        ndarr = np.arange(2 * 4 * 3 * 4 * 5, dtype='float64').reshape(
            (2, 4, 3, 4, 5))
        init(sarr)
        sarr[::2, ::2, ...] = ndarr_input[...]
        ndarr[::2, ::2, ...] = ndarr_input[...]
        check(sarr, ndarr)

        sarr = solvcon.SimpleArrayFloat64((2, 2, 3, 8, 10))
        ndarr = np.arange(
            2 * 2 * 3 * 8 * 10, dtype='float64').reshape((2, 2, 3, 8, 10))
        init(sarr)
        sarr[::2, ..., ::2, ::2] = ndarr_input[...]
        ndarr[::2, ..., ::2, ::2] = ndarr_input[...]
        check(sarr, ndarr)

        sarr = solvcon.SimpleArrayFloat64((1, 2, 3, 8, 10))
        ndarr = np.arange(
            1 * 2 * 3 * 8 * 10, dtype='float64').reshape((1, 2, 3, 8, 10))
        init(sarr)
        sarr[..., ::2, ::2] = ndarr_input[...]
        ndarr[..., ::2, ::2] = ndarr_input[...]
        check(sarr, ndarr)

        sarr = solvcon.SimpleArrayFloat64((2, 4, 6, 8, 10))
        ndarr = np.arange(
            2 * 4 * 6 * 8 * 10, dtype='float64').reshape((2, 4, 6, 8, 10))
        init(sarr)
        sarr[::2, ::2, ::2, ::2, ::2] = ndarr_input[...]
        ndarr[::2, ::2, ::2, ::2, ::2] = ndarr_input[...]
        check(sarr, ndarr)

        sarr = solvcon.SimpleArrayFloat64((2, 6, 3, 4, 5))
        ndarr = np.arange(
            2 * 6 * 3 * 4 * 5, dtype='float64').reshape((2, 6, 3, 4, 5))
        init(sarr)
        sarr[::2, ::3, ::, :, ::1] = ndarr_input[...]
        ndarr[::2, ::3, ::, :, ::1] = ndarr_input[...]
        check(sarr, ndarr)

    def test_SimpleArray_broadcast_slice_complex_ndarray(self):
        ndarr = np.zeros((2, 3), dtype='complex128')
        sarr = solvcon.SimpleArrayComplex128(array=ndarr)

        sarr[0, 1] = 1 + 2j
        ndarr[0, 1] = 1 + 2j
        sarr[1, 2] = solvcon.complex128(3, 4)
        ndarr[1, 2] = 3 + 4j
        np.testing.assert_array_equal(ndarr, sarr.ndarray)

        rhs = np.arange(2 * 3, dtype='float64').reshape((2, 3))
        rhs = (rhs + 3j).astype('complex128')
        sarr[:, :] = rhs
        np.testing.assert_array_equal(rhs, sarr.ndarray)

    def test_SimpleArray_broadcast_slice_shape(self):
        ndarr = np.arange(2 * 3 * 4, dtype='float64').reshape((2, 3, 4))

        sarr = solvcon.SimpleArrayFloat64((4, 6, 8))
        with self.assertRaisesRegex(
                RuntimeError,
                r"Broadcast input array from shape\(2, 3, 4\) "
                r"into shape\(2, 2, 2\)"
        ):
            sarr[::2, ::3, ::4] = ndarr[...]

        sarr = solvcon.SimpleArrayFloat64((4, 6, 8))
        with self.assertRaisesRegex(
                RuntimeError,
                r"Broadcast input array from shape\(2, 3, 4\) "
                r"into shape\(2, 6, 8\)"
        ):
            sarr[::2, ::1, ...] = ndarr[...]

    def test_SimpleArray_broadcast_slice_ghost_1d(self):
        import math
        N = 13
        G = 3
        STEP = 3

        sarr = solvcon.SimpleArrayFloat64(N)
        ndarr = np.arange(math.ceil(N / STEP), dtype='float64')
        ndarr2 = np.arange(N, dtype='float64')
        sarr.nghost = G

        sarr[::STEP] = ndarr[...]
        ndarr2[::STEP] = ndarr[...]

        print(ndarr2.shape)

        for i in range(0, N, STEP):
            self.assertEqual(ndarr2[i], sarr[i - G])

    def test_SimpleArray_broadcast_slice_ghost_md(self):
        import math
        N = 5
        G = 2
        STEP = 2

        sarr = solvcon.SimpleArrayFloat64((N, 3, 4))
        ndarr = np.arange(math.ceil(N / STEP) * 3 * 4, dtype='float64') \
            .reshape((math.ceil(N / STEP), 3, 4))
        ndarr2 = np.arange(N * 3 * 4, dtype='float64').reshape((N, 3, 4))
        sarr.nghost = G

        sarr[::STEP, ...] = ndarr[...]
        ndarr2[::STEP, ...] = ndarr[...]

        for i in range(0, N, STEP):
            for j in range(3):
                for k in range(4):
                    self.assertEqual(ndarr2[i, j, k], sarr[i - G, j, k])

    def test_SimpleArray_broadcast_from_list_list(self):
        sarr = solvcon.SimpleArrayFloat64((2, 3))
        sarr[:, :] = [[1, 2, 3], [4, 5, 6]]
        for i in range(2):
            for j in range(3):
                self.assertEqual(sarr[i, j], i * 3 + j + 1)

        sarr = solvcon.SimpleArrayFloat64((2, 3))
        sarr[:1, :2] = [[1, 2]]
        self.assertEqual(sarr[0, 0], 1)
        self.assertEqual(sarr[0, 1], 2)

    def test_SimpleArray_broadcast_from_tuple_list(self):
        sarr = solvcon.SimpleArrayFloat64((2, 3))
        sarr[:, :] = [(1, 2, 3), (4, 5, 6)]
        for i in range(2):
            for j in range(3):
                self.assertEqual(sarr[i, j], i * 3 + j + 1)

        sarr = solvcon.SimpleArrayFloat64((2, 3))
        sarr[:1, :2] = [(1, 2)]
        self.assertEqual(sarr[0, 0], 1)
        self.assertEqual(sarr[0, 1], 2)

    def test_SimpleArray_broadcast_from_tuple_tuple(self):
        sarr = solvcon.SimpleArrayFloat64((2, 3))
        sarr[:, :] = ((1, 2, 3), (4, 5, 6))
        for i in range(2):
            for j in range(3):
                self.assertEqual(sarr[i, j], i * 3 + j + 1)

        sarr = solvcon.SimpleArrayFloat64((2, 3))
        sarr[:1, :2] = ((1, 2),)
        self.assertEqual(sarr[0, 0], 1)
        self.assertEqual(sarr[0, 1], 2)

    def test_SimpleArray_SimpleArrayPlex_type_switch(self):
        arrayplex_int32 = solvcon.SimpleArray((2, 3, 4), dtype="int32")

        # from plex to typed
        array_int32 = arrayplex_int32.typed
        self.assertEqual(
            str(type(array_int32)), "<class '_solvcon.SimpleArrayInt32'>")

        # from typed to plex
        arrayplex_int32_2 = array_int32.plex
        self.assertEqual(
            str(type(arrayplex_int32_2)), "<class '_solvcon.SimpleArray'>")

    def test_sort(self):
        # Note: tests on array with ghost indices should be added
        #       in the future

        test_data = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            [1, 5, 10, 2, 6, 9, 7, 8, 4, 3],
            [1, 0, 1, -3, -4, -1, 1, 9, 5, -4],
            [-1.3, -4.8, 1.5, 0.3, 7.1, 2.5, 4.8, -0.1, 9.4, 7.6]
        ]

        def _check(arr, use_float=False):
            if use_float:
                narr = np.array(arr, dtype='float64')
                sarr = solvcon.SimpleArrayFloat64(array=narr)
            else:
                narr = np.array(arr, dtype='int32')
                sarr = solvcon.SimpleArrayInt32(array=narr)

            args = sarr.argsort()
            for i in range(1, len(args)):
                self.assertLessEqual(sarr[args[i - 1]], sarr[args[i]])

            sorted_arr = sarr.take_along_axis(args)
            for i in range(1, len(sorted_arr)):
                self.assertLessEqual(sorted_arr[i - 1], sorted_arr[i])

            sarr.sort()
            for i in range(1, len(sarr)):
                self.assertLessEqual(sarr[i - 1], sarr[i])

        _check(test_data[0])
        _check(test_data[1])
        _check(test_data[2])
        _check(test_data[3])
        _check(test_data[4], True)

    def test_take_along_axis(self):
        data = [1, 5, 10, 2, 6, 9, 7, 8, 4, 3]
        narr = np.array(data, dtype='int32')
        data_arr = solvcon.SimpleArrayInt32(array=narr)

        # test 1-D indices
        idx = [0, 2, 4, 6, 1, 3, 5, 7]
        narr = np.array(idx, dtype='uint64')
        idx_arr = solvcon.SimpleArrayUint64(array=narr)

        ret_arr = data_arr.take_along_axis(idx_arr)
        for i in range(len(idx)):
            self.assertEqual(ret_arr[i], data[idx[i]])

        ret_arr = data_arr.take_along_axis_simd(idx_arr)
        for i in range(len(idx)):
            self.assertEqual(ret_arr[i], data[idx[i]])

        # test 2-D indices
        idx = [[0, 2, 4, 6], [1, 3, 5, 7]]
        narr = np.array(idx, dtype='uint64')
        idx_arr = solvcon.SimpleArrayUint64(array=narr)

        ret_arr = data_arr.take_along_axis(idx_arr)
        for i in range(len(idx)):
            for j in range(len(idx[i])):
                self.assertEqual(ret_arr[i, j], data[idx_arr[i, j]])

        ret_arr = data_arr.take_along_axis_simd(idx_arr)
        for i in range(len(idx)):
            for j in range(len(idx[i])):
                self.assertEqual(ret_arr[i, j], data[idx_arr[i, j]])

        # test out-of-range index
        idx = [[0, 1], [2, 3], [4, 20]]
        narr = np.array(idx, dtype='uint64')
        idx_arr = solvcon.SimpleArrayUint64(array=narr)

        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::take_along_axis\(\): indices\[2, 1\] is 20, " +
            "which is out of range of the array size 10"
        ):
            ret_arr = data_arr.take_along_axis(idx_arr)

        with self.assertRaisesRegex(
            IndexError,
            r"SimpleArray::take_along_axis_simd\(\): indices\[2, 1\] is 20, " +
            "which is out of range of the array size 10"
        ):
            ret_arr = data_arr.take_along_axis_simd(idx_arr)


class SimpleArrayAlignmentTC(unittest.TestCase):
    """
    Test memory alignment support for SimpleArray.
    Valid alignment values are: 0 (default, no alignment), 16, 32, or 64 bytes.
    Tests cover various array types (Float64, Int32, Uint64) and shapes (1D, 2D, 3D).
    """  # noqa: E501

    def test_default_alignment(self):
        array = solvcon.SimpleArrayFloat64((2, 3, 4))  # default alignment
        self.assertEqual(0, array.alignment)
        self.assertEqual((2, 3, 4), array.shape)

    def test_alignment_creation(self):
        array = solvcon.SimpleArrayFloat64((4, 4), 16)
        self.assertEqual(16, array.alignment)
        self.assertEqual((4, 4), array.shape)

        array = solvcon.SimpleArrayFloat64((8, 8), 32)
        self.assertEqual(32, array.alignment)
        self.assertEqual((8, 8), array.shape)

        array = solvcon.SimpleArrayFloat64((16, 8), 64)
        self.assertEqual(64, array.alignment)
        self.assertEqual((16, 8), array.shape)

    def test_alignment_with_value_initialization(self):
        # default alignment 0
        array = solvcon.SimpleArrayFloat64((4, 4), 3.14159)
        self.assertEqual(0, array.alignment)
        self.assertEqual((4, 4), array.shape)
        self.assertEqual(3.14159, array[0, 0])

        # 16-byte alignment
        array = solvcon.SimpleArrayFloat64((4, 4), 2.71828, 16)
        self.assertEqual(16, array.alignment)
        self.assertEqual((4, 4), array.shape)
        self.assertEqual(2.71828, array[0, 0])

    def test_alignment_validation_valid_values(self):
        solvcon.SimpleArrayFloat64((4, 4), 16)
        solvcon.SimpleArrayFloat64((8, 8), 32)
        solvcon.SimpleArrayFloat64((16, 8), 64)

    def test_alignment_validation_invalid_values(self):
        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::ConcreteBuffer: alignment must be 0, 16, 32, or 64, but got 17"  # noqa E501
        ):
            solvcon.SimpleArrayFloat64((4, 4), 17)

        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::ConcreteBuffer: alignment must be 0, 16, 32, or 64, but got 100"  # noqa E501
        ):
            solvcon.SimpleArrayFloat64((4, 4), 100)

        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::ConcreteBuffer: alignment must be 0, 16, 32, or 64, but got 128"  # noqa E501
        ):
            solvcon.SimpleArrayFloat64((16, 16), 128)

    def test_alignment_size_validation(self):
        solvcon.SimpleArrayFloat64((2, 2), 32)  # 2*2*8=32, multiple of 32
        solvcon.SimpleArrayFloat64((4, 4), 64)  # 4*4*8=128, multiple of 64

        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::allocate: size .* must be a multiple of alignment 16"  # noqa E501
        ):
            solvcon.SimpleArrayFloat64((5, 1), 16)

        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::allocate: size .* must be a multiple of alignment 32"  # noqa E501
        ):
            solvcon.SimpleArrayFloat64((3, 5), 32)

    def test_alignment_with_operations(self):
        array1 = solvcon.SimpleArrayFloat64((4, 4), 16)
        array2 = solvcon.SimpleArrayFloat64((4, 4), 32)

        self.assertEqual(16, array1.alignment)
        self.assertEqual(32, array2.alignment)

        array1.fill(5.0)
        array2.fill(10.0)

        self.assertEqual(5.0, array1[0, 0])
        self.assertEqual(10.0, array2[0, 0])

        self.assertEqual(5.0 * 16, array1.sum())
        self.assertEqual(10.0 * 16, array2.sum())

    def test_alignment_with_ndarray_conversion(self):
        array = solvcon.SimpleArrayFloat64((4, 4), 16)
        self.assertEqual(16, array.alignment)

        ndarr = np.array(array, copy=False)
        self.assertEqual((4, 4), ndarr.shape)
        self.assertEqual(np.float64, ndarr.dtype)

        array[0, 0] = 42.0
        self.assertEqual(42.0, ndarr[0, 0])

    def test_alignment_with_clone(self):
        array = solvcon.SimpleArrayFloat64((4, 4), 3.14, 32)
        self.assertEqual(32, array.alignment)
        self.assertEqual(3.14, array[0, 0])

        cloned = array.clone()
        self.assertEqual(32, cloned.alignment)
        self.assertEqual((4, 4), cloned.shape)
        self.assertEqual(3.14, cloned[0, 0])

        array[0, 0] = 99.9
        self.assertEqual(99.9, array[0, 0])
        self.assertEqual(3.14, cloned[0, 0])

    def test_alignment_int32_array(self):
        array = solvcon.SimpleArrayInt32((8, 8), 16)
        self.assertEqual(16, array.alignment)
        self.assertEqual((8, 8), array.shape)

        array.fill(42)
        self.assertEqual(42, array[0, 0])
        self.assertEqual(42 * 64, array.sum())

    def test_alignment_uint64_array(self):
        array = solvcon.SimpleArrayUint64((4, 8), 32)
        self.assertEqual(32, array.alignment)
        self.assertEqual((4, 8), array.shape)

        array.fill(100)
        self.assertEqual(100, array[0, 0])
        self.assertEqual(100 * 32, array.sum())

    def test_alignment_with_different_shapes(self):
        array1d = solvcon.SimpleArrayFloat64((32,), 16)
        self.assertEqual(16, array1d.alignment)
        self.assertEqual((32,), array1d.shape)

        array2d = solvcon.SimpleArrayFloat64((4, 8), 32)
        self.assertEqual(32, array2d.alignment)
        self.assertEqual((4, 8), array2d.shape)

        array3d = solvcon.SimpleArrayFloat64((2, 4, 4), 64)
        self.assertEqual(64, array3d.alignment)
        self.assertEqual((2, 4, 4), array3d.shape)

    def test_alignment_with_simd_operations(self):
        size = 16
        alignments = [16, 32, 64]

        for alignment in alignments:
            arr1 = solvcon.SimpleArrayFloat64((size,), alignment)
            arr2 = solvcon.SimpleArrayFloat64((size,), alignment)

            self.assertEqual(alignment, arr1.alignment)
            self.assertEqual(alignment, arr2.alignment)

            for index in range(size):
                arr1[index] = index * 2.0
                arr2[index] = index * 3.0

            result_add = arr1.add_simd(arr2)
            self.assertEqual(0, result_add.alignment)
            for index in range(size):
                expected = index * 2.0 + index * 3.0
                self.assertAlmostEqual(expected, result_add[index])

            result_sub = arr1.sub_simd(arr2)
            self.assertEqual(0, result_sub.alignment)
            for index in range(size):
                expected = index * 2.0 - index * 3.0
                self.assertAlmostEqual(expected, result_sub[index])

            result_mul = arr1.mul_simd(arr2)
            self.assertEqual(0, result_mul.alignment)
            for index in range(size):
                expected = index * 2.0 * index * 3.0
                self.assertAlmostEqual(expected, result_mul[index])

            result_div = arr2.div_simd(arr1)
            self.assertEqual(0, result_div.alignment)
            for index in range(1, size):
                expected = index * 3.0 / (index * 2.0)
                self.assertAlmostEqual(expected, result_div[index])

    def test_alignment_with_simd_operations_multidimensional(self):
        alignments = [16, 32, 64]

        for alignment in alignments:
            array1_2d = solvcon.SimpleArrayFloat64((4, 8), alignment)
            array2_2d = solvcon.SimpleArrayFloat64((4, 8), alignment)

            self.assertEqual(alignment, array1_2d.alignment)
            self.assertEqual(alignment, array2_2d.alignment)
            self.assertEqual((4, 8), array1_2d.shape)
            self.assertEqual((4, 8), array2_2d.shape)

            for i in range(4):
                for j in range(8):
                    array1_2d[i, j] = (i * 8 + j) * 2.0
                    array2_2d[i, j] = (i * 8 + j) * 3.0

            result_add_2d = array1_2d.add_simd(array2_2d)
            self.assertEqual(0, result_add_2d.alignment)
            self.assertEqual((4, 8), result_add_2d.shape)
            for i in range(4):
                for j in range(8):
                    value = i * 8 + j
                    expected = value * 2.0 + value * 3.0
                    self.assertAlmostEqual(expected, result_add_2d[i, j])

            result_sub_2d = array1_2d.sub_simd(array2_2d)
            self.assertEqual(0, result_sub_2d.alignment)
            self.assertEqual((4, 8), result_sub_2d.shape)
            for i in range(4):
                for j in range(8):
                    value = i * 8 + j
                    expected = value * 2.0 - value * 3.0
                    self.assertAlmostEqual(expected, result_sub_2d[i, j])

            result_mul_2d = array1_2d.mul_simd(array2_2d)
            self.assertEqual(0, result_mul_2d.alignment)
            self.assertEqual((4, 8), result_mul_2d.shape)
            for i in range(4):
                for j in range(8):
                    value = i * 8 + j
                    expected = value * 2.0 * value * 3.0
                    self.assertAlmostEqual(expected, result_mul_2d[i, j])

            result_div_2d = array2_2d.div_simd(array1_2d)
            self.assertEqual(0, result_div_2d.alignment)
            self.assertEqual((4, 8), result_div_2d.shape)
            for i in range(4):
                for j in range(8):
                    value = i * 8 + j
                    if value > 0:
                        expected = value * 3.0 / (value * 2.0)
                        self.assertAlmostEqual(expected, result_div_2d[i, j])

            array1_3d = solvcon.SimpleArrayFloat64((2, 4, 4), alignment)
            array2_3d = solvcon.SimpleArrayFloat64((2, 4, 4), alignment)

            self.assertEqual(alignment, array1_3d.alignment)
            self.assertEqual(alignment, array2_3d.alignment)
            self.assertEqual((2, 4, 4), array1_3d.shape)
            self.assertEqual((2, 4, 4), array2_3d.shape)

            for i in range(2):
                for j in range(4):
                    for k in range(4):
                        array1_3d[i, j, k] = (i * 16 + j * 4 + k) * 2.0
                        array2_3d[i, j, k] = (i * 16 + j * 4 + k) * 3.0

            result_add_3d = array1_3d.add_simd(array2_3d)
            self.assertEqual(0, result_add_3d.alignment)
            self.assertEqual((2, 4, 4), result_add_3d.shape)
            for i in range(2):
                for j in range(4):
                    for k in range(4):
                        value = i * 16 + j * 4 + k
                        expected = value * 2.0 + value * 3.0
                        self.assertAlmostEqual(expected,
                                               result_add_3d[i, j, k])

            result_sub_3d = array1_3d.sub_simd(array2_3d)
            self.assertEqual(0, result_sub_3d.alignment)
            self.assertEqual((2, 4, 4), result_sub_3d.shape)
            for i in range(2):
                for j in range(4):
                    for k in range(4):
                        value = i * 16 + j * 4 + k
                        expected = value * 2.0 - value * 3.0
                        self.assertAlmostEqual(expected,
                                               result_sub_3d[i, j, k])

            result_mul_3d = array1_3d.mul_simd(array2_3d)
            self.assertEqual(0, result_mul_3d.alignment)
            self.assertEqual((2, 4, 4), result_mul_3d.shape)
            for i in range(2):
                for j in range(4):
                    for k in range(4):
                        value = i * 16 + j * 4 + k
                        expected = value * 2.0 * value * 3.0
                        self.assertAlmostEqual(expected,
                                               result_mul_3d[i, j, k])

            result_div_3d = array2_3d.div_simd(array1_3d)
            self.assertEqual(0, result_div_3d.alignment)
            self.assertEqual((2, 4, 4), result_div_3d.shape)
            for i in range(2):
                for j in range(4):
                    for k in range(4):
                        value = i * 16 + j * 4 + k
                        if value > 0:
                            expected = value * 3.0 / (value * 2.0)
                            self.assertAlmostEqual(expected,
                                                   result_div_3d[i, j, k])

    def test_alignment_size_validation_multidimensional(self):
        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::allocate: size .* must be a multiple of alignment 16"  # noqa E501
        ):
            solvcon.SimpleArrayFloat64((1, 3), 16)

        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::allocate: size .* must be a multiple of alignment 32"  # noqa E501
        ):
            solvcon.SimpleArrayFloat64((1, 3), 32)

        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::allocate: size .* must be a multiple of alignment 64"  # noqa E501
        ):
            solvcon.SimpleArrayFloat64((3, 3), 64)

        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::allocate: size .* must be a multiple of alignment 16"  # noqa E501
        ):
            solvcon.SimpleArrayFloat64((1, 1, 1), 16)

        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::allocate: size .* must be a multiple of alignment 32"  # noqa E501
        ):
            solvcon.SimpleArrayFloat64((1, 1, 1), 32)

        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::allocate: size .* must be a multiple of alignment 64"  # noqa E501
        ):
            solvcon.SimpleArrayFloat64((2, 3, 5), 64)

    def test_alignment_with_unaligned_rows(self):
        # 2D arrays that row is not aligned
        array1_2d = solvcon.SimpleArrayFloat64((2, 3), 16)
        array2_2d = solvcon.SimpleArrayFloat64((2, 3), 16)

        self.assertEqual(16, array1_2d.alignment)
        self.assertEqual(16, array2_2d.alignment)
        self.assertEqual((2, 3), array1_2d.shape)

        for i in range(2):
            for j in range(3):
                array1_2d[i, j] = (i * 3 + j) * 2.0
                array2_2d[i, j] = (i * 3 + j) * 3.0

        # SIMD ops must tolerate the unaligned row stride and still yield exact math.  # noqa: E501
        result_add = array1_2d.add_simd(array2_2d)
        result_sub = array1_2d.sub_simd(array2_2d)
        result_mul = array1_2d.mul_simd(array2_2d)

        for i in range(2):
            for j in range(3):
                value = i * 3 + j
                self.assertAlmostEqual(value * 5.0, result_add[i, j])
                self.assertAlmostEqual(value * -1.0, result_sub[i, j])
                self.assertAlmostEqual(value * value * 6.0, result_mul[i, j])

        # Repeat with 3D data that the innermost dimension is unaligned.
        array1_3d = solvcon.SimpleArrayFloat64((2, 2, 2), 32)
        array2_3d = solvcon.SimpleArrayFloat64((2, 2, 2), 32)

        self.assertEqual(32, array1_3d.alignment)
        self.assertEqual(32, array2_3d.alignment)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    array1_3d[i, j, k] = (i * 4 + j * 2 + k) * 2.0
                    array2_3d[i, j, k] = (i * 4 + j * 2 + k) * 3.0

        result_add_3d = array1_3d.add_simd(array2_3d)
        result_sub_3d = array1_3d.sub_simd(array2_3d)
        result_mul_3d = array1_3d.mul_simd(array2_3d)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    value = i * 4 + j * 2 + k
                    self.assertAlmostEqual(value * 5.0,
                                           result_add_3d[i, j, k])
                    self.assertAlmostEqual(value * -1.0,
                                           result_sub_3d[i, j, k])
                    self.assertAlmostEqual(value * value * 6.0,
                                           result_mul_3d[i, j, k])


class SimpleArrayCalculatorsTC(unittest.TestCase):

    def test_minmaxsum(self):
        sarr = solvcon.SimpleArrayFloat64(shape=(2, 4), value=10.0)

        self.assertEqual(sarr.sum(), 10.0 * 2 * 4)
        self.assertEqual(sarr.min(), 10.0)
        self.assertEqual(sarr.max(), 10.0)
        sarr.fill(1.0)
        self.assertEqual(sarr.sum(), 1.0 * 2 * 4)
        self.assertEqual(sarr.min(), 1.0)
        self.assertEqual(sarr.max(), 1.0)
        sarr[1, 0] = 9.2
        sarr[0, 3] = -2.3
        self.assertEqual(sarr.min(), -2.3)
        self.assertEqual(sarr.max(), 9.2)

    def test_sum_contiguous(self):
        # 1D contiguous: exercises the straight pointer loop in
        # sum_contiguous() with no shape/stride arithmetic involved.
        nparr = np.arange(1, 101, dtype='float64')
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        self.assertEqual(sarr.sum(), np.sum(nparr))

        # Multi-dim C-contiguous: still a single dense block, so sum()
        # takes the contiguous path even though shape is multi-dim.
        nparr = np.arange(60, dtype='float64').reshape((3, 4, 5))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        self.assertEqual(sarr.sum(), np.sum(nparr))

        # Multi-dim F-contiguous: dense block but with column-major
        # strides; sum_contiguous() walks the buffer in F order, which
        # still hits every element exactly once.
        nparr = np.asfortranarray(
            np.arange(60, dtype='float64').reshape((3, 4, 5)))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        self.assertEqual(sarr.sum(), np.sum(nparr))

    def test_sum_non_contiguous(self):
        # Strided slice that fails both C- and F-contiguity checks, so
        # sum() must take the sum_strided path. Distinct integer values
        # mean any indexing bug shifts the result.
        nparr = np.arange(625, dtype='float64').reshape((5, 5, 5, 5))
        nparr = nparr[:3:2, 1:4:2, :3:3, 3:4:2]
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        self.assertEqual(sarr.sum(), np.sum(nparr))

    def test_sum_empty_non_contiguous(self):
        # Empty and non-contiguous: shape (3, 0, 4) with strides
        # (0, 1, 0) is neither C- nor F-contiguous, so sum() must
        # short-circuit on n == 0 instead of falling into the strided
        # path and reading from an empty buffer.
        sarr = solvcon.SimpleArrayFloat64(shape=(3, 4, 0), value=0.0)
        sarr = sarr.transpose(axis=[0, 2, 1])
        self.assertEqual(sarr.sum(), 0.0)

    def test_mean_empty_raises(self):
        sarr = solvcon.SimpleArrayFloat64(shape=(0, 3), value=0.0)
        with self.assertRaisesRegex(RuntimeError, "empty array"):
            sarr.mean()

    def test_abs(self):
        sarr = solvcon.SimpleArrayInt64(shape=(3, 2), value=-2)
        self.assertEqual(sarr.sum(), -2 * 3 * 2)
        sarr = sarr.abs()
        self.assertEqual(sarr.sum(), 2 * 3 * 2)

        # Taking absolute value of unsigned type simply copies the data.
        sarr = solvcon.SimpleArrayInt8(shape=(3, 2), value=2)
        self.assertEqual(sarr.sum(), 2 * 3 * 2)
        sarr = sarr.abs()
        self.assertEqual(sarr.sum(), 2 * 3 * 2)

        # Absolute value of Boolean is special.
        sarr = solvcon.SimpleArrayBool(shape=(3, 2), value=1)
        self.assertEqual(sarr.sum(), True)
        sarr = sarr.abs()
        self.assertEqual(sarr.sum(), True)

    def test_median(self):
        nparr = np.arange(24, dtype='float64')
        np.random.shuffle(nparr)
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.median(nparr)
        smed = sarr.median()
        self.assertEqual(npmed, smed)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        np.random.shuffle(nparr)
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.median(nparr)
        smed = sarr.median()
        self.assertEqual(npmed, smed)

        nparr = np.arange(81, dtype='float64').reshape((3, 3, 3, 3))
        np.random.shuffle(nparr)
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.median(nparr)
        smed = sarr.median()
        self.assertEqual(npmed, smed)

        nparr = np.arange(24, dtype='complex128').reshape((2, 3, 4))
        npimg = nparr.copy()
        np.random.shuffle(nparr)
        np.random.shuffle(npimg)
        nparr = nparr + 1j * npimg
        sarr = solvcon.SimpleArrayComplex128(array=nparr)
        npmed = np.median(nparr)
        smed = sarr.median()
        self.assertEqual(npmed.real, smed.real)
        self.assertEqual(npmed.imag, smed.imag)

        # Reference: https://github.com/numpy/numpy/issues/12943
        nparr = np.array([1 + 10j, 2 + 1j, 3 + 0j, 0 + 3j], dtype='complex128')
        sarr = solvcon.SimpleArrayComplex128(array=nparr)
        npmed = np.median(nparr)
        smed = sarr.median()
        self.assertEqual(npmed.real, smed.real)
        self.assertEqual(npmed.imag, smed.imag)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        nparr = nparr[::2, ::2, ::2]
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npavg = np.median(nparr)
        savg = sarr.median()
        self.assertEqual(npavg, savg)

        nparr = np.arange(4 * 3 * 2, dtype='float64').reshape((4, 3, 2))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sarr.nghost = 1
        self.assertEqual(np.median(nparr), sarr.median())

    def test_median_with_axis(self):
        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.median(nparr, axis=0)
        smed = sarr.median(axis=0)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.median(nparr, axis=1)
        smed = sarr.median(axis=1)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.median(nparr, axis=2)
        smed = sarr.median(axis=2)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.median(nparr, axis=(0, 1))
        smed = sarr.median(axis=[0, 1])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.median(nparr, axis=(1, 2))
        smed = sarr.median(axis=[1, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.median(nparr, axis=(0, 2))
        smed = sarr.median(axis=[0, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(4 * 3 * 2, dtype='float64').reshape((4, 3, 2))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sarr.nghost = 1
        npmed = np.median(nparr, axis=1)
        smed = sarr.median(axis=1)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(4 * 3 * 2, dtype='float64').reshape((4, 3, 2))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sarr.nghost = 1
        npmed = np.median(nparr, axis=0)
        smed = sarr.median(axis=0)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

    def test_median_freq(self):
        bool_data_odd = np.array([True, False, True, False, True], dtype=bool)
        sarr_bool_odd = solvcon.SimpleArrayBool(array=bool_data_odd)
        expected_bool_odd = np.median(bool_data_odd)
        result_bool_odd = sarr_bool_odd.median()
        self.assertEqual(bool(expected_bool_odd), bool(result_bool_odd))

        bool_data_even = np.array([True, False, True, False], dtype=bool)
        sarr_bool_even = solvcon.SimpleArrayBool(array=bool_data_even)
        expected_bool_even = np.median(bool_data_even)
        result_bool_even = sarr_bool_even.median()
        self.assertEqual(bool(expected_bool_even), bool(result_bool_even))

        bool_data_all_true = np.array([True, True, True], dtype=bool)
        sarr_bool_all_true = solvcon.SimpleArrayBool(array=bool_data_all_true)
        expected_bool_all_true = np.median(bool_data_all_true)
        result_bool_all_true = sarr_bool_all_true.median()
        self.assertEqual(bool(expected_bool_all_true),
                         bool(result_bool_all_true))

        bool_data_all_false = np.array([False, False, False, False],
                                       dtype=bool)
        sarr_bool_all_false = solvcon.SimpleArrayBool(
            array=bool_data_all_false)
        expected_bool_all_false = np.median(bool_data_all_false)
        result_bool_all_false = sarr_bool_all_false.median()
        self.assertEqual(bool(expected_bool_all_false),
                         bool(result_bool_all_false))

        uint8_data_odd = np.array([10, 5, 20, 15, 25], dtype=np.uint8)
        sarr_uint8_odd = solvcon.SimpleArrayUint8(array=uint8_data_odd)
        expected_uint8_odd = np.median(uint8_data_odd)
        result_uint8_odd = sarr_uint8_odd.median()
        self.assertEqual(int(expected_uint8_odd), int(result_uint8_odd))

        uint8_data_even = np.array([10, 5, 20, 15], dtype=np.uint8)
        sarr_uint8_even = solvcon.SimpleArrayUint8(array=uint8_data_even)
        expected_uint8_even = np.median(uint8_data_even)
        result_uint8_even = sarr_uint8_even.median()
        self.assertEqual(int(expected_uint8_even), int(result_uint8_even))

        uint8_data_dup_odd = np.array([0, 0, 0, 0, 0], dtype=np.uint8)
        sarr_uint8_dup_odd = solvcon.SimpleArrayUint8(
            array=uint8_data_dup_odd)
        expected_uint8_dup_odd = np.median(uint8_data_dup_odd)
        result_uint8_dup_odd = sarr_uint8_dup_odd.median()
        self.assertEqual(int(expected_uint8_dup_odd),
                         int(result_uint8_dup_odd))

        uint8_data_dup_even = np.array([255, 255, 255, 255], dtype=np.uint8)
        sarr_uint8_dup_even = solvcon.SimpleArrayUint8(
            array=uint8_data_dup_even)
        expected_uint8_dup_even = np.median(uint8_data_dup_even)
        result_uint8_dup_even = sarr_uint8_dup_even.median()
        self.assertEqual(int(expected_uint8_dup_even),
                         int(result_uint8_dup_even))

        int8_data_odd_pos = np.array([10, 5, 20, 15, 25], dtype=np.int8)
        sarr_int8_odd_pos = solvcon.SimpleArrayInt8(array=int8_data_odd_pos)
        expected_int8_odd_pos = np.median(int8_data_odd_pos)
        result_int8_odd_pos = sarr_int8_odd_pos.median()
        self.assertEqual(int(expected_int8_odd_pos),
                         int(result_int8_odd_pos))

        int8_data_even_pos = np.array([10, 5, 20, 15], dtype=np.int8)
        sarr_int8_even_pos = solvcon.SimpleArrayInt8(array=int8_data_even_pos)
        expected_int8_even_pos = np.median(int8_data_even_pos)
        result_int8_even_pos = sarr_int8_even_pos.median()
        self.assertEqual(int(expected_int8_even_pos),
                         int(result_int8_even_pos))

        int8_data_odd_neg = np.array([-10, -5, -20, -15, -25], dtype=np.int8)
        sarr_int8_odd_neg = solvcon.SimpleArrayInt8(array=int8_data_odd_neg)
        expected_int8_odd_neg = np.median(int8_data_odd_neg)
        result_int8_odd_neg = sarr_int8_odd_neg.median()
        self.assertEqual(int(expected_int8_odd_neg),
                         int(result_int8_odd_neg))

        int8_data_even_neg = np.array([-10, -5, -20, -15], dtype=np.int8)
        sarr_int8_even_neg = solvcon.SimpleArrayInt8(array=int8_data_even_neg)
        expected_int8_even_neg = np.median(int8_data_even_neg)
        result_int8_even_neg = sarr_int8_even_neg.median()
        self.assertEqual(int(expected_int8_even_neg),
                         int(result_int8_even_neg))

        int8_data_edge = np.array([-128, -128, -128, -128], dtype=np.int8)
        sarr_int8_edge = solvcon.SimpleArrayInt8(array=int8_data_edge)
        expected_int8_edge = np.median(int8_data_edge)
        result_int8_edge = sarr_int8_edge.median()
        self.assertEqual(int(expected_int8_edge), int(result_int8_edge))

        int8_data_dup_even = np.array([127, 127, 127, 127], dtype=np.int8)
        sarr_int8_dup_even = solvcon.SimpleArrayInt8(array=int8_data_dup_even)
        expected_int8_dup_even = np.median(int8_data_dup_even)
        result_int8_dup_even = sarr_int8_dup_even.median()
        self.assertEqual(int(expected_int8_dup_even),
                         int(result_int8_dup_even))

    def test_average(self):
        nparr = np.arange(24, dtype='float64')
        np.random.shuffle(nparr)
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npavg = np.average(nparr)
        savg = sarr.average()
        self.assertEqual(npavg, savg)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        np.random.shuffle(nparr)
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npavg = np.average(nparr)
        savg = sarr.average()
        self.assertEqual(npavg, savg)

        nparr = np.arange(81, dtype='float64').reshape((3, 3, 3, 3))
        np.random.shuffle(nparr)
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npavg = np.average(nparr)
        savg = sarr.average()
        self.assertEqual(npavg, savg)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        nparr = nparr[::2, ::2, ::2]
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npavg = np.average(nparr)
        savg = sarr.average()
        self.assertEqual(npavg, savg)

        nparr = np.array([1, 2, 3, 4, 5], dtype='float64')
        weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2], dtype='float64')
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sweights = solvcon.SimpleArrayFloat64(array=weights)
        npavg = np.average(nparr, weights=weights)
        savg = sarr.average(weight=sweights)
        self.assertEqual(npavg, savg)

        nparr = np.arange(6, dtype='float64').reshape((2, 3))
        weights = np.array([0.5, 0.3, 0.2], dtype='float64')
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sweights = solvcon.SimpleArrayFloat64(array=weights)
        npavg = np.average(nparr, weights=weights, axis=1)
        savg = sarr.average(axis=1, weight=sweights)
        savg = savg.ndarray
        self.assertTrue(np.allclose(npavg, savg))

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        weights = np.array([0.25, 0.25, 0.25, 0.25], dtype='float64')
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sweights = solvcon.SimpleArrayFloat64(array=weights)
        npavg = np.average(nparr, weights=weights, axis=2)
        savg = sarr.average(axis=2, weight=sweights)
        savg = savg.ndarray
        self.assertTrue(np.allclose(npavg, savg))

        nparr = np.arange(12, dtype='float64').reshape((3, 4))
        weights = np.array([0.5, 0.3, 0.2], dtype='float64')
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sweights = solvcon.SimpleArrayFloat64(array=weights)
        npavg = np.average(nparr, weights=weights, axis=0)
        savg = sarr.average(axis=0, weight=sweights)
        savg = savg.ndarray
        self.assertTrue(np.allclose(npavg, savg))

        nparr = np.arange(4 * 3 * 2, dtype='float64').reshape((4, 3, 2))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sarr.nghost = 1
        npavg = np.average(nparr)
        savg = sarr.average()
        self.assertEqual(npavg, savg)

        weights = np.arange(1, 4 * 3 * 2 + 1, dtype='float64')
        weights = weights.reshape((4, 3, 2))
        sweights = solvcon.SimpleArrayFloat64(array=weights)
        sweights.nghost = 1
        npavg = np.average(nparr, weights=weights)
        savg = sarr.average(weight=sweights)
        self.assertEqual(npavg, savg)

    def test_average_with_axis(self):
        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.average(nparr, axis=0)
        smed = sarr.average(axis=0)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.average(nparr, axis=1)
        smed = sarr.average(axis=1)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.average(nparr, axis=2)
        smed = sarr.average(axis=2)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.average(nparr, axis=(0, 1))
        smed = sarr.average(axis=[0, 1])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.average(nparr, axis=(1, 2))
        smed = sarr.average(axis=[1, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.average(nparr, axis=(0, 2))
        smed = sarr.average(axis=[0, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        weights = np.array([0.1, 0.2, 0.3, 0.2, 0.2], dtype='float64')
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sweights = solvcon.SimpleArrayFloat64(array=weights)
        npmed = np.average(nparr, weights=weights, axis=1)
        smed = sarr.average(axis=1, weight=sweights)
        smed = smed.ndarray
        self.assertTrue(np.allclose(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        weights = np.array([
            [0.2, 0.25, 0.25, 0.25, 0.25, 0.25],
            [0.25, 0.2, 0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.2, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.2, 0.25, 0.25]
        ], dtype='float64')
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sweights = solvcon.SimpleArrayFloat64(array=weights)
        npmed = np.average(nparr, weights=weights, axis=(0, 2))
        smed = sarr.average(axis=[0, 2], weight=sweights)
        smed = smed.ndarray
        self.assertTrue(np.allclose(npmed, smed))

        nparr = np.arange(60, dtype='float64').reshape((3, 4, 5))
        weights_axis0 = np.array([0.4, 0.3, 0.3], dtype='float64')
        weights_axis2 = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype='float64')
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sweights_axis0 = solvcon.SimpleArrayFloat64(array=weights_axis0)
        sweights_axis2 = solvcon.SimpleArrayFloat64(array=weights_axis2)

        npmed = np.average(nparr, weights=weights_axis0, axis=0)
        smed = sarr.average(axis=0, weight=sweights_axis0)
        smed = smed.ndarray
        self.assertTrue(np.allclose(npmed, smed))

        npmed = np.average(nparr, weights=weights_axis2, axis=2)
        smed = sarr.average(axis=2, weight=sweights_axis2)
        smed = smed.ndarray
        self.assertTrue(np.allclose(npmed, smed))

        nparr = np.array([[1, 2], [3, 4]], dtype='float64')
        weights = np.array([0.6, 0.4], dtype='float64')
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sweights = solvcon.SimpleArrayFloat64(array=weights)
        npmed = np.average(nparr, weights=weights, axis=1)
        smed = sarr.average(axis=1, weight=sweights)
        smed = smed.ndarray
        self.assertTrue(np.allclose(npmed, smed))

        nparr = np.arange(4 * 3 * 2, dtype='float64').reshape((4, 3, 2))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sarr.nghost = 1
        npavg = np.average(nparr, axis=0)
        savg = sarr.average(axis=0)
        savg = savg.ndarray
        self.assertTrue(np.array_equal(npavg, savg))

        weights = np.array([0.5, 0.3, 0.2], dtype='float64')
        sweights = solvcon.SimpleArrayFloat64(array=weights)
        npavg = np.average(nparr, weights=weights, axis=1)
        savg = sarr.average(axis=1, weight=sweights)
        savg = savg.ndarray
        self.assertTrue(np.allclose(npavg, savg))

    def test_mean(self):
        nparr = np.arange(24, dtype='float64')
        np.random.shuffle(nparr)
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npavg = np.mean(nparr)
        savg = sarr.mean()
        self.assertEqual(npavg, savg)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        np.random.shuffle(nparr)
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmean = np.mean(nparr)
        smean = sarr.mean()
        self.assertEqual(npmean, smean)

        nparr = np.arange(81, dtype='float64').reshape((3, 3, 3, 3))
        np.random.shuffle(nparr)
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmean = np.mean(nparr)
        smean = sarr.mean()
        self.assertEqual(npmean, smean)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        nparr = nparr[::2, ::2, ::2]
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmean = np.mean(nparr)
        smean = sarr.mean()
        self.assertEqual(npmean, smean)

        nparr = np.arange(4 * 3 * 2, dtype='float64').reshape((4, 3, 2))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sarr.nghost = 1
        self.assertEqual(np.mean(nparr), sarr.mean())

    def test_mean_with_axis(self):
        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.mean(nparr, axis=0)
        smed = sarr.mean(axis=0)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.mean(nparr, axis=1)
        smed = sarr.mean(axis=1)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.mean(nparr, axis=2)
        smed = sarr.mean(axis=2)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.mean(nparr, axis=(0, 1))
        smed = sarr.mean(axis=[0, 1])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.mean(nparr, axis=(1, 2))
        smed = sarr.mean(axis=[1, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.mean(nparr, axis=(0, 2))
        smed = sarr.mean(axis=[0, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(4 * 3 * 2, dtype='float64').reshape((4, 3, 2))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sarr.nghost = 1
        npmean = np.mean(nparr, axis=0)
        smean = sarr.mean(axis=0)
        smean = smean.ndarray
        self.assertTrue(np.array_equal(npmean, smean))

    def test_var(self):
        nparr = np.arange(24, dtype='float64')
        np.random.shuffle(nparr)
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npavg = np.var(nparr)
        savg = sarr.var()
        self.assertEqual(npavg, savg)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        np.random.shuffle(nparr)
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npvar = np.var(nparr)
        svar = sarr.var()
        self.assertEqual(npvar, svar)

        nparr = np.arange(24, dtype='complex128').reshape((2, 3, 4))
        npimg = nparr.copy()
        np.random.shuffle(nparr)
        np.random.shuffle(npimg)
        nparr = npimg + 1j * npimg
        sarr = solvcon.SimpleArrayComplex128(array=nparr)
        npvar = np.var(nparr)
        svar = sarr.var()
        self.assertEqual(npvar, svar)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        nparr = nparr[::2, ::2, ::2]
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npvar = np.var(nparr)
        svar = sarr.var()
        self.assertEqual(npvar, svar)

        nparr = np.arange(4 * 3 * 2, dtype='float64').reshape((4, 3, 2))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sarr.nghost = 1
        self.assertEqual(np.var(nparr), sarr.var())

    def test_var_with_axis(self):
        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.var(nparr, axis=0)
        smed = sarr.var(axis=0)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.var(nparr, axis=1)
        smed = sarr.var(axis=1)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.var(nparr, axis=2)
        smed = sarr.var(axis=2)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.var(nparr, axis=(0, 1))
        smed = sarr.var(axis=[0, 1])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.var(nparr, axis=(1, 2))
        smed = sarr.var(axis=[1, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.var(nparr, axis=(0, 2))
        smed = sarr.var(axis=[0, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(4 * 3 * 2, dtype='float64').reshape((4, 3, 2))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sarr.nghost = 1
        npvar = np.var(nparr, axis=1)
        svar = sarr.var(axis=1)
        svar = svar.ndarray
        self.assertTrue(np.allclose(npvar, svar))

        nparr = np.arange(4 * 3 * 2, dtype='float64').reshape((4, 3, 2))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sarr.nghost = 1
        npvar = np.var(nparr, axis=0)
        svar = sarr.var(axis=0)
        svar = svar.ndarray
        self.assertTrue(np.allclose(npvar, svar))

    def test_std(self):
        nparr = np.arange(24, dtype='float64')
        np.random.shuffle(nparr)
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npavg = np.std(nparr)
        savg = sarr.std()
        self.assertEqual(npavg, savg)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        np.random.shuffle(nparr)
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npstd = np.std(nparr)
        sstd = sarr.std()
        self.assertEqual(npstd, sstd)

        nparr = np.arange(24, dtype='complex128').reshape((2, 3, 4))
        npimg = nparr.copy()
        np.random.shuffle(nparr)
        np.random.shuffle(npimg)
        nparr = npimg + 1j * npimg
        sarr = solvcon.SimpleArrayComplex128(array=nparr)
        npstd = np.std(nparr)
        sstd = sarr.std()
        self.assertEqual(npstd, sstd)

        nparr = np.arange(24, dtype='float64').reshape((2, 3, 4))
        nparr = nparr[::2, ::2, ::2]
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npstd = np.std(nparr)
        sstd = sarr.std()
        self.assertEqual(npstd, sstd)

        nparr = np.arange(4 * 3 * 2, dtype='float64').reshape((4, 3, 2))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sarr.nghost = 1
        self.assertEqual(np.std(nparr), sarr.std())

    def test_std_with_axis(self):
        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.std(nparr, axis=0)
        smed = sarr.std(axis=0)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.std(nparr, axis=1)
        smed = sarr.std(axis=1)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.std(nparr, axis=2)
        smed = sarr.std(axis=2)
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.std(nparr, axis=(0, 1))
        smed = sarr.std(axis=[0, 1])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.std(nparr, axis=(1, 2))
        smed = sarr.std(axis=[1, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(120, dtype='float64').reshape((4, 5, 6))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        npmed = np.std(nparr, axis=(0, 2))
        smed = sarr.std(axis=[0, 2])
        smed = smed.ndarray
        self.assertTrue(np.array_equal(npmed, smed))

        nparr = np.arange(4 * 3 * 2, dtype='float64').reshape((4, 3, 2))
        sarr = solvcon.SimpleArrayFloat64(array=nparr)
        sarr.nghost = 1
        npstd = np.std(nparr, axis=0)
        sstd = sarr.std(axis=0)
        sstd = sstd.ndarray
        self.assertTrue(np.allclose(npstd, sstd))

    def type_convertor(self, dtype):
        return {
            'int8': solvcon.SimpleArrayInt8,
            'int16': solvcon.SimpleArrayInt16,
            'int32': solvcon.SimpleArrayInt32,
            'int64': solvcon.SimpleArrayInt64,
            'uint8': solvcon.SimpleArrayUint8,
            'uint16': solvcon.SimpleArrayUint16,
            'uint32': solvcon.SimpleArrayUint32,
            'uint64': solvcon.SimpleArrayUint64,
            'float32': solvcon.SimpleArrayFloat32,
            'float64': solvcon.SimpleArrayFloat64,
        }[dtype]

    def test_add(self):
        # test integer
        def _check_add_type(type):
            arr1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            arr2 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            res = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
            narr1 = np.array(arr1, dtype=type)
            narr2 = np.array(arr2, dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)
            nres = np.add(narr1, narr2)
            sres = sarr1.add(sarr2)
            simdres = sarr1.add_simd(sarr2)
            sarr1.iadd(sarr2)
            for i in range(len(res)):
                self.assertEqual(sres[i], res[i])
                self.assertEqual(sarr1[i], res[i])
                self.assertEqual(simdres[i], res[i])
                self.assertEqual(sres[i], nres[i])

        _check_add_type('int8')
        _check_add_type('int16')
        _check_add_type('int32')
        _check_add_type('int64')
        _check_add_type('uint8')
        _check_add_type('uint16')
        _check_add_type('uint32')
        _check_add_type('uint64')
        _check_add_type('float32')
        _check_add_type('float64')

        # test boolean
        arr1 = [True, True, True, False, False, False]
        arr2 = [True, False, True, False, True, False]
        res = [True, True, True, False, True, False]
        narr1 = np.array(arr1, dtype='bool')
        narr2 = np.array(arr2, dtype='bool')
        sarr1 = solvcon.SimpleArrayBool(array=narr1)
        sarr2 = solvcon.SimpleArrayBool(array=narr2)
        nres = np.add(narr1, narr2)
        sres = sarr1.add(sarr2)
        sarr1.iadd(sarr2)
        for i in range(len(res)):
            self.assertEqual(sres[i], res[i])
            self.assertEqual(sarr1[i], res[i])
            self.assertEqual(sres[i], nres[i])

    def test_add_1d(self):
        def _check_add_1d_type(type):
            narr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=type)
            narr2 = np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)

            nres = np.add(narr1, narr2)
            sres = sarr1.add(sarr2)
            for i in range(len(narr1)):
                self.assertEqual(sres[i], nres[i])

            sarr1.iadd(sarr2)
            for i in range(len(narr1)):
                self.assertEqual(sarr1[i], nres[i])

        _check_add_1d_type('int8')
        _check_add_1d_type('int16')
        _check_add_1d_type('int32')
        _check_add_1d_type('int64')
        _check_add_1d_type('uint8')
        _check_add_1d_type('uint16')
        _check_add_1d_type('uint32')
        _check_add_1d_type('uint64')
        _check_add_1d_type('float32')
        _check_add_1d_type('float64')

        # bool
        narr1 = np.array([True, True, False, False, True, False, True, False],
                         dtype='bool')
        narr2 = np.array([True, False, True, False, True, False, True, False],
                         dtype='bool')
        sarr1 = solvcon.SimpleArrayBool(array=narr1)
        sarr2 = solvcon.SimpleArrayBool(array=narr2)
        nres = np.add(narr1, narr2)
        sres = sarr1.add(sarr2)
        for i in range(len(narr1)):
            self.assertEqual(sres[i], nres[i])
        sarr1.iadd(sarr2)
        for i in range(len(narr1)):
            self.assertEqual(sarr1[i], nres[i])

    def test_add_2d(self):
        def _check_add_2d_type(type):
            narr1 = np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12]], dtype=type)
            narr2 = np.array([[2, 2, 2, 2],
                              [3, 3, 3, 3],
                              [4, 4, 4, 4]], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)

            nres = np.add(narr1, narr2)
            sres = sarr1.add(sarr2)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    self.assertEqual(sres[i, j], nres[i, j])

            sarr1.iadd(sarr2)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    self.assertEqual(sarr1[i, j], nres[i, j])

        _check_add_2d_type('int8')
        _check_add_2d_type('int16')
        _check_add_2d_type('int32')
        _check_add_2d_type('int64')
        _check_add_2d_type('uint8')
        _check_add_2d_type('uint16')
        _check_add_2d_type('uint32')
        _check_add_2d_type('uint64')
        _check_add_2d_type('float32')
        _check_add_2d_type('float64')

        # bool
        narr1 = np.array([[True, False, True, False],
                          [False, True, False, True],
                          [True, True, False, False]], dtype='bool')
        narr2 = np.array([[True, True, False, False],
                          [False, False, True, True],
                          [True, False, True, False]], dtype='bool')
        sarr1 = solvcon.SimpleArrayBool(array=narr1)
        sarr2 = solvcon.SimpleArrayBool(array=narr2)
        nres = np.add(narr1, narr2)
        sres = sarr1.add(sarr2)
        for i in range(narr1.shape[0]):
            for j in range(narr1.shape[1]):
                self.assertEqual(sres[i, j], nres[i, j])
        sarr1.iadd(sarr2)
        for i in range(narr1.shape[0]):
            for j in range(narr1.shape[1]):
                self.assertEqual(sarr1[i, j], nres[i, j])

    def test_add_3d(self):
        def _check_add_3d_type(type):
            narr1 = np.array([[[1, 2],
                               [3, 4]],
                              [[5, 6],
                               [7, 8]]], dtype=type)
            narr2 = np.array([[[2, 2],
                               [2, 2]],
                              [[3, 3],
                               [3, 3]]], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)

            nres = np.add(narr1, narr2)
            sres = sarr1.add(sarr2)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    for k in range(narr1.shape[2]):
                        self.assertEqual(sres[i, j, k], nres[i, j, k])

            sarr1.iadd(sarr2)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    for k in range(narr1.shape[2]):
                        self.assertEqual(sarr1[i, j, k], nres[i, j, k])

        _check_add_3d_type('int8')
        _check_add_3d_type('int16')
        _check_add_3d_type('int32')
        _check_add_3d_type('int64')
        _check_add_3d_type('uint8')
        _check_add_3d_type('uint16')
        _check_add_3d_type('uint32')
        _check_add_3d_type('uint64')
        _check_add_3d_type('float32')
        _check_add_3d_type('float64')

        # bool
        narr1 = np.array([[[True, False],
                           [False, True]],
                          [[True, True],
                           [False, False]]], dtype='bool')
        narr2 = np.array([[[False, True],
                           [True, False]],
                          [[True, False],
                           [True, False]]], dtype='bool')
        sarr1 = solvcon.SimpleArrayBool(array=narr1)
        sarr2 = solvcon.SimpleArrayBool(array=narr2)
        nres = np.add(narr1, narr2)
        sres = sarr1.add(sarr2)
        for i in range(narr1.shape[0]):
            for j in range(narr1.shape[1]):
                for k in range(narr1.shape[2]):
                    self.assertEqual(sres[i, j, k], nres[i, j, k])
        sarr1.iadd(sarr2)
        for i in range(narr1.shape[0]):
            for j in range(narr1.shape[1]):
                for k in range(narr1.shape[2]):
                    self.assertEqual(sarr1[i, j, k], nres[i, j, k])

    def test_add_scalar_1d(self):
        def _check_add_scalar_1d_type(type):
            narr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            scalar = 3

            nres = np.add(narr1, scalar)
            sres = sarr1.add(scalar)
            for i in range(len(narr1)):
                self.assertEqual(sres[i], nres[i])

            sarr1.iadd(scalar)
            for i in range(len(narr1)):
                self.assertEqual(sarr1[i], nres[i])

        _check_add_scalar_1d_type('int8')
        _check_add_scalar_1d_type('int16')
        _check_add_scalar_1d_type('int32')
        _check_add_scalar_1d_type('int64')
        _check_add_scalar_1d_type('uint8')
        _check_add_scalar_1d_type('uint16')
        _check_add_scalar_1d_type('uint32')
        _check_add_scalar_1d_type('uint64')
        _check_add_scalar_1d_type('float32')
        _check_add_scalar_1d_type('float64')

        # bool
        narr1 = np.array([True, False, True, False, True, False, True, False],
                         dtype='bool')
        sarr1 = solvcon.SimpleArrayBool(array=narr1)
        scalar = True
        nres = np.add(narr1, scalar)
        sres = sarr1.add(scalar)
        for i in range(len(narr1)):
            self.assertEqual(sres[i], nres[i])
        sarr1.iadd(scalar)
        for i in range(len(narr1)):
            self.assertEqual(sarr1[i], nres[i])

    def test_add_scalar_2d(self):
        def _check_add_scalar_2d_type(type):
            narr1 = np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12]], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            scalar = 3

            nres = np.add(narr1, scalar)
            sres = sarr1.add(scalar)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    self.assertEqual(sres[i, j], nres[i, j])

            sarr1.iadd(scalar)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    self.assertEqual(sarr1[i, j], nres[i, j])

        _check_add_scalar_2d_type('int8')
        _check_add_scalar_2d_type('int16')
        _check_add_scalar_2d_type('int32')
        _check_add_scalar_2d_type('int64')
        _check_add_scalar_2d_type('uint8')
        _check_add_scalar_2d_type('uint16')
        _check_add_scalar_2d_type('uint32')
        _check_add_scalar_2d_type('uint64')
        _check_add_scalar_2d_type('float32')
        _check_add_scalar_2d_type('float64')

        # bool
        narr1 = np.array([[True, False, True, False],
                          [False, True, False, True],
                          [True, True, False, False]], dtype='bool')
        sarr1 = solvcon.SimpleArrayBool(array=narr1)
        scalar = True
        nres = np.add(narr1, scalar)
        sres = sarr1.add(scalar)
        for i in range(narr1.shape[0]):
            for j in range(narr1.shape[1]):
                self.assertEqual(sres[i, j], nres[i, j])
        sarr1.iadd(scalar)
        for i in range(narr1.shape[0]):
            for j in range(narr1.shape[1]):
                self.assertEqual(sarr1[i, j], nres[i, j])

    def test_add_scalar_3d(self):
        def _check_add_scalar_3d_type(type):
            narr1 = np.array([[[1, 2],
                               [3, 4]],
                              [[5, 6],
                               [7, 8]]], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            scalar = 3

            nres = np.add(narr1, scalar)
            sres = sarr1.add(scalar)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    for k in range(narr1.shape[2]):
                        self.assertEqual(sres[i, j, k], nres[i, j, k])

            sarr1.iadd(scalar)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    for k in range(narr1.shape[2]):
                        self.assertEqual(sarr1[i, j, k], nres[i, j, k])

        _check_add_scalar_3d_type('int8')
        _check_add_scalar_3d_type('int16')
        _check_add_scalar_3d_type('int32')
        _check_add_scalar_3d_type('int64')
        _check_add_scalar_3d_type('uint8')
        _check_add_scalar_3d_type('uint16')
        _check_add_scalar_3d_type('uint32')
        _check_add_scalar_3d_type('uint64')
        _check_add_scalar_3d_type('float32')
        _check_add_scalar_3d_type('float64')

        # bool
        narr1 = np.array([[[True, False],
                           [False, True]],
                          [[True, True],
                           [False, False]]], dtype='bool')
        sarr1 = solvcon.SimpleArrayBool(array=narr1)
        scalar = True
        nres = np.add(narr1, scalar)
        sres = sarr1.add(scalar)
        for i in range(narr1.shape[0]):
            for j in range(narr1.shape[1]):
                for k in range(narr1.shape[2]):
                    self.assertEqual(sres[i, j, k], nres[i, j, k])
        sarr1.iadd(scalar)
        for i in range(narr1.shape[0]):
            for j in range(narr1.shape[1]):
                for k in range(narr1.shape[2]):
                    self.assertEqual(sarr1[i, j, k], nres[i, j, k])

    def test_sub(self):
        # test integer
        def _check_sub_type(type):
            arr1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            arr2 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            narr1 = np.array(arr1, dtype=type)
            narr2 = np.array(arr2, dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)
            nres = np.subtract(narr2, narr1)
            sres = sarr2.sub(sarr1)
            simdres = sarr2.sub_simd(sarr1)
            sarr2.isub(sarr1)
            for i in range(len(arr1)):
                self.assertEqual(sres[i], arr1[i])
                self.assertEqual(sarr2[i], arr1[i])
                self.assertEqual(simdres[i], arr1[i])
                self.assertEqual(sres[i], nres[i])

        _check_sub_type('int8')
        _check_sub_type('int16')
        _check_sub_type('int32')
        _check_sub_type('int64')
        _check_sub_type('uint8')
        _check_sub_type('uint16')
        _check_sub_type('uint32')
        _check_sub_type('uint64')
        _check_sub_type('float32')
        _check_sub_type('float64')

        # test boolean
        arr1 = [True, True, True, False, False, False]
        arr2 = [True, False, True, False, True, False]
        narr1 = np.array(arr1, dtype='bool')
        narr2 = np.array(arr2, dtype='bool')
        sarr1 = solvcon.SimpleArrayBool(array=narr1)
        sarr2 = solvcon.SimpleArrayBool(array=narr2)
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::isub\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr2.sub(sarr1)

        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::isub\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr2.isub(sarr1)

    def test_sub_1d(self):
        def _check_sub_1d_type(type):
            narr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=type)
            narr2 = np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)

            nres = np.subtract(narr1, narr2)
            sres = sarr1.sub(sarr2)
            for i in range(len(narr1)):
                self.assertEqual(sres[i], nres[i])

            sarr1.isub(sarr2)
            for i in range(len(narr1)):
                self.assertEqual(sarr1[i], nres[i])

        _check_sub_1d_type('int8')
        _check_sub_1d_type('int16')
        _check_sub_1d_type('int32')
        _check_sub_1d_type('int64')
        _check_sub_1d_type('uint8')
        _check_sub_1d_type('uint16')
        _check_sub_1d_type('uint32')
        _check_sub_1d_type('uint64')
        _check_sub_1d_type('float32')
        _check_sub_1d_type('float64')

        # bool
        narr1 = np.array([True, True, False, False, True, False, True, False],
                         dtype='bool')
        narr2 = np.array([True, False, True, False, True, False, True, False],
                         dtype='bool')
        sarr1 = solvcon.SimpleArrayBool(array=narr1)
        sarr2 = solvcon.SimpleArrayBool(array=narr2)
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::isub\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr1.sub(sarr2)
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::isub\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr1.isub(sarr2)

    def test_sub_2d(self):
        def _check_sub_2d_type(type):
            narr1 = np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12]], dtype=type)
            narr2 = np.array([[2, 2, 2, 2],
                              [3, 3, 3, 3],
                              [4, 4, 4, 4]], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)

            nres = np.subtract(narr1, narr2)
            sres = sarr1.sub(sarr2)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    self.assertEqual(sres[i, j], nres[i, j])

            sarr1.isub(sarr2)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    self.assertEqual(sarr1[i, j], nres[i, j])

        _check_sub_2d_type('int8')
        _check_sub_2d_type('int16')
        _check_sub_2d_type('int32')
        _check_sub_2d_type('int64')
        _check_sub_2d_type('uint8')
        _check_sub_2d_type('uint16')
        _check_sub_2d_type('uint32')
        _check_sub_2d_type('uint64')
        _check_sub_2d_type('float32')
        _check_sub_2d_type('float64')

        # bool
        narr1 = np.array([[True, False, True, False],
                          [False, True, False, True],
                          [True, True, False, False]], dtype='bool')
        narr2 = np.array([[True, True, False, False],
                          [False, False, True, True],
                          [True, False, True, False]], dtype='bool')
        sarr1 = solvcon.SimpleArrayBool(array=narr1)
        sarr2 = solvcon.SimpleArrayBool(array=narr2)
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::isub\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr1.sub(sarr2)
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::isub\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr1.isub(sarr2)

    def test_sub_3d(self):
        def _check_sub_3d_type(type):
            narr1 = np.array([[[1, 2],
                               [3, 4]],
                              [[5, 6],
                               [7, 8]]], dtype=type)
            narr2 = np.array([[[2, 2],
                               [2, 2]],
                              [[3, 3],
                               [3, 3]]], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)

            nres = np.subtract(narr1, narr2)
            sres = sarr1.sub(sarr2)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    for k in range(narr1.shape[2]):
                        self.assertEqual(sres[i, j, k], nres[i, j, k])

            sarr1.isub(sarr2)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    for k in range(narr1.shape[2]):
                        self.assertEqual(sarr1[i, j, k], nres[i, j, k])

        _check_sub_3d_type('int8')
        _check_sub_3d_type('int16')
        _check_sub_3d_type('int32')
        _check_sub_3d_type('int64')
        _check_sub_3d_type('uint8')
        _check_sub_3d_type('uint16')
        _check_sub_3d_type('uint32')
        _check_sub_3d_type('uint64')
        _check_sub_3d_type('float32')
        _check_sub_3d_type('float64')

        # bool
        narr1 = np.array([[[True, False],
                           [False, True]],
                          [[True, True],
                           [False, False]]], dtype='bool')
        narr2 = np.array([[[False, True],
                           [True, False]],
                          [[True, False],
                           [True, False]]], dtype='bool')
        sarr1 = solvcon.SimpleArrayBool(array=narr1)
        sarr2 = solvcon.SimpleArrayBool(array=narr2)
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::isub\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr1.sub(sarr2)
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::isub\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr1.isub(sarr2)

    def test_sub_scalar_1d(self):
        def _check_sub_scalar_1d_type(type):
            narr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            scalar = 3

            nres = np.subtract(narr1, scalar)
            sres = sarr1.sub(scalar)
            for i in range(len(narr1)):
                self.assertEqual(sres[i], nres[i])

            sarr1.isub(scalar)
            for i in range(len(narr1)):
                self.assertEqual(sarr1[i], nres[i])

        _check_sub_scalar_1d_type('int8')
        _check_sub_scalar_1d_type('int16')
        _check_sub_scalar_1d_type('int32')
        _check_sub_scalar_1d_type('int64')
        _check_sub_scalar_1d_type('uint8')
        _check_sub_scalar_1d_type('uint16')
        _check_sub_scalar_1d_type('uint32')
        _check_sub_scalar_1d_type('uint64')
        _check_sub_scalar_1d_type('float32')
        _check_sub_scalar_1d_type('float64')

        # bool
        narr1 = np.array([True, False, True, False, True, False, True, False],
                         dtype='bool')
        sarr1 = solvcon.SimpleArrayBool(array=narr1)
        scalar = True
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::isub\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr1.sub(scalar)
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::isub\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr1.isub(scalar)

    def test_sub_scalar_2d(self):
        def _check_sub_scalar_2d_type(type):
            narr1 = np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12]], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            scalar = 3

            nres = np.subtract(narr1, scalar)
            sres = sarr1.sub(scalar)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    self.assertEqual(sres[i, j], nres[i, j])

            sarr1.isub(scalar)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    self.assertEqual(sarr1[i, j], nres[i, j])

        _check_sub_scalar_2d_type('int8')
        _check_sub_scalar_2d_type('int16')
        _check_sub_scalar_2d_type('int32')
        _check_sub_scalar_2d_type('int64')
        _check_sub_scalar_2d_type('uint8')
        _check_sub_scalar_2d_type('uint16')
        _check_sub_scalar_2d_type('uint32')
        _check_sub_scalar_2d_type('uint64')
        _check_sub_scalar_2d_type('float32')
        _check_sub_scalar_2d_type('float64')

        # bool
        narr1 = np.array([[True, False, True, False],
                          [False, True, False, True],
                          [True, True, False, False]], dtype='bool')
        sarr1 = solvcon.SimpleArrayBool(array=narr1)
        scalar = True
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::isub\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr1.sub(scalar)
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::isub\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr1.isub(scalar)

    def test_sub_scalar_3d(self):
        def _check_sub_scalar_3d_type(type):
            narr1 = np.array([[[1, 2],
                               [3, 4]],
                              [[5, 6],
                               [7, 8]]], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            scalar = 3

            nres = np.subtract(narr1, scalar)
            sres = sarr1.sub(scalar)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    for k in range(narr1.shape[2]):
                        self.assertEqual(sres[i, j, k], nres[i, j, k])

            sarr1.isub(scalar)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    for k in range(narr1.shape[2]):
                        self.assertEqual(sarr1[i, j, k], nres[i, j, k])

        _check_sub_scalar_3d_type('int8')
        _check_sub_scalar_3d_type('int16')
        _check_sub_scalar_3d_type('int32')
        _check_sub_scalar_3d_type('int64')
        _check_sub_scalar_3d_type('uint8')
        _check_sub_scalar_3d_type('uint16')
        _check_sub_scalar_3d_type('uint32')
        _check_sub_scalar_3d_type('uint64')
        _check_sub_scalar_3d_type('float32')
        _check_sub_scalar_3d_type('float64')

        # bool
        narr1 = np.array([[[True, False],
                           [False, True]],
                          [[True, True],
                           [False, False]]], dtype='bool')
        sarr1 = solvcon.SimpleArrayBool(array=narr1)
        scalar = True
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::isub\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr1.sub(scalar)
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::isub\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr1.isub(scalar)

    def test_mul(self):
        def _check_mul_type(type):
            arr1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            arr2 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
            res = [2, 8, 18, 32, 50, 72, 98, 128, 162, 200]
            narr1 = np.array(arr1, dtype=type)
            narr2 = np.array(arr2, dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)
            nres = np.multiply(narr2, narr1)
            sres = sarr1.mul(sarr2)
            simdres = sarr1.mul_simd(sarr2)
            sarr1.imul(sarr2)
            for i in range(len(res)):
                self.assertEqual(sres[i], res[i])
                self.assertEqual(sarr1[i], res[i])
                self.assertEqual(simdres[i], res[i])
                self.assertEqual(sres[i], nres[i])

        _check_mul_type('int16')
        _check_mul_type('int32')
        _check_mul_type('int64')
        _check_mul_type('uint8')
        _check_mul_type('uint16')
        _check_mul_type('uint32')
        _check_mul_type('uint64')
        _check_mul_type('float32')
        _check_mul_type('float64')

        # test boolean
        arr1 = [True, True, True, False, False, False]
        arr2 = [True, False, True, False, True, False]
        res = [True, False, True, False, False, False]
        narr1 = np.array(arr1, dtype='bool')
        narr2 = np.array(arr2, dtype='bool')
        sarr1 = solvcon.SimpleArrayBool(array=narr1)
        sarr2 = solvcon.SimpleArrayBool(array=narr2)
        nres = np.multiply(narr2, narr1)
        sres = sarr1.mul(sarr2)
        sarr1.imul(sarr2)
        for i in range(len(res)):
            self.assertEqual(sres[i], res[i])
            self.assertEqual(sarr1[i], res[i])
            self.assertEqual(sres[i], nres[i])

    def test_mul_1d(self):
        def _check_mul_1d_type(type):
            narr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=type)
            narr2 = np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)

            nres = np.multiply(narr1, narr2)
            sres = sarr1.mul(sarr2)
            for i in range(len(narr1)):
                self.assertEqual(sres[i], nres[i])

            sarr1.imul(sarr2)
            for i in range(len(narr1)):
                self.assertEqual(sarr1[i], nres[i])

        _check_mul_1d_type('int32')
        _check_mul_1d_type('int64')
        _check_mul_1d_type('float32')
        _check_mul_1d_type('float64')

    def test_mul_2d(self):
        def _check_mul_2d_type(type):
            narr1 = np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12]], dtype=type)
            narr2 = np.array([[2, 2, 2, 2],
                              [3, 3, 3, 3],
                              [4, 4, 4, 4]], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)

            nres = np.multiply(narr1, narr2)
            sres = sarr1.mul(sarr2)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    self.assertEqual(sres[i, j], nres[i, j])

            sarr1.imul(sarr2)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    self.assertEqual(sarr1[i, j], nres[i, j])

        _check_mul_2d_type('int32')
        _check_mul_2d_type('int64')
        _check_mul_2d_type('float32')
        _check_mul_2d_type('float64')

    def test_mul_3d(self):
        def _check_mul_3d_type(type):
            narr1 = np.arange(24, dtype=type).reshape((2, 3, 4))
            narr2 = np.arange(1, 25, dtype=type).reshape((2, 3, 4))
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)

            nres = np.multiply(narr1, narr2)
            sres = sarr1.mul(sarr2)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    for k in range(narr1.shape[2]):
                        self.assertEqual(sres[i, j, k], nres[i, j, k])

            sarr1.imul(sarr2)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    for k in range(narr1.shape[2]):
                        self.assertEqual(sarr1[i, j, k], nres[i, j, k])

        _check_mul_3d_type('int32')
        _check_mul_3d_type('int64')
        _check_mul_3d_type('float32')
        _check_mul_3d_type('float64')

    def test_mul_scalar_1d(self):
        """Test 1D array scalar multiplication"""
        def _check_mul_scalar_1d_type(type):
            narr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            scalar = 3

            nres = np.multiply(narr1, scalar)
            sres = sarr1.mul(scalar)

            for i in range(len(narr1)):
                self.assertEqual(sres[i], nres[i])

            sarr1.imul(scalar)
            for i in range(len(narr1)):
                self.assertEqual(sarr1[i], nres[i])

        _check_mul_scalar_1d_type('int32')
        _check_mul_scalar_1d_type('int64')
        _check_mul_scalar_1d_type('float32')
        _check_mul_scalar_1d_type('float64')

    def test_mul_scalar_2d(self):
        """Test 2D array (matrix) scalar multiplication"""
        def _check_mul_scalar_2d_type(type):
            narr1 = np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12]], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            scalar = 2

            nres = np.multiply(narr1, scalar)
            sres = sarr1.mul(scalar)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    self.assertEqual(sres[i, j], nres[i, j])

            sarr1.imul(scalar)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    self.assertEqual(sarr1[i, j], nres[i, j])

        _check_mul_scalar_2d_type('int32')
        _check_mul_scalar_2d_type('int64')
        _check_mul_scalar_2d_type('float32')
        _check_mul_scalar_2d_type('float64')

    def test_mul_scalar_3d(self):
        """Test 3D array scalar multiplication"""
        def _check_mul_scalar_3d_type(type):
            narr1 = np.arange(24, dtype=type).reshape((2, 3, 4))
            sarr1 = self.type_convertor(type)(array=narr1)
            scalar = 5

            nres = np.multiply(narr1, scalar)
            sres = sarr1.mul(scalar)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    for k in range(narr1.shape[2]):
                        self.assertEqual(sres[i, j, k], nres[i, j, k])

            sarr1.imul(scalar)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    for k in range(narr1.shape[2]):
                        self.assertEqual(sarr1[i, j, k], nres[i, j, k])

        _check_mul_scalar_3d_type('int32')
        _check_mul_scalar_3d_type('int64')
        _check_mul_scalar_3d_type('float32')
        _check_mul_scalar_3d_type('float64')

    def test_div(self):
        arr1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        arr2 = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

        def _check_div_type(type):
            res = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            narr1 = np.array(arr1, dtype=type)
            narr2 = np.array(arr2, dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)
            nres = np.divide(narr2, narr1)
            sres = sarr2.div(sarr1)
            simdres = sarr2.div_simd(sarr1)
            sarr2.idiv(sarr1)
            for i in range(len(res)):
                self.assertEqual(sres[i], res[i])
                self.assertEqual(sarr2[i], res[i])
                self.assertEqual(simdres[i], res[i])
                self.assertEqual(sres[i], nres[i])

        _check_div_type('int8')
        _check_div_type('int16')
        _check_div_type('int32')
        _check_div_type('int64')
        _check_div_type('uint8')
        _check_div_type('uint16')
        _check_div_type('uint32')
        _check_div_type('uint64')

        # test float
        res = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        narr1 = np.array(arr1, dtype='float64') / 5
        narr2 = np.array(arr2, dtype='float64') / 2
        sarr1 = solvcon.SimpleArrayFloat64(array=narr1)
        sarr2 = solvcon.SimpleArrayFloat64(array=narr2)
        nres = np.divide(narr2, narr1)
        sres = sarr2.div(sarr1)
        simdres = sarr2.div_simd(sarr1)
        sarr2.idiv(sarr1)
        for i in range(len(res)):
            self.assertEqual(sres[i], res[i])
            self.assertEqual(sarr2[i], res[i])
            self.assertEqual(simdres[i], res[i])
            self.assertEqual(sres[i], nres[i])

        # test boolean
        arr1 = [True, True, True, False, False, False]
        arr2 = [True, False, True, False, True, False]
        res = [True, True, True, False, True, False]
        narr1 = np.array(arr1, dtype='bool')
        narr2 = np.array(arr2, dtype='bool')
        sarr1 = solvcon.SimpleArrayBool(array=narr1)
        sarr2 = solvcon.SimpleArrayBool(array=narr2)
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::idiv\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr2.div(sarr1)

        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray<bool>::idiv\(\): "
            r"boolean value doesn't support this operation"
        ):
            sarr2.idiv(sarr1)

    def test_div_1d(self):
        def _check_div_1d_type(type):
            narr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=type)
            narr2 = np.array([2, 3, 4, 5, 6, 7, 8, 9], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)

            # cast the result array back to the tested data type
            # Numpy promotes any numerical types to floating point number when
            # performing division
            # ref: https://numpy.org/doc/stable/reference/arrays.promotion.html
            nres = np.divide(narr1, narr2).astype(type)
            sres = sarr1.div(sarr2)
            for i in range(len(narr1)):
                self.assertEqual(sres[i], nres[i])

            sarr1.idiv(sarr2)
            for i in range(len(narr1)):
                self.assertEqual(sarr1[i], nres[i])

        _check_div_1d_type('int32')
        _check_div_1d_type('int64')
        _check_div_1d_type('float32')
        _check_div_1d_type('float64')

    def test_div_2d(self):
        def _check_div_2d_type(type):
            narr1 = np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12]], dtype=type)
            narr2 = np.array([[2, 2, 2, 2],
                              [3, 3, 3, 3],
                              [4, 4, 4, 4]], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)

            # cast the result array back to the tested data type
            # Numpy promotes any numerical types to floating point number when
            # performing division
            # ref: https://numpy.org/doc/stable/reference/arrays.promotion.html
            nres = np.divide(narr1, narr2).astype(type)
            sres = sarr1.div(sarr2)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    self.assertEqual(sres[i, j], nres[i, j])

            sarr1.idiv(sarr2)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    self.assertEqual(sarr1[i, j], nres[i, j])

        _check_div_2d_type('int32')
        _check_div_2d_type('int64')
        _check_div_2d_type('float32')
        _check_div_2d_type('float64')

    def test_div_3d(self):
        def _check_div_3d_type(type):
            narr1 = np.arange(24, dtype=type).reshape((2, 3, 4))
            narr2 = np.arange(1, 25, dtype=type).reshape((2, 3, 4))
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)

            # cast the result array back to the tested data type
            # Numpy promotes any numerical types to floating point number when
            # performing division
            # ref: https://numpy.org/doc/stable/reference/arrays.promotion.html
            nres = np.divide(narr1, narr2).astype(type)
            sres = sarr1.div(sarr2)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    for k in range(narr1.shape[2]):
                        self.assertEqual(sres[i, j, k], nres[i, j, k])

            sarr1.idiv(sarr2)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    for k in range(narr1.shape[2]):
                        self.assertEqual(sarr1[i, j, k], nres[i, j, k])

        _check_div_3d_type('int32')
        _check_div_3d_type('int64')
        _check_div_3d_type('float32')
        _check_div_3d_type('float64')

    def test_div_scalar_1d(self):
        """Test 1D array scalar divtiplication"""
        def _check_div_scalar_1d_type(type):
            narr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            scalar = 3

            # cast the result array back to the tested data type
            # Numpy promotes any numerical types to floating point number when
            # performing division
            # ref: https://numpy.org/doc/stable/reference/arrays.promotion.html
            nres = np.divide(narr1, scalar).astype(type)
            sres = sarr1.div(scalar)

            for i in range(len(narr1)):
                self.assertEqual(sres[i], nres[i])

            sarr1.idiv(scalar)
            for i in range(len(narr1)):
                self.assertEqual(sarr1[i], nres[i])

        _check_div_scalar_1d_type('int32')
        _check_div_scalar_1d_type('int64')
        _check_div_scalar_1d_type('float32')
        _check_div_scalar_1d_type('float64')

    def test_div_scalar_2d(self):
        """Test 2D array (matrix) scalar divtiplication"""
        def _check_div_scalar_2d_type(type):
            narr1 = np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12]], dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            scalar = 2

            # cast the result array back to the tested data type
            # Numpy promotes any numerical types to floating point number when
            # performing division
            # ref: https://numpy.org/doc/stable/reference/arrays.promotion.html
            nres = np.divide(narr1, scalar).astype(type)
            sres = sarr1.div(scalar)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    self.assertEqual(sres[i, j], nres[i, j])

            sarr1.idiv(scalar)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    self.assertEqual(sarr1[i, j], nres[i, j])

        _check_div_scalar_2d_type('int32')
        _check_div_scalar_2d_type('int64')
        _check_div_scalar_2d_type('float32')
        _check_div_scalar_2d_type('float64')

    def test_div_scalar_3d(self):
        """Test 3D array scalar divtiplication"""
        def _check_div_scalar_3d_type(type):
            narr1 = np.arange(24, dtype=type).reshape((2, 3, 4))
            sarr1 = self.type_convertor(type)(array=narr1)
            scalar = 5

            # cast the result array back to the tested data type
            # Numpy promotes any numerical types to floating point number when
            # performing division
            # ref: https://numpy.org/doc/stable/reference/arrays.promotion.html
            nres = np.divide(narr1, scalar).astype(type)
            sres = sarr1.div(scalar)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    for k in range(narr1.shape[2]):
                        self.assertEqual(sres[i, j, k], nres[i, j, k])

            sarr1.idiv(scalar)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    for k in range(narr1.shape[2]):
                        self.assertEqual(sarr1[i, j, k], nres[i, j, k])

        _check_div_scalar_3d_type('int32')
        _check_div_scalar_3d_type('int64')
        _check_div_scalar_3d_type('float32')
        _check_div_scalar_3d_type('float64')

    def test_eq_1d(self):
        """Test 1D array element-wise equality"""
        def _check_eq_1d_type(type):
            arr1 = [1, 2, 3, 4, 5, 6, 7, 8]
            arr2 = [1, 0, 3, 0, 5, 0, 7, 0]
            res = [True, False, True, False, True, False, True, False]
            narr1 = np.array(arr1, dtype=type)
            narr2 = np.array(arr2, dtype=type)
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)
            nres = np.equal(narr1, narr2)
            sres = sarr1.eq(sarr2)
            self.assertEqual(sres.ndarray.dtype, np.bool_)
            for i in range(len(res)):
                self.assertEqual(sres[i], res[i])
                self.assertEqual(sres[i], nres[i])

        _check_eq_1d_type('int8')
        _check_eq_1d_type('int16')
        _check_eq_1d_type('int32')
        _check_eq_1d_type('int64')
        _check_eq_1d_type('uint8')
        _check_eq_1d_type('uint16')
        _check_eq_1d_type('uint32')
        _check_eq_1d_type('uint64')
        _check_eq_1d_type('float32')
        _check_eq_1d_type('float64')

        with self.assertRaisesRegex(
            ValueError,
            r"SimpleArray::eq\(\): shape mismatch: "
            r"this=\(8\) other=\(3\)"
        ):
            sarr1 = solvcon.SimpleArrayInt32(array=np.arange(8, dtype='int32'))
            sarr2 = solvcon.SimpleArrayInt32(array=np.arange(3, dtype='int32'))
            sarr1.eq(sarr2)

    def test_eq_2d(self):
        """Test 2D array element-wise equality"""
        def _check_eq_2d_type(type):
            narr1 = np.array([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12]], dtype=type)
            narr2 = np.array([[1, 0, 3, 0],
                              [5, 0, 7, 0],
                              [9, 0, 11, 0]], dtype=type)
            res = [[True, False, True, False],
                   [True, False, True, False],
                   [True, False, True, False]]
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)
            nres = np.equal(narr1, narr2)
            sres = sarr1.eq(sarr2)
            self.assertEqual(sres.ndarray.dtype, np.bool_)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    self.assertEqual(sres[i, j], res[i][j])
                    self.assertEqual(sres[i, j], nres[i, j])

        _check_eq_2d_type('int8')
        _check_eq_2d_type('int16')
        _check_eq_2d_type('int32')
        _check_eq_2d_type('int64')
        _check_eq_2d_type('uint8')
        _check_eq_2d_type('uint16')
        _check_eq_2d_type('uint32')
        _check_eq_2d_type('uint64')
        _check_eq_2d_type('float32')
        _check_eq_2d_type('float64')

    def test_eq_scalar_1d(self):
        """Test 1D array element-wise equality against a scalar"""
        def _check_eq_scalar_1d_type(type):
            narr1 = np.array([1, 2, 3, 4, 5, 3, 7, 3], dtype=type)
            scalar = 3
            res = [False, False, True, False, False, True, False, True]
            sarr1 = self.type_convertor(type)(array=narr1)
            nres = np.equal(narr1, scalar)
            sres = sarr1.eq(scalar)
            self.assertEqual(sres.ndarray.dtype, np.bool_)
            for i in range(len(res)):
                self.assertEqual(sres[i], res[i])
                self.assertEqual(sres[i], nres[i])

        _check_eq_scalar_1d_type('int8')
        _check_eq_scalar_1d_type('int16')
        _check_eq_scalar_1d_type('int32')
        _check_eq_scalar_1d_type('int64')
        _check_eq_scalar_1d_type('uint8')
        _check_eq_scalar_1d_type('uint16')
        _check_eq_scalar_1d_type('uint32')
        _check_eq_scalar_1d_type('uint64')
        _check_eq_scalar_1d_type('float32')
        _check_eq_scalar_1d_type('float64')

    def test_eq_scalar_2d(self):
        """Test 2D array element-wise equality against a scalar"""
        def _check_eq_scalar_2d_type(type):
            narr1 = np.array([[1, 5, 3, 5],
                              [5, 6, 7, 5],
                              [9, 5, 11, 12]], dtype=type)
            scalar = 5
            res = [[False, True, False, True],
                   [True, False, False, True],
                   [False, True, False, False]]
            sarr1 = self.type_convertor(type)(array=narr1)
            nres = np.equal(narr1, scalar)
            sres = sarr1.eq(scalar)
            self.assertEqual(sres.ndarray.dtype, np.bool_)
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    self.assertEqual(sres[i, j], res[i][j])
                    self.assertEqual(sres[i, j], nres[i, j])

        _check_eq_scalar_2d_type('int8')
        _check_eq_scalar_2d_type('int16')
        _check_eq_scalar_2d_type('int32')
        _check_eq_scalar_2d_type('int64')
        _check_eq_scalar_2d_type('uint8')
        _check_eq_scalar_2d_type('uint16')
        _check_eq_scalar_2d_type('uint32')
        _check_eq_scalar_2d_type('uint64')
        _check_eq_scalar_2d_type('float32')
        _check_eq_scalar_2d_type('float64')

    def test_eq_3d(self):
        """Test 3D array element-wise equality preserves shape"""
        def _check_eq_3d_type(type):
            narr1 = np.arange(24, dtype=type).reshape(2, 3, 4)
            narr2 = narr1.copy()
            narr2[0, 1, 2] = 0
            narr2[1, 2, 3] = 0
            res = [[[True, True, True, True],
                    [True, True, False, True],
                    [True, True, True, True]],
                   [[True, True, True, True],
                    [True, True, True, True],
                    [True, True, True, False]]]
            sarr1 = self.type_convertor(type)(array=narr1)
            sarr2 = self.type_convertor(type)(array=narr2)
            nres = np.equal(narr1, narr2)
            sres = sarr1.eq(sarr2)
            self.assertEqual(sres.ndarray.dtype, np.bool_)
            self.assertEqual(list(sres.ndarray.shape), [2, 3, 4])
            for i in range(narr1.shape[0]):
                for j in range(narr1.shape[1]):
                    for k in range(narr1.shape[2]):
                        self.assertEqual(sres[i, j, k], res[i][j][k])
                        self.assertEqual(sres[i, j, k], nres[i, j, k])

        _check_eq_3d_type('int32')
        _check_eq_3d_type('uint8')
        _check_eq_3d_type('float64')

    def test_eq_bool(self):
        """eq() on boolean source arrays still yields a boolean array"""
        narr1 = np.array([True, True, False, False], dtype='bool')
        narr2 = np.array([True, False, True, False], dtype='bool')
        res = [True, False, False, True]
        sarr1 = solvcon.SimpleArrayBool(array=narr1)
        sarr2 = solvcon.SimpleArrayBool(array=narr2)

        nres = np.equal(narr1, narr2)
        sres = sarr1.eq(sarr2)
        self.assertEqual(sres.ndarray.dtype, np.bool_)
        for i in range(len(res)):
            self.assertEqual(sres[i], res[i])
            self.assertEqual(sres[i], nres[i])

        res_scalar = [True, True, False, False]
        nres_scalar = np.equal(narr1, True)
        sres_scalar = sarr1.eq(True)
        self.assertEqual(sres_scalar.ndarray.dtype, np.bool_)
        for i in range(len(res_scalar)):
            self.assertEqual(sres_scalar[i], res_scalar[i])
            self.assertEqual(sres_scalar[i], nres_scalar[i])

    def test_eq_does_not_bind_operator(self):
        """__eq__ is intentionally left unbound (issue #810).

        Python's == must therefore keep its default object-identity
        behavior; element-wise comparison is reachable only through eq().
        """
        narr = np.arange(4, dtype='int32')
        sarr1 = solvcon.SimpleArrayInt32(array=narr)
        sarr2 = solvcon.SimpleArrayInt32(array=narr.copy())
        # Identity comparison returns a plain bool, not a SimpleArray.
        self.assertIsInstance(sarr1 == sarr2, bool)
        self.assertFalse(sarr1 == sarr2)
        self.assertTrue(sarr1 == sarr1)
        # The element-wise path is eq(), which returns a boolean array.
        self.assertEqual(sarr1.eq(sarr2).ndarray.dtype, np.bool_)

    def test_eye(self):
        """Test eye() static method for creating identity matrices"""
        eye2 = solvcon.SimpleArrayFloat64.eye(10)
        for i in range(10):
            for j in range(10):
                if i == j:
                    self.assertEqual(eye2[i, j], 1.0)
                else:
                    self.assertEqual(eye2[i, j], 0.0)

        with self.assertRaisesRegex(
            ValueError,
            r"SimpleArray::eye\(\): size must be greater than 0, but got 0"
        ):
            solvcon.SimpleArrayFloat64.eye(0)

        with self.assertRaisesRegex(
            ValueError,
            r"SimpleArray::eye\(\): size must be greater than 0, but got -1"
        ):
            solvcon.SimpleArrayFloat64.eye(-1)

    def test_scaled_eye(self):
        """Test scaled_eye() static method for creating scaled identity
        matrices"""
        scaled_eye = solvcon.SimpleArrayFloat64.scaled_eye(10, 3.5)
        for i in range(10):
            for j in range(10):
                if i == j:
                    self.assertEqual(scaled_eye[i, j], 3.5)
                else:
                    self.assertEqual(scaled_eye[i, j], 0.0)

        with self.assertRaisesRegex(
            ValueError,
            r"SimpleArray::scaled_eye\(\): size must be greater than 0, "
            r"but got 0"
        ):
            solvcon.SimpleArrayFloat64.scaled_eye(0, 2.0)

        with self.assertRaisesRegex(
            ValueError,
            r"SimpleArray::scaled_eye\(\): size must be greater than 0, "
            r"but got -1"
        ):
            solvcon.SimpleArrayFloat64.scaled_eye(-1, 2.0)

    def test_hermitian(self):
        """Test hermitian() method for creating conjugate transpose"""
        ndarr = np.array([[1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j],
                          [7.0 + 8.0j, 9.0 + 10.0j, 11.0 + 12.0j],
                          [13.0 + 14.0j, 15.0 + 16.0j, 17.0 + 18.0j]],
                         dtype='complex128')

        sarr = solvcon.SimpleArrayComplex128(array=ndarr)
        shermitian = sarr.hermitian()
        ndhermitian = ndarr.conj().T
        print(shermitian.ndarray)
        print(ndhermitian)
        self.assertEqual(shermitian.shape, ndhermitian.shape)
        np.testing.assert_equal(shermitian.ndarray, ndhermitian)

        vector = solvcon.SimpleArrayFloat64(5)
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray::hermitian\(\): operation requires 2D SimpleArray, "
            r"but got 1D SimpleArray"
        ):
            vector.hermitian()

    def test_symmetrize(self):
        """Test symmetrize() method for creating symmetric matrices"""
        ndarr = np.array([[1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j],
                         [7.0 + 8.0j, 9.0 + 10.0j, 11.0 + 12.0j],
                         [13.0 + 14.0j, 15.0 + 16.0j, 17.0 + 18.0j]],
                         dtype='complex128')
        sarr = solvcon.SimpleArrayComplex128(array=ndarr)
        symmetric = sarr.symmetrize()
        ndsymmetric = (ndarr + ndarr.conj().T) / 2.0
        self.assertEqual(symmetric.shape, ndsymmetric.shape)
        np.testing.assert_equal(symmetric.ndarray, ndsymmetric)

        vector = solvcon.SimpleArrayFloat64(5)
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray::symmetrize\(\): operation requires 2D SimpleArray, "
            r"but got 1D SimpleArray"
        ):
            vector.symmetrize()

        non_square = solvcon.SimpleArrayFloat64((3, 4))
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray::symmetrize\(\): operation requires square "
            r"SimpleArray, but got 3x4 shape"
        ):
            non_square.symmetrize()

    def test_trace(self):
        """Test trace() method for summing the diagonal of a matrix"""
        ndarr = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]], dtype='float64')
        sarr = solvcon.SimpleArrayFloat64(array=ndarr)
        self.assertEqual(sarr.trace(), 15.0)

        ndarr_int = np.array([[2, 0, 1],
                              [0, 3, 0],
                              [4, 0, 5]], dtype='int32')
        sarr_int = solvcon.SimpleArrayInt32(array=ndarr_int)
        self.assertEqual(sarr_int.trace(), 10)

        ndarr_cplx = np.array([[1.0 + 2.0j, 3.0 + 4.0j],
                               [5.0 + 6.0j, 7.0 + 8.0j]],
                              dtype='complex128')
        sarr_cplx = solvcon.SimpleArrayComplex128(array=ndarr_cplx)
        self.assertEqual(complex(sarr_cplx.trace()), 8.0 + 10.0j)

        vector = solvcon.SimpleArrayFloat64(5)
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray::trace\(\): operation requires 2D SimpleArray, "
            r"but got 1D SimpleArray"
        ):
            vector.trace()

        non_square = solvcon.SimpleArrayFloat64((3, 4))
        with self.assertRaisesRegex(
            RuntimeError,
            r"SimpleArray::trace\(\): operation requires square "
            r"SimpleArray, but got 3x4 shape"
        ):
            non_square.trace()


class SimpleArraySearchTC(unittest.TestCase):

    def test_argminmax(self):
        # test 1-D data
        data = [1, 3, 5, 7, 9]
        narr = np.array(data, dtype='uint64')
        sarr = solvcon.SimpleArrayUint64(array=narr)

        self.assertEqual(sarr.argmin(), 0)
        self.assertEqual(sarr.argmax(), 4)
        self.assertEqual(narr.argmin(), sarr.argmin())
        self.assertEqual(narr.argmax(), sarr.argmax())

        # test N-D data
        data = [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10], [1, 10, 1, 10, 1]]
        narr = np.array(data, dtype='float64')
        sarr = solvcon.SimpleArrayFloat64(array=narr)
        self.assertEqual(sarr.argmin(), 0)
        self.assertEqual(sarr.argmax(), 9)
        self.assertEqual(narr.argmin(), sarr.argmin())
        self.assertEqual(narr.argmax(), sarr.argmax())
    
    def test_argwhere(self):
        # test 1-D data
        data = [1, 3, 5, 7, 9]
        narr = np.array(data, dtype='uint64')
        sarr = solvcon.SimpleArrayUint64(array=narr)

        ret = sarr.eq(5).argwhere()
        np.testing.assert_array_equal(ret.ndarray, np.argwhere(narr == 5))

        # test N-D data
        data = [
            [1, 3, 5, 7, 9],
            [2, 4, 6, 8, 10],
            [1, 10, 1, 10, 1]
        ]
        narr = np.array(data, dtype='float64')
        sarr = solvcon.SimpleArrayFloat64(array=narr)

        ret = sarr.eq(10).argwhere()
        np.testing.assert_array_equal(ret.ndarray, np.argwhere(narr == 10))


class SimpleArrayPlexTC(unittest.TestCase):

    def test_SimpleArrayPlex_constructor(self):
        # 1. shape constructor
        dtype_list = [
            "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32",
            "int64", "uint64", "float32", "float64"
        ]
        for dtype in dtype_list:
            solvcon.SimpleArray((2, 3, 4), dtype=dtype)

        # 2. shape and value constructor
        solvcon.SimpleArray((2, 3, 4), dtype="bool", value=True)
        with self.assertRaisesRegex(
                TypeError,
                r"Data type mismatch, expected Python bool"
        ):
            solvcon.SimpleArray((2, 3, 4), dtype="bool", value=3.3)

        dtype_list_int = [
            "int8", "uint8", "int16", "uint16", "int32", "uint32",
            "int64", "uint64"
        ]
        for dtype in dtype_list_int:
            solvcon.SimpleArray((2, 3, 4), dtype=dtype, value=3)
            with self.assertRaisesRegex(
                    TypeError,
                    r"Data type mismatch, expected Python int"
            ):
                solvcon.SimpleArray((2, 3, 4), dtype=dtype, value=3.3)

        dtype_list_float = ["float32", "float64"]
        for dtype in dtype_list_float:
            solvcon.SimpleArray((2, 3, 4), dtype=dtype, value=3.0)
            with self.assertRaisesRegex(
                    TypeError,
                    r"Data type mismatch, expected Python float"
            ):
                solvcon.SimpleArray((2, 3, 4), dtype=dtype, value=3)

        # 3. np.ndarray constructor
        # exclude bool, since it cannot use np.arange
        dtype_list_no_bool = [
            "int8", "uint8", "int16", "uint16", "int32", "uint32",
            "int64", "uint64", "float32", "float64"
        ]
        for dtype in dtype_list_no_bool:
            ndarr = np.arange(2 * 3 * 4, dtype=dtype)
            solvcon.SimpleArray(ndarr)
        boolean_array = np.array([True, False, True], dtype='bool')
        solvcon.SimpleArray(boolean_array)

    def test_SimpleArray_clone(self):
        sarr = solvcon.SimpleArray((2, 3, 4), value=2.0, dtype='float64')
        sarr_ref = sarr
        sarr_clone = sarr.clone()

        self.assertTrue(sarr_ref is sarr)
        ref_ndarr = sarr_ref.typed.ndarray[...]
        ndarr = sarr.typed.ndarray[...]
        np.testing.assert_equal(ref_ndarr, ndarr)

        self.assertFalse(sarr_clone is sarr)
        clone_ndarr = sarr_clone.typed.ndarray[...]
        ndarr = sarr.typed.ndarray[...]
        np.testing.assert_equal(clone_ndarr, ndarr)

        sarr[0, 0, 3] = 3.0
        self.assertEqual(sarr_ref[0, 0, 3], 3.0)
        self.assertEqual(sarr_clone[0, 0, 3], 2.0)

    def test_SimpleArrayPlex_buffer(self):
        magic_number = 3.1415
        sarr = solvcon.SimpleArray(
            (2, 3, 4), value=magic_number, dtype='float64')
        ndarr = np.array(sarr, copy=False)
        self.assertEqual((2, 3, 4), ndarr.shape)
        self.assertEqual((ndarr == magic_number).all(), True)

    def test_SimpleArrayPlex_get_item(self):
        magic_number = 3.1415
        sarr = solvcon.SimpleArray(
            (2, 3, 4), value=magic_number, dtype='float64')
        self.assertEqual(sarr[1, 2, 3], magic_number)

    def test_SimpleArrayPlex_properties(self):
        magic_number = 3.1415
        shape = (2, 3, 4)
        sarr = solvcon.SimpleArray(shape, value=magic_number, dtype='float64')

        self.assertEqual(sarr.nbytes, 2 * 3 * 4 * 8)
        self.assertEqual(sarr.size, 2 * 3 * 4)
        self.assertEqual(sarr.itemsize, 8)
        self.assertEqual(sarr.stride, (12, 4, 1))
        self.assertEqual(len(sarr), 2 * 3 * 4)  # number of elements
        self.assertEqual(sarr.nbody, 2 - 0)
        self.assertEqual(sarr.has_ghost, False)
        self.assertEqual(sarr.nghost, 0)

    def test_minmaxsum(self):
        sarr = solvcon.SimpleArray((2, 4), value=10.0, dtype='float64')

        self.assertEqual(sarr.sum(), 10.0 * 2 * 4)
        self.assertEqual(sarr.min(), 10.0)
        self.assertEqual(sarr.max(), 10.0)
        sarr.fill(1.0)
        self.assertEqual(sarr.sum(), 1.0 * 2 * 4)
        self.assertEqual(sarr.min(), 1.0)
        self.assertEqual(sarr.max(), 1.0)
        sarr[1, 0] = 9.2
        sarr[0, 3] = -2.3
        self.assertEqual(sarr.min(), -2.3)
        self.assertEqual(sarr.max(), 9.2)

    def test_abs(self):
        sarr = solvcon.SimpleArray((3, 2), value=-2, dtype='int64')
        self.assertEqual(sarr.sum(), -2 * 3 * 2)
        sarr = sarr.abs()
        self.assertEqual(sarr.sum(), 2 * 3 * 2)

    def test_SimpleArrayPlex_set_item_simple(self):
        # just test if the SimpleArrayPlex works
        # more detailed tests are in SimpleArrayBasicTC
        magic_number = 1214

        sarr = solvcon.SimpleArray(20, dtype="uint32")
        sarr[7] = magic_number
        self.assertEqual(sarr[7], magic_number)

        magic_number = 12141618
        sarr = solvcon.SimpleArray((2, 3, 4), dtype="uint64")
        sarr[1, 2, 3] = magic_number
        self.assertEqual(sarr[1, 2, 3], magic_number)

    def test_SimpleArrayPlex_set_item_ellipse(self):
        # just test if the SimpleArrayPlex works
        # more detailed tests are in SimpleArrayBasicTC
        sarr = solvcon.SimpleArray((2, 3, 4), dtype="float64")
        ndarr = np.arange(
            (4 * 2) * (3 * 3) * (2 * 4), dtype='float64').reshape(
            (4 * 2, 3 * 3, 2 * 4))

        stride_arr = ndarr[::4, ::3, ::2]

        # point to the same data
        self.assertEqual(ndarr.__array_interface__['data'],
                         stride_arr.__array_interface__['data'])

        sarr[...] = stride_arr[...]

        for i in range(2):
            for j in range(3):
                for k in range(4):
                    self.assertEqual(stride_arr[i, j, k], sarr[i, j, k])

    # maps dtype with (typed class, typed class name, numpy dtype)
    _DTINFO = {
        "bool": (solvcon.SimpleArrayBool,
                 "_solvcon.SimpleArrayBool", np.bool_),
        "int8": (solvcon.SimpleArrayInt8,
                 "_solvcon.SimpleArrayInt8", np.int8),
        "int16": (solvcon.SimpleArrayInt16,
                  "_solvcon.SimpleArrayInt16", np.int16),
        "int32": (solvcon.SimpleArrayInt32,
                  "_solvcon.SimpleArrayInt32", np.int32),
        "int64": (solvcon.SimpleArrayInt64,
                  "_solvcon.SimpleArrayInt64", np.int64),
        "uint8": (solvcon.SimpleArrayUint8,
                  "_solvcon.SimpleArrayUint8", np.uint8),
        "uint16": (solvcon.SimpleArrayUint16,
                   "_solvcon.SimpleArrayUint16", np.uint16),
        "uint32": (solvcon.SimpleArrayUint32,
                   "_solvcon.SimpleArrayUint32", np.uint32),
        "uint64": (solvcon.SimpleArrayUint64,
                   "_solvcon.SimpleArrayUint64", np.uint64),
        "float32": (solvcon.SimpleArrayFloat32,
                    "_solvcon.SimpleArrayFloat32", np.float32),
        "float64": (solvcon.SimpleArrayFloat64,
                    "_solvcon.SimpleArrayFloat64", np.float64),
        "complex64": (solvcon.SimpleArrayComplex64,
                      "_solvcon.SimpleArrayComplex64",
                      np.complex64),
        "complex128": (solvcon.SimpleArrayComplex128,
                       "_solvcon.SimpleArrayComplex128",
                       np.complex128),
    }

    def test_SimpleArrayPlex_uniform_type(self):
        arrays = {
            dtype: solvcon.SimpleArray((2, 3, 4), dtype=dtype)
            for dtype in self._DTINFO
        }
        first = next(iter(arrays.values()))
        for dtype, arr in arrays.items():
            with self.subTest(dtype=dtype):
                self.assertTrue(type(arr) is type(first))
                self.assertEqual(str(type(arr)),
                                 "<class '_solvcon.SimpleArray'>")

    def test_SimpleArrayPlex_cast_to_typed(self):
        for dtype, (_, typed_name, np_dtype) in self._DTINFO.items():
            with self.subTest(dtype=dtype):
                plex = solvcon.SimpleArray((2, 3, 4), dtype=dtype)
                typed = plex.typed
                self.assertEqual(str(type(typed)),
                                 "<class '{}'>".format(typed_name))
                self.assertEqual(typed.ndarray.dtype, np_dtype)

    def test_SimpleArrayPlex_cast_dtype_mismatch(self):
        # Constructing a typed array from an ndarray whose
        # dtype doesn't match must raise RuntimeError.
        mismatch_cases = [
            ("bool", solvcon.SimpleArrayFloat64),
            ("int8", solvcon.SimpleArrayUint8),
            ("int16", solvcon.SimpleArrayInt32),
            ("int32", solvcon.SimpleArrayFloat64),
            ("int64", solvcon.SimpleArrayUint64),
            ("uint8", solvcon.SimpleArrayInt8),
            ("uint16", solvcon.SimpleArrayUint32),
            ("uint32", solvcon.SimpleArrayInt32),
            ("uint64", solvcon.SimpleArrayFloat64),
            ("float32", solvcon.SimpleArrayFloat64),
            ("float64", solvcon.SimpleArrayInt32),
            ("complex64", solvcon.SimpleArrayComplex128),
            ("complex128", solvcon.SimpleArrayFloat64),
        ]
        for dtype, wrong_cls in mismatch_cases:
            with self.subTest(dtype=dtype):
                plex = solvcon.SimpleArray((2, 3, 4), dtype=dtype)
                ndarr = np.array(plex, copy=False)
                with self.assertRaisesRegex(RuntimeError, r"dtype mismatch"):
                    wrong_cls(array=ndarr)

    def test_SimpleArrayPlex_typed_roundtrip(self):
        for dtype, (_, typed_name, _) in self._DTINFO.items():
            with self.subTest(dtype=dtype):
                plex = solvcon.SimpleArray((2, 3, 4), dtype=dtype)
                typed = plex.typed
                plex2 = typed.plex

                self.assertEqual(str(type(plex)),
                                 "<class '_solvcon.SimpleArray'>")
                self.assertEqual(str(type(typed)),
                                 "<class '{}'>".format(typed_name))
                self.assertEqual(str(type(plex2)),
                                 "<class '_solvcon.SimpleArray'>")

                # Verify data survives the roundtrip:
                # write through typed, read back from plex2
                ndarr = np.array(plex, copy=False)
                ndarr.flat[0] = 1
                self.assertEqual(plex2[0, 0, 0], typed[0, 0, 0])

    def test_SimpleArrayPlex_typed_preserves_data(self):
        # Test non-complex dtypes using value= constructor
        value_cases = {
            "bool": True,
            "int8": 7, "int16": 7, "int32": 7, "int64": 7,
            "uint8": 7, "uint16": 7, "uint32": 7, "uint64": 7,
            "float32": 1.5, "float64": 3.14,
        }
        for dtype, value in value_cases.items():
            with self.subTest(dtype=dtype):
                plex = solvcon.SimpleArray((4,), dtype=dtype, value=value)
                typed = plex.typed
                for i in range(4):
                    self.assertEqual(typed[i], value)

        # Complex dtypes: plex value= constructor does not
        # accept Python complex, use ndarray constructor
        for np_dtype in (np.complex64, np.complex128):
            with self.subTest(dtype=str(np_dtype)):
                ndarr = np.full((4,), 1.5 + 2.5j, dtype=np_dtype)
                plex = solvcon.SimpleArray(ndarr)
                typed = plex.typed
                for i in range(4):
                    self.assertEqual(complex(typed[i]), 1.5 + 2.5j)

    def test_SimpleArrayPlex_typed_from_typed_constructor(self):
        for dtype, (cls, typed_name, _) in self._DTINFO.items():
            with self.subTest(dtype=dtype):
                # typed array built from constructor
                direct = cls((2, 3, 4))
                # typed array obtained via plex.typed
                via_plex = solvcon.SimpleArray((2, 3, 4), dtype=dtype).typed
                self.assertTrue(type(direct) is type(via_plex))
                self.assertEqual(str(type(direct)),
                                 "<class '{}'>".format(typed_name))
                self.assertEqual(str(type(via_plex)),
                                 "<class '{}'>".format(typed_name))


class SimpleArrayPlexAlignmentTC(unittest.TestCase):

    def test_default_alignment(self):
        array = solvcon.SimpleArray((2, 3, 4), dtype='float64')
        self.assertEqual(0, array.alignment)
        self.assertEqual((2, 3, 4), array.typed.shape)

    def test_alignment_creation(self):
        array = solvcon.SimpleArray((4, 4), dtype='float64', alignment=16)
        self.assertEqual(16, array.alignment)
        self.assertEqual((4, 4), array.typed.shape)

        array = solvcon.SimpleArray((8, 8), dtype='float64', alignment=32)
        self.assertEqual(32, array.alignment)
        self.assertEqual((8, 8), array.typed.shape)

        array = solvcon.SimpleArray((16, 8), dtype='float64', alignment=64)
        self.assertEqual(64, array.alignment)
        self.assertEqual((16, 8), array.typed.shape)

    def test_alignment_with_value_initialization(self):
        array = solvcon.SimpleArray((4, 4), value=3.14159, dtype='float64')
        self.assertEqual(0, array.alignment)
        self.assertEqual((4, 4), array.typed.shape)
        self.assertEqual(3.14159, array[0, 0])

        array = solvcon.SimpleArray((4, 4), value=2.71828,
                                    dtype='float64', alignment=16)
        self.assertEqual(16, array.alignment)
        self.assertEqual((4, 4), array.typed.shape)
        self.assertEqual(2.71828, array[0, 0])

    def test_alignment_validation_valid_values(self):
        solvcon.SimpleArray((4, 4), dtype='float64', alignment=16)
        solvcon.SimpleArray((8, 8), dtype='float64', alignment=32)
        solvcon.SimpleArray((16, 8), dtype='float64', alignment=64)

    def test_alignment_validation_invalid_values(self):
        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::ConcreteBuffer: "
                "alignment must be 0, 16, 32, or 64, but got 17"
        ):
            solvcon.SimpleArray((4, 4), dtype='float64', alignment=17)

        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::ConcreteBuffer: "
                "alignment must be 0, 16, 32, or 64, but got 100"
        ):
            solvcon.SimpleArray((4, 4), dtype='float64', alignment=100)

        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::ConcreteBuffer: "
                "alignment must be 0, 16, 32, or 64, but got 128"
        ):
            solvcon.SimpleArray((16, 16), dtype='float64', alignment=128)

    def test_alignment_size_validation(self):
        solvcon.SimpleArray((2, 2), dtype='float64', alignment=32)
        solvcon.SimpleArray((4, 4), dtype='float64', alignment=64)

        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::allocate: "
                "size .* must be a multiple of alignment 16"
        ):
            solvcon.SimpleArray((5, 1), dtype='float64', alignment=16)

        with self.assertRaisesRegex(
                ValueError,
                "ConcreteBuffer::allocate: "
                "size .* must be a multiple of alignment 32"
        ):
            solvcon.SimpleArray((3, 5), dtype='float64', alignment=32)

    def test_alignment_with_operations(self):
        array1 = solvcon.SimpleArray((4, 4), dtype='float64', alignment=16)
        array2 = solvcon.SimpleArray((4, 4), dtype='float64', alignment=32)

        self.assertEqual(16, array1.alignment)
        self.assertEqual(32, array2.alignment)

        array1.fill(5.0)
        array2.fill(10.0)

        self.assertEqual(5.0, array1[0, 0])
        self.assertEqual(10.0, array2[0, 0])

        self.assertEqual(5.0 * 16, array1.sum())
        self.assertEqual(10.0 * 16, array2.sum())

    def test_alignment_with_ndarray_conversion(self):
        array = solvcon.SimpleArray((4, 4), dtype='float64', alignment=16)
        self.assertEqual(16, array.alignment)

        ndarr = np.array(array, copy=False)
        self.assertEqual((4, 4), ndarr.shape)
        self.assertEqual(np.float64, ndarr.dtype)

        array[0, 0] = 42.0
        self.assertEqual(42.0, ndarr[0, 0])

    def test_alignment_with_clone(self):
        array = solvcon.SimpleArray((4, 4), value=3.14,
                                    dtype='float64', alignment=32)
        self.assertEqual(32, array.alignment)
        self.assertEqual(3.14, array[0, 0])

        cloned = array.clone()
        self.assertEqual(32, cloned.alignment)
        self.assertEqual((4, 4), cloned.typed.shape)
        self.assertEqual(3.14, cloned[0, 0])

        array[0, 0] = 99.9
        self.assertEqual(99.9, array[0, 0])
        self.assertEqual(3.14, cloned[0, 0])

    def test_alignment_int32_array(self):
        array = solvcon.SimpleArray((8, 8), dtype='int32', alignment=16)
        self.assertEqual(16, array.alignment)
        self.assertEqual((8, 8), array.typed.shape)

        array.fill(42)
        self.assertEqual(42, array[0, 0])
        self.assertEqual(42 * 64, array.sum())

    def test_alignment_uint64_array(self):
        array = solvcon.SimpleArray((4, 8), dtype='uint64', alignment=32)
        self.assertEqual(32, array.alignment)
        self.assertEqual((4, 8), array.typed.shape)

        array.fill(100)
        self.assertEqual(100, array[0, 0])
        self.assertEqual(100 * 32, array.sum())

    def test_alignment_with_different_shapes(self):
        array1d = solvcon.SimpleArray((32,), dtype='float64', alignment=16)
        self.assertEqual(16, array1d.alignment)
        self.assertEqual((32,), array1d.typed.shape)

        array2d = solvcon.SimpleArray((4, 8), dtype='float64', alignment=32)
        self.assertEqual(32, array2d.alignment)
        self.assertEqual((4, 8), array2d.typed.shape)

        array3d = solvcon.SimpleArray((2, 4, 4), dtype='float64', alignment=64)
        self.assertEqual(64, array3d.alignment)
        self.assertEqual((2, 4, 4), array3d.typed.shape)

    def test_alignment_multiple_datatypes(self):
        dtype_list = ['int8', 'int16', 'int32', 'int64', 'uint8',
                      'uint16', 'uint32', 'uint64', 'float32', 'float64']

        for dtype in dtype_list:
            array = solvcon.SimpleArray((4, 4), dtype=dtype, alignment=16)
            self.assertEqual(16, array.alignment)

            array = solvcon.SimpleArray((8, 8), dtype=dtype, alignment=32)
            self.assertEqual(32, array.alignment)

            array = solvcon.SimpleArray((16, 8), dtype=dtype, alignment=64)
            self.assertEqual(64, array.alignment)


class SimpleCollectorTC(unittest.TestCase):

    def test_construct(self):
        ct = solvcon.SimpleCollectorFloat64(10)
        self.assertEqual(10, ct.capacity)
        self.assertEqual(10, len(ct))

        ct = solvcon.SimpleCollectorFloat64()
        self.assertEqual(0, ct.capacity)
        self.assertEqual(0, len(ct))

        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 0 is out of bounds with size 0"
        ):
            ct[0]

        ct.reserve(6)
        self.assertEqual(6, ct.capacity)
        self.assertEqual(0, len(ct))  # size unchanged

        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 0 is out of bounds with size 0"
        ):
            ct[0]

        ct.expand(6)
        self.assertEqual(6, ct.capacity)
        self.assertEqual(6, len(ct))  # size changed

        ct[5]  # should not raise an exception
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 6 is out of bounds with size 6"
        ):
            ct[6]

        # initialize
        for it in range(6):
            ct[it] = it

        arr = ct.as_array()
        self.assertEqual(6, len(arr))
        self.assertEqual(list(range(6)), list(arr))

        # prove ct and arr share memory
        for it in range(6):
            ct[it] = it + 10
        self.assertEqual(list(it + 10 for it in range(6)), list(ct))

    def test_push_back(self):
        # Starting from 0.
        ct = solvcon.SimpleCollectorFloat64()
        self.assertEqual(0, ct.capacity)
        self.assertEqual(0, len(ct))

        ct.push_back(3.14159)
        self.assertEqual(1, ct.capacity)
        self.assertEqual(1, len(ct))
        self.assertEqual(ct[0], 3.14159)

        ct.push_back(3.14159 * 2)
        self.assertEqual(2, ct.capacity)
        self.assertEqual(2, len(ct))
        self.assertEqual(ct[1], 3.14159 * 2)

        ct.push_back(3.14159 * 3)
        self.assertEqual(4, ct.capacity)
        self.assertEqual(3, len(ct))
        self.assertEqual(ct[2], 3.14159 * 3)

        for it in range(10):
            ct.push_back(3.14159 * (3 + it + 1))
            self.assertEqual(3 + it + 1, len(ct))
            self.assertEqual(ct[3 + it], 3.14159 * (3 + it + 1))

        # Starting from non-zero and not power of 2.
        ct = solvcon.SimpleCollectorFloat64(10)
        self.assertEqual(10, ct.capacity)
        self.assertEqual(10, len(ct))

        ct.push_back(3.14159 * 4)
        self.assertEqual(20, ct.capacity)  # double capacity but not power of 2
        self.assertEqual(11, len(ct))
        self.assertEqual(ct[10], 3.14159 * 4)

    def test_alignment_validation(self):
        # Valid alignments: 0, 16, 32, 64
        ct = solvcon.SimpleCollectorFloat64(16, 16)
        self.assertEqual(16, ct.alignment)

        ct = solvcon.SimpleCollectorFloat64(32, 32)
        self.assertEqual(32, ct.alignment)

        ct = solvcon.SimpleCollectorFloat64(64, 64)
        self.assertEqual(64, ct.alignment)

        ct = solvcon.SimpleCollectorFloat64(16, 0)
        self.assertEqual(0, ct.alignment)

        # Invalid alignments
        with self.assertRaisesRegex(
                ValueError,
                "BufferExpander::BufferExpander: alignment must be 0, 16, 32, or 64, but got 8"  # noqa E501
        ):
            solvcon.SimpleCollectorFloat64(16, 8)

        with self.assertRaisesRegex(
                ValueError,
                "BufferExpander::BufferExpander: alignment must be 0, 16, 32, or 64, but got 128"  # noqa E501
        ):
            solvcon.SimpleCollectorFloat64(128, 128)

    def test_alignment_size_validation(self):
        # Size must be a multiple of alignment
        ct = solvcon.SimpleCollectorFloat64(0, 16)
        self.assertEqual(16, ct.alignment)

        ct = solvcon.SimpleCollectorFloat64(16, 16)
        self.assertEqual(16, len(ct))

        ct = solvcon.SimpleCollectorFloat64(32, 16)
        self.assertEqual(32, len(ct))

        ct = solvcon.SimpleCollectorFloat64(64, 32)
        self.assertEqual(64, len(ct))

        # Invalid sizes (not multiple of alignment)
        with self.assertRaisesRegex(
                ValueError,
                "BufferExpander::allocate: size .* must be a multiple of alignment 16"  # noqa E501
        ):
            solvcon.SimpleCollectorFloat64(17, 16)

        with self.assertRaisesRegex(
                ValueError,
                "BufferExpander::allocate: size .* must be a multiple of alignment 32"  # noqa E501
        ):
            solvcon.SimpleCollectorFloat64(31, 32)

    def test_alignment_with_reserve(self):
        # Test that reserve maintains alignment
        ct = solvcon.SimpleCollectorFloat64(16, 16)
        self.assertEqual(16, ct.capacity)

        # Reserve with aligned size
        ct.reserve(32)
        self.assertEqual(32, ct.capacity)

        # Verify alignment is maintained
        self.assertEqual(16, ct.alignment)

        # Reserve with non-aligned size should fail
        with self.assertRaisesRegex(
                ValueError,
                "BufferExpander::allocate: size .* must be a multiple of alignment 16"  # noqa E501
        ):
            ct.reserve(33)

        ct = solvcon.SimpleCollectorFloat64(0, 32)  # start from 0 capacity
        self.assertEqual(0, ct.capacity)
        self.assertEqual(32, ct.alignment)
        ct.reserve(64)
        self.assertEqual(64, ct.capacity)

    def test_alignment_with_push_back(self):
        # Test that push_back works with alignment
        ct = solvcon.SimpleCollectorFloat64(16, 16)
        self.assertEqual(16, len(ct))
        self.assertEqual(16, ct.capacity)

        # Push back should expand with aligned size
        ct.push_back(3.14159)
        self.assertEqual(17, len(ct))
        # Capacity should be doubled and remain aligned (32 elements = 256 bytes aligned to 16)  # noqa E501
        self.assertEqual(32, ct.capacity)

        # Verify alignment is maintained
        self.assertEqual(16, ct.alignment)

    def test_alignment_preserved_in_as_array(self):
        ct = solvcon.SimpleCollectorFloat64(16, 32)
        self.assertEqual(32, ct.alignment)
        self.assertEqual(16, len(ct))

        for it in range(16):
            ct[it] = it * 3.14

        arr = ct.as_array()
        self.assertEqual(16, len(arr))
        self.assertEqual(32, arr.alignment)

        for it in range(16):
            self.assertEqual(it * 3.14, arr[it])

    def test_clear(self):
        # Clear an empty collector.
        ct = solvcon.SimpleCollectorFloat64()
        self.assertEqual(0, ct.capacity)
        self.assertEqual(0, len(ct))
        ct.clear()
        self.assertEqual(0, ct.capacity)
        self.assertEqual(0, len(ct))

        # Clear a collector with data from push_back.
        ct = solvcon.SimpleCollectorFloat64()
        for it in range(5):
            ct.push_back(it * 1.1)
        self.assertEqual(5, len(ct))
        old_capacity = ct.capacity
        self.assertGreater(old_capacity, 0)

        ct.clear()
        self.assertEqual(0, len(ct))
        self.assertEqual(old_capacity, ct.capacity)  # capacity preserved

        # After clear, accessing index 0 should raise.
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 0 is out of bounds with size 0"
        ):
            ct[0]

        # Can push_back again after clear.
        ct.push_back(42.0)
        self.assertEqual(1, len(ct))
        self.assertEqual(42.0, ct[0])
        self.assertEqual(old_capacity, ct.capacity)  # capacity still preserved

        # Clear a collector created with initial size.
        ct = solvcon.SimpleCollectorFloat64(10)
        self.assertEqual(10, len(ct))
        self.assertEqual(10, ct.capacity)
        ct.clear()
        self.assertEqual(0, len(ct))
        self.assertEqual(10, ct.capacity)

    def test_clear_with_alignment(self):
        ct = solvcon.SimpleCollectorFloat64(16, 16)
        self.assertEqual(16, ct.alignment)
        self.assertEqual(16, len(ct))

        ct.clear()
        self.assertEqual(0, len(ct))
        self.assertEqual(16, ct.capacity)
        self.assertEqual(16, ct.alignment)  # alignment preserved

    def test_clear_after_as_array(self):
        ct = solvcon.SimpleCollectorFloat64()
        for it in range(5):
            ct.push_back(it * 1.1)
        self.assertEqual(5, len(ct))

        arr = ct.as_array()
        self.assertEqual(5, len(arr))
        for it in range(5):
            self.assertEqual(it * 1.1, arr[it])

        # Clearing the collector resets its size but keeps its capacity,
        # and the SimpleArray returned earlier retains its data.
        old_capacity = ct.capacity
        ct.clear()
        self.assertEqual(0, len(ct))
        self.assertEqual(old_capacity, ct.capacity)
        self.assertEqual(5, len(arr))
        for it in range(5):
            self.assertEqual(it * 1.1, arr[it])

        # Collector can be reused after clear.
        ct.push_back(42.0)
        self.assertEqual(1, len(ct))
        self.assertEqual(42.0, ct[0])

    def test_alignment_preserved_in_as_concrete(self):
        ep = solvcon.BufferExpander(128, 16)
        self.assertEqual(16, ep.alignment)
        self.assertEqual(128, len(ep))

        for it in range(128):
            ep[it] = it

        cbuf = ep.as_concrete()
        self.assertEqual(128, len(cbuf))
        self.assertEqual(16, cbuf.alignment)

        for it in range(128):
            self.assertEqual(it, cbuf[it])

    def test_alignment_preserved_in_as_concrete_with_capacity(self):
        ep = solvcon.BufferExpander(64, 32)
        self.assertEqual(32, ep.alignment)
        self.assertEqual(64, len(ep))

        for it in range(64):
            ep[it] = it

        cbuf = ep.as_concrete(128)
        self.assertEqual(128, len(cbuf))
        self.assertEqual(32, cbuf.alignment)

        for it in range(64):
            self.assertEqual(it, cbuf[it])

    def test_alignment_validation_in_as_concrete(self):
        ep = solvcon.BufferExpander(64, 32)
        self.assertEqual(32, ep.alignment)

        with self.assertRaisesRegex(
                ValueError,
                "BufferExpander::as_concrete: size .* must be a multiple of alignment 32"  # noqa E501
        ):
            ep.as_concrete(100)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

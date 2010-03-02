# -*- coding: UTF-8 -*-

from unittest import TestCase

class TestNumpy(TestCase):
    def test_dot(self):
        from numpy import array, dot
        A = array([[1,2],[3,4]], dtype='int32')
        B = array([[5,6],[7,8]], dtype='int32')
        R = array([[19,22],[43,50]], dtype='int32')
        for val in (dot(A,B)-R).flat:
            self.assertEqual(val, 0)
        u = array([1,1], dtype='int32')
        Ru = array([3,7], dtype='int32')
        for val in (dot(A,u)-Ru).flat:
            self.assertEqual(val, 0)

    def test_eig(self):
        from numpy import array, dot
        from numpy.linalg import eig, inv
        A = array([[1,2],[3,4]], dtype='int32')
        vals, mat = eig(A)
        lbd = dot(dot(inv(mat), A), mat)
        for i in range(2):
            self.assertAlmostEqual(vals[i], lbd[i,i], places=14)

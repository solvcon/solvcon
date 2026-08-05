# Copyright (c) 2021, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import numpy as np

import solvcon


class StaticMeshTC(unittest.TestCase):

    def _check_shape(self, mh, ndim, nnode, nface, ncell,
                     nbound, ngstnode, ngstface, ngstcell, nedge):
        self.assertEqual(ndim, mh.ndim)
        self.assertEqual(nnode, mh.nnode)
        self.assertEqual(nface, mh.nface)
        self.assertEqual(ncell, mh.ncell)
        self.assertEqual(nbound, mh.nbound)
        self.assertEqual(ngstnode, mh.ngstnode)
        self.assertEqual(ngstface, mh.ngstface)
        self.assertEqual(ngstcell, mh.ngstcell)

        self.assertEqual(nedge, mh.nedge)

        self.assertEqual((mh.ngstnode + mh.nnode, ndim), mh.ndcrd.shape)
        self.assertEqual((mh.ngstface + mh.nface, ndim), mh.fccnd.shape)
        self.assertEqual((mh.ngstface + mh.nface, ndim), mh.fcnml.shape)
        self.assertEqual((mh.ngstface + mh.nface,), mh.fcara.shape)
        self.assertEqual((mh.ngstcell + mh.ncell, ndim), mh.clcnd.shape)
        self.assertEqual((mh.ngstcell + mh.ncell,), mh.clvol.shape)

        self.assertEqual((mh.ngstface + mh.nface,), mh.fctpn.shape)
        self.assertEqual((mh.ngstcell + mh.ncell,), mh.cltpn.shape)
        self.assertEqual((mh.ngstcell + mh.ncell,), mh.clgrp.shape)

        self.assertEqual(
            (mh.ngstface + mh.nface, mh.FCMND + 1), mh.fcnds.shape)
        self.assertEqual((mh.ngstface + mh.nface, mh.FCREL), mh.fccls.shape)
        self.assertEqual(
            (mh.ngstcell + mh.ncell, mh.CLMND + 1), mh.clnds.shape)
        self.assertEqual(
            (mh.ngstcell + mh.ncell, mh.CLMFC + 1), mh.clfcs.shape)

    def _check_metric_trivial(self, mh):
        self.assertTrue((mh.fccnd.ndarray[:, :] == 0).all())
        self.assertTrue((mh.fcnml.ndarray[:, :] == 0).all())
        self.assertTrue((mh.fcara.ndarray[:] == 0).all())
        self.assertTrue((mh.clcnd.ndarray[:, :] == 0).all())
        self.assertTrue((mh.clvol.ndarray[:] == 0).all())

    def test_construct(self):
        def _test(cls, ndim):
            mh = cls(ndim=ndim, nnode=0)
            self._check_shape(mh, ndim=ndim, nnode=0, nface=0, ncell=0,
                              nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                              nedge=0)

        _test(solvcon.StaticMesh, ndim=2)
        _test(solvcon.StaticMesh, ndim=3)

    def test_2d_trivial_triangles(self):
        mh = solvcon.StaticMesh(ndim=2, nnode=4, nface=0, ncell=3)
        mh.ndcrd.ndarray[:, :] = (0, 0), (-1, -1), (1, -1), (0, 1)
        mh.cltpn.ndarray[:] = solvcon.StaticMesh.TRIANGLE
        mh.clnds.ndarray[:, :4] = (3, 0, 1, 2), (3, 0, 2, 3), (3, 0, 3, 1)

        self._check_shape(mh, ndim=2, nnode=4, nface=0, ncell=3,
                          nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=0)
        self._check_metric_trivial(mh)

        # Test build interior data.
        mh.build_interior(do_metric=False, build_edge=False)
        self._check_shape(mh, ndim=2, nnode=4, nface=6, ncell=3,
                          nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=0)
        self._check_metric_trivial(mh)
        mh.build_interior()  # do_metric=True, build_edge=True
        self._check_shape(mh, ndim=2, nnode=4, nface=6, ncell=3,
                          nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=6)
        np.testing.assert_almost_equal(
            mh.fccnd,
            [[-0.5, -0.5], [0.0, -1.0], [0.5, -0.5],
             [0.5, 0.0], [0.0, 0.5], [-0.5, 0.0]])
        np.testing.assert_almost_equal(
            mh.fcnml,
            [[-0.7071068, 0.7071068], [0.0, -1.0], [0.7071068, 0.7071068],
             [0.8944272, 0.4472136], [-1.0, -0.0], [-0.8944272, 0.4472136]])
        np.testing.assert_almost_equal(
            mh.fcara, [1.4142136, 2.0, 1.4142136, 2.236068, 1.0, 2.236068])
        np.testing.assert_almost_equal(
            mh.clcnd, [[0.0, -0.6666667], [0.3333333, 0.0], [-0.3333333, 0.0]])
        np.testing.assert_almost_equal(
            mh.clvol, [1.0, 0.5, 0.5])

        # Build boundary data.
        self.assertEqual(0, mh.nbcs)
        self.assertEqual(0, mh.nbound)
        self.assertEqual((mh.nbound, 3), mh.bndfcs.shape)
        mh.build_boundary()
        self.assertEqual(1, mh.nbcs)
        self.assertEqual(3, mh.nbound)
        self.assertEqual((mh.nbound, 3), mh.bndfcs.shape)
        self.assertEqual(
            [[1, 0, -1], [3, 0, -1], [5, 0, -1]],
            mh.bndfcs.ndarray.tolist()
        )

        # Build ghost data.
        self._check_shape(mh, ndim=2, nnode=4, nface=6, ncell=3,
                          nbound=3, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=6)
        mh.build_ghost()
        self._check_shape(mh, ndim=2, nnode=4, nface=6, ncell=3,
                          nbound=3, ngstnode=3, ngstface=6, ngstcell=3,
                          nedge=6)

    def test_3d_single_tetrahedron(self):
        mh = solvcon.StaticMesh(ndim=3, nnode=4, nface=4, ncell=1)
        mh.ndcrd.ndarray[:, :] = (0, 0, 0), (0, 1, 0), (-1, 1, 0), (0, 1, 1)
        mh.cltpn.ndarray[:] = solvcon.StaticMesh.TETRAHEDRON
        mh.clnds.ndarray[:, :5] = [(4, 0, 1, 2, 3)]

        self._check_shape(mh, ndim=3, nnode=4, nface=4, ncell=1,
                          nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=0)
        self._check_metric_trivial(mh)

        mh.build_interior(do_metric=False, build_edge=False)
        self._check_shape(mh, ndim=3, nnode=4, nface=4, ncell=1,
                          nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=0)
        self._check_metric_trivial(mh)
        mh.build_interior()  # do_metric=True, build_edge=True
        self._check_shape(mh, ndim=3, nnode=4, nface=4, ncell=1,
                          nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=6)
        # TODO: I have not verified the numbers.  But the mesh looks OK in
        # pilot.
        np.testing.assert_almost_equal(
            mh.fccnd,
            [[-0.3333333,  0.6666667,  0.       ],  # noqa
             [ 0.       ,  0.6666667,  0.3333333],  # noqa
             [-0.3333333,  0.6666667,  0.3333333],  # noqa
             [-0.3333333,  1.       ,  0.3333333]])  # noqa
        np.testing.assert_almost_equal(
            mh.fcnml,
            [[ 0.       ,  0.       , -1.       ],  # noqa
             [ 1.       ,  0.       ,  0.       ],  # noqa
             [-0.5773503, -0.5773503,  0.5773503],  # noqa
             [ 0.       ,  1.       ,  0.       ]])  # noqa
        np.testing.assert_almost_equal(
            mh.fcara, [0.5      , 0.5      , 0.8660254, 0.5      ])  # noqa
        np.testing.assert_almost_equal(
            mh.clcnd, [[-0.25,  0.75,  0.25]])
        np.testing.assert_almost_equal(
            mh.clvol, [0.1666667])

        # Build boundary data.
        self.assertEqual(0, mh.nbcs)
        self.assertEqual(0, mh.nbound)
        self.assertEqual((mh.nbound, 3), mh.bndfcs.shape)
        mh.build_boundary()
        self.assertEqual(1, mh.nbcs)
        self.assertEqual(4, mh.nbound)
        self.assertEqual((mh.nbound, 3), mh.bndfcs.shape)
        self.assertEqual(
            [[0, 0, -1], [1, 0, -1], [2, 0, -1], [3, 0, -1]],
            mh.bndfcs.ndarray.tolist()
        )

        # Build ghost data.
        self._check_shape(mh, ndim=3, nnode=4, nface=4, ncell=1,
                          nbound=4, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=6)
        mh.build_ghost()
        self._check_shape(mh, ndim=3, nnode=4, nface=4, ncell=1,
                          nbound=4, ngstnode=4, ngstface=12, ngstcell=4,
                          nedge=6)

    def test_1d_single_line(self):
        mh = solvcon.StaticMesh(ndim=1, nnode=2, nface=0, ncell=1)
        mh.ndcrd.ndarray[:] = [[0], [1]]
        mh.cltpn.ndarray[:] = solvcon.StaticMesh.LINE
        mh.clnds.ndarray[:, :3] = [(2, 0, 1)]

        self._check_shape(mh, ndim=1, nnode=2, nface=0, ncell=1,
                          nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=0)
        self._check_metric_trivial(mh)

        mh.build_interior(do_metric=False, build_edge=False)
        self._check_shape(mh, ndim=1, nnode=2, nface=2, ncell=1,
                          nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=0)
        self._check_metric_trivial(mh)

        mh.build_interior()  # do_metric=True, build_edge=True
        self._check_shape(mh, ndim=1, nnode=2, nface=2, ncell=1,
                          nbound=0, ngstnode=0, ngstface=0, ngstcell=0,
                          nedge=2)

        # do_metric do nothing due to dim == 1
        self._check_metric_trivial(mh)
        # TODO: Need to add build_boundary and build_ghost to make sure
        #       Line type behavior.


class StaticMeshBcTC(unittest.TestCase):
    """Populate boundary-condition groups from Python.

    The three-triangle fan of StaticMeshTC has the boundary faces 1, 3,
    and 5; the other faces are interior.
    """

    def _make_mesh(self):
        mh = solvcon.StaticMesh(ndim=2, nnode=4, nface=0, ncell=3)
        mh.ndcrd.ndarray[:, :] = (0, 0), (-1, -1), (1, -1), (0, 1)
        mh.cltpn.ndarray[:] = solvcon.StaticMesh.TRIANGLE
        mh.clnds.ndarray[:, :4] = (3, 0, 1, 2), (3, 0, 2, 3), (3, 0, 3, 1)
        mh.build_interior()
        return mh

    def test_populate_and_build(self):
        mh = self._make_mesh()
        bottom = mh.add_bc('bottom', [1])
        flank = mh.add_bc('flank', [3, 5])
        self.assertEqual(2, mh.nbcs)
        self.assertEqual('bottom', bottom.name)
        self.assertEqual(1, bottom.nbound)
        self.assertEqual(2, flank.nbound)

        mh.build_boundary()
        # No leftover face: no catch-all group is appended.
        self.assertEqual(2, mh.nbcs)
        # Column 1 carries the group index in attach order.
        self.assertEqual(
            [[1, 0, -1], [3, 1, -1], [5, 1, -1]],
            mh.bndfcs.ndarray.tolist())
        # Column 1 of facn is back-filled with the bndfcs row; column 2
        # stays at its -1 idle value.
        self.assertEqual([[1, 0, -1]], bottom.facn.ndarray.tolist())
        self.assertEqual(
            [[3, 1, -1], [5, 2, -1]], flank.facn.ndarray.tolist())

    def test_catch_all_collects_leftover(self):
        mh = self._make_mesh()
        mh.add_bc('bottom', [1])
        mh.build_boundary()
        self.assertEqual(2, mh.nbcs)
        leftover = mh.bc(1)
        self.assertEqual(solvcon.StaticMeshBc.NONAME, leftover.name)
        self.assertEqual(
            [[3, 1, -1], [5, 2, -1]], leftover.facn.ndarray.tolist())
        self.assertEqual(
            [[1, 0, -1], [3, 1, -1], [5, 1, -1]],
            mh.bndfcs.ndarray.tolist())

    def test_read_back(self):
        mh = self._make_mesh()
        bottom = mh.add_bc('bottom', [1])
        self.assertIs(bottom, mh.bc(0))
        self.assertIs(bottom, mh.bc('bottom'))
        self.assertEqual(1, len(mh.bcs))
        self.assertIs(bottom, mh.bcs[0])

    def test_rename(self):
        mh = self._make_mesh()
        mh.add_bc('bottom', [1])
        mh.bc(0).name = 'floor'
        self.assertEqual('floor', mh.bc('floor').name)
        with self.assertRaises(KeyError):
            mh.bc('bottom')

    def test_add_bc_rejects_bad_faces(self):
        mh = self._make_mesh()
        with self.assertRaisesRegex(ValueError, 'not a boundary face'):
            mh.add_bc('interior', [0])
        with self.assertRaisesRegex(ValueError, 'out of range'):
            mh.add_bc('outside', [99])
        with self.assertRaisesRegex(ValueError, 'out of range'):
            mh.add_bc('negative', [-1])
        with self.assertRaisesRegex(ValueError, 'already claimed'):
            mh.add_bc('doubled', [3, 3])
        mh.add_bc('bottom', [1])
        with self.assertRaisesRegex(ValueError, 'already claimed'):
            mh.add_bc('again', [1])

    def test_add_bc_rejects_bad_names(self):
        mh = self._make_mesh()
        mh.add_bc('bottom', [1])
        with self.assertRaisesRegex(ValueError, 'duplicated name'):
            mh.add_bc('bottom', [3])
        with self.assertRaisesRegex(ValueError, 'empty name'):
            mh.add_bc('', [3])

    def test_add_bc_after_build_raises(self):
        mh = self._make_mesh()
        mh.build_boundary()
        with self.assertRaisesRegex(RuntimeError, 'already built'):
            mh.add_bc('late', [1])

    def test_double_build_boundary_raises(self):
        mh = self._make_mesh()
        mh.build_boundary()
        with self.assertRaisesRegex(RuntimeError, 'already built'):
            mh.build_boundary()

    def test_lookup_errors(self):
        mh = self._make_mesh()
        with self.assertRaises(IndexError):
            mh.bc(0)
        with self.assertRaises(KeyError):
            mh.bc('absent')
# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

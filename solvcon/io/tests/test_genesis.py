# -*- coding: UTF-8 -*-

from unittest import TestCase

class TestReadTetra(TestCase):
    import os
    from ...conf import env
    testfn = [env.datadir] + ['cubic_t200mm.g']
    testfn = os.path.join(*testfn)

    def test_dim(self):
        from ..genesis import Genesis
        gn = Genesis(self.testfn)
        self.assertEqual(gn.get_dim('num_dim'), 3)
        self.assertEqual(gn.get_dim('num_nodes'), 316)
        self.assertEqual(gn.get_dim('num_elem'), 1253)
        self.assertEqual(gn.get_dim('num_el_blk'), 2)
        self.assertEqual(gn.get_dim('num_side_sets'), 6)
        gn.close_file()

    def test_ss_names(self):
        from ..genesis import Genesis
        gn = Genesis(self.testfn)
        nbc = gn.get_dim('num_side_sets')
        slen = gn.get_dim('len_string')
        lines = gn.get_lines('ss_names', (nbc, slen))
        self.assertEqual(len(lines), 6)
        self.assertEqual(lines[0], 'front')
        self.assertEqual(gn.get_dim('num_side_ss1'), 62)
        self.assertEqual(lines[1], 'rear')
        self.assertEqual(gn.get_dim('num_side_ss2'), 62)
        self.assertEqual(lines[2], 'lower')
        self.assertEqual(gn.get_dim('num_side_ss3'), 60)
        self.assertEqual(lines[3], 'left')
        self.assertEqual(gn.get_dim('num_side_ss4'), 60)
        self.assertEqual(lines[4], 'upper')
        self.assertEqual(gn.get_dim('num_side_ss5'), 60)
        self.assertEqual(lines[5], 'right')
        self.assertEqual(gn.get_dim('num_side_ss6'), 60)
        gn.close_file()

    def test_coord(self):
        from ..genesis import Genesis
        gn = Genesis(self.testfn)
        ndim = gn.get_dim('num_dim')
        nnode = gn.get_dim('num_nodes')
        ndcrd = gn.get_array('coord', (ndim, nnode), 'float64').T.copy()
        self.assertTrue((ndcrd>=-0.5).all())
        self.assertTrue((ndcrd<=0.5).all())
        gn.close_file()

    def test_block(self):
        from ..genesis import Genesis
        gn = Genesis(self.testfn)
        # name.
        nblk = gn.get_dim('num_el_blk')
        slen = gn.get_dim('len_string')
        blks = gn.get_lines('eb_names', (nblk, slen))
        self.assertEqual(len(blks), 2)
        self.assertEqual(blks[0], 'rear')
        self.assertEqual(blks[1], 'front')
        # block 1.
        ncell = gn.get_dim('num_el_in_blk1')
        clnnd = gn.get_dim('num_nod_per_el1')
        clnds = gn.get_array('connect1', (ncell, clnnd), 'int32')
        self.assertEqual(clnds.min(), 1)
        self.assertEqual(clnds.max(), 181)
        self.assertEqual(gn.get_attr_text('elem_type', 'connect1'), 'TETRA')
        # block 2.
        ncell = gn.get_dim('num_el_in_blk2')
        clnnd = gn.get_dim('num_nod_per_el2')
        clnds = gn.get_array('connect2', (ncell, clnnd), 'int32')
        self.assertEqual(clnds.min(), 5)
        self.assertEqual(clnds.max(), 316)
        self.assertEqual(gn.get_attr_text('elem_type', 'connect2'), 'TETRA')
        gn.close_file()

    def test_load(self):
        from numpy import arange
        from ..genesis import Genesis
        # load from netCDF.
        gn = Genesis(self.testfn)
        gn.load()
        gn.close_file()
        # meta data.
        self.assertEqual(gn.ndim, 3)
        self.assertEqual(gn.nnode, 316)
        self.assertEqual(gn.ncell, 1253)
        # blocks.
        self.assertEqual(len(gn.blks), 2)
        self.assertEqual(gn.blks[0][0], 'rear')
        self.assertEqual(gn.blks[0][1], 'TETRA')
        self.assertEqual(gn.blks[0][2].shape, (635, 4))
        self.assertEqual(gn.blks[1][0], 'front')
        self.assertEqual(gn.blks[1][1], 'TETRA')
        self.assertEqual(gn.blks[1][2].shape, (618, 4))
        # BCs.
        self.assertEqual(len(gn.bcs), 6)
        self.assertEqual(gn.bcs[0][0], 'front')
        self.assertEqual(gn.bcs[0][1].shape, (62,))
        self.assertEqual(gn.bcs[0][2].shape, (62,))
        self.assertEqual(gn.bcs[1][0], 'rear')
        self.assertEqual(gn.bcs[1][1].shape, (62,))
        self.assertEqual(gn.bcs[1][2].shape, (62,))
        self.assertEqual(gn.bcs[2][0], 'lower')
        self.assertEqual(gn.bcs[2][1].shape, (60,))
        self.assertEqual(gn.bcs[2][2].shape, (60,))
        self.assertEqual(gn.bcs[3][0], 'left')
        self.assertEqual(gn.bcs[3][1].shape, (60,))
        self.assertEqual(gn.bcs[3][2].shape, (60,))
        self.assertEqual(gn.bcs[4][0], 'upper')
        self.assertEqual(gn.bcs[4][1].shape, (60,))
        self.assertEqual(gn.bcs[4][2].shape, (60,))
        self.assertEqual(gn.bcs[5][0], 'right')
        self.assertEqual(gn.bcs[5][1].shape, (60,))
        self.assertEqual(gn.bcs[5][2].shape, (60,))
        # coordinate.
        self.assertEqual(gn.ndcrd.shape, (316, 3))
        self.assertTrue((gn.ndcrd >= -0.5).all())
        self.assertTrue((gn.ndcrd <= 0.5).all())
        # mapper.
        self.assertTrue((gn.emap == arange(1253)+1).all())

    def test_convert_interior(self):
        from ...block import Block
        from ..genesis import Genesis
        # load from netCDF.
        gn = Genesis(self.testfn)
        gn.load()
        gn.close_file()
        # convert.
        blk = Block(ndim=gn.ndim, nnode=gn.nnode, ncell=gn.ncell,
            fpdtype='float64')
        gn._convert_interior_to(blk)
        # test cell type.
        self.assertTrue((blk.cltpn == 5).all())
        self.assertTrue((blk.clnds[:,0] == 4).all())
        # test index of node in cell.
        self.assertEqual(blk.clnds[:,1:5].min(), 0)
        self.assertEqual(blk.clnds[:,1:5].max(), 316-1)
        # test group.
        self.assertEqual(len(blk.grpnames), 2)
        self.assertEqual(blk.clgrp.min(), 0)
        self.assertEqual((blk.clgrp==0).sum(), 635)
        self.assertEqual(blk.clgrp.max(), 1)
        self.assertEqual((blk.clgrp==1).sum(), 618)

    def test_toblock(self):
        from ..genesis import Genesis
        # load from netCDF.
        gn = Genesis(self.testfn)
        gn.load()
        gn.close_file()
        # convert.
        blk = gn.toblock()
        # test BC.
        self.assertEqual(len(blk.bclist), 6)
        self.assertEqual(blk.bclist[0].name, 'front')
        self.assertEqual(len(blk.bclist[0]), 62)
        self.assertTrue((blk.fccnd[blk.bclist[0].facn[:,0],2] == 0.5).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[0].facn[:,0]].sum(), 1.0,
            15)
        self.assertEqual(blk.bclist[1].name, 'rear')
        self.assertEqual(len(blk.bclist[1]), 62)
        self.assertTrue((blk.fccnd[blk.bclist[1].facn[:,0],2] == -0.5).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[1].facn[:,0]].sum(), 1.0,
            15)
        self.assertEqual(blk.bclist[2].name, 'lower')
        self.assertEqual(len(blk.bclist[2]), 60)
        self.assertTrue((blk.fccnd[blk.bclist[2].facn[:,0],1] == -0.5).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[2].facn[:,0]].sum(), 1.0,
            15)
        self.assertEqual(blk.bclist[3].name, 'left')
        self.assertEqual(len(blk.bclist[3]), 60)
        self.assertTrue((blk.fccnd[blk.bclist[3].facn[:,0],0] == -0.5).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[3].facn[:,0]].sum(), 1.0,
            15)
        self.assertEqual(blk.bclist[4].name, 'upper')
        self.assertEqual(len(blk.bclist[4]), 60)
        self.assertTrue((blk.fccnd[blk.bclist[4].facn[:,0],1] == 0.5).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[4].facn[:,0]].sum(), 1.0,
            15)
        self.assertEqual(blk.bclist[5].name, 'right')
        self.assertEqual(len(blk.bclist[5]), 60)
        self.assertTrue((blk.fccnd[blk.bclist[5].facn[:,0],0] == 0.5).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[5].facn[:,0]].sum(), 1.0,
            15)
        # test geometry.
        self.assertEqual(blk.ndcrd.min(), -0.5)
        self.assertEqual(blk.ndcrd.max(), 0.5)
        self.assertAlmostEqual(blk.clvol.sum(), 1.0, 14)

class TestReadHex(TestCase):
    import os
    from ...conf import env
    testfn = [env.datadir] + ['cubic_h200mm.g']
    testfn = os.path.join(*testfn)

    def test_dim(self):
        from ..genesis import Genesis
        gn = Genesis(self.testfn)
        self.assertEqual(gn.get_dim('num_dim'), 3)
        self.assertEqual(gn.get_dim('num_nodes'), 252)
        self.assertEqual(gn.get_dim('num_elem'), 150)
        self.assertEqual(gn.get_dim('num_el_blk'), 2)
        self.assertEqual(gn.get_dim('num_side_sets'), 6)
        gn.close_file()

    def test_ss_names(self):
        from ..genesis import Genesis
        gn = Genesis(self.testfn)
        nbc = gn.get_dim('num_side_sets')
        slen = gn.get_dim('len_string')
        lines = gn.get_lines('ss_names', (nbc, slen))
        self.assertEqual(len(lines), 6)
        self.assertEqual(lines[0], 'front')
        self.assertEqual(gn.get_dim('num_side_ss1'), 25)
        self.assertEqual(lines[1], 'rear')
        self.assertEqual(gn.get_dim('num_side_ss2'), 25)
        self.assertEqual(lines[2], 'lower')
        self.assertEqual(gn.get_dim('num_side_ss3'), 30)
        self.assertEqual(lines[3], 'left')
        self.assertEqual(gn.get_dim('num_side_ss4'), 30)
        self.assertEqual(lines[4], 'upper')
        self.assertEqual(gn.get_dim('num_side_ss5'), 30)
        self.assertEqual(lines[5], 'right')
        self.assertEqual(gn.get_dim('num_side_ss6'), 30)
        gn.close_file()

    def test_coord(self):
        from ..genesis import Genesis
        gn = Genesis(self.testfn)
        ndim = gn.get_dim('num_dim')
        nnode = gn.get_dim('num_nodes')
        ndcrd = gn.get_array('coord', (ndim, nnode), 'float64').T.copy()
        self.assertTrue((ndcrd >= -0.5-1.e-15).all())
        self.assertTrue((ndcrd <= 0.5+1.e-15).all())
        gn.close_file()

    def test_block(self):
        from ..genesis import Genesis
        gn = Genesis(self.testfn)
        # name.
        nblk = gn.get_dim('num_el_blk')
        slen = gn.get_dim('len_string')
        blks = gn.get_lines('eb_names', (nblk, slen))
        self.assertEqual(len(blks), 2)
        self.assertEqual(blks[0], 'rear')
        self.assertEqual(blks[1], 'front')
        # block 1.
        ncell = gn.get_dim('num_el_in_blk1')
        clnnd = gn.get_dim('num_nod_per_el1')
        clnds = gn.get_array('connect1', (ncell, clnnd), 'int32')
        self.assertEqual(clnds.min(), 1)
        self.assertEqual(clnds.max(), 144)
        self.assertEqual(gn.get_attr_text('elem_type', 'connect1'), 'HEX8')
        # block 2.
        ncell = gn.get_dim('num_el_in_blk2')
        clnnd = gn.get_dim('num_nod_per_el2')
        clnds = gn.get_array('connect2', (ncell, clnnd), 'int32')
        self.assertEqual(clnds.min(), 109)
        self.assertEqual(clnds.max(), 252)
        self.assertEqual(gn.get_attr_text('elem_type', 'connect2'), 'HEX8')
        gn.close_file()

    def test_load(self):
        from numpy import arange
        from ..genesis import Genesis
        # load from netCDF.
        gn = Genesis(self.testfn)
        gn.load()
        gn.close_file()
        # meta data.
        self.assertEqual(gn.ndim, 3)
        self.assertEqual(gn.nnode, 252)
        self.assertEqual(gn.ncell, 150)
        # blocks.
        self.assertEqual(len(gn.blks), 2)
        self.assertEqual(gn.blks[0][0], 'rear')
        self.assertEqual(gn.blks[0][1], 'HEX8')
        self.assertEqual(gn.blks[0][2].shape, (75, 8))
        self.assertEqual(gn.blks[1][0], 'front')
        self.assertEqual(gn.blks[1][1], 'HEX8')
        self.assertEqual(gn.blks[1][2].shape, (75, 8))
        # BCs.
        self.assertEqual(len(gn.bcs), 6)
        self.assertEqual(gn.bcs[0][0], 'front')
        self.assertEqual(gn.bcs[0][1].shape, (25,))
        self.assertEqual(gn.bcs[0][2].shape, (25,))
        self.assertEqual(gn.bcs[1][0], 'rear')
        self.assertEqual(gn.bcs[1][1].shape, (25,))
        self.assertEqual(gn.bcs[1][2].shape, (25,))
        self.assertEqual(gn.bcs[2][0], 'lower')
        self.assertEqual(gn.bcs[2][1].shape, (30,))
        self.assertEqual(gn.bcs[2][2].shape, (30,))
        self.assertEqual(gn.bcs[3][0], 'left')
        self.assertEqual(gn.bcs[3][1].shape, (30,))
        self.assertEqual(gn.bcs[3][2].shape, (30,))
        self.assertEqual(gn.bcs[4][0], 'upper')
        self.assertEqual(gn.bcs[4][1].shape, (30,))
        self.assertEqual(gn.bcs[4][2].shape, (30,))
        self.assertEqual(gn.bcs[5][0], 'right')
        self.assertEqual(gn.bcs[5][1].shape, (30,))
        self.assertEqual(gn.bcs[5][2].shape, (30,))
        # coordinate.
        self.assertEqual(gn.ndcrd.shape, (252, 3))
        self.assertTrue((gn.ndcrd >= -0.5-1.e-15).all())
        self.assertTrue((gn.ndcrd <= 0.5+1.e-15).all())
        # mapper.
        self.assertTrue((gn.emap == arange(150)+1).all())

    def test_convert_interior(self):
        from ...block import Block
        from ..genesis import Genesis
        # load from netCDF.
        gn = Genesis(self.testfn)
        gn.load()
        gn.close_file()
        # convert.
        blk = Block(ndim=gn.ndim, nnode=gn.nnode, ncell=gn.ncell,
            fpdtype='float64')
        gn._convert_interior_to(blk)
        # test cell type.
        self.assertTrue((blk.cltpn == 4).all())
        self.assertTrue((blk.clnds[:,0] == 8).all())
        # test index of node in cell.
        self.assertEqual(blk.clnds[:,1:9].min(), 0)
        self.assertEqual(blk.clnds[:,1:9].max(), 252-1)
        # test group.
        self.assertEqual(len(blk.grpnames), 2)
        self.assertEqual(blk.clgrp.min(), 0)
        self.assertEqual((blk.clgrp==0).sum(), 75)
        self.assertEqual(blk.clgrp.max(), 1)
        self.assertEqual((blk.clgrp==1).sum(), 75)

    def test_toblock(self):
        from numpy import abs
        from ..genesis import Genesis
        # load from netCDF.
        gn = Genesis(self.testfn)
        gn.load()
        gn.close_file()
        # convert.
        blk = gn.toblock()
        # test BC.
        self.assertEqual(len(blk.bclist), 6)
        self.assertEqual(blk.bclist[0].name, 'front')
        self.assertEqual(len(blk.bclist[0]), 25)
        self.assertTrue((abs(blk.fccnd[blk.bclist[0].facn[:,0],2] - 0.5)
            < 1.e-15).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[0].facn[:,0]].sum(), 1.0,
            15)
        self.assertEqual(blk.bclist[1].name, 'rear')
        self.assertEqual(len(blk.bclist[1]), 25)
        self.assertTrue((abs(blk.fccnd[blk.bclist[1].facn[:,0],2] + 0.5) 
            < 1.e-15).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[1].facn[:,0]].sum(), 1.0,
            15)
        self.assertEqual(blk.bclist[2].name, 'lower')
        self.assertEqual(len(blk.bclist[2]), 30)
        self.assertTrue((abs(blk.fccnd[blk.bclist[2].facn[:,0],1] + 0.5)
            < 1.e-15).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[2].facn[:,0]].sum(), 1.0,
            15)
        self.assertEqual(blk.bclist[3].name, 'left')
        self.assertEqual(len(blk.bclist[3]), 30)
        self.assertTrue((abs(blk.fccnd[blk.bclist[3].facn[:,0],0] + 0.5)
            < 1.e-15).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[3].facn[:,0]].sum(), 1.0,
            15)
        self.assertEqual(blk.bclist[4].name, 'upper')
        self.assertEqual(len(blk.bclist[4]), 30)
        self.assertTrue((abs(blk.fccnd[blk.bclist[4].facn[:,0],1] - 0.5)
            < 1.e-15).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[4].facn[:,0]].sum(), 1.0,
            15)
        self.assertEqual(blk.bclist[5].name, 'right')
        self.assertEqual(len(blk.bclist[5]), 30)
        self.assertTrue((abs(blk.fccnd[blk.bclist[5].facn[:,0],0] - 0.5)
            < 1.e-15).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[5].facn[:,0]].sum(), 1.0,
            15)
        # test geometry.
        self.assertAlmostEqual(blk.ndcrd.min(), -0.5, 15)
        self.assertAlmostEqual(blk.ndcrd.max(), 0.5, 15)
        self.assertAlmostEqual(blk.clvol.sum(), 1.0, 14)

class TestReadTri(TestCase):
    import os
    from ...conf import env
    testfn = [env.datadir] + ['square_t200mm.g']
    testfn = os.path.join(*testfn)

    def test_dim(self):
        from ..genesis import Genesis
        gn = Genesis(self.testfn)
        self.assertEqual(gn.get_dim('num_dim'), 2)
        self.assertEqual(gn.get_dim('num_nodes'), 41)
        self.assertEqual(gn.get_dim('num_elem'), 60)
        self.assertEqual(gn.get_dim('num_el_blk'), 1)
        self.assertEqual(gn.get_dim('num_side_sets'), 4)
        gn.close_file()

    def test_ss_names(self):
        from ..genesis import Genesis
        gn = Genesis(self.testfn)
        nbc = gn.get_dim('num_side_sets')
        slen = gn.get_dim('len_string')
        lines = gn.get_lines('ss_names', (nbc, slen))
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], 'upper')
        self.assertEqual(gn.get_dim('num_side_ss1'), 5)
        self.assertEqual(lines[1], 'left')
        self.assertEqual(gn.get_dim('num_side_ss2'), 5)
        self.assertEqual(lines[2], 'lower')
        self.assertEqual(gn.get_dim('num_side_ss3'), 5)
        self.assertEqual(lines[3], 'right')
        self.assertEqual(gn.get_dim('num_side_ss4'), 5)
        gn.close_file()

    def test_coord(self):
        from ..genesis import Genesis
        gn = Genesis(self.testfn)
        ndim = gn.get_dim('num_dim')
        nnode = gn.get_dim('num_nodes')
        ndcrd = gn.get_array('coord', (ndim, nnode), 'float64').T.copy()
        self.assertTrue((ndcrd >= -0.5-1.e-15).all())
        self.assertTrue((ndcrd <= 0.5+1.e-15).all())
        gn.close_file()

    def test_block(self):
        from ..genesis import Genesis
        gn = Genesis(self.testfn)
        # name.
        nblk = gn.get_dim('num_el_blk')
        slen = gn.get_dim('len_string')
        blks = gn.get_lines('eb_names', (nblk, slen))
        self.assertEqual(len(blks), 1)
        # block 1.
        ncell = gn.get_dim('num_el_in_blk1')
        clnnd = gn.get_dim('num_nod_per_el1')
        clnds = gn.get_array('connect1', (ncell, clnnd), 'int32')
        self.assertEqual(clnds.min(), 1)
        self.assertEqual(clnds.max(), 41)
        self.assertEqual(gn.get_attr_text('elem_type', 'connect1'), 'TRI3')
        gn.close_file()

    def test_load(self):
        from numpy import arange
        from ..genesis import Genesis
        # load from netCDF.
        gn = Genesis(self.testfn)
        gn.load()
        gn.close_file()
        # meta data.
        self.assertEqual(gn.ndim, 2)
        self.assertEqual(gn.nnode, 41)
        self.assertEqual(gn.ncell, 60)
        # blocks.
        self.assertEqual(len(gn.blks), 1)
        self.assertEqual(gn.blks[0][1], 'TRI3')
        self.assertEqual(gn.blks[0][2].shape, (60, 3))
        # BCs.
        self.assertEqual(len(gn.bcs), 4)
        self.assertEqual(gn.bcs[0][0], 'upper')
        self.assertEqual(gn.bcs[0][1].shape, (5,))
        self.assertEqual(gn.bcs[0][2].shape, (5,))
        self.assertEqual(gn.bcs[1][0], 'left')
        self.assertEqual(gn.bcs[1][1].shape, (5,))
        self.assertEqual(gn.bcs[1][2].shape, (5,))
        self.assertEqual(gn.bcs[2][0], 'lower')
        self.assertEqual(gn.bcs[2][1].shape, (5,))
        self.assertEqual(gn.bcs[2][2].shape, (5,))
        self.assertEqual(gn.bcs[3][0], 'right')
        self.assertEqual(gn.bcs[3][1].shape, (5,))
        self.assertEqual(gn.bcs[3][2].shape, (5,))
        # coordinate.
        self.assertEqual(gn.ndcrd.shape, (41, 2))
        self.assertTrue((gn.ndcrd >= -0.5-1.e-15).all())
        self.assertTrue((gn.ndcrd <= 0.5+1.e-15).all())
        # mapper.
        self.assertTrue((gn.emap == arange(60)+1).all())

    def test_convert_interior(self):
        from ...block import Block
        from ..genesis import Genesis
        # load from netCDF.
        gn = Genesis(self.testfn)
        gn.load()
        gn.close_file()
        # convert.
        blk = Block(ndim=gn.ndim, nnode=gn.nnode, ncell=gn.ncell,
            fpdtype='float64')
        gn._convert_interior_to(blk)
        # test cell type.
        self.assertTrue((blk.cltpn == 3).all())
        self.assertTrue((blk.clnds[:,0] == 3).all())
        # test index of node in cell.
        self.assertEqual(blk.clnds[:,1:4].min(), 0)
        self.assertEqual(blk.clnds[:,1:4].max(), 41-1)
        # test group.
        self.assertEqual(len(blk.grpnames), 1)
        self.assertEqual(blk.clgrp.min(), 0)
        self.assertEqual((blk.clgrp==0).sum(), 60)

    def test_toblock(self):
        from numpy import abs
        from ..genesis import Genesis
        # load from netCDF.
        gn = Genesis(self.testfn)
        gn.load()
        gn.close_file()
        # convert.
        blk = gn.toblock()
        # test BC.
        self.assertEqual(len(blk.bclist), 4)
        self.assertEqual(blk.bclist[0].name, 'upper')
        self.assertEqual(len(blk.bclist[0]), 5)
        self.assertTrue((abs(blk.fccnd[blk.bclist[0].facn[:,0],1] - 0.5)
            < 1.e-15).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[0].facn[:,0]].sum(), 1.0,
            15)
        self.assertEqual(blk.bclist[1].name, 'left')
        self.assertEqual(len(blk.bclist[1]), 5)
        self.assertTrue((abs(blk.fccnd[blk.bclist[1].facn[:,0],0] + 0.5) 
            < 1.e-15).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[1].facn[:,0]].sum(), 1.0,
            15)
        self.assertEqual(blk.bclist[2].name, 'lower')
        self.assertEqual(len(blk.bclist[2]), 5)
        self.assertTrue((abs(blk.fccnd[blk.bclist[2].facn[:,0],1] + 0.5)
            < 1.e-15).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[2].facn[:,0]].sum(), 1.0,
            15)
        self.assertEqual(blk.bclist[3].name, 'right')
        self.assertEqual(len(blk.bclist[3]), 5)
        self.assertTrue((abs(blk.fccnd[blk.bclist[3].facn[:,0],0] - 0.5)
            < 1.e-15).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[3].facn[:,0]].sum(), 1.0,
            15)
        # test geometry.
        self.assertAlmostEqual(blk.ndcrd.min(), -0.5, 15)
        self.assertAlmostEqual(blk.ndcrd.max(), 0.5, 15)
        self.assertAlmostEqual(blk.clvol.sum(), 1.0, 14)

class TestReadQuad(TestCase):
    import os
    from ...conf import env
    testfn = [env.datadir] + ['square_q200mm.g']
    testfn = os.path.join(*testfn)

    def test_dim(self):
        from ..genesis import Genesis
        gn = Genesis(self.testfn)
        self.assertEqual(gn.get_dim('num_dim'), 2)
        self.assertEqual(gn.get_dim('num_nodes'), 36)
        self.assertEqual(gn.get_dim('num_elem'), 25)
        self.assertEqual(gn.get_dim('num_el_blk'), 1)
        self.assertEqual(gn.get_dim('num_side_sets'), 4)
        gn.close_file()

    def test_ss_names(self):
        from ..genesis import Genesis
        gn = Genesis(self.testfn)
        nbc = gn.get_dim('num_side_sets')
        slen = gn.get_dim('len_string')
        lines = gn.get_lines('ss_names', (nbc, slen))
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], 'upper')
        self.assertEqual(gn.get_dim('num_side_ss1'), 5)
        self.assertEqual(lines[1], 'left')
        self.assertEqual(gn.get_dim('num_side_ss2'), 5)
        self.assertEqual(lines[2], 'lower')
        self.assertEqual(gn.get_dim('num_side_ss3'), 5)
        self.assertEqual(lines[3], 'right')
        self.assertEqual(gn.get_dim('num_side_ss4'), 5)
        gn.close_file()

    def test_coord(self):
        from ..genesis import Genesis
        gn = Genesis(self.testfn)
        ndim = gn.get_dim('num_dim')
        nnode = gn.get_dim('num_nodes')
        ndcrd = gn.get_array('coord', (ndim, nnode), 'float64').T.copy()
        self.assertTrue((ndcrd >= -0.5-1.e-15).all())
        self.assertTrue((ndcrd <= 0.5+1.e-15).all())
        gn.close_file()

    def test_block(self):
        from ..genesis import Genesis
        gn = Genesis(self.testfn)
        # name.
        nblk = gn.get_dim('num_el_blk')
        slen = gn.get_dim('len_string')
        blks = gn.get_lines('eb_names', (nblk, slen))
        self.assertEqual(len(blks), 1)
        # block 1.
        ncell = gn.get_dim('num_el_in_blk1')
        clnnd = gn.get_dim('num_nod_per_el1')
        clnds = gn.get_array('connect1', (ncell, clnnd), 'int32')
        self.assertEqual(clnds.min(), 1)
        self.assertEqual(clnds.max(), 36)
        self.assertEqual(gn.get_attr_text('elem_type', 'connect1'), 'SHELL4')
        gn.close_file()

    def test_load(self):
        from numpy import arange
        from ..genesis import Genesis
        # load from netCDF.
        gn = Genesis(self.testfn)
        gn.load()
        gn.close_file()
        # meta data.
        self.assertEqual(gn.ndim, 2)
        self.assertEqual(gn.nnode, 36)
        self.assertEqual(gn.ncell, 25)
        # blocks.
        self.assertEqual(len(gn.blks), 1)
        self.assertEqual(gn.blks[0][1], 'SHELL4')
        self.assertEqual(gn.blks[0][2].shape, (25, 4))
        # BCs.
        self.assertEqual(len(gn.bcs), 4)
        self.assertEqual(gn.bcs[0][0], 'upper')
        self.assertEqual(gn.bcs[0][1].shape, (5,))
        self.assertEqual(gn.bcs[0][2].shape, (5,))
        self.assertEqual(gn.bcs[1][0], 'left')
        self.assertEqual(gn.bcs[1][1].shape, (5,))
        self.assertEqual(gn.bcs[1][2].shape, (5,))
        self.assertEqual(gn.bcs[2][0], 'lower')
        self.assertEqual(gn.bcs[2][1].shape, (5,))
        self.assertEqual(gn.bcs[2][2].shape, (5,))
        self.assertEqual(gn.bcs[3][0], 'right')
        self.assertEqual(gn.bcs[3][1].shape, (5,))
        self.assertEqual(gn.bcs[3][2].shape, (5,))
        # coordinate.
        self.assertEqual(gn.ndcrd.shape, (36, 2))
        self.assertTrue((gn.ndcrd >= -0.5-1.e-15).all())
        self.assertTrue((gn.ndcrd <= 0.5+1.e-15).all())
        # mapper.
        self.assertTrue((gn.emap == arange(25)+1).all())

    def test_convert_interior(self):
        from ...block import Block
        from ..genesis import Genesis
        # load from netCDF.
        gn = Genesis(self.testfn)
        gn.load()
        gn.close_file()
        # convert.
        blk = Block(ndim=gn.ndim, nnode=gn.nnode, ncell=gn.ncell,
            fpdtype='float64')
        gn._convert_interior_to(blk)
        # test cell type.
        self.assertTrue((blk.cltpn == 2).all())
        self.assertTrue((blk.clnds[:,0] == 4).all())
        # test index of node in cell.
        self.assertEqual(blk.clnds[:,1:5].min(), 0)
        self.assertEqual(blk.clnds[:,1:5].max(), 36-1)
        # test group.
        self.assertEqual(len(blk.grpnames), 1)
        self.assertEqual(blk.clgrp.min(), 0)
        self.assertEqual((blk.clgrp==0).sum(), 25)

    def test_toblock(self):
        from numpy import abs
        from ..genesis import Genesis
        # load from netCDF.
        gn = Genesis(self.testfn)
        gn.load()
        gn.close_file()
        # convert.
        blk = gn.toblock()
        # test BC.
        self.assertEqual(len(blk.bclist), 4)
        self.assertEqual(blk.bclist[0].name, 'upper')
        self.assertEqual(len(blk.bclist[0]), 5)
        self.assertTrue((abs(blk.fccnd[blk.bclist[0].facn[:,0],1] - 0.5)
            < 1.e-15).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[0].facn[:,0]].sum(), 1.0,
            15)
        self.assertEqual(blk.bclist[1].name, 'left')
        self.assertEqual(len(blk.bclist[1]), 5)
        self.assertTrue((abs(blk.fccnd[blk.bclist[1].facn[:,0],0] + 0.5) 
            < 1.e-15).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[1].facn[:,0]].sum(), 1.0,
            15)
        self.assertEqual(blk.bclist[2].name, 'lower')
        self.assertEqual(len(blk.bclist[2]), 5)
        self.assertTrue((abs(blk.fccnd[blk.bclist[2].facn[:,0],1] + 0.5)
            < 1.e-15).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[2].facn[:,0]].sum(), 1.0,
            15)
        self.assertEqual(blk.bclist[3].name, 'right')
        self.assertEqual(len(blk.bclist[3]), 5)
        self.assertTrue((abs(blk.fccnd[blk.bclist[3].facn[:,0],0] - 0.5)
            < 1.e-15).all())
        self.assertAlmostEqual(blk.fcara[blk.bclist[3].facn[:,0]].sum(), 1.0,
            15)
        # test geometry.
        self.assertAlmostEqual(blk.ndcrd.min(), -0.5, 15)
        self.assertAlmostEqual(blk.ndcrd.max(), 0.5, 15)
        self.assertAlmostEqual(blk.clvol.sum(), 1.0, 14)

class TestLarge(TestCase):
    def test_compare(self):
        import os
        from ...conf import env
        from ..genesis import Genesis
        gn1 = Genesis(os.path.join(env.datadir, 'cubic_t200mm.g'))
        gn1.load()
        gn1.close_file()
        gn2 = Genesis(os.path.join(env.datadir, 'cubic_t200mm_large.g'))
        gn2.load()
        gn2.close_file()
        self.assertTrue((gn1.ndcrd == gn2.ndcrd).all())

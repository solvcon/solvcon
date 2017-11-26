# -*- coding: UTF-8 -*-
#
# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>


"""
Surface.
"""


import numpy as np

import pythreejs as ptjs

import solvcon as sc


class Surface(ptjs.Mesh):

    def __init__(self, source, *args, **kw):
        self.source = source
        if isinstance(self.source, sc.Block):
            kw['geometry'] = self.make_geometry_from_block(self.source)
        elif isinstance(self.source, sc.BC):
            kw['geometry'] = self.make_geometry_from_bc(self.source)
        else:
            raise TypeError("support only Block or BC")

        material = kw.pop('material', None)
        color = kw.pop('color', 0x636DC9)
        side = kw.pop('side', 'DoubleSide')
        if not material:
            material = ptjs.BasicMaterial(color=color, side=side)
        kw['material'] = material

        super(Surface, self).__init__(*args, **kw)

    @staticmethod
    def make_geometry_from_block(blk):
        if blk.ndim != 2:
            raise ValueError("Block must be 2D")

        geometry = ptjs.FaceGeometry()

        ndcrd = np.zeros((blk.nnode, 3), dtype=blk.ndcrd.dtype)
        ndcrd[:,:2] = blk.ndcrd
        geometry.vertices = ndcrd.ravel().tolist()

        slct = blk.clnds[:,0] == 3
        geometry.face3 = blk.clnds[slct,1:4].ravel().tolist()
        slct = blk.clnds[:,0] == 4
        geometry.face4 = blk.clnds[slct,1:5].ravel().tolist()

        return geometry

    @staticmethod
    def make_geometry_from_bc(bc):
        if bc.blk.ndim != 3:
            raise ValueError("BC must be of 3D Block")

        nface = len(bc)
        fcs_orig = bc.facn[:,0]
        fcnds = bc.blk.fcnds[fcs_orig]
        nds = fcnds[:,1:].flatten()
        nds.sort()
        nds = np.unique(nds)
        it = 0
        while nds[it] < 0: # Skip leading negative.
            it += 1
        nds = nds[it:].copy()
        nnode = nds.shape[0]
        ndcrd = bc.blk.ndcrd[nds]
        ndmap = dict((val, it) for it, val in enumerate(nds))
        it = 0
        while it < fcnds.shape[0]:
            jt = 1
            while jt <= fcnds[it,0] and fcnds[it,jt] >= 0:
                # Reindex the nodes.  FIXME: slow loops?
                fcnds[it,jt] = ndmap[fcnds[it,jt]]
                jt += 1
            it += 1

        geometry = ptjs.FaceGeometry()
        geometry.vertices = ndcrd.ravel().tolist()

        slct = fcnds[:,0] == 3
        geometry.face3 = fcnds[slct,1:4].ravel().tolist()
        slct = fcnds[:,0] == 4
        geometry.face4 = fcnds[slct,1:5].ravel().tolist()

        return geometry

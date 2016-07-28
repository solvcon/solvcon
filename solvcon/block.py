# -*- coding: UTF-8 -*-
#
# Copyright (c) 2008, Yung-Yu Chen <yyc@solvcon.net>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Unstructured mesh definition."""


from __future__ import absolute_import, division, print_function


__all__ = [
    'Block',
]


import warnings
import importlib
import json

import numpy as np

from .py3kcompat import with_metaclass
from . import boundcond
from . import dependency
dependency.import_module_may_fail('.mesh')
dependency.import_module_may_fail('.march')


# metadata for unstructured mesh.
elemtype = np.array([
    # index, dim, node, edge, surface,    name
    [     0,   0,    1,    0,       0, ], # node/point/vertex
    [     1,   1,    2,    0,       0, ], # line/edge
    [     2,   2,    4,    4,       0, ], # quadrilateral
    [     3,   2,    3,    3,       0, ], # triangle
    [     4,   3,    8,   12,       6, ], # hexahedron/brick
    [     5,   3,    4,    6,       4, ], # tetrahedron
    [     6,   3,    6,    9,       5, ], # prism/wedge
    [     7,   3,    5,    8,       5, ], # pyramid
], dtype='int32')
MAX_FCNND = elemtype[elemtype[:,1]<3,2].max()
MAX_CLNND = elemtype[:,2].max()
MAX_CLNFC = max(elemtype[elemtype[:,1]==2,3].max(),
                elemtype[elemtype[:,1]==3,4].max())


class BlockJSONEncoder(json.JSONEncoder):
    """
    JSON serialization helper for :py:class:`Block`.  Only interior data are
    serialized.

    >>> # build a 2D block.
    >>> blk = Block(ndim=2, nnode=4, nface=6, ncell=3, nbound=3)
    >>> blk.ndcrd[:,:] = (0,0), (-1,-1), (1,-1), (0,1)
    >>> blk.cltpn[:] = 3
    >>> blk.clnds[:,:4] = (3, 0,1,2), (3, 0,2,3), (3, 0,3,1)
    >>> blk.build_interior()
    >>> blk.build_boundary()
    >>> blk.build_ghost()
    >>> # without this encoder, Block isn't JSON serializable.
    >>> import json
    >>> json.dumps(blk) # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    TypeError: ... is not JSON serializable
    >>> # with the encoder, Block can be turned to a JSON string.
    >>> import json
    >>> line = json.dumps(blk, cls=BlockJSONEncoder)
    >>> # what we serialized to JSON.  note 2/3 compatibility.
    >>> sorted(str(key) for key in json.loads(line).keys()) # doctest: +NORMALIZE_WHITESPACE
    ['bndfcs',
     'clcnd', 'clfcs', 'clgrp', 'clnds', 'cltpn', 'clvol',
     'fcara', 'fccls', 'fccnd', 'fcnds', 'fcnml', 'fctpn',
     'nbound', 'ncell', 'ndcrd', 'ndim', 'nface', 'nnode']
    """

    def default(self, blk):
        # Let the base class to raise TypeError.
        if not isinstance(blk, Block):
            return super(BlockJSONEncoder, self).default(blk)
        # Convert.
        dataset = {
            key: getattr(blk, key) for key in
            ("ndim", "nnode", "nface", "ncell", "nbound")
        }
        dataset.update({
            key: getattr(blk, key).tolist() for key in
            ("ndcrd", "fccnd", "fcnml", "fcara", "clcnd", "clvol",
             "fctpn", "cltpn", "clgrp",
             "fcnds", "fccls", "clnds", "clfcs",
             "bndfcs")
        })
        return dataset


class _TableDescriptor(object):
    """
    Control the access of array attributes in :py:mod:`Block`.
    """

    def __init__(self, name, prefix, collector_name):
        self.name = name
        self.prefix = prefix
        self.collector_name = collector_name

    def _get_collector(self, ins):
        """This indirection prevents cyclic reference to *ins*."""
        if not hasattr(ins, self.collector_name):
            setattr(ins, self.collector_name, dict())
        return getattr(ins, self.collector_name)

    def __get__(self, ins, cls):
        if self.name not in ins.TABLE_NAMES:
            raise AttributeError('"%s" is not in Block.TABLE_NAME'%self.name)
        collector = self._get_collector(ins)
        return collector[self.name]

    def __set__(self, ins, val):
        if not isinstance(val, (march.Table, np.ndarray)):
            raise TypeError('only Table and ndarray are acceptable')
        if self.name not in ins.TABLE_NAMES:
            raise AttributeError('"%s" is not in Block.TABLE_NAME'%self.name)
        collector = self._get_collector(ins)
        collector[self.name] = val

    def __delete__(self, ins):
        raise AttributeError("can't delete attribute %s" % self.name)


class BlockMeta(type):
    def __new__(cls, name, bases, namespace):
        # Table names.
        namespace['GEOMETRY_TABLE_NAMES'] = (
            'ndcrd', 'fccnd', 'fcnml', 'fcara', 'clcnd', 'clvol')
        namespace['META_TABLE_NAMES'] = (
            'fctpn', 'cltpn', 'clgrp')
        namespace['CONNECTIVITY_TABLE_NAMES'] = (
            'fcnds', 'fccls', 'clnds', 'clfcs')
        namespace['TABLE_NAMES'] = TABLE_NAMES = (
            namespace['GEOMETRY_TABLE_NAMES']
          + namespace['META_TABLE_NAMES']
          + namespace['CONNECTIVITY_TABLE_NAMES'])
        # Table descriptors.
        for cname, prefix in (('_tables','tb'), ('_shared_arrays','sh'),
                              ('_body_arrays',''), ('_ghost_arrays','gst')):
            for tname in TABLE_NAMES:
                descr = _TableDescriptor(tname, prefix, cname)
                namespace[prefix+tname] = descr
        return super(BlockMeta, cls).__new__(cls, name, bases, namespace)


class Block(with_metaclass(BlockMeta)):
    """
    :ivar use_incenter: specify using incenter or not.
    :itype use_incenter: bool
    :ivar blkn: serial number of the block.
    :ivar bclist: list of associated BC objects.
    :ivar bndfcs: list of BC faces, contains BC face index and BC class serial
        number, respectively.  The type number definition follows Nasa 2D CESE 
        code.
    :ivar grpnames: list of names of cell groups.
    :ivar ndcrd: Node croodinate data.
    :ivar fccnd: Central coordinates of face.
    :ivar fcnml: Unit-normal vector of face.
    :ivar fcara: Area of face.
    :ivar clcnd: Central coordinates of cell.
    :ivar clvol: Volume of cell.
    :ivar fctpn: Type of face.
    :ivar cltpn: Type of cell.
    :ivar clgrp: Group index of cell.
    :ivar fcnds: List of nodes in face; arr[:,0] for the number.
    :ivar fccls: Related cells for each face, contains belong, neibor (ghost as
        negative), neiblk, and neibcl (cell index in neighboring block), 
        respectively.
    :ivar clnds: List of nodes in cell; arr[:,0] for the number.
    :ivar clfcs: List of faces in cell; arr[:,0] for the number.

    Provide geometry and connectivity information for unstructured-mesh block.
    In terms of APIs in the Block class, to build a block requires the
    following actions:

    1. build_interior: build up inner connectivity and call calc_metric to
      calculate metrics for interior meshes.
    2. build_boundary: build up information of boundary faces according to
       boundary conditions objects (BCs).  Also "patch" the block with
       "unspecified" BC for boundary faces without related BC objects.
    3. build_ghost: build up information for ghost cells.  Ghost cells is for
       treatment of boundary conditions in a solving code/logic.  The
       build-up includes create ghost entities (nodes, faces, and cells
       itself) and their connectivity and metrics.  The storage of interior
       meshed will be changed to make the storgae for both interior and ghost
       information contiguous.

    .. note:

        Prefixes: nd = node, fc = face, cl = cell; gst = ghost; sh = shared.

    >>> # build a 2D block.
    >>> blk = Block(ndim=2, nnode=4, nface=6, ncell=3, nbound=3)
    >>> blk.ndcrd[:,:] = (0,0), (-1,-1), (1,-1), (0,1)
    >>> blk.cltpn[:] = 3
    >>> blk.clnds[:,:4] = (3, 0,1,2), (3, 0,2,3), (3, 0,3,1)
    >>> blk.build_interior()
    >>> blk.build_boundary()
    >>> blk.build_ghost()
    >>> # also test for a short-hand method.
    >>> from .testing import create_trivial_2d_blk
    >>> tblk = create_trivial_2d_blk()
    >>> (tblk.shndcrd == blk.shndcrd).all()
    True
    >>> (tblk.shfccnd == blk.shfccnd).all()
    True
    >>> (tblk.shfccls == blk.shfccls).all()
    True
    """

    FCMND = MAX_FCNND
    CLMND = MAX_CLNND
    CLMFC = MAX_CLNFC

    JSONEncoder = BlockJSONEncoder

    def __init__(self, *args, **kw):
        """
        :keyword fpdtype: dtype for the floating point data.  Deprecated.
        :keyword ndim: spatial dimension.
        :type ndim: int
        :keyword nnode: number of nodes.
        :type nnode: int
        :keyword nface: number of faces.
        :type nface: int
        :keyword ncell: number of cells.
        :type ncell: int
        :keyword nbound: number of BC faces.
        :type nbound: int
        :keyword use_incenter: specify using incenter or not.
        :type use_incenter: bool

        Initialization.
        """
        # get rid of fpdtype setting.
        kw.pop('fpdtype', None)
        # get parameters.
        ndim = kw.setdefault('ndim', 0)
        nnode = kw.setdefault('nnode', 0)
        nface = kw.setdefault('nface', 0)
        ncell = kw.setdefault('ncell', 0)
        nbound = kw.setdefault('nbound', 0)
        self.use_incenter = kw.setdefault('use_incenter', False)
        # serial number of the block.
        self.blkn = None
        # boundary conditions and boundary faces information.
        self.bclist = list()
        self.bndfcs = np.empty((nbound, 2), dtype='int32')
        # group names.
        self.grpnames = list()
        # all tables.
        self._build_tables(ndim, nnode, nface, ncell, 0, 0, 0)
        for name in self.TABLE_NAMES:
            table = getattr(self, 'tb'+name)
            setattr(self, name, table.B)
            setattr(self, 'gst'+name, table.G)
            setattr(self, 'sh'+name, table.F)
        # keep initialization sequence.
        super(Block, self).__init__()
        # sanity check.
        self.check_sanity()

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state
        for name in self.TABLE_NAMES:
            table = getattr(self, 'tb'+name)
            setattr(self, name, table.B)
            setattr(self, 'gst'+name, table.G)
            setattr(self, 'sh'+name, table.F)

    def check_sanity(self):
        self.create_msh()

    def _build_tables(self, ndim, nnode, nface, ncell,
                      ngstnode, ngstface, ngstcell):
        """
        Allocate memory by creating solvcon.march.Table objects.

        This method alters content in self object.

        :param ndim: number of dimensionality.
        :param nnode: number of body nodes.
        :param nface: number of body faces.
        :param ncell: number of body cells.
        :param ngstnode: number of nodes for ghost cells.
        :param ngstcell: number of faces for ghost cells.
        :param ngstcell: number of ghost cells.
        :return: nothing.
        """
        # metrics.
        self.tbndcrd = march.Table(ngstnode, nnode, ndim, dtype=self.fpdtype)
        self.tbfccnd = march.Table(ngstface, nface, ndim, dtype=self.fpdtype)
        self.tbfcnml = march.Table(ngstface, nface, ndim, dtype=self.fpdtype)
        self.tbfcara = march.Table(ngstface, nface, dtype=self.fpdtype)
        self.tbclcnd = march.Table(ngstcell, ncell, ndim, dtype=self.fpdtype)
        self.tbclvol = march.Table(ngstcell, ncell, dtype=self.fpdtype)
        # meta/type.
        self.tbfctpn = march.Table(ngstface, nface, dtype='int32')
        self.tbcltpn = march.Table(ngstcell, ncell, dtype='int32')
        self.tbclgrp = march.Table(ngstcell, ncell, dtype='int32')
        self.tbclgrp.F.fill(-1)
        # connectivities.
        self.tbfcnds = march.Table(ngstface, nface, self.FCMND+1, dtype='int32')
        self.tbfccls = march.Table(ngstface, nface, 4, dtype='int32')
        self.tbclnds = march.Table(ngstcell, ncell, self.CLMND+1, dtype='int32')
        self.tbclfcs = march.Table(ngstcell, ncell, self.CLMFC+1, dtype='int32')
        for name in ('fcnds', 'fccls', 'clnds', 'clfcs'):
            getattr(self, 'tb'+name).F.fill(-1)

    def __str__(self):
        return ', '.join([
            '[Block (%dD/%s): %d nodes'%(self.ndim,
                'incenter' if self.use_incenter else 'centroid',
                self.nnode),
            '%d faces (%d BC)'%(self.nface, self.nbound),
            '%d cells]'%self.ncell
        ])

    @property
    def fpdtype(self):
        return np.dtype('float64')
    @property
    def fpdtypestr(self):
        return dependency.str_of(self.fpdtype)

    @property
    def ndim(self):
        return self.ndcrd.shape[1]
    @property
    def nnode(self):
        return self.ndcrd.shape[0]
    @property
    def nface(self):
        return self.fcnds.shape[0]
    @property
    def ncell(self):
        return self.clnds.shape[0]
    @property
    def nbound(self):
        return self.bndfcs.shape[0]
    @property
    def ngstnode(self):
        return self.gstndcrd.shape[0]
    @property
    def ngstface(self):
        return self.gstfcnds.shape[0]
    @property
    def ngstcell(self):
        return self.gstclnds.shape[0]

    def check_simplex(self):
        """
        Check whether or not the block is composed purely by simplices, i.e.,
        triangles in two-dimensional space, or tetrahedra in three-dimensional
        space.

        @return: True if the block is composed purely by simplices; otherwise
            False.
        @rtype: bool
        """
        if self.ndim == 2:
            ret = (self.cltpn == 3).all()
        elif self.ndim == 3:
            ret = (self.cltpn == 5).all()
        else:
            raise ValueError('ndim %d != 2 or 3' % self.ndim)
        return ret

    def bind(self):
        for bc in self.bclist:
            bc.bind()
    def unbind(self):
        for bc in self.bclist:
            bc.unbind()

    def create_msh(self):
        """
        :return: An object contains the :c:type:`sc_mesh_t` variable for C code to
          use data in the :py:class:`Block` object.
        :rtype: :py:class:`solvcon.mesh.Mesh`

        The following code shows how and when to use this method:

        >>> blk = Block(ndim=2, nnode=4, nface=6, ncell=3, nbound=3)
        >>> blk.ndcrd[:,:] = (0,0), (-1,-1), (1,-1), (0,1)
        >>> blk.cltpn[:] = 3
        >>> blk.clnds[:,:4] = (3, 0,1,2), (3, 0,2,3), (3, 0,3,1)
        >>> blk.build_interior()
        >>> # it's OK to get a msh when its content is still invalid.
        >>> msh = blk.create_msh()
        >>> blk.build_boundary()
        >>> blk.build_ghost()
        >>> # now the msh is valid for the blk is fully built-up.
        >>> msh = blk.create_msh()
        """
        msh = mesh.Mesh()
        msh.setup_mesh(self)
        return msh

    def calc_metric(self):
        """
        Calculate metrics including normal vector and area of faces, and
        centroid coordinates and volume of cells.

        @return: nothing.
        """
        self.create_msh().calc_metric(self.use_incenter)

    def build_interior(self):
        """
        :return: Nothing.
        """
        # prepare to build connectivity: calculate max number of faces.
        max_nfc = 0
        for sig in elemtype[1:2]:   # 1D cells.
            max_nfc += (self.cltpn == sig[0]).sum()*sig[2]
        for sig in elemtype[2:4]:   # 2D cells.
            max_nfc += (self.cltpn == sig[0]).sum()*sig[3]
        for sig in elemtype[4:]:    # 3D cells.
            max_nfc += (self.cltpn == sig[0]).sum()*sig[4]
        # build connectivity information: get face definition from node list 
        # of cells.
        msh = self.create_msh()
        clfcs, fctpn, fcnds, fccls = msh.extract_faces_from_cells(max_nfc)
        nface = fctpn.shape[0]
        # check for initialization of information for faces.
        if self.nface != nface:
            # face arrays used in this method.
            self.tbfctpn = march.Table(0, nface, dtype='int32')
            self.tbfcnds = march.Table(0, nface, self.FCMND+1, dtype='int32')
            self.tbfccls = march.Table(0, nface, 4, dtype='int32')
            self.tbfccnd = march.Table(0, nface, self.ndim, dtype=self.fpdtype)
            self.tbfcnml = march.Table(0, nface, self.ndim, dtype=self.fpdtype)
            self.tbfcara = march.Table(0, nface, dtype=self.fpdtype)
            for name in ('fctpn', 'fcnds', 'fccls', 'fccnd', 'fcnml', 'fcara'):
                table = getattr(self, 'tb'+name)
                setattr(self, name, table.B)
                setattr(self, 'gst'+name, table.G)
                setattr(self, 'sh'+name, table.F)
        # assign extracted data to block.
        self.clfcs[:,:] = clfcs[:,:]
        self.fctpn[:] = fctpn[:]
        self.fcnds[:,:] = fcnds[:,:]
        self.fccls[:,:] = fccls[:,:]
        # calculate metric information.
        self.calc_metric()

    def build_boundary(self, unspec_type=None, unspec_name="unspecified"):
        """
        :keyword unspec_type: BC type for the unspecified boundary faces.
            Set to :py:obj:`None` indicates the default to
            :py:class:`solvcon.boundcond.unspecified`.
        :type unspec_type: :py:class:`type`
        :keyword unspec_name: Name for the unspecified BC.
        :type unspec_name: :py:class:`str`
        :return: Nothing.
        """
        if unspec_type == None:
            unspec_type = boundcond.bctregy.unspecified
        # count all possible boundary faces.
        slct = self.fccls[:,1] < 0
        nbound = slct.sum()
        allfacn = np.arange(self.nface, dtype='int32')[slct]
        # collect type definition from bclist.
        specified = np.empty(nbound, dtype='bool')
        specified.fill(False)
        bndfcs = np.empty((nbound, 2), dtype='int32')
        ibfc = 0
        for bc in self.bclist:
            try:
                # save face indices and type number to array.
                bndfcs[ibfc:ibfc+len(bc),0] = bc.facn[:,0]
                bndfcs[ibfc:ibfc+len(bc),1] = bc.sern
                # save bndfc indices back to bc object.
                bc.facn[:,1] = np.arange(ibfc,ibfc+len(bc))
                # mark specified indices.
                slct = allfacn.searchsorted(bc.facn[:,0])
                specified[slct] = True
                # advance counter.
                ibfc += len(bc)
            except Exception as e:
                e.args = tuple([str(bc)] + list(e.args))
                raise
        # collect unspecified boundary faces.
        leftfcs = allfacn[specified == False]
        if leftfcs.shape[0] != 0:
            # add unspecified boundary faces to an additional BC object.
            bc = unspec_type()
            bc.name = unspec_name
            bc.sern = len(self.bclist)
            bc.blk = self
            bc.facn = np.empty((len(leftfcs),3), dtype='int32')
            bc.facn[:,0] = leftfcs[:]
            bc.facn[:,1] = np.arange(ibfc,ibfc+len(bc))
            self.bclist.append(bc)
            bndfcs[ibfc:,0] = bc.facn[:,0]
            bndfcs[ibfc:,1] = bc.sern
        # finish, save array to block object.
        self.bndfcs = bndfcs

    def build_ghost(self):
        """
        :return: Nothing.
        """
        # initialize data structure (arrays) for ghost information.
        ngstnode, ngstface, ngstcell = self._count_ghost()
        self._build_tables(self.ndim, self.nnode, self.nface, self.ncell,
                           ngstnode, ngstface, ngstcell)
        for name in self.TABLE_NAMES:
            table = getattr(self, 'tb'+name)
            # simply assign names of shared (full) and ghost arrays.
            setattr(self, 'sh'+name, table.F)
            setattr(self, 'gst'+name, table.G)
            # reassign both names and contents of body arrays.
            table.B = getattr(self, name)
            setattr(self, name, table.B)
        # build ghost information, including connectivities and metrics.
        self.create_msh().build_ghost(self.bndfcs)
        # to this point the object should be sane.
        self.check_sanity()

    def _count_ghost(self):
        """
        Count number of ghost entities.  Number of nodes, faces, and cells for
        ghost can be determined according to boundary connectivities.

        Nubmer of ghost cells is exactly the number of boundary faces.

        Number of exterior faces for ghost cells is the total number of faces of
        ghost cells minus the number of boundary faces (which is shared between
        ghost cells and adjacent interior cells).

        Number of exterior nodes for ghost cells is the total number of nodes of 
        ghost cells minus the total number of nodes of boundary faces.

        @return: node, face, and cell number for ghost.
        @rtype: tuple
        """
        bfcs = self.bndfcs[:,0]
        bcls = self.fccls[bfcs,0]    # interior cells next to boundary.
        # count/get the number of total ghost faces.
        ngstcell = self.nbound
        # count the number of total ghost faces.
        gstcltpn = self.cltpn[bcls]
        ngstface = 0
        for thetype in elemtype[1:]:
            dim = thetype[1]
            nfc = thetype[1+dim]
            ncl = (gstcltpn==thetype[0]).sum()
            ngstface += (nfc-1)*ncl # only one concrete face for each ghost cell.
        # count the number of total ghost nodes.
        ngstnode = self.clnds[bcls,0].sum() - self.fcnds[bfcs,0].sum()
        # return result.
        return ngstnode, ngstface, ngstcell

    def partition(self, npart):
        msh = self.create_msh()
        return msh.partition(npart)

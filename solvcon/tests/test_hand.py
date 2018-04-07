import unittest
import math

from .. import dependency
dependency.import_module_may_fail('..march')

# FIXME: the geometry in the tests below hasn't been verified by visualization.
# It's kept here just to anchor what I developed in code.

class TestTriangles(unittest.TestCase):

    def setUp(self):
        self.msh = march.UnstructuredBlock2D()
        self.msh = march.UnstructuredBlock2D(4, 6, 3, False)
        self.msh.ndcrd[0] = ( 0,  0)
        self.msh.ndcrd[1] = (-1, -1)
        self.msh.ndcrd[2] = ( 1, -1)
        self.msh.ndcrd[3] = ( 0,  1)
        self.msh.cltpn.fill(3)
        self.msh.clnds[0,:4] = (3, 0, 1, 2)
        self.msh.clnds[1,:4] = (3, 0, 2, 3)
        self.msh.clnds[2,:4] = (3, 0, 3, 1)
        self.msh.build_interior()
        self.msh.build_boundary()
        self.msh.build_ghost()

    def test_shape(self):
        self.assertEqual(4, self.msh.nnode)
        self.assertEqual(6, self.msh.nface)
        self.assertEqual(3, self.msh.ncell)
        self.assertEqual((4, 2), self.msh.ndcrd.shape)
        self.assertEqual((6, 2), self.msh.fccnd.shape)

    def test_node0_metrics(self):
        node = march.NodeHand2D(self.msh, 0)
        self.assertEqual(march.Vector2D(0,0), node.crd)

    def test_face0_metrics(self):
        face = march.FaceHand2D(self.msh, 0)
        self.assertEqual(march.Vector2D(-0.5,-0.5), face.cnd)
        self.assertEqual(march.Vector2D(-1./math.sqrt(2), 1./math.sqrt(2)),
                         face.nml)
        self.assertEqual(math.sqrt(2), face.ara)

    def test_face0_nds(self):
        face = march.FaceHand2D(self.msh, 0)
        self.assertEqual(2, face.nnd)
        self.assertEqual(march.NodeHand2D(self.msh, 0), face.nds(1))
        self.assertEqual(march.NodeHand2D(self.msh, 1), face.nds(2))

    def test_face0_clb_cln(self):
        face = march.FaceHand2D(self.msh, 0)
        self.assertNotEqual(face.clb, face.cln)
        self.assertEqual(face.clb, march.CellHand2D(self.msh, 0))
        self.assertEqual(face.cln, march.CellHand2D(self.msh, 2))
        self.assertNotEqual(march.CellHand2D(self.msh, 0),
                            march.CellHand2D(self.msh, 2))

    def test_cell0_metrics(self):
        cell = march.CellHand2D(self.msh, 0)
        self.assertTrue(march.Vector2D(0,-2./3).is_close_to(cell.cnd, 1.e-15))
        self.assertEqual(1, cell.vol)

    def test_cell0_nds(self):
        cell = march.CellHand2D(self.msh, 0)
        self.assertEqual(3, cell.nnd)
        self.assertEqual(march.NodeHand2D(self.msh, 0), cell.nds(1))
        self.assertEqual(march.NodeHand2D(self.msh, 1), cell.nds(2))
        self.assertEqual(march.NodeHand2D(self.msh, 2), cell.nds(3))

    def test_cell0_fcs(self):
        cell = march.CellHand2D(self.msh, 0)
        self.assertEqual(3, cell.nfc)
        self.assertEqual(march.FaceHand2D(self.msh, 0), cell.fcs(1))
        self.assertEqual(march.FaceHand2D(self.msh, 1), cell.fcs(2))
        self.assertEqual(march.FaceHand2D(self.msh, 2), cell.fcs(3))

    def test_cell0_cls(self):
        cell = march.CellHand2D(self.msh, 0)
        self.assertEqual(3, cell.nfc)
        self.assertEqual(march.CellHand2D(self.msh, 2), cell.cls(1))
        self.assertEqual(march.CellHand2D(self.msh, -1), cell.cls(2))
        self.assertEqual(march.CellHand2D(self.msh, 1), cell.cls(3))

    def test_node0_repr(self):
        node = march.NodeHand2D(self.msh, 0)
        golden = "NodeHand2D(index=0, crd=Vector2D(0.00000000000000000000e+00,0.00000000000000000000e+00))"
        self.assertEqual(golden, node.repr(indent=2, precision=20))

    def test_face0_repr(self):
        self.maxDiff = None
        face = march.FaceHand2D(self.msh, 0)
        golden = """FaceHand2D(
  index=0,
  type=1:line,
  belong_cell=0;3:triangle,
  neighbor_cell=2;3:triangle,
  cnd=Vector2D(-5.00000000000000000000e-01,-5.00000000000000000000e-01),
  nml=Vector2D(-7.07106781186547461715e-01,7.07106781186547461715e-01),
  ara=1.41421356237309514547e+00,
  nds=[
    NodeHand2D(index=0, crd=Vector2D(0.00000000000000000000e+00,0.00000000000000000000e+00)),
    NodeHand2D(index=1, crd=Vector2D(-1.00000000000000000000e+00,-1.00000000000000000000e+00))
  ]
)"""
        self.assertEqual(golden, face.repr(indent=2, precision=20))

    def test_cell0_repr(self):
        self.maxDiff = None
        cell = march.CellHand2D(self.msh, 0)
        golden = """CellHand2D(
  index=0,
  type=3:triangle,
  cnd=Vector2D(0.00000000000000000000e+00,-6.66666666666666740682e-01),
  vol=1.00000000000000000000e+00,
  nds=[
    NodeHand2D(index=0, crd=Vector2D(0.00000000000000000000e+00,0.00000000000000000000e+00)),
    NodeHand2D(index=1, crd=Vector2D(-1.00000000000000000000e+00,-1.00000000000000000000e+00)),
    NodeHand2D(index=2, crd=Vector2D(1.00000000000000000000e+00,-1.00000000000000000000e+00))
  ],
  fcs=[0:(0,1);2, 1:(1,2);-1, 2:(2,0);1]
)"""
        self.assertEqual(golden, cell.repr(indent=2, precision=20))

        golden_nd1 = "NodeHand2D(index=0, crd=Vector2D(0,0))"
        self.assertEqual(golden_nd1, cell.nds(1).repr(indent=2))

        golden_fc1 = """FaceHand2D(
  index=0,
  type=1:line,
  belong_cell=0;3:triangle,
  neighbor_cell=2;3:triangle,
  cnd=Vector2D(-0.5,-0.5),
  nml=Vector2D(-0.707107,0.707107),
  ara=1.41421,
  nds=[
    NodeHand2D(index=0, crd=Vector2D(0,0)),
    NodeHand2D(index=1, crd=Vector2D(-1,-1))
  ]
)"""
        self.assertEqual(golden_fc1, cell.fcs(1).repr(indent=2))

        golden_cl1 = """CellHand2D(
  index=2,
  type=3:triangle,
  cnd=Vector2D(-0.333333,0),
  vol=0.5,
  nds=[
    NodeHand2D(index=0, crd=Vector2D(0,0)),
    NodeHand2D(index=3, crd=Vector2D(0,1)),
    NodeHand2D(index=1, crd=Vector2D(-1,-1))
  ],
  fcs=[4:(3,0);1, 5:(3,1);-3, 0:(0,1);0]
)"""
        self.assertEqual(golden_cl1, cell.cls(1).repr(indent=2))

    def test_ce0_repr(self):
        ce = march.ConservationElement2D(self.msh, 0)
        golden = """ConservationElement2D(
  cnd=Vector2D(-1.66533453693773474901e-17,-6.88888888888888883955e-01),
  vol=1.66666666666666674068e+00,
  bces[0]=BasicCE2D(
    cnd=Vector2D(-3.70370370370370405322e-01,-4.81481481481481510265e-01),
    vol=5.00000000000000000000e-01,
    sfcnd[0]=Vector2D(-1.66666666666666685170e-01,0.00000000000000000000e+00),
    sfcnd[1]=Vector2D(-6.66666666666666740682e-01,-5.00000000000000000000e-01),
    sfcnd[2]=Vector2D(-inf,-inf),
    sfcnd[3]=Vector2D(-inf,-inf),
    sfnml[0]=Vector2D(0.00000000000000000000e+00,3.33333333333333370341e-01),
    sfnml[1]=Vector2D(-1.00000000000000000000e+00,6.66666666666666629659e-01),
    sfnml[2]=Vector2D(-inf,-inf),
    sfnml[3]=Vector2D(-inf,-inf)
  ),
  bces[1]=BasicCE2D(
    cnd=Vector2D(0.00000000000000000000e+00,-1.00000000000000000000e+00),
    vol=6.66666666666666740682e-01,
    sfcnd[0]=Vector2D(-5.00000000000000000000e-01,-1.16666666666666674068e+00),
    sfcnd[1]=Vector2D(5.00000000000000000000e-01,-1.16666666666666674068e+00),
    sfcnd[2]=Vector2D(-inf,-inf),
    sfcnd[3]=Vector2D(-inf,-inf),
    sfnml[0]=Vector2D(-3.33333333333333481363e-01,-1.00000000000000000000e+00),
    sfnml[1]=Vector2D(3.33333333333333481363e-01,-1.00000000000000000000e+00),
    sfnml[2]=Vector2D(-inf,-inf),
    sfnml[3]=Vector2D(-inf,-inf)
  ),
  bces[2]=BasicCE2D(
    cnd=Vector2D(3.70370370370370349811e-01,-4.81481481481481510265e-01),
    vol=5.00000000000000000000e-01,
    sfcnd[0]=Vector2D(6.66666666666666629659e-01,-5.00000000000000000000e-01),
    sfcnd[1]=Vector2D(1.66666666666666657415e-01,0.00000000000000000000e+00),
    sfcnd[2]=Vector2D(-inf,-inf),
    sfcnd[3]=Vector2D(-inf,-inf),
    sfnml[0]=Vector2D(1.00000000000000000000e+00,6.66666666666666740682e-01),
    sfnml[1]=Vector2D(-0.00000000000000000000e+00,3.33333333333333314830e-01),
    sfnml[2]=Vector2D(-inf,-inf),
    sfnml[3]=Vector2D(-inf,-inf)
  ),
  bces[3]=BasicCE2D(
    cnd=Vector2D(-inf,-inf),
    vol=-inf,
    sfcnd[0]=Vector2D(-inf,-inf),
    sfcnd[1]=Vector2D(-inf,-inf),
    sfcnd[2]=Vector2D(-inf,-inf),
    sfcnd[3]=Vector2D(-inf,-inf),
    sfnml[0]=Vector2D(-inf,-inf),
    sfnml[1]=Vector2D(-inf,-inf),
    sfnml[2]=Vector2D(-inf,-inf),
    sfnml[3]=Vector2D(-inf,-inf)
  ),
  bces[4]=BasicCE2D(
    cnd=Vector2D(-inf,-inf),
    vol=-inf,
    sfcnd[0]=Vector2D(-inf,-inf),
    sfcnd[1]=Vector2D(-inf,-inf),
    sfcnd[2]=Vector2D(-inf,-inf),
    sfcnd[3]=Vector2D(-inf,-inf),
    sfnml[0]=Vector2D(-inf,-inf),
    sfnml[1]=Vector2D(-inf,-inf),
    sfnml[2]=Vector2D(-inf,-inf),
    sfnml[3]=Vector2D(-inf,-inf)
  ),
  bces[5]=BasicCE2D(
    cnd=Vector2D(-inf,-inf),
    vol=-inf,
    sfcnd[0]=Vector2D(-inf,-inf),
    sfcnd[1]=Vector2D(-inf,-inf),
    sfcnd[2]=Vector2D(-inf,-inf),
    sfcnd[3]=Vector2D(-inf,-inf),
    sfnml[0]=Vector2D(-inf,-inf),
    sfnml[1]=Vector2D(-inf,-inf),
    sfnml[2]=Vector2D(-inf,-inf),
    sfnml[3]=Vector2D(-inf,-inf)
  )
)"""
        self.assertEqual(golden, ce.repr(indent=2, precision=20))

class TestTetrahedra(unittest.TestCase):

    def setUp(self):
        self.msh = march.UnstructuredBlock3D(5, 10, 4, False)
        self.msh.ndcrd[0] = ( 0,  0,  0)
        self.msh.ndcrd[1] = (10,  0,  0)
        self.msh.ndcrd[2] = ( 0, 10,  0)
        self.msh.ndcrd[3] = ( 0,  0, 10)
        self.msh.ndcrd[4] = ( 1,  1,  1)
        self.msh.cltpn.fill(5)
        self.msh.clnds[0,:5] = (4, 0, 1, 2, 4)
        self.msh.clnds[1,:5] = (4, 0, 2, 3, 4)
        self.msh.clnds[2,:5] = (4, 0, 3, 1, 4)
        self.msh.clnds[3,:5] = (4, 1, 2, 3, 4)
        self.msh.build_interior()
        self.msh.build_boundary()
        self.msh.build_ghost()

    def test_shape(self):
        self.assertEqual(5, self.msh.nnode)
        self.assertEqual(10, self.msh.nface)
        self.assertEqual(4, self.msh.ncell)
        self.assertEqual((5, 3), self.msh.ndcrd.shape)
        self.assertEqual((10, 3), self.msh.fccnd.shape)

    def test_node0_metrics(self):
        node = march.NodeHand3D(self.msh, 0)
        self.assertEqual(march.Vector3D(0,0,0), node.crd)

    def test_face0_metrics(self):
        face = march.FaceHand3D(self.msh, 0)
        self.assertTrue(
            march.Vector3D(10./3,10./3,0).is_close_to(face.cnd, 1.e-15))
        self.assertEqual(march.Vector3D(0,0,-1), face.nml)
        self.assertEqual(50, face.ara)

    def test_face0_nds(self):
        face = march.FaceHand3D(self.msh, 0)
        self.assertEqual(3, face.nnd)
        self.assertEqual(march.NodeHand3D(self.msh, 0), face.nds(1))
        self.assertEqual(march.NodeHand3D(self.msh, 2), face.nds(2))
        self.assertEqual(march.NodeHand3D(self.msh, 1), face.nds(3))

    def test_face0_clb_cln(self):
        face = march.FaceHand3D(self.msh, 0)
        self.assertNotEqual(face.clb, face.cln)
        self.assertEqual(face.clb, march.CellHand3D(self.msh, 0))
        self.assertEqual(face.cln, march.CellHand3D(self.msh, -1))
        self.assertNotEqual(march.CellHand3D(self.msh, -1),
                            march.CellHand3D(self.msh, 0))

    def test_cell0_metrics(self):
        cell = march.CellHand3D(self.msh, 0)
        self.assertTrue(
            march.Vector3D(2.75,2.75,0.25).is_close_to(cell.cnd, 1.e-15))
        self.assertEqual(1.66666666666666678509e+01, cell.vol)

    def test_cell0_nds(self):
        cell = march.CellHand3D(self.msh, 0)
        self.assertEqual(4, cell.nnd)
        self.assertEqual(march.NodeHand3D(self.msh, 0), cell.nds(1))
        self.assertEqual(march.NodeHand3D(self.msh, 1), cell.nds(2))
        self.assertEqual(march.NodeHand3D(self.msh, 2), cell.nds(3))
        self.assertEqual(march.NodeHand3D(self.msh, 4), cell.nds(4))

    def test_cell0_fcs(self):
        cell = march.CellHand3D(self.msh, 0)
        self.assertEqual(4, cell.nfc)
        self.assertEqual(march.FaceHand3D(self.msh, 0), cell.fcs(1))
        self.assertEqual(march.FaceHand3D(self.msh, 1), cell.fcs(2))
        self.assertEqual(march.FaceHand3D(self.msh, 2), cell.fcs(3))
        self.assertEqual(march.FaceHand3D(self.msh, 3), cell.fcs(4))

    def test_cell0_cls(self):
        cell = march.CellHand3D(self.msh, 0)
        self.assertEqual(4, cell.nfc)
        self.assertEqual(march.CellHand3D(self.msh, -1), cell.cls(1))
        self.assertEqual(march.CellHand3D(self.msh, 2), cell.cls(2))
        self.assertEqual(march.CellHand3D(self.msh, 1), cell.cls(3))
        self.assertEqual(march.CellHand3D(self.msh, 3), cell.cls(4))

    def test_node0_repr(self):
        node = march.NodeHand3D(self.msh, 0)
        golden = "NodeHand3D(index=0, crd=Vector3D(0.00000000000000000000e+00,0.00000000000000000000e+00,0.00000000000000000000e+00))"
        self.assertEqual(golden, node.repr(indent=2, precision=20))

    def test_face0_repr(self):
        self.maxDiff = None
        face = march.FaceHand3D(self.msh, 0)
        golden = """FaceHand3D(
  index=0,
  type=3:triangle,
  belong_cell=0;5:tetrahedron,
  neighbor_cell=-1;5:tetrahedron,
  cnd=Vector3D(3.33333333333333259318e+00,3.33333333333333259318e+00,0.00000000000000000000e+00),
  nml=Vector3D(0.00000000000000000000e+00,0.00000000000000000000e+00,-1.00000000000000000000e+00),
  ara=5.00000000000000000000e+01,
  nds=[
    NodeHand3D(index=0, crd=Vector3D(0.00000000000000000000e+00,0.00000000000000000000e+00,0.00000000000000000000e+00)),
    NodeHand3D(index=2, crd=Vector3D(0.00000000000000000000e+00,1.00000000000000000000e+01,0.00000000000000000000e+00)),
    NodeHand3D(index=1, crd=Vector3D(1.00000000000000000000e+01,0.00000000000000000000e+00,0.00000000000000000000e+00))
  ]
)"""
        self.assertEqual(golden, face.repr(indent=2, precision=20))

    def test_cell0_repr(self):
        self.maxDiff = None
        cell = march.CellHand3D(self.msh, 0)
        golden = """CellHand3D(
  index=0,
  type=5:tetrahedron,
  cnd=Vector3D(2.74999999999999911182e+00,2.75000000000000000000e+00,2.50000000000000000000e-01),
  vol=1.66666666666666678509e+01,
  nds=[
    NodeHand3D(index=0, crd=Vector3D(0.00000000000000000000e+00,0.00000000000000000000e+00,0.00000000000000000000e+00)),
    NodeHand3D(index=1, crd=Vector3D(1.00000000000000000000e+01,0.00000000000000000000e+00,0.00000000000000000000e+00)),
    NodeHand3D(index=2, crd=Vector3D(0.00000000000000000000e+00,1.00000000000000000000e+01,0.00000000000000000000e+00)),
    NodeHand3D(index=4, crd=Vector3D(1.00000000000000000000e+00,1.00000000000000000000e+00,1.00000000000000000000e+00))
  ],
  fcs=[0:(0,2,1);-1, 1:(0,1,4);2, 2:(0,4,2);1, 3:(1,2,4);3]
)"""
        self.assertEqual(golden, cell.repr(indent=2, precision=20))

        golden_nd1 = "NodeHand3D(index=0, crd=Vector3D(0,0,0))"
        self.assertEqual(golden_nd1, cell.nds(1).repr(indent=2))

        golden_fc1 = """FaceHand3D(
  index=0,
  type=3:triangle,
  belong_cell=0;5:tetrahedron,
  neighbor_cell=-1;5:tetrahedron,
  cnd=Vector3D(3.33333,3.33333,0),
  nml=Vector3D(0,0,-1),
  ara=50,
  nds=[
    NodeHand3D(index=0, crd=Vector3D(0,0,0)),
    NodeHand3D(index=2, crd=Vector3D(0,10,0)),
    NodeHand3D(index=1, crd=Vector3D(10,0,0))
  ]
)"""
        self.assertEqual(golden_fc1, cell.fcs(1).repr(indent=2))

        golden_cl1 = """CellHand3D(
  index=-1,
  type=5:tetrahedron,
  cnd=Vector3D(2.75,2.75,-0.25),
  vol=16.6667,
  nds=[
    NodeHand3D(index=0, crd=Vector3D(0,0,0)),
    NodeHand3D(index=1, crd=Vector3D(10,0,0)),
    NodeHand3D(index=2, crd=Vector3D(0,10,0)),
    NodeHand3D(index=-1, crd=Vector3D(1,1,-1))
  ],
  fcs=[0:(0,2,1);0, -1:(0,1,-1);-1, -2:(0,-1,2);-1, -3:(1,2,-1);-1]
)"""
        self.assertEqual(golden_cl1, cell.cls(1).repr(indent=2))

    def test_ce0_repr(self):
        ce = march.ConservationElement3D(self.msh, 0)
        golden = """ConservationElement3D(
  cnd=Vector3D(2.99999999999999955591e+00,2.99999999999999955591e+00,6.69642857142856984254e-01),
  vol=5.83333333333333428072e+01,
  bces[0]=BasicCE3D(
    cnd=Vector3D(3.18749999999999955591e+00,3.18749999999999955591e+00,0.00000000000000000000e+00),
    vol=8.33333333333333392545e+00,
    sfcnd[0]=Vector3D(9.16666666666666407615e-01,4.25000000000000000000e+00,-8.33333333333333287074e-02),
    sfcnd[1]=Vector3D(4.25000000000000000000e+00,4.25000000000000000000e+00,-8.33333333333333287074e-02),
    sfcnd[2]=Vector3D(4.25000000000000000000e+00,9.16666666666666629659e-01,-8.33333333333333287074e-02),
    sfcnd[3]=Vector3D(-inf,-inf,-inf),
    sfnml[0]=Vector3D(-1.25000000000000000000e+00,0.00000000000000000000e+00,-1.37499999999999946709e+01),
    sfnml[1]=Vector3D(1.25000000000000000000e+00,1.25000000000000000000e+00,-2.25000000000000035527e+01),
    sfnml[2]=Vector3D(0.00000000000000000000e+00,-1.25000000000000000000e+00,-1.37500000000000000000e+01),
    sfnml[3]=Vector3D(-inf,-inf,-inf)
  ),
  bces[1]=BasicCE3D(
    cnd=Vector3D(3.43749999999999911182e+00,6.25000000000000000000e-01,6.25000000000000000000e-01),
    vol=8.33333333333333392545e+00,
    sfcnd[0]=Vector3D(4.24999999999999911182e+00,8.33333333333333287074e-02,9.16666666666666629659e-01),
    sfcnd[1]=Vector3D(4.58333333333333303727e+00,4.16666666666666685170e-01,1.25000000000000000000e+00),
    sfcnd[2]=Vector3D(1.24999999999999955591e+00,4.16666666666666685170e-01,1.25000000000000000000e+00),
    sfcnd[3]=Vector3D(-inf,-inf,-inf),
    sfnml[0]=Vector3D(0.00000000000000000000e+00,-1.37500000000000000000e+01,1.25000000000000000000e+00),
    sfnml[1]=Vector3D(1.25000000000000000000e+00,8.75000000000000000000e+00,2.50000000000000088818e+00),
    sfnml[2]=Vector3D(-1.25000000000000000000e+00,4.44089209850062616169e-16,1.24999999999999933387e+00),
    sfnml[3]=Vector3D(-inf,-inf,-inf)
  ),
  bces[2]=BasicCE3D(
    cnd=Vector3D(6.24999999999999888978e-01,3.43750000000000000000e+00,6.25000000000000111022e-01),
    vol=8.33333333333333214910e+00,
    sfcnd[0]=Vector3D(4.16666666666666685170e-01,1.25000000000000000000e+00,1.25000000000000000000e+00),
    sfcnd[1]=Vector3D(4.16666666666666685170e-01,4.58333333333333303727e+00,1.25000000000000000000e+00),
    sfcnd[2]=Vector3D(8.33333333333333287074e-02,4.25000000000000000000e+00,9.16666666666666629659e-01),
    sfcnd[3]=Vector3D(-inf,-inf,-inf),
    sfnml[0]=Vector3D(0.00000000000000000000e+00,-1.25000000000000000000e+00,1.25000000000000000000e+00),
    sfnml[1]=Vector3D(8.75000000000000000000e+00,1.25000000000000000000e+00,2.50000000000000000000e+00),
    sfnml[2]=Vector3D(-1.37500000000000000000e+01,0.00000000000000000000e+00,1.24999999999999977796e+00),
    sfnml[3]=Vector3D(-inf,-inf,-inf)
  ),
  bces[3]=BasicCE3D(
    cnd=Vector3D(3.43749999999999911182e+00,3.43749999999999911182e+00,8.59374999999999777955e-01),
    vol=3.33333333333333428072e+01,
    sfcnd[0]=Vector3D(4.25000000000000000000e+00,4.25000000000000000000e+00,9.16666666666666629659e-01),
    sfcnd[1]=Vector3D(1.25000000000000000000e+00,4.58333333333333303727e+00,1.25000000000000000000e+00),
    sfcnd[2]=Vector3D(4.58333333333333303727e+00,1.25000000000000000000e+00,1.25000000000000000000e+00),
    sfcnd[3]=Vector3D(-inf,-inf,-inf),
    sfnml[0]=Vector3D(1.37500000000000000000e+01,1.37500000000000000000e+01,2.25000000000000000000e+01),
    sfnml[1]=Vector3D(-8.75000000000000000000e+00,0.00000000000000000000e+00,8.75000000000000000000e+00),
    sfnml[2]=Vector3D(0.00000000000000000000e+00,-8.75000000000000000000e+00,8.75000000000000000000e+00),
    sfnml[3]=Vector3D(-inf,-inf,-inf)
  ),
  bces[4]=BasicCE3D(
    cnd=Vector3D(-inf,-inf,-inf),
    vol=-inf,
    sfcnd[0]=Vector3D(-inf,-inf,-inf),
    sfcnd[1]=Vector3D(-inf,-inf,-inf),
    sfcnd[2]=Vector3D(-inf,-inf,-inf),
    sfcnd[3]=Vector3D(-inf,-inf,-inf),
    sfnml[0]=Vector3D(-inf,-inf,-inf),
    sfnml[1]=Vector3D(-inf,-inf,-inf),
    sfnml[2]=Vector3D(-inf,-inf,-inf),
    sfnml[3]=Vector3D(-inf,-inf,-inf)
  ),
  bces[5]=BasicCE3D(
    cnd=Vector3D(-inf,-inf,-inf),
    vol=-inf,
    sfcnd[0]=Vector3D(-inf,-inf,-inf),
    sfcnd[1]=Vector3D(-inf,-inf,-inf),
    sfcnd[2]=Vector3D(-inf,-inf,-inf),
    sfcnd[3]=Vector3D(-inf,-inf,-inf),
    sfnml[0]=Vector3D(-inf,-inf,-inf),
    sfnml[1]=Vector3D(-inf,-inf,-inf),
    sfnml[2]=Vector3D(-inf,-inf,-inf),
    sfnml[3]=Vector3D(-inf,-inf,-inf)
  )
)"""
        self.assertEqual(golden, ce.repr(indent=2, precision=20))


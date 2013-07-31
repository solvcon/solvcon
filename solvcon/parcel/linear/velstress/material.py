# -*- coding: UTF-8 -*-
#
# Copyright (c) 2013, Yung-Yu Chen <yyc@solvcon.net>
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
# - Neither the name of the SOLVCON nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
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

"""
Material definition.
"""


import numpy as np

from solvcon import gendata


#: Registry singleton.
mltregy = gendata.TypeNameRegistry()


class MaterialMeta(type):
    """
    Meta class for material class.
    """
    def __new__(cls, name, bases, namespace):
        newcls = super(MaterialMeta, cls).__new__(cls, name, bases, namespace)
        # register.
        mltregy.register(newcls)
        return newcls


class Material(object):
    """Material properties.  The constitutive relation needs not be symmetric.
    """

    __metaclass__ = MaterialMeta

    #: :py:class:`list` of :py:class:`tuple` for indices where the content
    #: should be zero.
    _zeropoints_ = []

    K = np.array([ [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ], [
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ], [
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], ], dtype='float64')

    def __init__(self, rho=None, al=None, be=None, ga=None, **kw):
        assert None is not rho
        assert None is not al
        assert None is not be
        assert None is not ga
        #: Density.
        self.rho = rho
        #: Alpha angle.
        self.al = al
        #: Beta angle.
        self.be = be
        #: Gamma angle.
        self.ga = ga
        # set stiffness matrix.
        origstiff = np.empty((6,6), dtype='float64')
        origstiff.fill(0.0)
        for key in kw.keys():   # becaues I pop out the key.
            if len(key) == 4 and key[:2] == 'co':
                try:
                    i = int(key[2])-1
                    j = int(key[3])-1
                except:
                    continue
                assert i < origstiff.shape[0]
                assert j < origstiff.shape[1]
                val = kw.pop(key)
                origstiff[i,j] = val
        #: Stiffness matrix in the crystal coordinate.
        self.origstiff = origstiff
        # check for zeros.
        self._check_origstiffzero(self.origstiff)
        # compute the stiffness matrix in the transformed global coordinate
        # system.
        bondmat = self.get_bondmat()
        #: Stiffness matrix in the transformed global coordinate.
        self.stiff = np.dot(bondmat, np.dot(self.origstiff, bondmat.T))
        super(Material, self).__init__(**kw)

    def __getattr__(self, key):
        if len(key) == 4 and key[:2] == 'co':
            i = int(key[2])
            j = int(key[3])
            if 1 <= i <= 6 and 1 <= j <= 6:
                return self.origstiff[i-1,j-1]
        elif len(key) == 3 and key[0] == 'c':
            i = int(key[1])
            j = int(key[2])
            if 1 <= i <= 6 and 1 <= j <= 6:
                return self.stiff[i-1,j-1]
        else:
            raise AttributeError

    def __str__(self):
        return '[%s: al=%.2f be=%.2f ga=%.2f (deg)]' % (self.__class__.__name__,
            self.al/(np.pi/180), self.be/(np.pi/180), self.ga/(np.pi/180))

    @classmethod
    def _check_origstiffzero(cls, origstiff):
        """
        Check for zero in original stiffness matrix.

        @note: no assumed symmetry.
        """
        for i, j in cls._zeropoints_:
            assert origstiff[i,j] == 0.0

    def get_rotmat(self):
        """
        Coordinate transformation matrix for three successive rotations through
        the Euler angles.

        @return: the transformation matrix.
        @rtype: numpy.ndarray
        """
        al = self.al; be = self.be; ga = self.ga
        almat = np.array([
            [np.cos(al), np.sin(al), 0],
            [-np.sin(al), np.cos(al), 0],
            [0, 0, 1],
        ], dtype='float64')
        bemat = np.array([
            [1, 0, 0],
            [0, np.cos(be), np.sin(be)],
            [0, -np.sin(be), np.cos(be)],
        ], dtype='float64')
        gamat = np.array([
            [np.cos(ga), np.sin(ga), 0],
            [-np.sin(ga), np.cos(ga), 0],
            [0, 0, 1],
        ], dtype='float64')
        return np.dot(gamat, np.dot(bemat, almat))

    def get_bondmat(self):
        """
        The Bond's matrix M as a shorthand of coordinate transformation for the 
        6-component stress vector.

        @return: the Bond's matrix.
        @rtype: numpy.ndarray
        """
        rotmat = self.get_rotmat()
        bond = np.empty((6,6), dtype='float64')
        # upper left.
        bond[:3,:3] = rotmat[:,:]**2
        # upper right.
        bond[0,3] = 2*rotmat[0,1]*rotmat[0,2]
        bond[0,4] = 2*rotmat[0,2]*rotmat[0,0]
        bond[0,5] = 2*rotmat[0,0]*rotmat[0,1]
        bond[1,3] = 2*rotmat[1,1]*rotmat[1,2]
        bond[1,4] = 2*rotmat[1,2]*rotmat[1,0]
        bond[1,5] = 2*rotmat[1,0]*rotmat[1,1]
        bond[2,3] = 2*rotmat[2,1]*rotmat[2,2]
        bond[2,4] = 2*rotmat[2,2]*rotmat[2,0]
        bond[2,5] = 2*rotmat[2,0]*rotmat[2,1]
        # lower left.
        bond[3,0] = rotmat[1,0]*rotmat[2,0]
        bond[3,1] = rotmat[1,1]*rotmat[2,1]
        bond[3,2] = rotmat[1,2]*rotmat[2,2]
        bond[4,0] = rotmat[2,0]*rotmat[0,0]
        bond[4,1] = rotmat[2,1]*rotmat[0,1]
        bond[4,2] = rotmat[2,2]*rotmat[0,2]
        bond[5,0] = rotmat[0,0]*rotmat[1,0]
        bond[5,1] = rotmat[0,1]*rotmat[1,1]
        bond[5,2] = rotmat[0,2]*rotmat[1,2]
        # lower right.
        bond[3,3] = rotmat[1,1]*rotmat[2,2] + rotmat[1,2]*rotmat[2,1]
        bond[3,4] = rotmat[1,0]*rotmat[2,2] + rotmat[1,2]*rotmat[2,0]
        bond[3,5] = rotmat[1,1]*rotmat[2,0] + rotmat[1,0]*rotmat[2,1]
        bond[4,3] = rotmat[0,1]*rotmat[2,2] + rotmat[0,2]*rotmat[2,1]
        bond[4,4] = rotmat[0,0]*rotmat[2,2] + rotmat[0,2]*rotmat[2,0]
        bond[4,5] = rotmat[0,1]*rotmat[2,0] + rotmat[0,0]*rotmat[2,1]
        bond[5,3] = rotmat[0,1]*rotmat[1,2] + rotmat[0,2]*rotmat[1,1]
        bond[5,4] = rotmat[0,0]*rotmat[1,2] + rotmat[0,2]*rotmat[1,0]
        bond[5,5] = rotmat[0,1]*rotmat[1,0] + rotmat[0,0]*rotmat[1,1]
        return bond

    def get_jacos(self):
        """
        Obtain the Jacobian matrices for the solid.

        @param K: the K matrix.
        @type K: numpy.ndarray
        @return: the Jacobian matrices
        @rtype: numpy.ndarray
        """
        rho = self.rho
        sf = self.stiff
        jacos = np.zeros((3,9,9), dtype='float64')
        for idm in range(3):
            K = self.K[idm]
            jaco = jacos[idm]
            jaco[:3,3:] = K/(-rho)  # the upper right submatrix.
            jaco[3:,:3] = -np.dot(sf, K.T) # the lower left submatrix.
        return jacos


################################################################################
# Begin material symmetry group.
class Triclinic(Material):
    """
    The stiffness matrix has to be symmetric.
    """
    _zeropoints_ = []
    def __init__(self, *args, **kw):
        for key in kw.keys():   # becaues I modify the key.
            if len(key) == 4 and key[:2] == 'co':
                try:
                    i = int(key[2])
                    j = int(key[3])
                except:
                    continue
                symkey = 'co%d%d' % (j, i)
                if i != j:
                    assert symkey not in kw
                kw[symkey] = kw[key]
        super(Triclinic, self).__init__(*args, **kw)
    @classmethod
    def _check_origstiffzero(cls, origstiff):
        for i, j in cls._zeropoints_:
            assert origstiff[i,j] == origstiff[j,i] == 0.0

class Monoclinic(Triclinic):
    _zeropoints_ = [
        (0,3), (0,5),
        (1,3), (1,5),
        (2,3), (2,5),
        (3,4), (4,5),
    ]

class Orthorhombic(Triclinic):
    _zeropoints_ = [
        (0,3), (0,4), (0,5),
        (1,3), (1,4), (1,5),
        (2,3), (2,4), (2,5),
        (3,4), (3,5), (4,5),
    ]

class Tetragonal(Triclinic):
    _zeropoints_ = [
        (0,3), (0,4),
        (1,3), (1,4),
        (2,3), (2,4), (2,5),
        (3,4), (3,5), (4,5),
    ]
    def __init__(self, *args, **kw):
        kw['co22'] = kw['co11']
        kw['co23'] = kw['co13']
        kw['co26'] = -kw.get('co16', 0.0)
        kw['co55'] = kw['co44']
        super(Tetragonal, self).__init__(*args, **kw)

class Trigonal(Triclinic):
    _zeropoints_ = [
        (0,5), (1,5),
        (2,3), (2,4), (2,5),
        (3,4),
    ]
    def __init__(self, *args, **kw):
        kw['co15'] = -kw.get('co25', 0.0)
        kw['co22'] = kw['co11']
        kw['co23'] = kw['co13']
        kw['co24'] = -kw.get('co14', 0.0)
        kw['co46'] = kw.get('co25', 0.0)
        kw['co55'] = kw['co44']
        kw['co56'] = kw.get('co14', 0.0)
        kw['co66'] = (kw['co11'] - kw['co12'])/2
        super(Trigonal, self).__init__(*args, **kw)

class Hexagonal(Trigonal):
    _zeropoints_ = [
        (0,3), (0,4), (0,5),
        (1,3), (1,4), (1,5),
        (2,3), (2,4), (2,5),
        (3,4), (3,5), (4,5),
    ]

class Cubic(Triclinic):
    _zeropoints_ = [
        (0,3), (0,4), (0,5),
        (1,3), (1,4), (1,5),
        (2,3), (2,4), (2,5),
        (3,4), (3,5), (4,5),
    ]
    def __init__(self, *args, **kw):
        kw['co13'] = kw['co12']
        kw['co22'] = kw['co11']
        kw['co23'] = kw['co12']
        kw['co33'] = kw['co11']
        kw['co55'] = kw['co44']
        kw['co66'] = kw['co44']
        super(Cubic, self).__init__(*args, **kw)

class Isotropic(Triclinic):
    _zeropoints_ = [
        (0,3), (0,4), (0,5),
        (1,3), (1,4), (1,5),
        (2,3), (2,4), (2,5),
        (3,4), (3,5), (4,5),
    ]
    def __init__(self, *args, **kw):
        kw['co12'] = kw['co11']-2*kw['co44']
        kw['co13'] = kw['co11']-2*kw['co44']
        kw['co22'] = kw['co11']
        kw['co23'] = kw['co11']-2*kw['co44']
        kw['co33'] = kw['co11']
        kw['co55'] = kw['co44']
        kw['co66'] = kw['co44']
        super(Isotropic, self).__init__(*args, **kw)
# End material symmetry group.
################################################################################


################################################################################
# Begin real material properties.
class GaAs(Cubic):
    def __init__(self, *args, **kw):
        kw.setdefault('rho', 5307.0)
        kw.setdefault('co11', 11.88e10)
        kw.setdefault('co12', 5.38e10)
        kw.setdefault('co44', 5.94e10)
        super(GaAs, self).__init__(*args, **kw)

class ZnO(Hexagonal):
    def __init__(self, *args, **kw):
        kw.setdefault('rho', 5680.0)
        kw.setdefault('co11', 20.97e10)
        kw.setdefault('co12', 12.11e10)
        kw.setdefault('co13', 10.51e10)
        kw.setdefault('co33', 21.09e10)
        kw.setdefault('co44', 4.247e10)
        super(ZnO, self).__init__(*args, **kw)

class CdS(Hexagonal):
    def __init__(self, *args, **kw):
        kw.setdefault('rho', 4820.0)
        kw.setdefault('co11', 9.07e10)
        kw.setdefault('co12', 5.81e10)
        kw.setdefault('co13', 5.1e10)
        kw.setdefault('co33', 9.38e10)
        kw.setdefault('co44', 1.504e10)
        super(CdS, self).__init__(*args, **kw)

class Zinc(Hexagonal):
    def __init__(self, *args, **kw):
        kw.setdefault('rho', 7.1*1.e-3/(1.e-2**3))
        kw.setdefault('co11', 14.3e11*1.e-5/(1.e-2**2))
        kw.setdefault('co12', 1.7e11*1.e-5/(1.e-2**2))
        kw.setdefault('co13', 3.3e11*1.e-5/(1.e-2**2))
        kw.setdefault('co33', 5.0e11*1.e-5/(1.e-2**2))
        kw.setdefault('co44', 4.0e11*1.e-5/(1.e-2**2))
        super(Zinc, self).__init__(*args, **kw)

class Beryl(Hexagonal):
    def __init__(self, *args, **kw):
        kw.setdefault('rho', 2.7*1.e-3/(1.e-2**3))
        kw.setdefault('co11', 26.94e11*1.e-5/(1.e-2**2))
        kw.setdefault('co12', 9.61e11*1.e-5/(1.e-2**2))
        kw.setdefault('co13', 6.61e11*1.e-5/(1.e-2**2))
        kw.setdefault('co33', 23.63e11*1.e-5/(1.e-2**2))
        kw.setdefault('co44', 6.53e11*1.e-5/(1.e-2**2))
        super(Beryl, self).__init__(*args, **kw)

class Albite(Triclinic):
    def __init__(self, *args, **kw):
        #kw.setdefault('rho', )
        kw.setdefault('co11', 69.9e9)
        kw.setdefault('co22', 183.5e9)
        kw.setdefault('co33', 179.5e9)
        kw.setdefault('co44', 24.9e9)
        kw.setdefault('co55', 26.8e9)
        kw.setdefault('co66', 33.5e9)
        kw.setdefault('co12', 34.0e9)
        kw.setdefault('co13', 30.8e9)
        kw.setdefault('co14', 5.1e9)
        kw.setdefault('co15', -2.4e9)
        kw.setdefault('co16', -0.9e9)
        kw.setdefault('co23', 5.5e9)
        kw.setdefault('co24', -3.9e9)
        kw.setdefault('co25', -7.7e9)
        kw.setdefault('co26', -5.8e9)
        kw.setdefault('co34', -8.7e9)
        kw.setdefault('co35', 7.1e9)
        kw.setdefault('co36', -9.8e9)
        kw.setdefault('co45', -2.4e9)
        kw.setdefault('co46', -7.2e9)
        kw.setdefault('co56', 0.5e9)
        super(Albite, self).__init__(*args, **kw)

class Acmite(Monoclinic):
    def __init__(self, *args, **kw):
        kw.setdefault('rho', 3.5e3)
        kw.setdefault('co11', 185.8e9)
        kw.setdefault('co22', 181.3e9)
        kw.setdefault('co33', 234.4e9)
        kw.setdefault('co44', 62.9e9)
        kw.setdefault('co55', 51.0e9)
        kw.setdefault('co66', 47.4e9)
        kw.setdefault('co12', 68.5e9)
        kw.setdefault('co13', 70.7e9)
        kw.setdefault('co15', 9.8e9)
        kw.setdefault('co23', 62.9e9)
        kw.setdefault('co25', 9.4e9)
        kw.setdefault('co35', 21.4e9)
        kw.setdefault('co46', 7.7e9)
        super(Acmite, self).__init__(*args, **kw)

class AlphaUranium(Orthorhombic):
    def __init__(self, *args, **kw):
        #kw.setdefault('rho', )
        kw.setdefault('rho', 8.2e3) # a false value.
        kw.setdefault('co11', 215.e9)
        kw.setdefault('co22', 199.e9)
        kw.setdefault('co33', 267.e9)
        kw.setdefault('co44', 124.e9)
        kw.setdefault('co55', 73.e9)
        kw.setdefault('co66', 74.e9)
        kw.setdefault('co12', 46.e9)
        kw.setdefault('co13', 22.e9)
        kw.setdefault('co23', 107.e9)
        super(AlphaUranium, self).__init__(*args, **kw)

class BariumTitanate(Tetragonal):
    def __init__(self, *args, **kw):
        kw.setdefault('rho', 6.2e3)
        kw.setdefault('co11', 275.0e9)
        kw.setdefault('co33', 165.0e9)
        kw.setdefault('co44', 54.3e9)
        kw.setdefault('co66', 113.0e9)
        kw.setdefault('co12', 179.0e9)
        kw.setdefault('co13', 151.0e9)
        super(BariumTitanate, self).__init__(*args, **kw)

class AlphaQuartz(Trigonal):
    def __init__(self, *args, **kw):
        kw.setdefault('rho', 2.651e3)
        kw.setdefault('co11', 87.6e9)
        kw.setdefault('co33', 106.8e9)
        kw.setdefault('co44', 57.2e9)
        kw.setdefault('co12', 6.1e9)
        kw.setdefault('co13', 13.3e9)
        kw.setdefault('co14', 17.3e9)
        super(AlphaQuartz, self).__init__(*args, **kw)

class RickerSample(Isotropic):
    def __init__(self, *args, **kw):
        kw.setdefault('rho', 2200.e0)
        kw.setdefault('co11', 3200.e0**2*2200.e0)
        kw.setdefault('co44', 1847.5e0**2*2200.e0)
        super(RickerSample, self).__init__(*args, **kw)

class RickerSampleLight(Isotropic):
    def __init__(self, *args, **kw):
        scale = 1.e-3
        kw.setdefault('rho', 2200.e0*scale)
        kw.setdefault('co11', 3200.e0**2*2200.e0*scale)
        kw.setdefault('co44', 1847.5e0**2*2200.e0*scale)
        super(RickerSampleLight, self).__init__(*args, **kw)
# End real material properties.
################################################################################

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:

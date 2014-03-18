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


__all__ = ['mltregy', 'Material']


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
    """Material properties.
    """

    __metaclass__ = MaterialMeta

    def __init__(self, **kw):
        self.rho = kw['rho']
        self.vp  = kw['vp']
        self.Gep = kw['Gep']
        self.Gem = kw['Gem']
        self.Glp = kw['Glp']
        self.Glm = kw['Glm']
        self.tau = kw['tau']

    def get_jacos(self, ndim):
        sumGlp = 0.0
        sumGlm = 0.0
        rho = self.rho
        Gep = self.Gep
        Gem = self.Gem
        Glp = self.Glp
        Glm = self.Glm
        tau = self.tau
        for i in range(6):
            sumGlp += Glp[i]
            sumGlm += Glm[i]
        if ndim == 3:
            jacos = np.zeros((3,45,45), dtype='float64')
            jacos[0][0][3] = -1.0/rho 
            jacos[0][1][8] = -1.0/rho 
            jacos[0][2][7] = -1.0/rho
            jacos[0][3][0] = (-Gep-sumGlp)
            jacos[0][4][0] = (2.0*(Gem+sumGlm)-Gep-sumGlp) 
            jacos[0][5][0] = (2.0*(Gem+sumGlm)-Gep-sumGlp)
            jacos[0][7][2] = (-Gem-sumGlm)
            jacos[0][8][1] = (-Gem-sumGlm)
                
            jacos[1][0][8] = -1.0/rho
            jacos[1][1][4] = -1.0/rho
            jacos[1][2][6] = -1.0/rho
            jacos[1][3][1] = (2.0*(Gem+sumGlm)-Gep-sumGlp)
            jacos[1][4][1] = (-Gep-sumGlp)
            jacos[1][5][1] = (2.0*(Gem+sumGlm)-Gep-sumGlp)
            jacos[1][6][2] = (-Gem-sumGlm)
            jacos[1][8][0] = (-Gem-sumGlm)
                
            jacos[2][0][7] = -1.0/rho
            jacos[2][1][6] = -1.0/rho
            jacos[2][2][5] = -1.0/rho
            jacos[2][3][2] = (2.0*(Gem+sumGlm)-Gep-sumGlp)
            jacos[2][4][2] = (2.0*(Gem+sumGlm)-Gep-sumGlp)
            jacos[2][5][2] = (-Gep-sumGlp)
            jacos[2][6][1] = (-Gem-sumGlm)
            jacos[2][7][0] = (-Gem-sumGlm)
            for i in range(6):
                jacos[0][9+i][0]= (Glp[i]/tau[i])
                jacos[0][15+i][0] = (Glp[i]/tau[i]-2.0*Glm[i]/tau[i])
                jacos[0][21+i][0] = (Glp[i]/tau[i]-2.0*Glm[i]/tau[i])
                jacos[0][33+i][2] = Glm[i]/tau[i]
                jacos[0][39+i][1] = Glm[i]/tau[i]

                jacos[1][9+i][1]= (Glp[i]/tau[i]-2.0*Glm[i]/tau[i])
                jacos[1][15+i][1] = (Glp[i]/tau[i])
                jacos[1][21+i][1] = (Glp[i]/tau[i]-2.0*Glm[i]/tau[i])
                jacos[1][27+i][2] = Glm[i]/tau[i]
                jacos[1][39+i][0] = Glm[i]/tau[i]

                jacos[2][9+i][2]= (Glp[i]/tau[i]-2.0*Glm[i]/tau[i])
                jacos[2][15+i][2] = (Glp[i]/tau[i]-2.0*Glm[i]/tau[i])
                jacos[2][21+i][2] = (Glp[i]/tau[i])
                jacos[2][27+i][1] = Glm[i]/tau[i]
                jacos[2][33+i][0] = Glm[i]/tau[i]
        if ndim == 2:
            jacos = np.zeros((2,45,45), dtype='float64')
            jacos[0][0][3] = -1.0/rho 
            jacos[0][1][8] = -1.0/rho
            jacos[0][2][7] = -1.0/rho
            jacos[0][3][0] = (-Gep-sumGlp)
            jacos[0][4][0] = (2.0*(Gem+sumGlm)-Gep-sumGlp) 
            jacos[0][5][0] = (2.0*(Gem+sumGlm)-Gep-sumGlp)
            jacos[0][7][2] = (-Gem-sumGlm) 
            jacos[0][8][1] = (-Gem-sumGlm)
                
            jacos[1][0][8] = -1.0/rho
            jacos[1][1][4] = -1.0/rho
            jacos[1][2][6] = -1.0/rho
            jacos[1][3][1] = (2.0*(Gem+sumGlm)-Gep-sumGlp)
            jacos[1][4][1] = (-Gep-sumGlp)
            jacos[1][5][1] = (2.0*(Gem+sumGlm)-Gep-sumGlp)
            jacos[1][6][2] = (-Gem-sumGlm)
            jacos[1][8][0] = (-Gem-sumGlm)

            for i in range(6):
                jacos[0][9+i][0]= (Glp[i]/tau[i])
                jacos[0][15+i][0] = (Glp[i]/tau[i]-2.0*Glm[i]/tau[i])
                jacos[0][21+i][0] = (Glp[i]/tau[i]-2.0*Glm[i]/tau[i])
                jacos[0][33+i][2] = Glm[i]/tau[i]
                jacos[0][39+i][1] = Glm[i]/tau[i]

                jacos[1][9+i][1]= (Glp[i]/tau[i]-2.0*Glm[i]/tau[i])
                jacos[1][15+i][1] = (Glp[i]/tau[i])
                jacos[1][21+i][0] = (Glp[i]/tau[i]-2.0*Glm[i]/tau[i])
                jacos[1][27+i][0] = Glm[i]/tau[i]
                jacos[1][39+i][0] = Glm[i]/tau[i]
        return jacos


class SoftTissue(Material):
    def __init__(self, **kw):
        kw.setdefault('rho', 1060.0)
        kw.setdefault('vp', 1578.0)
        gamma = 1060.0*1578.0**2
        kw.setdefault('Gep', 0.983643335902*gamma)
        kw.setdefault('Gem', 0.0)
        kw.setdefault('Glp', [0.0015579*gamma, 0.04412869*gamma, -0.22507493*gamma, 
                              0.52691894*gamma, -0.5573311*gamma, 0.22615716*gamma])
        kw.setdefault('Glm', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        kw.setdefault('tau', [7.95774715e-08, 5.03654883e-08, 3.68414220e-08, 2.90428728e-08, 
                              2.39691179e-08, 1.98943679e-08])
        super(SoftTissue, self). __init__(**kw)

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:

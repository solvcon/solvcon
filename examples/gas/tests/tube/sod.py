# -*- coding: UTF-8 -*-
#
# Copyright (c) 2017, Taihsiang Ho <tai271828@gmail.com>
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


import os  # Python standard library
import tempfile
import fnmatch
import math
import numpy as np  # http://www.numpy.org
import solvcon as sc  # SOLVCON
from solvcon.conf import env
from solvcon.parcel import gas  # A specific SOLVCON solver package we'll use
from .analytic import Sod1D

# prepare the temp folder to accommodate data
DIR_TMP = tempfile.mkdtemp(prefix='solvcon-')
DIR_DATA = os.path.abspath(os.path.join(DIR_TMP, 'result'))


class DiaphragmIAnchor(sc.MeshAnchor):
    """
    diaphragm initialization anchor
    """
    def __init__(self, svr, **kw):
        self.rho1 = float(kw.pop('rho1'))
        self.rho2 = float(kw.pop('rho2'))
        self.p1 = float(kw.pop('p1'))
        self.p2 = float(kw.pop('p2'))
        self.gamma = float(kw.pop('gamma'))
        self.tube_height = float(kw.pop('tube_height'))
        super(DiaphragmIAnchor, self).__init__(svr, **kw)

    def provide(self):
        super(DiaphragmIAnchor, self).provide()
        gamma = self.gamma
        svr = self.svr
        svr.soln[:,0].fill(self.rho1)
        svr.soln[:,1].fill(0.0)
        svr.soln[:,2].fill(0.0)
        if svr.ndim == 3:
            svr.soln[:,3].fill(0.0)
        # total energy of perfect gas without kinetic energy
        svr.soln[:,svr.ndim+1].fill(self.p1/(gamma-1))
        # the diaphragm localtion along the tube
        tube_middle = self.tube_height / 2
        # get coordination of cells in one side of the tube
        slct_int = svr.blk.clcnd[:,2] > tube_middle
        # get coordination of cells in one side of the tube
        slct_gst = svr.blk.gstclcnd[:,2] > tube_middle
        slct = np.concatenate((slct_gst, slct_int))
        # initialize the data in one side of the tube
        svr.soln[slct,0] = self.rho2
        # total energy of pergect gas without kinetic energy
        svr.soln[slct,svr.ndim+1] = self.p2/(gamma-1)
        # update.
        svr.sol[:] = svr.soln[:]


class TubeProbe(gas.ProbeHook):
    """
    Place a probe for the flow properties in the tube.
    """

    def __init__(self, cse, **kw):
        """
        Provide basic information to the probe hook.
        """
        # define where to probe
        pois = []
        for z_axis in range(99):
            poi = (str(z_axis), 0.0, 0.0, z_axis/100.0)
            pois.append(poi)
        # provide coordination and spec list information to the probe hook.
        kw['coords'] = pois
        kw['speclst'] = ['rho', 'v', 'p']
        super(TubeProbe, self).__init__(cse, **kw)

    def postloop(self):
        """
        Dump the probe data in the end.
        """
        super(TubeProbe, self).postloop()
        rho, v, p = self.points[0].vals[-1][1:]
        self.info('Probe result at %s:\n' % self.points[0])
        self.info('- rho = %.3f \n' % rho)
        self.info('- v = %.3f \n' % v)
        self.info('- p = %.3f \n' % p)


class SodtubeMesher(object):
    """
    Representation of a Sod tube and the Gmsh meshing helper.

    :ivar
    :type
    """

    GMSH_SCRIPT = """
    // draw a circle
    // vertices.
    // center of the circle
    Point(1) = {0,0,0,%(edgelength)g};
    // vertices on the arc
    Point(2) = {%(radius)g,0,0,%(edgelength)g};
    Point(3) = {0,%(radius)g,0,%(edgelength)g};
    Point(4) = {-%(radius)g,0,0,%(edgelength)g};
    Point(5) = {0,-%(radius)g,0,%(edgelength)g};
    // lines.
    // draw the arc
    Circle(1) = {2,1,3};
    Circle(2) = {3,1,4};
    Circle(3) = {4,1,5};
    Circle(4) = {5,1,2};
    // connect the arc to get a circle
    Line Loop(5) = {1,2,3,4};

    // surface.
    Plane Surface(6) = {5};
    // extude the surface toward z
    // to get a cylinder with height 2
    Extrude {0,0,%(height)g} {
      Surface{6};
    }
    // physics.
    Physical Surface("wall") = {15, 19, 23, 27};
    Physical Surface("left") = {6};
    Physical Surface("right") = {28};
    Physical Volume("tube") = {1};
    """.strip()

    def __init__(self, center, radius, height, edgelength):
        self.center = center
        self.radius = radius
        self.height = height
        self.edgelength = edgelength

    def __call__(self, cse):
        script_arguments = dict(
            radius=self.radius, height=self.height, edgelength=self.edgelength)
        cmds = self.GMSH_SCRIPT % script_arguments
        cmds = [cmd.rstrip() for cmd in cmds.strip().split('\n')]
        gmh = sc.Gmsh(cmds)()
        return gmh.toblock(bcname_mapper=cse.condition.bcmap)


def generate_bcmap():
    # BC map.
    bcmap = {
        b'wall': (sc.bctregy.GasWall, {},),
        b'left': (sc.bctregy.GasNonrefl, {},),
        b'right': (sc.bctregy.GasNonrefl, {},),
    }
    return bcmap


def tube_base(casename=None,
              psteps=None, ssteps=None, edgelength=None,
              gamma=None,
              rho1=None, p1=None,
              rho2=None, p2=None,
              **kw):
    """
    Base configuration of the simulation and return the case object.

    :return: The created Case object.
    :rtype: solvcon.parcel.gas.GasCase
    """
    ############################################################################
    # Step 1: Obtain the analytical solution.
    ############################################################################

    # Calculate the flow properties in all zones separated by the shock.
    # relation = DiaphragmIAnchor(gamma=gamma,
    #                            rho1=rho1, p1=p1, rho2=rho2, p2=p2)

    ############################################################################
    # Step 2: Instantiate the simulation case.
    ############################################################################

    # Create the mesh generator.  Keep it for later use.
    tube_height = 1.0
    mesher = SodtubeMesher(center=0.0, radius=0.05, height=tube_height,
                           edgelength=edgelength)
    # Set up case.
    cse = gas.GasCase(
        # Mesh generator.
        mesher=mesher,
        # Mapping boundary-condition treatments.
        bcmap=generate_bcmap(),
        # Use the case name to be the basename for all generated files.
        basefn=casename,
        # Use `cwd`/result to store all generated files.
        basedir=DIR_DATA,
        # Debug and capture-all.
        debug=False, **kw)

    ############################################################################
    # Step 3: Set up delayed callbacks.
    ############################################################################

    # Field initialization and derived calculations.
    cse.defer(gas.FillAnchor, mappers={'soln': gas.GasSolver.ALMOST_ZERO,
                                       'dsoln': 0.0, 'amsca': gamma})
    # cse.defer(gas.DensityInitAnchor, rho=density)
    cse.defer(DiaphragmIAnchor,
              gamma=gamma, rho1=rho1, p1=p1, rho2=rho2, p2=p2,
              tube_height=tube_height)
    cse.defer(gas.PhysicsAnchor, rsteps=ssteps)
    # probe the data, so this anchor should follow gas.PhysicsAnchor
    # Report information while calculating.
    cse.defer(gas.ProgressHook, linewidth=ssteps / psteps, psteps=psteps)
    #cse.defer(gas.CflHook, fullstop=False, cflmax=10.0, psteps=ssteps)
    # summarize the data fetched by TubeAnchor
    cse.defer(TubeProbe, psteps=ssteps)


    ############################################################################
    # Final: Return the configured simulation case.
    ############################################################################
    return cse


@gas.register_arrangement
def tube(casename, **kw):
    return tube_base(
        # Required positional argument for the name of the simulation case.
        casename,
        # Arguments to the base configuration.
        gamma=1.4,
        ssteps=8, psteps=4, edgelength=0.015,
        rho1=1.0, p1=1.0,
        rho2=0.125, p2=0.1,
        # Arguments to GasCase.
        time_increment=158000.e-8, steps_run=100, **kw)


def load_from_np(data_path=DIR_DATA):
    # sort the listed file names so the associated location
    # make more sense.
    np_files = sorted(os.listdir(data_path))
    points = []
    for np_file in fnmatch.filter(np_files, 'tube_pt_ppank*'):
        points.append(np.load(os.path.join(data_path, np_file)))

    return points


def load_from_analytic(points_target):
    points_base = []
    locations = []
    for idx in range(len(points_target)):
        locations.append(idx/float(len(points_target)) - 0.5)

    list_time = []
    for data in points_target[0]:
        list_time.append(data[0])

    sod = Sod1D()
    for idx in range(len(points_target)):
        point_by_time = []
        for time in list_time:
            analytic = sod.get_analytic_solution((locations[idx],), t=time)[0]
            point_by_time.append([time, analytic[1], analytic[2], analytic[3]])
        points_base.append(point_by_time)

    return points_base

def false_number_counter_reset(false_counter, false_derived_idx):
    false_derived_idx -= 1
    false_counter[false_derived_idx] = 0
    return false_counter

def false_number_counter_add(false_counter, false_derived_idx):
    false_derived_idx -= 1
    false_counter[false_derived_idx] += 1
    return false_counter

def compare_probe_data_by_counter(tolerance=1.5e-1, tolerance_number=9):
    """
    Judge the probe data ith analytic solutions by a counter method.

    The discontinuity region is very likely to has deviation. It does not make
    too much sense if only one point deviates from the analytic data at only few moments.

    :param tolerance:
    :param tolerance_number:
    :return:
    """
    points_target = load_from_np(DIR_DATA)
    #points_base = load_from_np(os.path.join(env.datadir, 'gas.sod_tube'))
    points_base = load_from_analytic(points_target)
    for idx_point in range(len(points_base)):
        point = points_base[idx_point]
        false_number_counter = [0]*3
        for idx_stride_step in range(len(point)):
            for idx_derived in range(1,4):
                # idx zero for time stride_step
                target = points_target[idx_point][idx_stride_step][idx_derived]
                base = point[idx_stride_step][idx_derived]
                delta = target - base
                if math.fabs(delta) < tolerance:
                    false_number_counter = false_number_counter_reset(false_number_counter, idx_derived)
                elif math.fabs(delta) > tolerance and max(false_number_counter) < tolerance_number:
                    false_number_counter = false_number_counter_add(false_number_counter, idx_derived)
                    print("%i %i %i %f %f %f" % (idx_point, idx_stride_step, idx_derived, target, base, delta))
                    print(false_number_counter)
                else:
                    false_number_counter = false_number_counter_add(false_number_counter, idx_derived)
                    print("%i %i %i %f %f %f" % (idx_point, idx_stride_step, idx_derived, target, base, delta))
                    print(false_number_counter)
                    return False

    return True

def compare_probe_data_by_snapshot(tolerance=2.0e-2):
    """
    Snapshot the final status of specific probed points.

    :param tolerrance:
    :return:
    """
    points_target = load_from_np(DIR_DATA)
    points_base = load_from_analytic(points_target)
    all_point_number = len(points_target)
    idx_point_to_check = [0, round(all_point_number/3), round(all_point_number/3*2), -1]
    for idx_point in idx_point_to_check:
        for idx_derived in range(1,4):
            target = points_target[idx_point][-1][idx_derived]
            base = points_base[idx_point][-1][idx_derived]
            delta = target - base
            if math.fabs(delta) > tolerance:
                print("%i %i %f %f %f" % (idx_point, idx_derived, target, base, delta))
                return False

    return True

def run():
    from solvcon.batch import batregy
    from solvcon import domain
    from solvcon.case import arrangements
    funckw ={
        'envar': {},
        'runlevel': 0,
        'solver_output': False,
        'batch': batregy['Batch'],
        'npart': None,
        'domantype': domain
    }
    func = arrangements['tube']
    func(submit=False, **funckw)

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:

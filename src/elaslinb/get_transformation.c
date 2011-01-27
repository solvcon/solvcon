/*
 * Copyright (C) 2010-2011 Yung-Yu Chen <yyc@solvcon.net>.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <stddef.h>
#include <math.h>
#include "elaslin.h"

int get_transformation(double *nvec, double *svec,
        double rotm[][3], double bondm[][6], double bondminv[][6]) {
    double len;
    double bondn[6][6];
    // iterators.
    int it, jt;
    // construct transformation (Q) for the boundary coordinate system.
    /// x'1-axis; the (unit) normal vector of the boundary surface.
    for (it=0; it<3; it++) {
        rotm[0][0] = nvec[0];
        rotm[0][1] = nvec[1];
        rotm[0][2] = nvec[2];
    };
    /// x'3-axis; constructed from the cross product of the normal (x'1) vector
    /// and a vector parallel to the boundary surface.
    rotm[2][0] = nvec[1]*svec[2] - nvec[2]*svec[1];
    rotm[2][1] = nvec[2]*svec[0] - nvec[0]*svec[2];
    rotm[2][2] = nvec[0]*svec[1] - nvec[1]*svec[0];
    len = sqrt(rotm[2][0]*rotm[2][0]
             + rotm[2][1]*rotm[2][1]
             + rotm[2][2]*rotm[2][2]);
    rotm[2][0] /= len;
    rotm[2][1] /= len;
    rotm[2][2] /= len;
    /// x'2-axis; the cross product of x'1 and x'3.
    rotm[1][0] = rotm[0][1]*rotm[2][2] - rotm[0][2]*rotm[2][1];
    rotm[1][1] = rotm[0][2]*rotm[2][0] - rotm[0][0]*rotm[2][2];
    rotm[1][2] = rotm[0][0]*rotm[2][1] - rotm[0][1]*rotm[2][0];
    // construct the Bond's matrices for the transformation matrix rotm (Q).
    /// upper left.
    bondm[0][0] = rotm[0][0] * rotm[0][0];
    bondm[0][1] = rotm[0][1] * rotm[0][1];
    bondm[0][2] = rotm[0][2] * rotm[0][2];
    bondm[1][0] = rotm[1][0] * rotm[1][0];
    bondm[1][1] = rotm[1][1] * rotm[1][1];
    bondm[1][2] = rotm[1][2] * rotm[1][2];
    bondm[2][0] = rotm[2][0] * rotm[2][0];
    bondm[2][1] = rotm[2][1] * rotm[2][1];
    bondm[2][2] = rotm[2][2] * rotm[2][2];
    /// upper right.
    bondm[0][3] = 2*rotm[0][1]*rotm[0][2];
    bondm[0][4] = 2*rotm[0][2]*rotm[0][0];
    bondm[0][5] = 2*rotm[0][0]*rotm[0][1];
    bondm[1][3] = 2*rotm[1][1]*rotm[1][2];
    bondm[1][4] = 2*rotm[1][2]*rotm[1][0];
    bondm[1][5] = 2*rotm[1][0]*rotm[1][1];
    bondm[2][3] = 2*rotm[2][1]*rotm[2][2];
    bondm[2][4] = 2*rotm[2][2]*rotm[2][0];
    bondm[2][5] = 2*rotm[2][0]*rotm[2][1];
    /// lower left.
    bondm[3][0] = rotm[1][0]*rotm[2][0];
    bondm[3][1] = rotm[1][1]*rotm[2][1];
    bondm[3][2] = rotm[1][2]*rotm[2][2];
    bondm[4][0] = rotm[2][0]*rotm[0][0];
    bondm[4][1] = rotm[2][1]*rotm[0][1];
    bondm[4][2] = rotm[2][2]*rotm[0][2];
    bondm[5][0] = rotm[0][0]*rotm[1][0];
    bondm[5][1] = rotm[0][1]*rotm[1][1];
    bondm[5][2] = rotm[0][2]*rotm[1][2];
    /// lower right.
    bondm[3][3] = rotm[1][1]*rotm[2][2] + rotm[1][2]*rotm[2][1];
    bondm[3][4] = rotm[1][0]*rotm[2][2] + rotm[1][2]*rotm[2][0];
    bondm[3][5] = rotm[1][1]*rotm[2][0] + rotm[1][0]*rotm[2][1];
    bondm[4][3] = rotm[0][1]*rotm[2][2] + rotm[0][2]*rotm[2][1];
    bondm[4][4] = rotm[0][0]*rotm[2][2] + rotm[0][2]*rotm[2][0];
    bondm[4][5] = rotm[0][1]*rotm[2][0] + rotm[0][0]*rotm[2][1];
    bondm[5][3] = rotm[0][1]*rotm[1][2] + rotm[0][2]*rotm[1][1];
    bondm[5][4] = rotm[0][0]*rotm[1][2] + rotm[0][2]*rotm[1][0];
    bondm[5][5] = rotm[0][1]*rotm[1][0] + rotm[0][0]*rotm[1][1];
    // Bond's matrix for strain.
    for (it=0; it<3; it++){
        for (jt=0; jt<3; jt++){
            bondn[it][jt] = bondm[it][jt];
        };
        for (jt=3; jt<6; jt++){
            bondn[it][jt] = bondm[it][jt] / 2.0;
        };
    };
    for (it=3; it<6; it++){
        for (jt=0; jt<3; jt++){
            bondn[it][jt] = bondm[it][jt] * 2.0;
        };
        for (jt=3; jt<6; jt++){
            bondn[it][jt] = bondm[it][jt];
        };
    };
    // Inverse of M is N^T.
    for (it=0; it<6; it++){
        for (jt=0; jt<6; jt++){
            bondminv[jt][it] = bondn[it][jt];
        };
    };
    return 0;
};
// vim: set ts=4 et:

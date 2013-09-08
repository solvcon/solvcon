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

#include "bulk.h"

#ifdef __CUDACC__
__global__ void cuda_bound_wall_soln(exedata *exd, int nbnd, int *facn) {
    int ibnd = blockDim.x * blockIdx.x + threadIdx.x;
#else
int bound_wall_soln(exedata *exd, int nbnd, int *facn) {
    int ibnd;
#endif
    // pointers.
    int *pfacn, *pfccls, *pfcnds;
    double *pfcnml, *pndcrd, *pfccnd;
    double *pisoln, *pjsoln;
    // scalars.
    double len;
    // arrays.
    double nx, ny;
    // iterators.
    int ifc, icl, jcl;
#ifndef __CUDACC__
    #pragma omp parallel for default(shared) private(ibnd, \
    pfacn, pfccls, pfcnds, pfcnml, pndcrd, pfccnd, \
    pisoln, pjsoln, len, ifc, icl, jcl, nx ,ny)
    for (ibnd=0; ibnd<nbnd; ibnd++) {
#else
    if (ibnd < nbnd) {
#endif
        pfacn = facn + ibnd*BFREL;
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc*FCREL;
        icl = pfccls[0];
        jcl = pfccls[1];
        pisoln = exd->soln + icl*NEQ;
        pjsoln = exd->soln + jcl*NEQ;
        // rotation and inverse rotation matrices.
        pfcnml = exd->fcnml + ifc*NDIM;
        pfcnds = exd->fcnds + ifc*(FCMND+1);
        pfccnd = exd->fccnd + ifc*NDIM;
        nx = pfcnml[0];
        ny = pfcnml[1];
        pjsoln[0] = pisoln[0];
        pjsoln[1] = pisoln[1] - 2*nx*(nx*pisoln[1]+ny*pisoln[2]);
        pjsoln[2] = pisoln[2] - 2*ny*(nx*pisoln[1]+ny*pisoln[2]);
#ifndef __CUDACC__
    };
    return 0;
};
#else
    };
};
extern "C" int bound_wall_soln(int nthread, void *gexc,
        int nbnd, void *gfacn) {
    int nblock = (nbnd + nthread-1) / nthread;
    cuda_bound_wall_soln<<<nblock, nthread>>>((exedata *)gexc,
        nbnd, (int *)gfacn);
    cudaThreadSynchronize();
    return 0;
};
#endif

#ifdef __CUDACC__
__global__ void cuda_bound_wall_dsoln(exedata *exd, int nbnd, int *facn) {
    int ibnd = blockDim.x * blockIdx.x + threadIdx.x;
#else
int bound_wall_dsoln(exedata *exd, int nbnd, int *facn) {
    int ibnd;
#endif
    // pointers.
    int *pfacn, *pfccls, *pfcnds;
    double *pfcnml, *pndcrd, *pfccnd, (*pten)[NDIM];
    double *pidsoln, *pjdsoln, *pdsoln;
    // scalars.
    double len, x, y, deg, ux, uy, vx, vy, pi;
    // arrays.
    double vec[NEQ][NDIM];
    double vmt[NDIM][NDIM], mat[NDIM][NDIM], mvt[NDIM][NDIM];
    // iterators.
    int ifc, icl, jcl, ieq, it, jt;
#ifndef __CUDACC__
    #pragma omp parallel for default(shared) private(ibnd, \
    pfacn, pfccls, pfcnds, pfcnml, pndcrd, pfccnd, pten, \
    pidsoln, pjdsoln, pdsoln, \
    len, vec, vmt, mat, mvt, ifc, icl, jcl, ieq, it, jt)
    for (ibnd=0; ibnd<nbnd; ibnd++) {
#else
    if (ibnd < nbnd) {
#endif
        pi = 3.14159265358979323846;
        pfacn = facn + ibnd*BFREL;
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc*FCREL;
        icl = pfccls[0];
        jcl = pfccls[1];
        pidsoln = exd->dsoln + icl*NEQ*NDIM;
        pjdsoln = exd->dsoln + jcl*NEQ*NDIM;
        // coordinate transformation and set transformed vectors.
        pfcnml = exd->fcnml + ifc*NDIM;
        pfcnds = exd->fcnds + ifc*(FCMND+1);
        pfccnd = exd->fccnd + ifc*NDIM;
        if(pfcnml[0] >= 0)
        {    x = pfcnml[0];
             y = pfcnml[1];
             if(x == 0 && y>0) deg = 0.0;
             else if (x == 0 && y<0) deg = pi;
             else if(pfcnml[1] > 0) 
             {    deg = acos(y/sqrt(x*x+y*y));
                  deg = -deg;}
             else
             {    deg = acos(x/sqrt(x*x+y*y));
                  deg = 3*pi/2 - deg;}
        }
        else
        {    x = pfcnml[0];
             y = pfcnml[1];
             if(y==0 && x>0) deg = -pi/2;
             else if(y==0 && x<0) deg = pi/2;
             else if(pfcnml[1] > 0)
             {    deg = acos(y/sqrt(x*x+y*y));}
             else 
             {    deg = acos(-y/sqrt(x*x+y*y));
                  deg = 2*pi - deg;} 
        }
        mat[0][0] = cos(2*deg);
        mat[0][1] = sin(2*deg);
        mat[1][0] = sin(2*deg);
        mat[1][1] = -cos(2*deg);
        pdsoln = pjdsoln;
        //eq1  density(continuity equation)  
        pdsoln[0] = mat[0][0]*pidsoln[0] + mat[0][1]*pidsoln[1];
        pdsoln[1] = mat[1][0]*pidsoln[0] - mat[1][1]*pidsoln[1];
        pdsoln += NDIM; pidsoln += NDIM;
        ux = pidsoln[0]; uy = pidsoln[1];
        pidsoln += NDIM;
        vx = pidsoln[0]; vy = pidsoln[1];
        //momentum equation eq2~3
        pdsoln[0] = mat[0][0]*mat[0][0]*ux+mat[0][1]*mat[0][1]*vy+
                    mat[0][0]*mat[0][1]*(vx+uy);
        pdsoln[1] = -mat[0][0]*mat[0][0]*uy+mat[0][1]*mat[0][1]*vx+
                    mat[0][0]*mat[0][1]*(ux-vy);
        pdsoln += NDIM;
        pdsoln[0] = -mat[0][0]*mat[0][0]*vx+mat[0][1]*mat[0][1]*uy+
                    mat[0][0]*mat[0][1]*(ux-vy);
        pdsoln[1] = mat[0][0]*mat[0][0]*vy+mat[0][1]*mat[0][1]*ux+
                    mat[0][0]*mat[0][1]*(-uy-vx);
        
#ifndef __CUDACC__
    };
    return 0;
};
#else
    };
};
extern "C" int bound_wall_dsoln(int nthread, void *gexc,
        int nbnd, void *gfacn) {
    int nblock = (nbnd + nthread-1) / nthread;
    cuda_bound_wall_dsoln<<<nblock, nthread>>>((exedata *)gexc,
        nbnd, (int *)gfacn);
    cudaThreadSynchronize();
    return 0;
};
#endif

// vim: set ts=4 et:

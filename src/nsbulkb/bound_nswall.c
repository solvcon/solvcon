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
__global__ void cuda_bound_nswall_soln(exedata *exd, int nbnd, int *facn) {
    int ibnd = blockDim.x * blockIdx.x + threadIdx.x;
#else
int bound_nswall_soln(exedata *exd, int nbnd, int *facn) {
    int ibnd;
#endif
    // pointers.
    int *pfacn, *pfccls, *pfcnds;
    double *pfcnml, *pndcrd, *pfccnd;
    double *pisoln, *pjsoln;
    // scalars.
    double len;
    // arrays.
    double mat[NDIM][NDIM], mvt[NDIM][NDIM];
    double mom[NDIM];
    // iterators.
    int ifc, icl, jcl;
#ifndef __CUDACC__
    #pragma omp parallel for default(shared) private(ibnd, \
    pfacn, pfccls, pfcnds, pfcnml, pndcrd, pfccnd, \
    pisoln, pjsoln, len, mat, mvt, mom, ifc, icl, jcl)
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
        pjsoln[0] = pisoln[0];
        pjsoln[1] = -pisoln[1];
        pjsoln[2] = -pisoln[2];
#ifndef __CUDACC__
    };
    return 0;
};
#else
    };
};
extern "C" int bound_nswall_soln(int nthread, void *gexc,
        int nbnd, void *gfacn) {
    int nblock = (nbnd + nthread-1) / nthread;
    cuda_bound_nswall_soln<<<nblock, nthread>>>((exedata *)gexc,
        nbnd, (int *)gfacn);
    cudaThreadSynchronize();
    return 0;
};
#endif

#ifdef __CUDACC__
__global__ void cuda_bound_nswall_dsoln(exedata *exd, int nbnd, int *facn) {
    int ibnd = blockDim.x * blockIdx.x + threadIdx.x;
#else
int bound_nswall_dsoln(exedata *exd, int nbnd, int *facn) {
    int ibnd;
#endif
    int *pfacn, *pfccls, *pfcnds;
    double *pidsoln, *pjdsoln, *pdsoln;
    double *pndcrd, *pfccnd, *pfcnml;
    // scalars.
    double len, x, y, deg, ux, uy, vx, vy, pi, q2s, q2t, q3s, q3t, nx, ny;
    // arrays.
    double vec[NEQ][NDIM];
    double mat[NDIM][NDIM], matinv[NDIM][NDIM];
    // iterators.
    int ifc, icl, jcl, ieq;
    pi = 3.14159265358979323846;
#ifndef __CUDACC__
    #pragma omp parallel for default(shared) private(ibnd, \
    pfacn, pfccls, pfcnds, pidsoln, pjdsoln, pdsoln, pndcrd, pfccnd, pfcnml, \
    len, x, y, deg, ux, uy, vx, vy, pi, q2s, q2t, q3s, q3t, nx, ny, vec, mat, \
    matinv, ifc, icl, jcl, ieq)
    for (ibnd=0; ibnd<nbnd; ibnd++) {
#else
    if (ibnd < nbnd) {
#endif
        pfacn = facn + ibnd*BFREL;
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc*FCREL;
        icl = pfccls[0];
        jcl = pfccls[1];
        pfcnds = exd->fcnds + ifc*(FCMND+1);
        pidsoln = exd->dsoln + icl*NEQ*NDIM;
        pjdsoln = exd->dsoln + jcl*NEQ*NDIM;
        pfccnd = exd->fccnd + ifc*NDIM;
        pfcnml = exd->fcnml + ifc*NDIM;
        if(pfcnml[0] >= 0)
        {    x = pfcnml[0];
             y = pfcnml[1];
             nx = x; ny = y;
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
             nx = x; ny = y;
             if(y==0 && x>0) deg = -pi/2;
             else if(y==0 && x<0) deg = pi/2;
             else if(pfcnml[1] > 0)
             {    deg = acos(y/sqrt(x*x+y*y));}
             else 
             {    deg = acos(-y/sqrt(x*x+y*y));
                  deg = 2*pi - deg;} 
        }
        //mat = T matrix in 99 paper
        mat[0][0] = cos(deg);
        mat[0][1] = -sin(deg);
        mat[1][0] = sin(deg);
        mat[1][1] = cos(deg);
        //transfer to sigma axis
        
        pdsoln = pjdsoln;
        //eq1  density(continuity equation)  
        pdsoln[0] = pidsoln[0] - 2*nx*(nx*pidsoln[0]+ny*pidsoln[1]);
        pdsoln[1] = pidsoln[1] - 2*ny*(nx*pidsoln[0]+ny*pidsoln[1]);
        pdsoln += NDIM; pidsoln += NDIM;
        ux = pidsoln[0]; uy = pidsoln[1];
        pidsoln += NDIM;
        vx = pidsoln[0]; vy = pidsoln[1];
        //momentum equation eq2~3
        q2s = mat[0][0]*(ux*mat[0][0]+vx*mat[1][0])+mat[1][0]*(uy*mat[0][0]+vy*mat[1][0]);
        q2t = -mat[1][0]*(ux*mat[0][0]+vx*mat[1][0])+mat[0][0]*(uy*mat[0][0]+vy*mat[1][0]);
        q3s = mat[0][0]*(vx*mat[0][0]-ux*mat[1][0])+mat[1][0]*(vy*mat[0][0]-uy*mat[1][0]);
        q3t = -mat[1][0]*(vx*mat[0][0]-ux*mat[1][0])+mat[0][0]*(vy*mat[0][0]-uy*mat[1][0]);
        q2s = -q2s;
        q3s = -q3s;
        pdsoln[0] = mat[0][0]*(q2s*mat[0][0]-q3s*mat[1][0])-mat[1][0]*(q2t*mat[0][0]-q3t*mat[1][0]);
        pdsoln[1] = mat[0][0]*(q2s*mat[0][0]-q3s*mat[1][0])-mat[1][0]*(q2t*mat[0][0]-q3t*mat[1][0]);
        pdsoln += NDIM;
        pdsoln[0] = mat[0][0]*(q3s*mat[0][0]+q2s*mat[1][0])-mat[1][0]*(q3t*mat[0][0]+q2t*mat[1][0]);
        pdsoln[1] = mat[1][0]*(q3s*mat[0][0]+q2s*mat[1][0])+mat[0][0]*(q3t*mat[0][0]+q2t*mat[1][0]);
#ifndef __CUDACC__
    };
    return 0;
};
#else
    };
};
extern "C" int bound_nswall_dsoln(int nthread, void *gexc,
        int nbnd, void *gfacn) {
    int nblock = (nbnd + nthread-1) / nthread;
    cuda_bound_nswall_dsoln<<<nblock, nthread>>>((exedata *)gexc,
        nbnd, (int *)gfacn);
    cudaThreadSynchronize();
    return 0;
};
#endif

// vim: set ts=4 et:

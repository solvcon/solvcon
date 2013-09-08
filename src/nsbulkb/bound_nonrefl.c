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
__global__ void cuda_bound_nonrefl_soln(exedata *exd, int nbnd, int *facn,
    ) {
    int ibnd = blockDim.x * blockIdx.x + threadIdx.x;
#else
int bound_nonrefl_soln(exedata *exd, int nbnd, int *facn
    ) {
    int ibnd;
#endif
    // pointers.
    int *pfacn, *pfccls;
    double *pisol, *pisoln, *pjsoln, *pjsol;
    double *pvalue, *pamsca;
    // iterators.
    int ifc, icl, jcl, ieq;
    double p, pn, v1n, v2n, pini, bulk, dp;
#ifndef __CUDACC__
    #pragma omp parallel for default(shared) private(ibnd, pfacn, pfccls, pisol,\
     pisoln, pjsoln, pjsol, pvalue, ifc, icl, jcl, ieq, \
     p, pn, v1n, v2n, pini, bulk)
    for (ibnd=0; ibnd<nbnd; ibnd++) {
#else
    if (ibnd < nbnd) {
#endif
        pfacn = facn + ibnd*BFREL;
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc*FCREL;
        icl = pfccls[0];
        jcl = pfccls[1];
        pamsca = exd->amsca + icl*NSCA;
        bulk = pamsca[0];
        pini = pamsca[5];
        // set solutions.
        pisol = exd->sol + icl*NEQ;
        pisoln = exd->soln + icl*NEQ;
        pjsoln = exd->soln + jcl*NEQ;
        p = pisol[0];
        pn = pisoln[0];
        v1n = pisoln[1]/pn;
        v2n = pisoln[2]/pn;
        p = bulk*log(p);
        pn = bulk*log(pn);
        dp = pn - p;
        pn = dp + pini;
        pn = exp(pn/bulk);
        //pjsoln[0] = pn;
        //pjsoln[1] = pn*v1n;
        //pjsoln[2] = pn*v2n;
        
        for (ieq=0; ieq<NEQ; ieq++) {
            pjsoln[ieq] = pisoln[ieq] + exd->taylor*(pisol[ieq] - pisoln[ieq]);
        };
                

#ifndef __CUDACC__
    };
    return 0;
};
#else
    };
};
extern "C" int bound_nonrefl_soln(int nthread, void *gexc,
    int nbnd, void *gfacn) {
    int nblock = (nbnd + nthread-1) / nthread;
    cuda_bound_nonrefl_soln<<<nblock, nthread>>>((exedata *)gexc,
        nbnd, (int *)gfacn);
    cudaThreadSynchronize();
    return 0;
};
#endif

#ifdef __CUDACC__
__global__ void cuda_bound_nonrefl_dsoln(exedata *exd, int nbnd, int *facn) {
    int ibnd = blockDim.x * blockIdx.x + threadIdx.x;
#else
int bound_nonrefl_dsoln(exedata *exd, int nbnd, int *facn) {
    int ibnd;
#endif
    // pointers.
    int *pfacn, *pfccls, *pfcnds;
    double *pidsol, *pidsoln, *pjdsoln, *pdsol, *pdsoln;
    double *pndcrd, *pfccnd, *pfcnml;
    // scalars.
    double len, nx, ny, x ,y, deg, pi;
    // arrays.
    double dif[NDIM];
    double vec[NEQ][NDIM];
    double mat[NDIM][NDIM], matinv[NDIM][NDIM];
    // iterators.
    int ifc, icl, jcl, ieq;
    pfacn = facn;
    pi = 3.14159265358979323846;
#ifndef __CUDACC__
    #pragma omp parallel for default(shared) private(ibnd, pfacn, pfccls, \
    pfcnds, pidsol, pidsoln, pjdsoln, pdsol, pdsoln, pndcrd, pfccnd, pfcnml, \
    len, nx, ny, x, y, deg, pi, dif, vec, mat, matinv, ifc, icl, jcl, ieq)
    for (ibnd=0; ibnd<nbnd; ibnd++) {
#else
    if (ibnd < nbnd) {
#endif
        pfacn = facn + ibnd*BFREL;
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc*FCREL;
        pfcnds = exd->fcnds + ifc*(FCMND+1);
        icl = pfccls[0];
        jcl = pfccls[1];
        pidsol = exd->dsol + icl*NEQ*NDIM;
        pidsoln = exd->dsoln + icl*NEQ*NDIM;
        pjdsoln = exd->dsoln + jcl*NEQ*NDIM;
        // coordinate transformation and set transformed vectors.
        pfcnml = exd->fcnml + ifc*NDIM;
        pfccnd = exd->fccnd + ifc*NDIM;
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
        mat[0][0] = cos(deg);
        mat[0][1] = -sin(deg);
        mat[1][0] = sin(deg);
        mat[1][1] = cos(deg);
        pdsoln = pjdsoln;
        if(fabs(nx) == 1.0)
        {    for(ieq=0; ieq<NEQ; ieq++)
             {    pdsoln[0]=0;
                  pdsoln[1]=pidsoln[1];
                  pdsoln += NDIM;
                  pidsoln += NDIM;}
        }
        else
        {    for(ieq=0; ieq<NEQ; ieq++)
             {    pdsoln[0]=pidsoln[0];
                  pdsoln[1]=0;
                  pdsoln += NDIM;
                  pidsoln += NDIM;}
        }
        
#ifndef __CUDACC__
    };
    return 0;
};
#else
    };
};
extern "C" int bound_nonrefl_dsoln(int nthread, void *gexc,
    int nbnd, void *gfacn) {
    int nblock = (nbnd + nthread-1) / nthread;
    cuda_bound_nonrefl_dsoln<<<nblock, nthread>>>((exedata *)gexc,
        nbnd, (int *)gfacn);
    cudaThreadSynchronize();
    return 0;
};
#endif

// vim: set ts=4 et:

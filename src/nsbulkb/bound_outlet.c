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
__global__ void cuda_bound_outlet_soln(exedata *exd, int nbnd, int *facn,
    int nvalue, double *value) {
    int ibnd = blockDim.x * blockIdx.x + threadIdx.x;
#else
int bound_outlet_soln(exedata *exd, int nbnd, int *facn,
    int nvalue, double *value) {
    int ibnd;
#endif
    // pointers.
    int *pfacn, *pfccls;
    double *pamsca;
    double *pvalue, *pjsoln, *pisol, *pjsol;
    // scalars.
    double rhor, rhol, bulk, rho;
    double v1l, v2l, v3l, v1r, v2r, v3r, pl, pr;
    double v1, v2, v3;
    double left, right;
    // pressure base
    double pini, p, eta;
    // iterators.
    int ifc, jcl, icl;
#ifndef __CUDACC__
    #pragma omp parallel for default(shared) private(ibnd, pfacn, pfccls,\
    pamsca, pvalue, pjsoln, pisol, pjsol, rhor, rhol, bulk, rho, v1l, v2l, v3l, \
    v1r, v2r, v3r, pl, pr, v1, v2, v3, left, right, pini, p, eta, ifc, jcl, icl)
    for (ibnd=0; ibnd<nbnd; ibnd++) {
#else
    if (ibnd < nbnd) {
#endif
        pfacn = facn + ibnd*BFREL;
        pvalue = value + ibnd*nvalue;
        pamsca = exd->amsca + ibnd*NSCA;
        ifc = pfacn[0];
        pfccls = exd->fccls + ifc*FCREL;
        icl = pfccls[0];
        jcl = pfccls[1];
        pisol = exd->sol + icl*NEQ;
        pjsol = exd->sol + jcl*NEQ;
        // extract parameters.
        bulk = pamsca[0];
        eta  = pamsca[3];
        pini = pamsca[5];
        

        v1l  = pisol[1]/pisol[0];
        v2l  = pisol[2]/pisol[0];
        v1r  = pjsol[1]/pjsol[0];
        v2r  = pjsol[2]/pjsol[0];
#if NDIM == 3
        v3l = pisol[3]/pisol[0];
        v3r = pjsol[3]/pjsol[0];
#endif
        // density base
        /*
        rhol = pisol[0];
        rhor = pjsol[0];
        right = -pow(rhol,-0.5) + v1l/(2*sqrt(bulk));
        left = -pow(rhor,-0.5) - v1r/(2*sqrt(bulk));
        rho = 4/pow(right+left,2);
        v1 = (right-left)*sqrt(bulk);
        v2 = rhol*(v1l+v2l)/rho - v1;
        pjsoln = exd->soln + jcl*NEQ;
        pjsoln[0] = rho;
        pjsoln[1] = rho*v1;
        pjsoln[2] = rho*v2;
        */
        // pressure base 
        //
        pl = pisol[0];
        pr = pjsol[0];
        right = -pow(pl,-0.5) + v1l*sqrt(eta/bulk)/2;
        left = -pow(pr,-0.5) - v1r*sqrt(eta/bulk)/2;
        p = 4/pow(right+left,2);
        v1 = (right-left)*sqrt(bulk/eta);
        v2 = 2*(-1/sqrt(pl)+1/sqrt(p)+v2l*sqrt(eta/bulk)/2)*sqrt(bulk/eta);
        //p = bulk*log(p);
        //pr = bulk*log(pr);
        //p = (pr-p) + pini;
        //p = exp(p/bulk);
        pjsoln = exd->soln + jcl*NEQ;
        pjsoln[0] = p;
        pjsoln[1] = p*v1;
        pjsoln[2] = p*v2;
        //
        
#if NDIM == 3
        pjsoln[3] = 0.0;
#endif
       
#ifndef __CUDACC__
    };
    return 0;
};
#else
    };
};
extern "C" int bound_outlet_soln(int nthread, void *gexc,
    int nbnd, void *gfacn, int nvalue, void *value) {
    int nblock = (nbnd + nthread-1) / nthread;
    cuda_bound_outlet_soln<<<nblock, nthread>>>((exedata *)gexc,
        nbnd, (int *)gfacn, nvalue, (double *)value);
    cudaThreadSynchronize();
    return 0;
};
#endif

#ifdef __CUDACC__
__global__ void cuda_bound_outlet_dsoln(exedata *exd, int nbnd, int *facn) {
    int ibnd = blockDim.x * blockIdx.x + threadIdx.x;
#else
int bound_outlet_dsoln(exedata *exd, int nbnd, int *facn) {
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
    #pragma omp parallel for default(shared) private(ibnd, pfacn, pfccls, pfcnds, \
    pidsol, pidsoln, pjdsoln, pdsol, pdsoln, pndcrd, pfccnd, pfcnml, len, \
    nx, ny, x, y, deg, pi, dif, vec, mat, matinv, ifc, icl, jcl, ieq)
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
        
        /*for(ieq=0; ieq<NEQ; ieq++)
        {    pdsoln[0]=pidsoln[0];
             pdsoln[1]=pidsoln[1];
             pdsoln+=NDIM;
             pidsoln+=NDIM;}*/
#ifndef __CUDACC__
    };
    return 0;
};
#else
    };
};
extern "C" int bound_outlet_dsoln(int nthread, void *gexc,
    int nbnd, void *gfacn) {
    int nblock = (nbnd + nthread-1) / nthread;
    cuda_bound_outlet_dsoln<<<nblock, nthread>>>((exedata *)gexc,
        nbnd, (int *)gfacn);
    cudaThreadSynchronize();
    return 0;
};
#endif

// vim: set ts=4 et:

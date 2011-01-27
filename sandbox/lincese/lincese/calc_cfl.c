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

#include "lincese.h"

int calc_cfl(exedata *exd, int istart, int iend) {
    int cputicks;
    struct tms timm0, timm1;
    int clnfc;
    // pointers.
    int *pclfcs, *pfccls;
    double *pcfl, *picecnd, *pjcecnd, *pcecnd, *pjac;
    // scalars.
    double hdt, dist, cfl;
    int lwork=4*NEQ;
    int eiginfo;
    int nswap;
    // strings.
    char *jobvl = "N";
    char *jobvr = "N";
    // arrays.
    double wdir[NDIM];
    double jacos[NEQ*NEQ*NDIM];
    double jaco[NEQ][NEQ];
    double fcn[NEQ][NEQ];
    double evcl[NEQ][NEQ], evcr[NEQ][NEQ];
    double wr[NEQ], wi[NEQ];
    double work[lwork];
    int argsort[NEQ];
    // iterators.
    int icl, jcl, ifl, ifc, ieq, jeq;
    times(&timm0);
    hdt = exd->time_increment / 2.0;
    pcfl = exd->cfl + istart;
    picecnd = exd->cecnd + istart * (CLMFC+1) * NDIM;
    pclfcs = exd->clfcs + istart * (CLMFC+1);
    for (icl=istart; icl<iend; icl++) {
        pcfl[0] = 0.0;
        exd->jacofunc(exd, icl, (double *)fcn, jacos);
        clnfc = pclfcs[0];
        for (ifl=1; ifl<=clnfc; ifl++) {
            ifc = pclfcs[ifl];
            pfccls = exd->fccls + ifc * FCREL;
            jcl = pfccls[0] + pfccls[1] - icl;
            // wave direction and distance.
            pjcecnd = exd->cecnd + jcl * (CLMFC+1) * NDIM;
            pcecnd = picecnd + NDIM;
            wdir[0] = picecnd[0] - pcecnd[0];
            wdir[1] = picecnd[1] - pcecnd[1];
#if NDIM == 3
            wdir[2] = picecnd[2] - pcecnd[2];
#endif
            dist = sqrt(wdir[0]*wdir[0] + wdir[1]*wdir[1]
#if NDIM == 3
                + wdir[2]*wdir[2]
#endif
                    );
            wdir[0] /= dist;
            wdir[1] /= dist;
#if NDIM == 3
            wdir[2] /= dist;
#endif
            // construct jacobian.
            pjac = jacos;
            for (ieq=0; ieq<NEQ; ieq++) {
                for (jeq=0; jeq<NEQ; jeq++) {
                    jaco[jeq][ieq] = wdir[0]*pjac[0] + wdir[1]*pjac[1]
#if NDIM == 3
                        + wdir[2]*pjac[2]
#endif
                        ;
                    pjac += NDIM;
                };
            };
            // solve eigen problem.
            lapack_dgeev(jobvl, jobvr,
                    &exd->neq, (double *)jaco, &exd->neq, wr, wi,
                    (double *)evcl, &exd->neq, (double *)evcr, &exd->neq,
                    work, &lwork, &eiginfo);
            // bubble sort the eigenvalues and put result in indices.
            for (ieq=0; ieq<NEQ; ieq++) {
                argsort[ieq] = ieq;
            };
            nswap = 1;
            while (nswap) {
                nswap = 0;
                for (ieq=1; ieq<NEQ; ieq++) {
                    if (wr[argsort[ieq-1]] > wr[argsort[ieq]]) {
                        eiginfo = argsort[ieq];
                        argsort[ieq] = argsort[ieq-1];
                        argsort[ieq-1] = eiginfo;
                        nswap += 1;
                    };
                };
            };
            // calculate CFL number.
            cfl = hdt * wr[argsort[NEQ-1]] / dist;
            pcfl[0] = max(pcfl[0], cfl);
        };
        // advance.
        pcfl += 1;
        picecnd += (CLMFC+1) * NDIM;
        pclfcs += CLMFC+1;
    };
    times(&timm1);
    cputicks = (int)((timm1.tms_utime+timm1.tms_stime)
                   - (timm0.tms_utime+timm0.tms_stime));
    return cputicks;
};
// vim: set ts=4 et:

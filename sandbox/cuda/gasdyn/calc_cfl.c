/*
 * Copyright (C) 2010 Yung-Yu Chen <yyc@solvcon.net>.
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

#include "gasdyn.h"

int calc_cfl(exedata *exd, int istart, int iend) {
    int cputicks;
    struct tms timm0, timm1;
    int clnfc;
    // pointers.
    int *pclfcs;
    double *pamsca, *pcfl, *pocfl, *psoln, *picecnd, *pcecnd;
    // scalars.
    double hdt, dist, wspd, ga, ga1, pr, ke;
    // arrays.
    double vec[NDIM];
    // iterators.
    int icl, ifl;
    times(&timm0);
#ifdef SOLVCESE_FE
    feenableexcept(SOLVCESE_FE);
#endif
    hdt = exd->time_increment / 2.0;
    pamsca = exd->amsca + istart*NSCA;
    pcfl = exd->cfl + istart;
    pocfl = exd->ocfl + istart;
    psoln = exd->soln + istart*NEQ;
    picecnd = exd->cecnd + istart*(CLMFC+1)*NDIM;
    pclfcs = exd->clfcs + istart*(CLMFC+1);
    for (icl=istart; icl<iend; icl++) {
        // estimate distance.
        dist = 1.e200;
        pcecnd = picecnd;
        clnfc = pclfcs[0];
        for (ifl=1; ifl<=clnfc; ifl++) {
            pcecnd += NDIM;
            // distance.
            vec[0] = picecnd[0] - pcecnd[0];
            vec[1] = picecnd[1] - pcecnd[1];
#if NDIM == 3
            vec[2] = picecnd[2] - pcecnd[2];
#endif
            wspd = sqrt(vec[0]*vec[0] + vec[1]*vec[1]
#if NDIM == 3
                      + vec[2]*vec[2]
#endif
            );
            // minimal value.
            dist = min(wspd, dist);
        };
        // wave speed.
        ga = pamsca[0];
        ga1 = ga - 1.0;
        wspd = psoln[1]*psoln[1] + psoln[2]*psoln[2]
#if NDIM == 3
             + psoln[3]*psoln[3]
#endif
        ;
        ke = wspd/(2.0*psoln[0]);
        pr = ga1 * (psoln[1+NDIM] - ke);
#if SOLVCESE_DEBUG
        if (pr < 0.0) printf("%d: pr = %g\n", icl, pr);   // usual cause.
#endif
        pr = (pr+fabs(pr))/2.0;
        wspd = sqrt(ga*pr/psoln[0]) + sqrt(wspd)/psoln[0];
        // CFL.
        pocfl[0] = hdt*wspd/dist;
        // if pressure is null, make CFL to be 1.
        pcfl[0] = (pocfl[0]-1.0) * pr/(pr+SOLVCESE_TINY) + 1.0;
        // correct negative pressure.
        psoln[1+NDIM] = pr/ga1 + ke + SOLVCESE_TINY;
        // advance.
        pamsca += NSCA;
        pcfl += 1;
        pocfl += 1;
        psoln += NEQ;
        picecnd += (CLMFC+1)*NDIM;
        pclfcs += CLMFC+1;
    };
    times(&timm1);
    cputicks = (int)((timm1.tms_utime+timm1.tms_stime)
                   - (timm0.tms_utime+timm0.tms_stime));
    return cputicks;
};
// vim: set ts=4 et:

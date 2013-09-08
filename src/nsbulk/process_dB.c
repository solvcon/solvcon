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
// FIXME: this function shouldn't go to CUDA, doesn't make sense.
__global__ void cuda_process_dB(exedata *exd, double *predif,
        double *dB) {
    // and this starting index is incorrect.
    int istart = blockDim.x * blockIdx.x + threadIdx.x;
#else
int process_dB(exedata *exd, double *predif,
        double *dB) {
#ifdef SOLVCON_FE
    feenableexcept(SOLVCON_FE);
#endif
#endif
    // pointers.
    double *pclcnd, *pcecnd;
    double *pamsca, *psoln, *pdsoln;
    double (*pvd)[NDIM];    // shorthand for derivative.
    double *ppredif, *pdB;
    // scalars.
    double bulk, p0, rho0, eta, pref, pini, p, rho;
    double xmax, xmin, ymax, ymin;
    double time;
    double *pt;
    FILE *pre;
    // arrays.
    double sft[NDIM];
    // iterators.
    int icl;
#ifndef __CUDACC__
    #pragma omp parallel for private(pclcnd, pcecnd, pamsca, psoln, pdsoln,\
    ppredif, pdB, bulk, p0, rho0, eta, pref, pini, p, sft, icl, rho, pt)
    for (icl=-exd->ngstcell; icl<exd->ncell; icl++) {
#else
    icl = istart;
    if (icl < exd->ncell) {
#endif
        pclcnd = exd->clcnd + icl*NDIM;
        pcecnd = exd->cecnd + icl*(CLMFC+1)*NDIM;
        pamsca = exd->amsca + icl*NSCA;
        psoln = exd->soln + icl*NEQ;
        ppredif = predif + icl+exd->ngstcell;
        pdB = dB + icl+exd->ngstcell;
        // obtain flow parameters.
        bulk = pamsca[0];
        p0 = pamsca[1];
        rho0 = pamsca[2];
        eta = pamsca[3];
        pref = pamsca[4];
        pini = pamsca[5];
        xmax = pamsca[8];
        xmin = pamsca[9];
        ymax = pamsca[10];
        ymin = pamsca[11];
        pdsoln = exd->dsoln + icl*NEQ*NDIM;
        pvd = (double (*)[NDIM])pdsoln;
        pt = exd->cecnd + icl*(CLMFC+1)*NDIM;
        // shift from solution point to cell center.
        sft[0] = pclcnd[0] - pcecnd[0];
        sft[1] = pclcnd[1] - pcecnd[1];
#if NDIM == 3
        sft[2] = pclcnd[2] - pcecnd[2];
#endif
        // rho is density for density base, and is p' for pressure base
        rho = psoln[0] + pdsoln[0]*sft[0] + pdsoln[1]*sft[1];
#if NDIM == 3
        rho += pdsoln[2]*sft[2];
#endif
        // density base
        /*
        p = p0 + bulk*log(rho/rho0);
        ppredif[0] = p - pini;
        pdB[0] = 10 * log10(pow(ppredif[0]/pref,2));
        */
        // pressure base
        //
        p = bulk*log(rho);
        ppredif[0] = p - pini;
        pdB[0] = 10 * log10(pow(ppredif[0]/pref,2));
        //
        // cavity_parkhi
        /*
        if((pt[0]>0.1103 && pt[0]<0.1104) && (pt[1]>0.469 && pt[1]<0.47)) {
            time = exd->time;
            pre = fopen("pressure_parkhi.txt","a");
            fprintf(pre,"%.10lf %.10lf\n", ppredif[0], time);
            //printf("%lf %lf %lf\n", pt[0], pt[1], sqrt(pow(pt[0]-0.11,2)+pow(pt[1]-0.47,2)));
            fclose(pre);
        }
        */
        // cavity_parkhi1
        /*
        if((pt[0]>0.109 && pt[0]<0.1095) && (pt[1]>0.4705 && pt[1]<0.4706)) {
            time = exd->time;
            pre = fopen("pressure_parkhi.txt","a");
            fprintf(pre,"%.10lf %.10lf\n", ppredif[0], time);
            //printf("%lf %lf %lf\n", pt[0], pt[1], sqrt(pow(pt[0]-0.11,2)+pow(pt[1]-0.47,2)));
            fclose(pre);
        }
        */
        // cavity_ahuja_p88
        /*
        if((pt[0]>0.00019 && pt[0]<0.000198) && (pt[1]>3.6603 && pt[1]<3.66031)) {
            time = exd->time;
            pre = fopen("pressure_ahuja_p88.txt","a");
            fprintf(pre,"%.10lf %.10lf\n", ppredif[0], time);
            //printf("%lf %lf %lf\n", pt[0], pt[1], sqrt(pow(pt[0]-0.0,2)+pow(pt[1]-3.66,2)));
            fclose(pre);
        }
        */
        // cavity_ahuja_p88 short (height == 0.3)
        //
        //if((pt[0]>-0.000315 && pt[0]<-0.000311) && (pt[1]>0.24987 && pt[1]<0.24988)) {
        //    time = exd->time;
        //    pre = fopen("pressure_ahuja_p88_pt1.txt","a");
        //    fprintf(pre,"%.10lf %.10lf\n", ppredif[0], time);
            //printf("%lf %lf %lf\n", pt[0], pt[1], sqrt(pow(pt[0]-0.0,2)+pow(pt[1]-0.25,2)));
        //    fclose(pre);
        //}
        if((pt[0]>xmin && pt[0]<xmax) && (pt[1]>ymin && pt[1]<ymax)) {
            time = exd->time;
            pre = fopen("pressure.txt","a");
            fprintf(pre,"%.10lf %.10lf\n", ppredif[0], time);
            //printf("%lf %lf %lf\n", pt[0], pt[1],sqrt(pow(pt[0]-0.0,2)+pow(pt[1]-1.2,2)));
            fclose(pre);
        }
        //
        // cavity_ahuja_p88_cut
        /*
        if((pt[0]>-0.00052 && pt[0]<-0.00051) && (pt[1]>0.2496 && pt[1]<0.2497)) {
            time = exd->time;
            pre = fopen("pressure_ahuja_p88_cut.txt","a");
            fprintf(pre,"%.10lf %.10lf\n", ppredif[0], time);
            //printf("%lf %lf %lf\n", pt[0], pt[1], sqrt(pow(pt[0]-0.0,2)+pow(pt[1]-0.25,2)));
            fclose(pre);
        }
        */
#ifndef __CUDACC__
    };
    return 0;
};
#else
    };
};
extern "C" int process_physics(int nthread, exedata *exc, void *gexc,
        double gasconst,
        double *vel, double *vor, double *vorm, double *rho, double *pre,
        double *tem, double *ken, double *sos, double *mac) {
    int nblock = (exc->ncell + nthread-1) / nthread;
    cuda_process_physics<<<nblock, nthread>>>((exedata *)gexc, gasconst,
        vel, vor, vorm, rho, pre, tem, ken, sos, mac);
    cudaThreadSynchronize();
    return 0;
};
#endif

// vim: set ts=4 et:

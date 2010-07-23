// Copyright (C) 2008-2010 by Yung-Yu Chen.  See LICENSE.txt for terms of
// usage.
#include "solvcon.h"
int calc_dsoln(MeshData *msd, ExecutionData *exd,
        FPTYPE *clcnd, FPTYPE *dsol, FPTYPE *dsoln) {
    FPTYPE *pdsol, *pdsoln, *pclcnd;
    int icl, ieq, idm;
    pdsol = dsol + msd->ngstcell*exd->neq*msd->ndim;
    pdsoln = dsoln + msd->ngstcell*exd->neq*msd->ndim;
    pclcnd = clcnd + msd->ngstcell*msd->ndim;
    for (icl=0; icl<msd->ncell; icl++) {
        for (ieq=0; ieq<exd->neq; ieq++) {
            for (idm=0; idm<msd->ndim; idm++) {
                pdsoln[idm] = pdsol[idm]
                            + pclcnd[idm] * exd->time_increment / 2.0;
            };
            pdsol += msd->ndim;
            pdsoln += msd->ndim;
        };
        pclcnd += msd->ndim;
    };
    return 0;
};
// vim: set ts=4 et:

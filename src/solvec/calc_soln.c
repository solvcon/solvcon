// Copyright (C) 2008-2010 by Yung-Yu Chen.  See LICENSE.txt for terms of
// usage.
#include "solvcon.h"
int calc_soln(MeshData *msd, ExecutionData *exd,
        FPTYPE *clvol, FPTYPE *sol, FPTYPE *soln) {
    FPTYPE *psol, *psoln, *pclvol;
    int icl, ieq;
    psol = sol + msd->ngstcell*exd->neq;
    psoln = soln + msd->ngstcell*exd->neq;
    pclvol = clvol + msd->ngstcell;
    for (icl=0; icl<msd->ncell; icl++) {
        for (ieq=0; ieq<exd->neq; ieq++) {
            psoln[ieq] = psol[ieq] + pclvol[0] * exd->time_increment / 2.0;
        };
        psol += exd->neq;
        psoln += exd->neq;
        pclvol += 1;
    };
    return 0;
};
// vim: set ts=4 et:

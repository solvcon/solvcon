/*
 * Copyright (C) 2008-2011 Yung-Yu Chen <yyc@solvcon.net>.
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

#include "cuse.h"

double calc_tau_scale_mesh(exedata *exd, int icl) {
    int *pclfcs, *pfccls;
    double mq, mqa;
    int ifl, jcl;
    mq = exd->mqlty[icl] - exd->mqmin;
    mq = (mq+fabs(mq))/2;   // must be non-negative.
    mqa = mq;
    pclfcs = exd->clfcs + icl*(CLMFC+1);
    for (ifl=1; ifl<=pclfcs[0]; ifl++) {
        pfccls = exd->fccls + pclfcs[ifl]*FCREL;
        jcl = pfccls[0] + pfccls[1] - icl;
        mq = exd->mqlty[jcl] - exd->mqmin;
        mq = (mq+fabs(mq))/2;   // must be non-negative.
        mqa += mq;
    };
    mqa /= (pclfcs[0]+1);
    return exd->taumin + fabs(exd->cfl[icl]) * exd->tauscale
                       + mqa * exd->mqscale;
};
// vim: set ts=4 et:

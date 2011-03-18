/*
 * Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.
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

__global__ void cuda_slct_io(int down, int nelm, int stride,
        int *slct, char *garr, char *gbrr) {
    int ielm = blockDim.x * blockIdx.x + threadIdx.x;
    char *psrr, *pdrr;
    int it;
    if (ielm < nelm) {
        if (down == 0) {    // upload to device (through buffer).
            psrr = gbrr + ielm*stride;
            pdrr = garr + slct[ielm]*stride;
        } else {    // download from device (through buffer).
            psrr = garr + slct[ielm]*stride;
            pdrr = gbrr + ielm*stride;
        };
        for (it=0; it<stride; it++) {
            pdrr[it] = psrr[it];
        };
    };
};
extern "C" int slct_io(int nthread, int down, int nelm, int stride,
        int *slct, char *garr, char *gbrr) {
    int nblock = (nelm + nthread-1) / nthread;
    cuda_slct_io<<<nblock, nthread>>>(down, nelm, stride, slct, garr, gbrr);
    cudaThreadSynchronize();
    return 0;
};

// vim: set ft=cuda ts=4 et:

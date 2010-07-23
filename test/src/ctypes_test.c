// Copyright (C) 2008-2010 by Yung-Yu Chen.  See LICENSE.txt for terms of
// usage.
#include "ctypes.h"
void ctypes_test(int *par_i, double *par_d, int *n_i, int *par_arr_i,
		int *n_d2, int *n_d1, double *par_arr_d) {
	int it, jt;
	for (it=0; it<n_i[0]; it++) {
		par_arr_i[it] += par_i[0];
	};
	for (it=0; it<n_d1[0]; it++) {
		for (jt=0; jt<n_d2[0]; jt++) {
			par_arr_d[it*n_d2[0]+jt] += par_d[0];
		};
	};
	for (it=0; it<n_d2[0]; it++) {
		par_arr_d[it] -= par_d[0];
	};
	return;
};
// vim: set ts=4 et:

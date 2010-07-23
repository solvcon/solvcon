// Copyright (C) 2008-2010 by Yung-Yu Chen.  See LICENSE.txt for terms of
// usage.
#include "ctypes.h"
void ctypes_type_test(record *par_t, record *ret_t) {
	double *locarr;
	int it;
	ret_t->idx = -par_t->idx;
	for (it=0; it<10; it++) {
		ret_t->arr[it] = -par_t->arr[it];
	};
	locarr = par_t->arr_ptr;
	for (it=0; it<10; it++) {
		locarr[it] += 200.0;
	};
	return;
};
// vim: set ts=4 et:

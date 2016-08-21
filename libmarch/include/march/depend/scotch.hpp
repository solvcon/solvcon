#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

extern "C" {
#include <scotch.h>
#include <metis.h>
}

namespace march {

namespace depend {

namespace scotch {

typedef SCOTCH_Num num_type;

} /* end namespace scotch */

} /* end namespace depend */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:

#pragma once

/*
 * Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

extern "C" {
#include <scotch.h>

void METIS_PartGraphKway(
    const SCOTCH_Num * const
  , const SCOTCH_Num * const
  , const SCOTCH_Num * const
  , const SCOTCH_Num * const
  , const SCOTCH_Num * const
  , const SCOTCH_Num * const
  , const SCOTCH_Num * const
  , const SCOTCH_Num * const
  , const SCOTCH_Num * const
  , SCOTCH_Num * const
  , SCOTCH_Num * const
);
}

namespace march {

namespace depend {

namespace scotch {

using num_type = SCOTCH_Num;

} /* end namespace scotch */

} /* end namespace depend */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:

# -*- coding: UTF-8 -*-
#
# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
#
# All rights reserved.


from __future__ import absolute_import, division, print_function

import atexit

from IPython.utils import io


def _fix_ipython_utils_io():
    """
    Sometimes IPython.utils.io.devnull isn't closed at the end of a nose test
    session of SOLVCON, and I see::

        sys:1: ResourceWarning: unclosed file <_io.TextIOWrapper name='/dev/null' mode='w' encoding='UTF-8'>

    Register this function at interpreter exit.
    """
    if not io.devnull.closed:
        io.devnull.close()
atexit.register(_fix_ipython_utils_io)

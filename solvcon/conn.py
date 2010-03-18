# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Remote connection and communication.
"""

try:
    from multiprocessing import Process
except ImportError:
    from processing import Process
try:
    from multiprocessing.connection import Client
except ImportError:
    from processing.connection import Client
try:
    from multiprocessing.connection import Listener
except ImportError:
    from processing.connection import Listener

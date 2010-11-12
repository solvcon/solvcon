# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2010 Yung-Yu Chen <yyc@solvcon.net>.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""
Multi-threading.
"""

class ThreadPool(object):
    """
    A synchronized thread pool.  The number of pre-created threads is not
    changeable.

    @ivar nthread: number of threads in the pool.
    @itype nthread: int
    """
    def __init__(self, nthread):
        """
        @param nthread: number of threads for the pool.
        @type nthread: int
        """
        from thread import allocate_lock, start_new_thread
        self.func = None
        self.nthread = nthread
        self.__threadids = [None] * nthread
        self.__threads = [None] * nthread
        self.__returns = [None] * nthread
        for it in range(nthread):
            mlck = allocate_lock(); mlck.acquire()
            wlck = allocate_lock(); wlck.acquire()
            tdata = [mlck, wlck, None, None]
            self.__threads[it] = tdata
            tid = start_new_thread(self.eventloop, (tdata,))
            self.__threadids[it] = tid
    def eventloop(self, tdata):
        """
        Event loop for the pre-created threads.
        """
        from thread import exit
        while True:
            tdata[0].acquire()
            if tdata[2] == None:
                exit()
            else:
                tdata[3] = self.func(*tdata[2])
            tdata[1].release()
    def __call__(self, func, arglists):
        """
        @param func: a callable to be dispatched to the thread pool
        @type func: callable
        @param arglists: a list of arguments for the callable.
        @type arglists: list
        """
        self.func = func
        nthread = self.nthread
        it = 0
        while it < nthread:
            self.__returns[it] = None
            it += 1
        it = 0
        while it < nthread:
            tdata = self.__threads[it]
            tdata[2] = arglists[it]
            tdata[0].release()
            it += 1
        it = 0
        while it < nthread:
            tdata = self.__threads[it]
            tdata[1].acquire()
            self.__returns[it] = tdata[3]
            it += 1
        return self.__returns

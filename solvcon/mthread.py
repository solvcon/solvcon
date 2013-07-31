# -*- coding: UTF-8 -*-
#
# Copyright (c) 2008, Yung-Yu Chen <yyc@solvcon.net>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the SOLVCON nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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

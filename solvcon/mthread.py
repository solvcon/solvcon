# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2010 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Multi-threading.
"""

class ThreadPool(object):
    def __init__(self, nthread):
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
        from thread import exit
        while True:
            tdata[0].acquire()
            if tdata[2] == None:
                exit()
            else:
                tdata[3] = self.func(*tdata[2])
            tdata[1].release()
    def __call__(self, func, arglists):
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

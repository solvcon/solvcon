# -*- coding: UTF-8 -*-

from unittest import TestCase

class TestPool(TestCase):
    def work(self, istart, iend):
        return iend-istart
    def test_run_multi_thread(self):
        from ..mthread import ThreadPool
        ncore = 4
        tpool = ThreadPool(ncore)
        iter_end = 1000
        iter_start = -1000
        incre = (iter_end-iter_start)/ncore + 1
        args = list()
        istart = iter_start
        for it in range(ncore):
            iend = min(istart+incre, iter_end)
            args.append((istart, iend))
            istart = iend
        ret = tpool(self.work, args)
        self.assertEqual(sum(ret), iter_end-iter_start)

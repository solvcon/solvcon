# Copyright (c) 2024, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

import numpy as np

import solvcon


class Plot3dTC(unittest.TestCase):

    def test_plot3d_parsing(self):

        data = """1
2 2 2
0 0 0 0 1 1 1 1
0 0 1 1 0 0 1 1
0 1 0 1 0 1 0 1
"""
        plot3d_instance = solvcon.core.Plot3d(data.encode('utf-8'))
        blk = plot3d_instance.to_block()

        # Check nodes information
        self.assertEqual(blk.nnode, 8)
        # Due to ghost cell and ghost node had been created, the real body
        # had been shifted and start with index 24
        np.testing.assert_almost_equal(blk.ndcrd.ndarray[24:, :].tolist(),
                                       [[0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 1.0, 1.0],
                                        [1.0, 0.0, 0.0],
                                        [1.0, 0.0, 1.0],
                                        [1.0, 1.0, 0.0],
                                        [1.0, 1.0, 1.0],
                                        ])
        # Check cells information
        self.assertEqual(blk.ncell, 1)
        self.assertEqual(blk.cltpn.ndarray[6:].tolist(), [5])
        self.assertEqual(blk.clnds.ndarray[6:, :].tolist(),
                         [[8, 0, 2, 6, 4, 1, 3, 7, 5]])

    def test_multi_block_cells_stay_in_their_block(self):
        # Three disjoint blocks along x; the first has a different shape
        # so a per-block size cannot be treated as a cumulative offset.
        # Per block the file lists x, then y, then z of every node, with
        # the x grid index running fastest; the node at grid (i, j, k)
        # sits at that coordinate plus the block's x offset (0, 10, 20).
        data = """3
3 2 2
2 2 2
2 2 2
0 1 2 0 1 2 0 1 2 0 1 2
0 0 0 1 1 1 0 0 0 1 1 1
0 0 0 0 0 0 1 1 1 1 1 1
10 11 10 11 10 11 10 11
0 0 1 1 0 0 1 1
0 0 0 0 1 1 1 1
20 21 20 21 20 21 20 21
0 0 1 1 0 0 1 1
0 0 0 0 1 1 1 1
"""
        blk = solvcon.core.Plot3d(data.encode('utf-8')).to_block()

        self.assertEqual(blk.nnode, 28)
        self.assertEqual(blk.ncell, 4)
        # The body rows sit after the ghost rows, as in the test above.
        ndcrd = blk.ndcrd.ndarray[-28:, :]
        clnds = blk.clnds.ndarray[-4:, :]

        np.testing.assert_almost_equal(
            ndcrd.tolist(),
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
             [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0],
             [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0],
             [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 1.0, 1.0],
             [10.0, 0.0, 0.0], [11.0, 0.0, 0.0],
             [10.0, 1.0, 0.0], [11.0, 1.0, 0.0],
             [10.0, 0.0, 1.0], [11.0, 0.0, 1.0],
             [10.0, 1.0, 1.0], [11.0, 1.0, 1.0],
             [20.0, 0.0, 0.0], [21.0, 0.0, 0.0],
             [20.0, 1.0, 0.0], [21.0, 1.0, 0.0],
             [20.0, 0.0, 1.0], [21.0, 0.0, 1.0],
             [20.0, 1.0, 1.0], [21.0, 1.0, 1.0]])
        # Two cells in the 3x2x2 block, then one per 2x2x2 block, whose
        # ids start at the cumulative node counts 12 and 20.
        self.assertEqual(clnds.tolist(),
                         [[8, 0, 3, 9, 6, 1, 4, 10, 7],
                          [8, 1, 4, 10, 7, 2, 5, 11, 8],
                          [8, 12, 14, 18, 16, 13, 15, 19, 17],
                          [8, 20, 22, 26, 24, 21, 23, 27, 25]])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

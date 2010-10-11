# -*- coding: UTF-8 -*-

from unittest import TestCase

class TestSpanningTree(TestCase):
    def test_spanningtree(self):
        from ..connection import SpanningTreeNode
        # graph data.
        graph = [
            [1, 3, 5],
            [0, 2, 3, 4],
            [1, 4, 7],
            [0, 1, 5, 6],
            [1, 2, 6, 9],
            [0, 3, 8],
            [1, 3, 4, 8, 9],
            [2, 9],
            [5, 6],
            [4, 6, 7],
        ]
        # construct spanning tree.
        head = SpanningTreeNode(val=0, level=0)
        visited = dict()
        head.traverse(graph, visited)
        # test results.
        self.assertEqual(len(visited), len(graph))

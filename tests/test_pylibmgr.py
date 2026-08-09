# Copyright (c) 2024, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

import os

import sys

from solvcon import pylibmgr


class pylibmgrTC(unittest.TestCase):

    def test_pylibmgr_search_library_root(self):
        # This test case assumes that solvcon third-party lib root's name is
        # thirdparty, it is located at solvcon project root: /path/to/solvcon.
        pylibmgr.search_library_root(os.getcwd(), 'thirdparty')
        finder = next(finder for finder in sys.meta_path
                      if isinstance(finder, pylibmgr.SolvconPathFinder))
        self.assertNotEqual(finder.lib_paths, {})

        # Reset library patch record
        finder.lib_paths = {}

        pylibmgr.search_library_root(os.getcwd(), "Can_not_find")
        self.assertEqual(finder.lib_paths, {})

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

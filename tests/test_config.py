# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for the user-level configuration file.
"""


import os
import json
import tempfile
import unittest

from solvcon.config import Config


class ConfigPathTC(unittest.TestCase):
    """Where the default configuration file resolves."""

    #: Every variable the resolution consults, saved and restored per test.
    VARIABLES = ("SOLVCON_CONFIG_HOME", "XDG_CONFIG_HOME")

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self.VARIABLES}
        for name in self.VARIABLES:
            os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_honors_solvcon_config_home(self):
        os.environ["SOLVCON_CONFIG_HOME"] = "/sc/here"
        self.assertEqual(Config.config_home(), "/sc/here")
        self.assertEqual(Config.default_path(), "/sc/here/pilot.json")

    def test_solvcon_config_home_precedes_xdg(self):
        os.environ["SOLVCON_CONFIG_HOME"] = "/sc/here"
        os.environ["XDG_CONFIG_HOME"] = "/xdg/here"
        self.assertEqual(Config.config_home(), "/sc/here")

    def test_empty_solvcon_config_home_falls_through(self):
        os.environ["SOLVCON_CONFIG_HOME"] = ""
        os.environ["XDG_CONFIG_HOME"] = "/xdg/here"
        self.assertEqual(Config.config_home(), "/xdg/here/solvcon")

    def test_honors_xdg_config_home(self):
        os.environ["XDG_CONFIG_HOME"] = "/xdg/here"
        self.assertEqual(Config.config_home(), "/xdg/here/solvcon")
        self.assertEqual(
            Config.default_path(), "/xdg/here/solvcon/pilot.json")

    def test_falls_back_to_dot_config(self):
        home = os.path.expanduser("~")
        self.assertEqual(
            Config.config_home(),
            os.path.join(home, ".config", "solvcon"))

    def test_default_path_is_used_without_an_argument(self):
        self.assertEqual(Config().path, Config.default_path())


class ConfigIoTC(unittest.TestCase):
    """Reading, writing, and the resilience of both."""

    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.path = os.path.join(self._dir, "sub", "pilot.json")

    def tearDown(self):
        for root, _, files in os.walk(self._dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            os.rmdir(root)

    def test_missing_file_loads_empty(self):
        cfg = Config(self.path).load()
        self.assertIsNone(cfg.get("window"))
        self.assertEqual(cfg.get("window", "fallback"), "fallback")

    def test_save_creates_parent_directory(self):
        Config(self.path).set("window", {"width": 800}).save()
        self.assertTrue(os.path.isfile(self.path))

    def test_round_trip(self):
        Config(self.path).set(
            "window", {"width": 800, "height": 600, "x": 10, "y": 20}).save()
        reloaded = Config(self.path).load()
        self.assertEqual(
            reloaded.get("window"),
            {"width": 800, "height": 600, "x": 10, "y": 20})

    def test_save_leaves_no_temporary_file(self):
        Config(self.path).set("window", {"width": 800}).save()
        self.assertFalse(os.path.exists(self.path + ".tmp"))

    def test_corrupt_file_loads_empty(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w", encoding="utf-8") as fobj:
            fobj.write("{not valid json")
        cfg = Config(self.path).load()
        self.assertIsNone(cfg.get("window"))

    def test_non_object_file_loads_empty(self):
        os.makedirs(os.path.dirname(self.path))
        with open(self.path, "w", encoding="utf-8") as fobj:
            json.dump([1, 2, 3], fobj)
        cfg = Config(self.path).load()
        self.assertIsNone(cfg.get("window"))


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

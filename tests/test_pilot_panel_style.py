# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests that the pilot's dock panels take their colors from the palette.
"""

import re
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.panel import _profiling, _tree_panel
    from PySide6 import QtGui, QtWidgets
except ImportError:
    pilot = None

#: Two palettes far enough apart that no shade mixed from one can fall in the
#: other, so a color the rules name themselves shows up in both.
LIGHT = ("#f4f4f4", "#101010", "#ffffff", "#1a5fd0", "#ffffff")
DARK = ("#242424", "#e8e8e8", "#1a1a1a", "#6ea8ff", "#08121f")


def _palette(window, text, base, accent, on_accent):
    """A palette of the colors the panel rules are mixed from."""
    palette = QtGui.QPalette()
    for role, color in ((QtGui.QPalette.Window, window),
                        (QtGui.QPalette.WindowText, text),
                        (QtGui.QPalette.Base, base),
                        (QtGui.QPalette.Text, text),
                        (QtGui.QPalette.Highlight, accent),
                        (QtGui.QPalette.HighlightedText, on_accent)):
        palette.setColor(role, QtGui.QColor(color))
    return palette


def _colors(sheet):
    """Every color the style rules name."""
    return set(re.findall(r"#[0-9a-fA-F]{6}", sheet.lower()))


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class ProfilingResultTreeTC(unittest.TestCase):
    """That a profiling result is colored by the palette it is read under.

    The palette is set on the tree rather than switched on the application,
    which would repaint the whole pilot window; on macOS the native style
    faults painting one into a backing store that no screen is behind. What
    a theme switch does to the application palette is
    ``tests/gui/test_theme.py``'s subject.
    """

    @classmethod
    def setUpClass(cls):
        # No window needed, only a live QGuiApplication to hold a widget.
        pilot.RManager.instance.setUp()

    def _tree(self, palette):
        """A tree wearing the ``palette``.

        A fresh one each time, because the style sheet a tree applies for its
        first palette pins the colors Qt re-polishes it with, and a second
        palette on the same tree snaps back to them.
        """
        tree = _profiling._ResultTree()
        tree.setPalette(_palette(*palette))
        return tree

    def test_the_tree_mixes_its_shades_from_the_palette(self):
        # A color the rules name themselves survives both palettes, so it
        # shows up on both sides of this and the sets intersect.
        self.assertFalse(_colors(self._tree(LIGHT).styleSheet())
                         & _colors(self._tree(DARK).styleSheet()))

    def test_the_selected_row_wears_the_accent(self):
        tree = self._tree(DARK)
        accent = tree.palette().color(QtGui.QPalette.Highlight)
        self.assertIn(accent.name().lower(), _colors(tree.styleSheet()))


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class CollapsibleSectionTC(unittest.TestCase):
    """That the section header is colored by the palette it sits on."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def setUp(self):
        self.section = _tree_panel._CollapsibleSection(
            "Mesh", QtWidgets.QWidget())

    def _sheet(self, palette):
        self.section.setPalette(_palette(*palette))
        return self.section._toggle.styleSheet()

    def test_the_header_takes_its_hover_from_the_palette(self):
        # A translucent grey would read as a hover on either palette while
        # Qt mixed it against whatever sat behind it, not against the theme.
        self.assertNotIn("rgba", self._sheet(LIGHT))
        self.assertFalse(_colors(self._sheet(LIGHT))
                         & _colors(self._sheet(DARK)))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

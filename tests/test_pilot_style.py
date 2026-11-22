# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the colors and the number font the pilot's style module mixes.

The floors are the ones WCAG 2.1 asks, 4.5:1 for text, written out here
rather than read from the module under test.
"""

import unittest
from unittest import mock

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from PySide6.QtGui import QColor, QFont, QPalette
        from PySide6.QtWidgets import QWidget
        from solvcon.pilot import _style
except ImportError:
    pilot = None

#: The dark curated palette of ``cpp/solvcon/pilot/theme/theme.cpp``, whose
#: white on the accent falls short of 4.5:1.  The roles are named rather
#: than given as ``QPalette`` members, which a build without Qt lacks.
DARK = {'Window': '#2d2f33', 'WindowText': '#e6e6e7', 'Text': '#e6e6e7',
        'Base': '#232427', 'Highlight': '#3d82e0',
        'HighlightedText': '#ffffff'}


@testing.requires_pilot
class LegibleTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def setUp(self):
        palette = QPalette()
        for role, color in DARK.items():
            palette.setColor(getattr(QPalette, role), QColor(color))
        widget = QWidget()
        widget.setPalette(palette)
        self.shades = pilot.Shades(widget)

    def test_the_contrast_is_the_one_wcag_defines(self):
        black, white = QColor('#000000'), QColor('#ffffff')
        self.assertAlmostEqual(21.0, _style.contrast(black, white))
        self.assertAlmostEqual(1.0, _style.contrast(white, white))

    def test_a_short_color_steps_only_as_far_as_it_must(self):
        shades = self.shades
        self.assertLess(_style.contrast(shades.accent, shades.on_accent),
                        4.5)
        fill = pilot.legible(shades, shades.accent, shades.on_accent)
        ratio = _style.contrast(fill, shades.on_accent)
        self.assertGreaterEqual(ratio, 4.5)
        self.assertLess(ratio, 1.1 * 4.5)
        self.assertAlmostEqual(shades.accent.hueF(), fill.hueF(), delta=0.02)
        # A color that reads already is left as it is.
        self.assertEqual(shades.text.name(), pilot.legible(
            shades, shades.text, shades.panel).name())


@testing.requires_pilot
class NumberFontTC(unittest.TestCase):

    def test_the_number_font_sets_tabular_figures_on_a_copy(self):
        base = QFont()
        base.setPixelSize(17)
        font = pilot.number_font(base, 12, QFont.DemiBold)
        self.assertEqual((12, QFont.DemiBold), (font.pixelSize(),
                                                font.weight()))
        self.assertEqual(17, base.pixelSize())
        if hasattr(QFont, 'Tag'):
            self.assertEqual(1, font.featureValue(QFont.Tag("tnum")))

    def test_a_qt_without_font_features_keeps_the_figures(self):
        with mock.patch.object(_style, '_TABULAR', None):
            font = pilot.number_font(QFont(), 11)
        self.assertEqual(11, font.pixelSize())
        if hasattr(QFont, 'Tag'):
            self.assertFalse(font.isFeatureSet(QFont.Tag("tnum")))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

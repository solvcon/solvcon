# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.painter import _icons
    from PySide6 import QtGui
except ImportError:
    pilot = None


def _drawn_pixels(image):
    """The (x, y) of every pixel an icon actually painted."""
    return [(x, y)
            for x in range(image.width())
            for y in range(image.height())
            if image.pixelColor(x, y).alpha() > 0]


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class PainterIconTC(unittest.TestCase):
    """The tool icons rasterize to the box they are asked for.

    An unscaled test screen reports ratio 1, which hides the viewport trap
    ``render`` guards against, so the ratio is driven explicitly here rather
    than taken from wherever the suite happens to run.
    """

    _SIZE = 17

    def setUp(self):
        # No window needed, only a live QGuiApplication to hold a QPixmap.
        pilot.RManager.instance.setUp()

    def _ink(self, name, ratio):
        """The bounding box of an icon's drawn pixels as (x0, y0, x1, y1)."""
        image = _icons.render(
            name, QtGui.QColor("black"), self._SIZE, ratio).toImage()
        marks = _drawn_pixels(image)
        return (min(x for x, _y in marks), min(y for _x, y in marks),
                max(x for x, _y in marks), max(y for _x, y in marks))

    def test_icon_scales_with_the_device_pixel_ratio(self):
        for name in _icons.ICONS:
            with self.subTest(name=name):
                plain = self._ink(name, 1.0)
                scaled = self._ink(name, 2.0)
                # An icon drawn too large loses its far side, so the edge
                # checks below are the half that catches a bad viewport.
                for flat, deep in zip(plain, scaled):
                    self.assertAlmostEqual(deep, 2 * flat, delta=2)
                self.assertLess(scaled[2], 2 * self._SIZE - 1)
                self.assertLess(scaled[3], 2 * self._SIZE - 1)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

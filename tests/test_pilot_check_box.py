# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The check box of the app forms, whose box and check are drawn by the
pilot rather than by the platform style.
"""

import unittest

from solvcon import pilot
from solvcon import testing


def _near(color, other, slack=48):
    return all(abs(one - two) <= slack for one, two in
               zip(color.getRgb()[:3], other.getRgb()[:3]))


def _square(box):
    """Return the square the box of ``box`` is drawn in, in its own
    coordinates, whatever room the platform gives the indicator."""
    from PySide6.QtCore import QRectF
    from PySide6.QtWidgets import QStyle, QStyleOptionButton
    from solvcon.pilot.panel import _app_widgets
    option = QStyleOptionButton()
    box.initStyleOption(option)
    rect = box.style().subElementRect(QStyle.SE_CheckBoxIndicator, option,
                                      box)
    side = min(rect.width(), rect.height(), _app_widgets._CHECK_SIDE)
    square = QRectF(0, 0, side, side)
    square.moveCenter(QRectF(rect).center())
    return square


@testing.requires_pilot
class CheckBoxTC(unittest.TestCase):

    #: The pixel ratio a widget is rendered at, whatever the screen's.  A
    #: fixed one keeps the antialiased edge of the check from weighing more
    #: on a screen of coarse pixels.
    RATIO = 4

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def _render(self, widget):
        """Return ``widget`` rendered at :attr:`RATIO`."""
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QImage
        image = QImage(widget.size() * self.RATIO,
                       QImage.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(self.RATIO)
        image.fill(0)
        widget.render(image, QPoint())
        return image

    def _indicator(self, checked):
        """Return the pixels of the square the box of a check box is drawn
        in, and the colors of its palette it is filled and checked in."""
        from solvcon.pilot import panel
        box = panel.CheckBox("density")
        self.addCleanup(testing.dispose_widget, box)
        box.setChecked(checked)
        box.adjustSize()
        square = _square(box)
        image = self._render(box)
        ratio = self.RATIO
        pixels = [image.pixelColor(x, y)
                  for x in range(round(square.left() * ratio),
                                 round(square.right() * ratio))
                  for y in range(round(square.top() * ratio),
                                 round(square.bottom() * ratio))]
        shades = pilot.Shades(box)
        return pixels, shades.accent, shades.on_accent

    def test_a_checked_box_fills_with_the_accent_under_a_thin_check(self):
        pixels, accent, on_accent = self._indicator(True)
        fill = sum(_near(pixel, accent) for pixel in pixels)
        check = sum(_near(pixel, on_accent) for pixel in pixels)
        self.assertGreater(fill, len(pixels) // 2)
        self.assertGreater(check, 0)
        # The check, a tenth of the side thick, takes about an eighth as
        # many pixels as the fill around it.
        self.assertLess(check, fill * 0.2)

    def test_the_boxes_of_stacked_rows_keep_apart(self):
        # The macOS layout stacks rows closer than a check box is tall; the
        # boxes must still read one per row, not as one bar.
        from PySide6.QtWidgets import QGridLayout, QWidget
        from solvcon.pilot import panel
        form = QWidget()
        self.addCleanup(testing.dispose_widget, form)
        grid = QGridLayout(form)
        grid.setVerticalSpacing(4)
        boxes = [panel.CheckBox(text) for text in ("density", "pressure")]
        for row, box in enumerate(boxes):
            box.setChecked(True)
            grid.addWidget(box, row, 0)
        form.adjustSize()
        image = self._render(form)
        ratio = self.RATIO
        accent = pilot.Shades(boxes[0]).accent
        squares = [_square(box).translated(box.pos()) for box in boxes]
        # Left of where the check starts, so the line crosses only fill.
        x = round((squares[0].left() + 0.15 * squares[0].width()) * ratio)
        between = [_near(image.pixelColor(x, y), accent)
                   for y in range(round(squares[0].center().y() * ratio),
                                  round(squares[1].center().y() * ratio))]
        self.assertTrue(between[0] and between[-1])
        self.assertFalse(all(between))

    def test_an_unchecked_box_is_empty(self):
        pixels, accent, on_accent = self._indicator(False)
        self.assertFalse(any(_near(pixel, accent, slack=24)
                             for pixel in pixels))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

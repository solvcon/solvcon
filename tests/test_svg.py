# Copyright (c) 2025, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

import os

import numpy as np

from solvcon import testing
from solvcon.plot import svg


class SvgParserTB(testing.TestBase, unittest.TestCase):

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)


class SvgParserGeneralTC(SvgParserTB):

    def test_single_closed_path(self):
        d_attr = "M10 10 L60 10 L60 60 L10 60 Z"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        self.assertEqual(len(path_element.get_cmds()), 5)
        self.assertEqual(
            path_element.get_cmds(),
            [('M', [10.0, 10.0]),
             ('L', [60.0, 10.0]),
             ('L', [60.0, 60.0]),
             ('L', [10.0, 60.0]),
             ('Z', [])])

    def test_multiple_closed_paths(self):
        d_attr = "M10 10 L60 10 L60 60 L10 60 Z M100 10 L150 60 L100 60 Z"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        self.assertEqual(len(path_element.get_cmds()), 9)
        self.assertEqual(
            path_element.get_cmds(),
            [('M', [10.0, 10.0]),
             ('L', [60.0, 10.0]),
             ('L', [60.0, 60.0]),
             ('L', [10.0, 60.0]),
             ('Z', []),
             ('M', [100.0, 10.0]),
             ('L', [150.0, 60.0]),
             ('L', [100.0, 60.0]),
             ('Z', [])])


class SvgMoveToCommandTC(SvgParserTB):
    """
    Test 'M' and 'm' commands in SVG path 'd' attribute.
    See more: https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Attribute/d#moveto_path_commands
    """  # noqa: E501

    def test_moveto_absolute(self):
        d_attr = "M 10 10 h 10"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 1)
        self.assertEqual(len(cpad), 0)
        self.assertEqual(list(spad.x0), [10.0])
        self.assertEqual(list(spad.y0), [10.0])
        self.assertEqual(list(spad.x1), [20.0])
        self.assertEqual(list(spad.y1), [10.0])

    def test_moveto_relative(self):
        d_attr = "M 10 10 h 10 m 0 10 h 10"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 2)
        self.assertEqual(len(cpad), 0)
        self.assertEqual(list(spad.x0), [10.0, 20.0])
        self.assertEqual(list(spad.y0), [10.0, 20.0])
        self.assertEqual(list(spad.x1), [20.0, 30.0])
        self.assertEqual(list(spad.y1), [10.0, 20.0])

    def test_moveto_absolute_and_relative(self):
        d_attr = "M 10 10 h 10 m  0 10 h 10"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 2)
        self.assertEqual(len(cpad), 0)
        self.assertEqual(list(spad.x0), [10.0, 20.0])
        self.assertEqual(list(spad.y0), [10.0, 20.0])
        self.assertEqual(list(spad.x1), [20.0, 30.0])
        self.assertEqual(list(spad.y1), [10.0, 20.0])

    def test_moveto_implicit_lineto(self):
        d_attr = "M 10 10 20 10 m 0 10 10 0"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 2)
        self.assertEqual(len(cpad), 0)
        self.assertEqual(list(spad.x0), [10.0, 20.0])
        self.assertEqual(list(spad.y0), [10.0, 20.0])
        self.assertEqual(list(spad.x1), [20.0, 30.0])
        self.assertEqual(list(spad.y1), [10.0, 20.0])


class SvgLineToCommandTC(SvgParserTB):
    """
    Test 'L', 'l', 'H', 'h', 'V', and 'v' commands in SVG path 'd' attribute.
    See more: https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Attribute/d#lineto_path_commands
    """  # noqa: E501

    def test_lineto_absolute(self):
        d_attr = "M 10 10 L 30 20 L 50 30"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 2)
        self.assertEqual(len(cpad), 0)
        self.assertEqual(list(spad.x0), [10.0, 30.0])
        self.assertEqual(list(spad.y0), [10.0, 20.0])
        self.assertEqual(list(spad.x1), [30.0, 50.0])
        self.assertEqual(list(spad.y1), [20.0, 30.0])

    def test_lineto_relative(self):
        d_attr = "M 10 10 l 20 10 l 20 10"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 2)
        self.assertEqual(len(cpad), 0)
        self.assertEqual(list(spad.x0), [10.0, 30.0])
        self.assertEqual(list(spad.y0), [10.0, 20.0])
        self.assertEqual(list(spad.x1), [30.0, 50.0])
        self.assertEqual(list(spad.y1), [20.0, 30.0])

    def test_horizontal_lineto_absolute(self):
        d_attr = "M 10 10 H 30 H 50"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 2)
        self.assertEqual(len(cpad), 0)
        self.assertEqual(list(spad.x0), [10.0, 30.0])
        self.assertEqual(list(spad.y0), [10.0, 10.0])
        self.assertEqual(list(spad.x1), [30.0, 50.0])
        self.assertEqual(list(spad.y1), [10.0, 10.0])

    def test_horizontal_lineto_relative(self):
        d_attr = "M 10 10 h 20 h 20"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 2)
        self.assertEqual(len(cpad), 0)
        self.assertEqual(list(spad.x0), [10.0, 30.0])
        self.assertEqual(list(spad.y0), [10.0, 10.0])
        self.assertEqual(list(spad.x1), [30.0, 50.0])
        self.assertEqual(list(spad.y1), [10.0, 10.0])

    def test_vertical_lineto_absolute(self):
        d_attr = "M 10 10 V 30 V 50"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 2)
        self.assertEqual(len(cpad), 0)
        self.assertEqual(list(spad.x0), [10.0, 10.0])
        self.assertEqual(list(spad.y0), [10.0, 30.0])
        self.assertEqual(list(spad.x1), [10.0, 10.0])
        self.assertEqual(list(spad.y1), [30.0, 50.0])

    def test_vertical_lineto_relative(self):
        d_attr = "M 10 10 v 20 v 20"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 2)
        self.assertEqual(len(cpad), 0)
        self.assertEqual(list(spad.x0), [10.0, 10.0])
        self.assertEqual(list(spad.y0), [10.0, 30.0])
        self.assertEqual(list(spad.x1), [10.0, 10.0])
        self.assertEqual(list(spad.y1), [30.0, 50.0])

    def test_lineto_all_variants(self):
        d_attr = "M 10 10 L 20 10 l 10 0 V 20 v 10 H 20 h -10"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 6)
        self.assertEqual(len(cpad), 0)
        self.assertEqual(list(spad.x0), [10.0, 20.0, 30.0, 30.0, 30.0, 20.0])
        self.assertEqual(list(spad.y0), [10.0, 10.0, 10.0, 20.0, 30.0, 30.0])
        self.assertEqual(list(spad.x1), [20.0, 30.0, 30.0, 30.0, 20.0, 10.0])
        self.assertEqual(list(spad.y1), [10.0, 10.0, 20.0, 30.0, 30.0, 30.0])

    def test_lineto_multiple_coordinates(self):
        d_attr = "M 10 10 L 20 10 30 10 l 10 0 10 0 V 20 30 v 10 10 H 40 30 h -10 -10"  # noqa: E501
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 12)
        self.assertEqual(len(cpad), 0)
        self.assertEqual(list(spad.x0), [10.0, 20.0, 30.0, 40.0, 50.0, 50.0,
                                         50.0, 50.0, 50.0, 40.0, 30.0, 20.0])
        self.assertEqual(list(spad.y0), [10.0, 10.0, 10.0, 10.0, 10.0, 20.0,
                                         30.0, 40.0, 50.0, 50.0, 50.0, 50.0])
        self.assertEqual(list(spad.x1), [20.0, 30.0, 40.0, 50.0, 50.0, 50.0,
                                         50.0, 50.0, 40.0, 30.0, 20.0, 10.0])
        self.assertEqual(list(spad.y1), [10.0, 10.0, 10.0, 10.0, 20.0, 30.0,
                                         40.0, 50.0, 50.0, 50.0, 50.0, 50.0])


class SvgCubicBezierCurveCommandTC(SvgParserTB):
    """
    Test 'C', 'c', 'S', and 's' commands in SVG path 'd' attribute.
    See more: https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Attribute/d#cubic_b%C3%A9zier_curve
    """  # noqa: E501

    def test_absolute(self):
        d_attr = "M 10 90 C 30 90 25 10 50 10"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 0)
        self.assertEqual(len(cpad), 1)
        self.assertEqual(list(cpad[0][0]), [10, 90, 0])
        self.assertEqual(list(cpad[0][1]), [30, 90, 0])
        self.assertEqual(list(cpad[0][2]), [25, 10, 0])
        self.assertEqual(list(cpad[0][3]), [50, 10, 0])

    def test_relative(self):
        d_attr = "M 10 90 c 20 0 15 -80 40 -80"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 0)
        self.assertEqual(len(cpad), 1)
        self.assertEqual(list(cpad[0][0]), [10, 90, 0])
        self.assertEqual(list(cpad[0][1]), [30, 90, 0])
        self.assertEqual(list(cpad[0][2]), [25, 10, 0])
        self.assertEqual(list(cpad[0][3]), [50, 10, 0])

    def test_smooth_cubic_bezier_absolute(self):
        d_attr = "M 10 90 C 30 90 25 10 50 10 S 70 90 90 90"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 0)
        self.assertEqual(len(cpad), 2)
        self.assertEqual(list(cpad[1][0]), [50, 10, 0])
        self.assertEqual(list(cpad[1][1]), [75, 10, 0])
        self.assertEqual(list(cpad[1][2]), [70, 90, 0])
        self.assertEqual(list(cpad[1][3]), [90, 90, 0])

    def test_smooth_cubic_bezier_relative(self):
        d_attr = "M 10 90 C 30 90 25 10 50 10 s 20 80 40 80"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 0)
        self.assertEqual(len(cpad), 2)
        self.assertEqual(list(cpad[1][0]), [50, 10, 0])
        self.assertEqual(list(cpad[1][1]), [75, 10, 0])
        self.assertEqual(list(cpad[1][2]), [70, 90, 0])
        self.assertEqual(list(cpad[1][3]), [90, 90, 0])

    def test_cubic_bezier_basic(self):
        d_attr = "M 10 90 C 30 90 25 10 50 10 S 70 90 90 90 c 20 0 15 -80 40 -80 s 20 80 40 80"  # noqa: E501
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 0)
        self.assertEqual(len(cpad), 4)
        self.assertEqual(list(cpad[0][0]), [10, 90, 0])
        self.assertEqual(list(cpad[0][1]), [30, 90, 0])
        self.assertEqual(list(cpad[0][2]), [25, 10, 0])
        self.assertEqual(list(cpad[0][3]), [50, 10, 0])

        self.assertEqual(list(cpad[1][0]), [50, 10, 0])
        self.assertEqual(list(cpad[1][1]), [75, 10, 0])
        self.assertEqual(list(cpad[1][2]), [70, 90, 0])
        self.assertEqual(list(cpad[1][3]), [90, 90, 0])

        self.assertEqual(list(cpad[2][0]), [90, 90, 0])
        self.assertEqual(list(cpad[2][1]), [110, 90, 0])
        self.assertEqual(list(cpad[2][2]), [105, 10, 0])
        self.assertEqual(list(cpad[2][3]), [130, 10, 0])

        self.assertEqual(list(cpad[3][0]), [130, 10, 0])
        self.assertEqual(list(cpad[3][1]), [155, 10, 0])
        self.assertEqual(list(cpad[3][2]), [150, 90, 0])
        self.assertEqual(list(cpad[3][3]), [170, 90, 0])

    def test_cubic_bezier_implicit_curves(self):
        d_attr = "M 10 90 C 30 90 25 10 50 10 75 10 70 90 90 90 S 105 10 130 10 150 90 170 90 c 20 0 15 -80 40 -80 25 0 20 80 40 80 s 15 -80 40 -80 20 80 40 80"  # noqa: E501
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 0)
        self.assertEqual(len(cpad), 8)

        # two "C" curves
        self.assertEqual(list(cpad[0][0]), [10, 90, 0])
        self.assertEqual(list(cpad[0][1]), [30, 90, 0])
        self.assertEqual(list(cpad[0][2]), [25, 10, 0])
        self.assertEqual(list(cpad[0][3]), [50, 10, 0])

        self.assertEqual(list(cpad[1][0]), [50, 10, 0])
        self.assertEqual(list(cpad[1][1]), [75, 10, 0])
        self.assertEqual(list(cpad[1][2]), [70, 90, 0])
        self.assertEqual(list(cpad[1][3]), [90, 90, 0])

        # two "S" curves
        self.assertEqual(list(cpad[2][0]), [90, 90, 0])
        self.assertEqual(list(cpad[2][1]), [110, 90, 0])
        self.assertEqual(list(cpad[2][2]), [105, 10, 0])
        self.assertEqual(list(cpad[2][3]), [130, 10, 0])

        self.assertEqual(list(cpad[3][0]), [130, 10, 0])
        self.assertEqual(list(cpad[3][1]), [155, 10, 0])
        self.assertEqual(list(cpad[3][2]), [150, 90, 0])
        self.assertEqual(list(cpad[3][3]), [170, 90, 0])

        # two "c" curves
        self.assertEqual(list(cpad[4][0]), [170, 90, 0])
        self.assertEqual(list(cpad[4][1]), [190, 90, 0])
        self.assertEqual(list(cpad[4][2]), [185, 10, 0])
        self.assertEqual(list(cpad[4][3]), [210, 10, 0])

        self.assertEqual(list(cpad[5][0]), [210, 10, 0])
        self.assertEqual(list(cpad[5][1]), [235, 10, 0])
        self.assertEqual(list(cpad[5][2]), [230, 90, 0])
        self.assertEqual(list(cpad[5][3]), [250, 90, 0])

        # two "s" curves
        self.assertEqual(list(cpad[6][0]), [250, 90, 0])
        self.assertEqual(list(cpad[6][1]), [270, 90, 0])
        self.assertEqual(list(cpad[6][2]), [265, 10, 0])
        self.assertEqual(list(cpad[6][3]), [290, 10, 0])

        self.assertEqual(list(cpad[7][0]), [290, 10, 0])
        self.assertEqual(list(cpad[7][1]), [315, 10, 0])
        self.assertEqual(list(cpad[7][2]), [310, 90, 0])
        self.assertEqual(list(cpad[7][3]), [330, 90, 0])

    def test_cubic_bezier_point_continuity(self):
        d_attr = "M 10 90 C 30 90 25 10 50 10 S 70 90 90 90 c 20 0 15 -80 40 -80 s 20 80 40 80"  # noqa: E501
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 0)
        self.assertEqual(len(cpad), 4)

        for i in range(len(cpad) - 1):
            self.assert_allclose(cpad[i][3][0], cpad[i + 1][0][0])
            self.assert_allclose(cpad[i][3][1], cpad[i + 1][0][1])
            self.assert_allclose(cpad[i][3][2], cpad[i + 1][0][2])


class SvgQuadraticBezierCurveCommandTC(SvgParserTB):
    """
    Test 'Q', 'q', 'T', and 't' commands in SVG path 'd' attribute.
    See more: https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Attribute/d#quadratic_b%C3%A9zier_curve
    """  # noqa: E501

    def test_quadratic_bezier_absolute(self):
        d_attr = "M 10 50 Q 25 25 40 50"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 0)
        self.assertEqual(len(cpad), 1)
        self.assertEqual(list(cpad[0][0]), [10, 50, 0])
        self.assertEqual(list(cpad[0][1]), [25, 25, 0])
        self.assertEqual(list(cpad[0][2]), [40, 50, 0])
        self.assertEqual(list(cpad[0][3]), [40, 50, 0])

    def test_quadratic_bezier_relative(self):
        d_attr = "M 10 50 q 15 -25 30 0"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 0)
        self.assertEqual(len(cpad), 1)
        self.assertEqual(list(cpad[0][0]), [10, 50, 0])
        self.assertEqual(list(cpad[0][1]), [25, 25, 0])
        self.assertEqual(list(cpad[0][2]), [40, 50, 0])
        self.assertEqual(list(cpad[0][3]), [40, 50, 0])

    def test_smooth_quadratic_bezier_absolute(self):
        d_attr = "M 10 50 Q 25 25 40 50 T 70 50"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 0)
        self.assertEqual(len(cpad), 2)
        self.assertEqual(list(cpad[1][0]), [40, 50, 0])
        self.assertEqual(list(cpad[1][1]), [55, 75, 0])
        self.assertEqual(list(cpad[1][2]), [70, 50, 0])
        self.assertEqual(list(cpad[1][3]), [70, 50, 0])

    def test_smooth_quadratic_bezier_relative(self):
        d_attr = "M 10 50 Q 25 25 40 50 t 30 0"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 0)
        self.assertEqual(len(cpad), 2)
        self.assertEqual(list(cpad[1][0]), [40, 50, 0])
        self.assertEqual(list(cpad[1][1]), [55, 75, 0])
        self.assertEqual(list(cpad[1][2]), [70, 50, 0])
        self.assertEqual(list(cpad[1][3]), [70, 50, 0])

    def test_quadratic_bezier_basic(self):
        d_attr = "M 10 50 Q 25 25 40 50 q 15 25 30 0 T 100 50 t 30 0"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 0)
        self.assertEqual(len(cpad), 4)

        self.assertEqual(list(cpad[0][0]), [10, 50, 0])
        self.assertEqual(list(cpad[0][1]), [25, 25, 0])
        self.assertEqual(list(cpad[0][2]), [40, 50, 0])
        self.assertEqual(list(cpad[0][3]), [40, 50, 0])

        self.assertEqual(list(cpad[1][0]), [40, 50, 0])
        self.assertEqual(list(cpad[1][1]), [55, 75, 0])
        self.assertEqual(list(cpad[1][2]), [70, 50, 0])
        self.assertEqual(list(cpad[1][3]), [70, 50, 0])

        self.assertEqual(list(cpad[2][0]), [70, 50, 0])
        self.assertEqual(list(cpad[2][1]), [85, 25, 0])
        self.assertEqual(list(cpad[2][2]), [100, 50, 0])
        self.assertEqual(list(cpad[2][3]), [100, 50, 0])

        self.assertEqual(list(cpad[3][0]), [100, 50, 0])
        self.assertEqual(list(cpad[3][1]), [115, 75, 0])
        self.assertEqual(list(cpad[3][2]), [130, 50, 0])
        self.assertEqual(list(cpad[3][3]), [130, 50, 0])

    def test_quadratic_bezier_implicit_curves(self):
        d_attr = "M 10 50 Q 25 25 40 50 55 75 70 50 q 15 -25 30 0 15 25 30 0 T 160 50 190 50 t 30 0 30 0"  # noqa: E501
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 0)
        self.assertEqual(len(cpad), 8)

        # two 'Q' curves
        self.assertEqual(list(cpad[0][0]), [10, 50, 0])
        self.assertEqual(list(cpad[0][1]), [25, 25, 0])
        self.assertEqual(list(cpad[0][2]), [40, 50, 0])
        self.assertEqual(list(cpad[0][3]), [40, 50, 0])

        self.assertEqual(list(cpad[1][0]), [40, 50, 0])
        self.assertEqual(list(cpad[1][1]), [55, 75, 0])
        self.assertEqual(list(cpad[1][2]), [70, 50, 0])
        self.assertEqual(list(cpad[1][3]), [70, 50, 0])

        # two 'q' curves
        self.assertEqual(list(cpad[2][0]), [70, 50, 0])
        self.assertEqual(list(cpad[2][1]), [85, 25, 0])
        self.assertEqual(list(cpad[2][2]), [100, 50, 0])
        self.assertEqual(list(cpad[2][3]), [100, 50, 0])

        self.assertEqual(list(cpad[3][0]), [100, 50, 0])
        self.assertEqual(list(cpad[3][1]), [115, 75, 0])
        self.assertEqual(list(cpad[3][2]), [130, 50, 0])
        self.assertEqual(list(cpad[3][3]), [130, 50, 0])

        # two 'T' curves
        self.assertEqual(list(cpad[4][0]), [130, 50, 0])
        self.assertEqual(list(cpad[4][1]), [145, 25, 0])
        self.assertEqual(list(cpad[4][2]), [160, 50, 0])
        self.assertEqual(list(cpad[4][3]), [160, 50, 0])

        self.assertEqual(list(cpad[5][0]), [160, 50, 0])
        self.assertEqual(list(cpad[5][1]), [175, 75, 0])
        self.assertEqual(list(cpad[5][2]), [190, 50, 0])
        self.assertEqual(list(cpad[5][3]), [190, 50, 0])

        # two 't' curves
        self.assertEqual(list(cpad[6][0]), [190, 50, 0])
        self.assertEqual(list(cpad[6][1]), [205, 25, 0])
        self.assertEqual(list(cpad[6][2]), [220, 50, 0])
        self.assertEqual(list(cpad[6][3]), [220, 50, 0])

        self.assertEqual(list(cpad[7][0]), [220, 50, 0])
        self.assertEqual(list(cpad[7][1]), [235, 75, 0])
        self.assertEqual(list(cpad[7][2]), [250, 50, 0])
        self.assertEqual(list(cpad[7][3]), [250, 50, 0])

    def test_quadratic_bezier_point_continuity(self):
        d_attr = "M 10 50 Q 25 25 40 50 q 15 25 30 0 T 100 50 t 30 0"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 0)
        self.assertEqual(len(cpad), 4)

        for i in range(len(cpad) - 1):
            self.assert_allclose(cpad[i][2][0], cpad[i + 1][0][0])
            self.assert_allclose(cpad[i][2][1], cpad[i + 1][0][1])
            self.assert_allclose(cpad[i][2][2], cpad[i + 1][0][2])


class SvgEllipticalArcCurveCommandTC(SvgParserTB):
    """
    Test 'A' and 'a' commands in SVG path 'd' attribute.
    See more: https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/d#elliptical_arc_curve
    """  # noqa: E501

    def test_arc_basic(self):
        d_attr = "M 6 10 A 6 4 10 1 0 14 10 A 6 4 10 0 1 20 10"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 39 * 2)
        self.assertEqual(len(cpad), 0)

        self.assert_allclose(spad.x0_at(0), 6)
        self.assert_allclose(spad.y0_at(0), 10)
        self.assert_allclose(spad.x1_at(77), 20)
        self.assert_allclose(spad.y1_at(77), 10)

        self.assert_allclose(spad.x1_at(38), 14)
        self.assert_allclose(spad.y1_at(38), 10)

        for i in range(len(spad) - 1):
            self.assert_allclose(spad.x1_at(i), spad.x0_at(i + 1))
            self.assert_allclose(spad.y1_at(i), spad.y0_at(i + 1))

    def test_arc_implicit_curves(self):
        d_attr = "M 6 10 A 6 4 10 1 0 14 10 6 4 10 0 1 20 10"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 39 * 2)
        self.assertEqual(len(cpad), 0)

        self.assert_allclose(spad.x0_at(0), 6)
        self.assert_allclose(spad.y0_at(0), 10)
        self.assert_allclose(spad.x1_at(77), 20)
        self.assert_allclose(spad.y1_at(77), 10)

        self.assert_allclose(spad.x1_at(38), 14)
        self.assert_allclose(spad.y1_at(38), 10)

        for i in range(len(spad) - 1):
            self.assert_allclose(spad.x1_at(i), spad.x0_at(i + 1))
            self.assert_allclose(spad.y1_at(i), spad.y0_at(i + 1))

    def test_arc_relative(self):
        d_attr = "M 6 10 a 6 4 10 1 0 8 0 a 6 4 10 0 1 6 0"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 39 * 2)
        self.assertEqual(len(cpad), 0)

        self.assert_allclose(spad.x0_at(0), 6)
        self.assert_allclose(spad.y0_at(0), 10)
        self.assert_allclose(spad.x1_at(77), 20)
        self.assert_allclose(spad.y1_at(77), 10)

        self.assert_allclose(spad.x1_at(38), 14)
        self.assert_allclose(spad.y1_at(38), 10)

        for i in range(len(spad) - 1):
            self.assert_allclose(spad.x1_at(i), spad.x0_at(i + 1))
            self.assert_allclose(spad.y1_at(i), spad.y0_at(i + 1))

    def test_arc_mixed_absolute_relative(self):
        d_attr = "M 6 10 A 6 4 10 1 0 14 10 a 6 4 10 0 1 6 0"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 39 * 2)
        self.assertEqual(len(cpad), 0)

        self.assert_allclose(spad.x0_at(0), 6)
        self.assert_allclose(spad.y0_at(0), 10)
        self.assert_allclose(spad.x1_at(77), 20)
        self.assert_allclose(spad.y1_at(77), 10)

        self.assert_allclose(spad.x1_at(38), 14)
        self.assert_allclose(spad.y1_at(38), 10)

        for i in range(len(spad) - 1):
            self.assert_allclose(spad.x1_at(i), spad.x0_at(i + 1))
            self.assert_allclose(spad.y1_at(i), spad.y0_at(i + 1))

    def test_arc_point_continuity(self):
        d_attr = "M 0 0 A 10 10 0 0 1 20 0"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 39)
        self.assertEqual(len(cpad), 0)

        self.assert_allclose(spad.x0_at(0), 0, atol=1e-14)
        self.assert_allclose(spad.y0_at(0), 0, atol=1e-14)
        self.assert_allclose(spad.x1_at(38), 20, atol=1e-14)
        self.assert_allclose(spad.y1_at(38), 0, atol=1e-14)

        for i in range(len(spad) - 1):
            self.assert_allclose(spad.x1_at(i), spad.x0_at(i + 1))
            self.assert_allclose(spad.y1_at(i), spad.y0_at(i + 1))

    def test_arc_circular(self):
        d_attr = "M 100 100 A 50 50 0 1 1 200 100"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 39)
        self.assertEqual(len(cpad), 0)

        self.assert_allclose(spad.x0_at(0), 100)
        self.assert_allclose(spad.y0_at(0), 100)
        self.assert_allclose(spad.x1_at(38), 200)
        self.assert_allclose(spad.y1_at(38), 100)

        center_x = 150
        center_y = 100
        radius = 50

        for i in range(len(spad)):
            x0 = spad.x0_at(i)
            y0 = spad.y0_at(i)
            distance = ((x0 - center_x)**2 + (y0 - center_y)**2)**0.5
            self.assert_allclose(distance, radius, rtol=1e-2)

            x1 = spad.x1_at(i)
            y1 = spad.y1_at(i)
            distance = ((x1 - center_x)**2 + (y1 - center_y)**2)**0.5
            self.assert_allclose(distance, radius, rtol=1e-2)


class SvgPathCommandTC(SvgParserTB):
    """
    Test 'Z' and 'z' commands in SVG path 'd' attribute.
    See more: https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/d#clossvg.epath
    """  # noqa: E501

    def test_close_path_uppercase(self):
        d_attr = "M 10 10 L 20 10 l 10 0 Z"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad, cpad = path_element.spads[0], path_element.cpads[0]

        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 3)
        self.assertEqual(len(cpad), 0)
        self.assertEqual(list(spad.x0), [10.0, 20.0, 30.0])
        self.assertEqual(list(spad.y0), [10.0, 10.0, 10.0])
        self.assertEqual(list(spad.x1), [20.0, 30.0, 10.0])
        self.assertEqual(list(spad.y1), [10.0, 10.0, 10.0])

    def test_close_path_lowercase(self):
        d_attr = "M 10 10 L 20 10 l 10 0 z"
        fill_attr = "none"
        path_element = svg.EPath(d_attr=d_attr, fill_attr=fill_attr)
        spad = path_element.spads[0]
        cpad = path_element.cpads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(cpad.ndim, 2)
        self.assertEqual(len(spad), 3)
        self.assertEqual(len(cpad), 0)
        self.assertEqual(list(spad.x0), [10.0, 20.0, 30.0])
        self.assertEqual(list(spad.y0), [10.0, 10.0, 10.0])
        self.assertEqual(list(spad.x1), [20.0, 30.0, 10.0])
        self.assertEqual(list(spad.y1), [10.0, 10.0, 10.0])


class SvgTransformAttributeTC(SvgParserTB):
    """
    Test EShapeBase._parse_transform_attribute() for the SVG `transform`
    attribute.
    See more: https://www.w3.org/TR/css-transforms-1/#svg-syntax
    """

    def _parse(self, transform_attr):
        shape = svg.EShapeBase(transform_attr=transform_attr)
        return shape._parse_transform_attribute()

    def test_empty_string(self):
        self.assertEqual(self._parse(""), [])

    def test_matrix(self):
        result = self._parse("matrix(1,0,0,1,10,20)")
        self.assertEqual(
            result, [('matrix', [1.0, 0.0, 0.0, 1.0, 10.0, 20.0])])

    def test_translate_one_arg(self):
        result = self._parse("translate(10)")
        self.assertEqual(result, [('translate', [10.0])])

    def test_translate_two_args(self):
        result = self._parse("translate(10, 20)")
        self.assertEqual(result, [('translate', [10.0, 20.0])])

    def test_translate_comma_only(self):
        result = self._parse("translate(10,20)")
        self.assertEqual(result, [('translate', [10.0, 20.0])])

    def test_translate_space_only(self):
        result = self._parse("translate(10 20)")
        self.assertEqual(result, [('translate', [10.0, 20.0])])

    def test_scale_one_arg(self):
        result = self._parse("scale(2)")
        self.assertEqual(result, [('scale', [2.0])])

    def test_scale_two_args(self):
        result = self._parse("scale(2, 3)")
        self.assertEqual(result, [('scale', [2.0, 3.0])])

    def test_rotate_one_arg(self):
        result = self._parse("rotate(45)")
        self.assertEqual(result, [('rotate', [45.0])])

    def test_rotate_three_args(self):
        result = self._parse("rotate(45, 10, 20)")
        self.assertEqual(result, [('rotate', [45.0, 10.0, 20.0])])

    def test_skewx(self):
        result = self._parse("skewX(30)")
        self.assertEqual(result, [('skewX', [30.0])])

    def test_skewy(self):
        result = self._parse("skewY(15)")
        self.assertEqual(result, [('skewY', [15.0])])

    def test_negative_and_decimal_numbers(self):
        result = self._parse("translate(-10.5, .5)")
        self.assertEqual(result, [('translate', [-10.5, 0.5])])

    def test_scientific_notation_numbers(self):
        result = self._parse("translate(1e2, -2.5e-3)")
        self.assertEqual(result, [('translate', [100.0, -0.0025])])

    def test_extra_whitespace(self):
        result = self._parse("  translate( 10 ,  20 )  ")
        self.assertEqual(result, [('translate', [10.0, 20.0])])

    def test_multiple_transforms(self):
        result = self._parse("translate(10,20) rotate(45)")
        self.assertEqual(
            result,
            [('translate', [10.0, 20.0]), ('rotate', [45.0])])

    def test_unsupported_function_name(self):
        with self.assertRaises(ValueError):
            self._parse("foo(1,2)")

    def test_matrix_wrong_arg_count(self):
        with self.assertRaises(ValueError):
            self._parse("matrix(1,2,3)")

    def test_translate_too_many_args(self):
        with self.assertRaises(ValueError):
            self._parse("translate(1,2,3)")

    def test_rotate_two_args_invalid(self):
        with self.assertRaises(ValueError):
            self._parse("rotate(1,2)")

    def test_skewx_too_many_args(self):
        with self.assertRaises(ValueError):
            self._parse("skewX(1,2)")


class SvgTransformationMatrixTC(SvgParserTB):
    """
    Test EShapeBase._calc_transformation_matrix() for the SVG `transform`
    attribute functions.
    """

    def _calc(self, transform_funs):
        shape = svg.EShapeBase()
        return shape._calc_transformation_matrix(transform_funs)

    def test_empty_is_identity(self):
        result = self._calc([])
        self.assert_allclose(result, np.identity(3))

    def test_matrix(self):
        result = self._calc([('matrix', [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])])
        expected = np.array([[1.0, 3.0, 5.0],
                             [2.0, 4.0, 6.0],
                             [0.0, 0.0, 1.0]])
        self.assert_allclose(result, expected)

    def test_translate_one_arg(self):
        result = self._calc([('translate', [10.0])])
        expected = np.array([[1.0, 0.0, 10.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]])
        self.assert_allclose(result, expected)

    def test_translate_two_args(self):
        result = self._calc([('translate', [10.0, 20.0])])
        expected = np.array([[1.0, 0.0, 10.0],
                             [0.0, 1.0, 20.0],
                             [0.0, 0.0, 1.0]])
        self.assert_allclose(result, expected)

    def test_scale_one_arg(self):
        result = self._calc([('scale', [2.0])])
        expected = np.array([[2.0, 0.0, 0.0],
                             [0.0, 2.0, 0.0],
                             [0.0, 0.0, 1.0]])
        self.assert_allclose(result, expected)

    def test_scale_two_args(self):
        result = self._calc([('scale', [2.0, 3.0])])
        expected = np.array([[2.0, 0.0, 0.0],
                             [0.0, 3.0, 0.0],
                             [0.0, 0.0, 1.0]])
        self.assert_allclose(result, expected)

    def test_rotate_one_arg(self):
        result = self._calc([('rotate', [90.0])])
        expected = np.array([[0.0, -1.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0]])
        self.assert_allclose(result, expected, atol=1e-14)

    def test_rotate_three_args(self):
        # Rotate 90 degrees about center (1, 1).
        result = self._calc([('rotate', [90.0, 1.0, 1.0])])
        expected = np.array([[0.0, -1.0, 2.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0]])
        self.assert_allclose(result, expected, atol=1e-14)

    def test_skewx(self):
        result = self._calc([('skewX', [45.0])])
        expected = np.array([[1.0, 1.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]])
        self.assert_allclose(result, expected, atol=1e-14)

    def test_skewy(self):
        result = self._calc([('skewY', [45.0])])
        expected = np.array([[1.0, 0.0, 0.0],
                             [1.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]])
        self.assert_allclose(result, expected, atol=1e-14)

    def test_multiple_transforms(self):
        result = self._calc([('translate', [10.0, 0.0]), ('scale', [2.0])])
        expected = np.array([[2.0, 0.0, 10.0],
                             [0.0, 2.0, 0.0],
                             [0.0, 0.0, 1.0],])
        self.assert_allclose(result, expected)

    def test_multiple_transforms_with_different_order(self):
        result = self._calc([('scale', [2.0]), ('translate', [10.0, 0.0])])
        expected = np.array([[2.0, 0.0, 20.0],
                             [0.0, 2.0, 0.0],
                             [0.0, 0.0, 1.0]])
        self.assert_allclose(result, expected)


class SvgAffineTransformTC(SvgParserTB):
    """
    Test EShapeBase._transform() applying the parsed SVG `transform`
    attribute to a shape's SegmentPad and CurvePad point buffers.
    """

    def test_no_transform_attr(self):
        d_attr = "M10 10 L20 10"
        path_element = svg.EPath(d_attr=d_attr, fill_attr="none",
                                 transform_attr=None)
        spad = path_element.spads[0]
        self.assertEqual(list(spad.x0), [10.0])
        self.assertEqual(list(spad.y0), [10.0])
        self.assertEqual(list(spad.x1), [20.0])
        self.assertEqual(list(spad.y1), [10.0])

    def test_translate_segment(self):
        d_attr = "M10 10 L20 10"
        path_element = svg.EPath(d_attr=d_attr, fill_attr="none",
                                 transform_attr="translate(5,3)")
        spad = path_element.spads[0]
        self.assertEqual(list(spad.x0), [15.0])
        self.assertEqual(list(spad.y0), [13.0])
        self.assertEqual(list(spad.x1), [25.0])
        self.assertEqual(list(spad.y1), [13.0])

    def test_scale_segment(self):
        d_attr = "M10 10 L20 10"
        path_element = svg.EPath(d_attr=d_attr, fill_attr="none",
                                 transform_attr="scale(2,3)")
        spad = path_element.spads[0]
        self.assertEqual(list(spad.x0), [20.0])
        self.assertEqual(list(spad.y0), [30.0])
        self.assertEqual(list(spad.x1), [40.0])
        self.assertEqual(list(spad.y1), [30.0])

    def test_rotate_segment(self):
        d_attr = "M1 0 L2 0"
        path_element = svg.EPath(d_attr=d_attr, fill_attr="none",
                                 transform_attr="rotate(90)")
        spad = path_element.spads[0]
        self.assert_allclose(list(spad.x0), [0.0], atol=1e-14)
        self.assert_allclose(list(spad.y0), [1.0], atol=1e-14)
        self.assert_allclose(list(spad.x1), [0.0], atol=1e-14)
        self.assert_allclose(list(spad.y1), [2.0], atol=1e-14)

    def test_translate_curve(self):
        d_attr = "M10 90 C30 90 25 10 50 10"
        path_element = svg.EPath(d_attr=d_attr, fill_attr="none",
                                 transform_attr="translate(5,5)")
        cpad = path_element.cpads[0]
        self.assertEqual(list(cpad.p0_at(0)), [15.0, 95.0, 0.0])
        self.assertEqual(list(cpad.p1_at(0)), [35.0, 95.0, 0.0])
        self.assertEqual(list(cpad.p2_at(0)), [30.0, 15.0, 0.0])
        self.assertEqual(list(cpad.p3_at(0)), [55.0, 15.0, 0.0])

    def test_rotate_curve(self):
        d_attr = "M10 90 C30 90 25 10 50 10"

        # Rotate curve about the origin of the element
        path_element = svg.EPath(d_attr=d_attr, fill_attr="none",
                                 transform_attr="rotate(90)")
        cpad = path_element.cpads[0]
        self.assert_allclose(list(cpad.p0_at(0)), [-90.0, 10.0, 0.0],
                             atol=1e-12)
        self.assert_allclose(list(cpad.p1_at(0)), [-90.0, 30.0, 0.0],
                             atol=1e-12)
        self.assert_allclose(list(cpad.p2_at(0)), [-10.0, 25.0, 0.0],
                             atol=1e-12)
        self.assert_allclose(list(cpad.p3_at(0)), [-10.0, 50.0, 0.0],
                             atol=1e-12)

        # Rotate curve about (1, 1)
        path_element = svg.EPath(d_attr=d_attr, fill_attr="none",
                                 transform_attr="rotate(90, 1, 1)")
        cpad = path_element.cpads[0]
        self.assert_allclose(list(cpad.p0_at(0)), [-88.0, 10.0, 0.0],
                             atol=1e-12)
        self.assert_allclose(list(cpad.p1_at(0)), [-88.0, 30.0, 0.0],
                             atol=1e-12)
        self.assert_allclose(list(cpad.p2_at(0)), [-8.0, 25.0, 0.0],
                             atol=1e-12)
        self.assert_allclose(list(cpad.p3_at(0)), [-8.0, 50.0, 0.0],
                             atol=1e-12)

    def test_scale_curve(self):
        d_attr = "M10 90 C30 90 25 10 50 10"
        path_element = svg.EPath(d_attr=d_attr, fill_attr="none",
                                 transform_attr="scale(2)")
        cpad = path_element.cpads[0]
        self.assertEqual(list(cpad.p0_at(0)), [20.0, 180.0, 0.0])
        self.assertEqual(list(cpad.p1_at(0)), [60.0, 180.0, 0.0])
        self.assertEqual(list(cpad.p2_at(0)), [50.0, 20.0, 0.0])
        self.assertEqual(list(cpad.p3_at(0)), [100.0, 20.0, 0.0])

    def test_skew_curve_along_x_axis(self):
        d_attr = "M10 90 C30 90 25 10 50 10"
        tan30 = np.tan(np.radians(30))

        path_element = svg.EPath(d_attr=d_attr, fill_attr="none",
                                 transform_attr="skewX(30)")
        cpad = path_element.cpads[0]
        self.assert_allclose(list(cpad.p0_at(0)),
                             [10 + 90 * tan30, 90.0, 0.0], atol=1e-9)
        self.assert_allclose(list(cpad.p1_at(0)),
                             [30 + 90 * tan30, 90.0, 0.0], atol=1e-9)
        self.assert_allclose(list(cpad.p2_at(0)),
                             [25 + 10 * tan30, 10.0, 0.0], atol=1e-9)
        self.assert_allclose(list(cpad.p3_at(0)),
                             [50 + 10 * tan30, 10.0, 0.0], atol=1e-9)

    def test_skew_curve_along_y_axis(self):
        d_attr = "M10 90 C30 90 25 10 50 10"
        tan30 = np.tan(np.radians(30))

        path_element = svg.EPath(d_attr=d_attr, fill_attr="none",
                                 transform_attr="skewY(30)")
        cpad = path_element.cpads[0]
        self.assert_allclose(list(cpad.p0_at(0)),
                             [10.0, 90 + 10 * tan30, 0.0], atol=1e-9)
        self.assert_allclose(list(cpad.p1_at(0)),
                             [30.0, 90 + 30 * tan30, 0.0], atol=1e-9)
        self.assert_allclose(list(cpad.p2_at(0)),
                             [25.0, 10 + 25 * tan30, 0.0], atol=1e-9)
        self.assert_allclose(list(cpad.p3_at(0)),
                             [50.0, 10 + 50 * tan30, 0.0], atol=1e-9)

    def test_multiple_transformation(self):
        d_attr = "M0 0 L1 0"
        path_element = svg.EPath(d_attr=d_attr, fill_attr="none",
                                 transform_attr="translate(10,0) scale(2)")
        spad = path_element.spads[0]
        self.assertEqual(list(spad.x0), [10.0])
        self.assertEqual(list(spad.y0), [0.0])
        self.assertEqual(list(spad.x1), [12.0])
        self.assertEqual(list(spad.y1), [0.0])


class SvgShapeTC(SvgParserTB):
    """
    Test SVG rectangle shape parsing.
    See more: https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorials/SVG_from_scratch/Basic_shapes
    """  # noqa: E501

    def test_rectangle_shape(self):
        rectangle = svg.ERectangle(x=10, y=10, width=30, height=20,
                                   fill_attr="none")
        self.assertEqual(len(rectangle.spads), 1)
        spad = rectangle.spads[0]
        self.assertEqual(spad.ndim, 2)
        self.assertEqual(len(spad), 4)
        self.assertEqual(list(spad.x0), [10.0, 40.0, 40.0, 10.0])
        self.assertEqual(list(spad.y0), [10.0, 10.0, 30.0, 30.0])
        self.assertEqual(list(spad.x1), [40.0, 40.0, 10.0, 10.0])
        self.assertEqual(list(spad.y1), [10.0, 30.0, 30.0, 10.0])

    def test_circle_shape(self):
        cx, cy, r = 50.0, 50.0, 20.0
        kappa = 0.5522847498

        circle = svg.ECircle(cx=cx, cy=cy, r=r, fill_attr="none")
        self.assertEqual(len(circle.cpads), 1)
        cpad = circle.cpads[0]

        # Check Curve 0: Top-right quadrant (0 to 90 degrees)
        self.assert_allclose(list(cpad.p0_at(0)), [cx + r, cy, 0])
        self.assert_allclose(list(cpad.p1_at(0)), [cx + r, cy + r * kappa, 0])
        self.assert_allclose(list(cpad.p2_at(0)), [cx + r * kappa, cy + r, 0])
        self.assert_allclose(list(cpad.p3_at(0)), [cx, cy + r, 0])

        # Check Curve 1: Top-left quadrant (90 to 180 degrees)
        self.assert_allclose(list(cpad.p0_at(1)), [cx, cy + r, 0])
        self.assert_allclose(list(cpad.p1_at(1)), [cx - r * kappa, cy + r, 0])
        self.assert_allclose(list(cpad.p2_at(1)), [cx - r, cy + r * kappa, 0])
        self.assert_allclose(list(cpad.p3_at(1)), [cx - r, cy, 0])

        # Check Curve 2: Bottom-left quadrant (180 to 270 degrees)
        self.assert_allclose(list(cpad.p0_at(2)), [cx - r, cy, 0])
        self.assert_allclose(list(cpad.p1_at(2)), [cx - r, cy - r * kappa, 0])
        self.assert_allclose(list(cpad.p2_at(2)), [cx - r * kappa, cy - r, 0])
        self.assert_allclose(list(cpad.p3_at(2)), [cx, cy - r, 0])

        # Check Curve 3: Bottom-right quadrant (270 to 360 degrees)
        self.assert_allclose(list(cpad.p0_at(3)), [cx, cy - r, 0])
        self.assert_allclose(list(cpad.p1_at(3)), [cx + r * kappa, cy - r, 0])
        self.assert_allclose(list(cpad.p2_at(3)), [cx + r, cy - r * kappa, 0])
        self.assert_allclose(list(cpad.p3_at(3)), [cx + r, cy, 0])

    def test_ellipse_shape(self):
        cx, cy, rx, ry = 50.0, 50.0, 30.0, 20.0
        kappa = 0.5522847498

        ellipse = svg.EEllipse(cx=cx, cy=cy, rx=rx, ry=ry, fill_attr="none")
        self.assertEqual(len(ellipse.cpads), 1)
        cpad = ellipse.cpads[0]

        # Check Curve 0: Top-right quadrant (0 to 90 degrees)
        self.assert_allclose(list(cpad.p0_at(0)), [cx + rx, cy, 0])
        self.assert_allclose(list(cpad.p1_at(0)),
                             [cx + rx, cy + ry * kappa, 0])
        self.assert_allclose(list(cpad.p2_at(0)),
                             [cx + rx * kappa, cy + ry, 0])
        self.assert_allclose(list(cpad.p3_at(0)), [cx, cy + ry, 0])

        # Check Curve 1: Top-left quadrant (90 to 180 degrees)
        self.assert_allclose(list(cpad.p0_at(1)), [cx, cy + ry, 0])
        self.assert_allclose(list(cpad.p1_at(1)),
                             [cx - rx * kappa, cy + ry, 0])
        self.assert_allclose(list(cpad.p2_at(1)),
                             [cx - rx, cy + ry * kappa, 0])
        self.assert_allclose(list(cpad.p3_at(1)), [cx - rx, cy, 0])

        # Check Curve 2: Bottom-left quadrant (180 to 270 degrees)
        self.assert_allclose(list(cpad.p0_at(2)), [cx - rx, cy, 0])
        self.assert_allclose(list(cpad.p1_at(2)),
                             [cx - rx, cy - ry * kappa, 0])
        self.assert_allclose(list(cpad.p2_at(2)),
                             [cx - rx * kappa, cy - ry, 0])
        self.assert_allclose(list(cpad.p3_at(2)), [cx, cy - ry, 0])

        # Check Curve 3: Bottom-right quadrant (270 to 360 degrees)
        self.assert_allclose(list(cpad.p0_at(3)), [cx, cy - ry, 0])
        self.assert_allclose(list(cpad.p1_at(3)),
                             [cx + rx * kappa, cy - ry, 0])
        self.assert_allclose(list(cpad.p2_at(3)),
                             [cx + rx, cy - ry * kappa, 0])
        self.assert_allclose(list(cpad.p3_at(3)), [cx + rx, cy, 0])

    def test_line_shape(self):
        line = svg.ELine(x1=10, y1=20, x2=30, y2=40, fill_attr="none")
        self.assertEqual(len(line.spads), 1)
        spad = line.spads[0]

        self.assertEqual(spad.ndim, 2)
        self.assertEqual(len(spad), 1)
        self.assertEqual(list(spad.x0), [10.0])
        self.assertEqual(list(spad.y0), [20.0])
        self.assertEqual(list(spad.x1), [30.0])
        self.assertEqual(list(spad.y1), [40.0])

    def test_polyline_shape(self):
        polyline = svg.EPolyline(points=[(10, 10), (20, 20), (30, 10)],
                                 fill_attr="none")
        self.assertEqual(len(polyline.spads), 1)
        spad = polyline.spads[0]

        self.assertEqual(spad.ndim, 2)
        self.assertEqual(len(spad), 2)
        self.assertEqual(list(spad.x0), [10.0, 20.0])
        self.assertEqual(list(spad.y0), [10.0, 20.0])
        self.assertEqual(list(spad.x1), [20.0, 30.0])
        self.assertEqual(list(spad.y1), [20.0, 10.0])

    def test_polygon_shape(self):
        polygon = svg.EPolygon(points=[(10, 10), (20, 20), (30, 10)],
                               fill_attr="none")
        self.assertEqual(len(polygon.spads), 1)
        spad = polygon.spads[0]

        self.assertEqual(spad.ndim, 2)
        self.assertEqual(len(spad), 3)
        self.assertEqual(list(spad.x0), [10.0, 20.0, 30.0])
        self.assertEqual(list(spad.y0), [10.0, 20.0, 10.0])
        self.assertEqual(list(spad.x1), [20.0, 30.0, 10.0])
        self.assertEqual(list(spad.y1), [20.0, 10.0, 10.0])


class SvgFileTC(SvgParserTB):
    """
    Test loading SVG files.
    """

    TESTDIR = os.path.abspath(os.path.dirname(__file__))
    DATADIR = os.path.join(TESTDIR, "data/svg")

    def _check_svg_file(self, filename, spads_count, cpads_count,
                        total_segments, total_curves, spad_lengths,
                        cpad_lengths, sample_points):
        """Generic test method for SVG files using test data."""
        file_path = os.path.join(self.DATADIR, filename)

        parser = svg.SvgParser(file_path)
        parser.parse()
        spads, cpads = parser.get_pads()

        self.assertEqual(len(spads), spads_count)
        self.assertEqual(len(cpads), cpads_count)

        total_segments_actual = sum(len(spad) for spad in spads)
        total_curves_actual = sum(len(cpad) for cpad in cpads)
        self.assertEqual(total_segments_actual, total_segments)
        self.assertEqual(total_curves_actual, total_curves)

        for index, expected_length in spad_lengths:
            self.assertEqual(len(spads[index]), expected_length)
            self.assertEqual(spads[index].ndim, 2)

        for index, expected_length in cpad_lengths:
            self.assertEqual(len(cpads[index]), expected_length)
            self.assertEqual(cpads[index].ndim, 2)

        spad_sample_points = sample_points['spad']
        for pad_index, segment_index, x0, y0, x1, y1 in spad_sample_points:
            if x0 is not None:
                self.assert_allclose(spads[pad_index].x0_at(segment_index), x0)
            if y0 is not None:
                self.assert_allclose(spads[pad_index].y0_at(segment_index), y0)
            if x1 is not None:
                self.assert_allclose(spads[pad_index].x1_at(segment_index), x1)
            if y1 is not None:
                self.assert_allclose(spads[pad_index].y1_at(segment_index), y1)

        cpad_sample_points = sample_points['cpad']
        for pad_index, curve_index, p0, p3 in cpad_sample_points:
            if p0 is not None:
                self.assert_allclose(list(cpads[pad_index].p0_at(curve_index)),
                                     p0)
            if p3 is not None:
                self.assert_allclose(list(cpads[pad_index].p3_at(curve_index)),
                                     p3)

    def test_load_android_svg(self):
        self._check_svg_file(
            filename='android.svg',
            spads_count=3,
            cpads_count=5,
            total_segments=14,
            total_curves=9,
            spad_lengths=[(0, 8), (1, 2), (2, 4)],
            cpad_lengths=[(2, 1), (3, 4), (4, 4)],
            sample_points={
                'spad': [
                    (0, 0, 14.0, 40.0, 14.0, 64.0),  # M14,40v24
                    (0, 1, 81.0, 40.0, 81.0, 64.0),  # M81,40v24
                    (0, 2, 38.0, 68.0, 38.0, 92.0),  # M38,68v24
                    (0, 3, 57.0, 68.0, 57.0, 92.0),  # M57,68v24
                    (0, 4, 28.0, 42.0, 28.0, 73.0),  # M28,42v31
                    (0, 5, 28.0, 73.0, 67.0, 73.0),  # h39
                    (0, 6, 67.0, 73.0, 67.0, 42.0),  # v-31
                    (0, 7, 67.0, 42.0, 28.0, 42.0),  # z (close)
                    (1, 0, 32.0, 5.0, 37.0, 15.0),   # M32,5l5,10
                    (1, 1, 64.0, 5.0, 58.0, 15.0),   # M64,5l-6,10
                    (2, 0, 22.0, 35.0, 73.0, 35.0),  # M22,35h51
                    (2, 1, 73.0, 35.0, 73.0, 45.0),  # v10
                    (2, 2, 73.0, 45.0, 22.0, 45.0),  # h-51
                    (2, 3, 22.0, 45.0, 22.0, 35.0),  # z (close)
                ],
                'cpad': [
                    (2, 0, [22.0, 33.0, 0.0], [73.0, 33.0, 0.0]),  # M22,33c0-31,51-31,51,0  # noqa: E501
                    (3, 0, [38.0, 22.0, 0.0], [36.0, 24.0, 0.0]),  # circle cx="36" cy="22" r="2"  # noqa: E501
                    (3, 1, [36.0, 24.0, 0.0], [34.0, 22.0, 0.0]),
                    (3, 2, [34.0, 22.0, 0.0], [36.0, 20.0, 0.0]),
                    (3, 3, [36.0, 20.0, 0.0], [38.0, 22.0, 0.0]),
                    (4, 0, [61.0, 22.0, 0.0], [59.0, 24.0, 0.0]),  # circle cx="59" cy="22" r="2"  # noqa: E501
                    (4, 1, [59.0, 24.0, 0.0], [57.0, 22.0, 0.0]),
                    (4, 2, [57.0, 22.0, 0.0], [59.0, 20.0, 0.0]),
                    (4, 3, [59.0, 20.0, 0.0], [61.0, 22.0, 0.0]),
                ],
            }
        )

    def test_load_beacon_svg(self):
        self._check_svg_file(
            filename='beacon.svg',
            spads_count=4,
            cpads_count=4,
            total_segments=20,
            total_curves=0,
            spad_lengths=[(0, 5), (1, 5), (2, 5), (3, 5)],
            cpad_lengths=[],
            sample_points={
                'spad': [
                    (0, 0, 28.0, 6.0, 72.0, 6.0),    # M28,6h44
                    (0, 1, 72.0, 6.0, 72.0, 22.0),   # v16
                    (0, 2, 72.0, 22.0, 50.0, 43.0),  # l-22,21
                    (0, 3, 50.0, 43.0, 28.0, 22.0),  # l-22-21
                    (0, 4, 28.0, 22.0, 28.0, 6.0),   # z (close)
                    (1, 0, 28.0, 95.0, 72.0, 95.0),  # M28,95h44
                    (1, 1, 72.0, 95.0, 72.0, 79.0),  # v-16
                    (1, 2, 72.0, 79.0, 50.0, 58.0),  # l-22-21
                    (1, 3, 50.0, 58.0, 28.0, 79.0),  # l-22,21
                    (1, 4, 28.0, 79.0, 28.0, 95.0),  # z (close)
                    (2, 0, 6.0, 30.0, 6.0, 72.0),    # M6,30v42
                    (2, 1, 6.0, 72.0, 21.0, 72.0),   # h15
                    (2, 2, 21.0, 72.0, 42.0, 51.0),  # l21-21
                    (2, 3, 42.0, 51.0, 21.0, 30.0),  # l-21-21
                    (2, 4, 21.0, 30.0, 6.0, 30.0),   # z (close)
                    (3, 0, 95.0, 30.0, 95.0, 72.0),  # M95,30v42
                    (3, 1, 95.0, 72.0, 80.0, 72.0),  # h-15
                    (3, 2, 80.0, 72.0, 59.0, 51.0),  # l-21-21
                    (3, 3, 59.0, 51.0, 80.0, 30.0),  # l21-21
                    (3, 4, 80.0, 30.0, 95.0, 30.0),  # z (close)
                ],
                'cpad': [],
            }
        )

    def test_load_bozo_svg(self):
        self._check_svg_file(
            filename='bozo.svg',
            spads_count=4,
            cpads_count=9,
            total_segments=1,
            total_curves=30,
            spad_lengths=[(1, 1)],
            cpad_lengths=[(0, 4), (1, 2), (2, 2), (3, 2), (4, 4), (5, 4),
                          (6, 4), (7, 4), (8, 4)],
            sample_points={
                'spad': [
                    (1, 0, 35.0, 45.0, 35.0, 45.0),
                ],
                'cpad': [
                    (0, 0, [10.0, 15.0, 0.0], [50.0, 15.0, 0.0]),
                    (0, 1, [50.0, 15.0, 0.0], [90.0, 15.0, 0.0]),
                    (1, 0, [35.0, 45.0, 0.0], [65.0, 45.0, 0.0]),
                    (2, 0, [35.0, 30.0, 0.0], [45.0, 30.0, 0.0]),
                    (3, 0, [55.0, 30.0, 0.0], [65.0, 30.0, 0.0]),
                    (4, 0, [55.0, 40.0, 0.0], [50.0, 45.0, 0.0]),
                    (5, 0, [49.0, 38.0, 0.0], [48.0, 39.0, 0.0]),
                    (6, 0, [42.0, 30.0, 0.0], [40.0, 32.0, 0.0]),
                    (7, 0, [62.0, 30.0, 0.0], [60.0, 32.0, 0.0]),
                    (8, 0, [72.0, 40.0, 0.0], [50.0, 75.0, 0.0]),
                ],
            }
        )

    def test_load_caution_svg(self):
        self._check_svg_file(
            filename='caution.svg',
            spads_count=3,
            cpads_count=4,
            total_segments=7,
            total_curves=7,
            spad_lengths=[(0, 3), (1, 3), (2, 1)],
            cpad_lengths=[(0, 3), (3, 4)],
            sample_points={
                'spad': [
                    (0, 0, 13.0, 89.0, 91.0, 89.0),
                    (0, 1, 96.0, 80.0, 56.0, 9.0),
                    (1, 0, 52.0, 10.0, 10.0, 85.0),
                    (1, 1, 10.0, 85.0, 93.0, 85.0),
                    (2, 0, 52.0, 32.0, 52.0, 58.0),
                ],
                'cpad': [
                    (0, 0, [8.0, 80.0, 0.0], [13.0, 89.0, 0.0]),
                    (0, 1, [91.0, 89.0, 0.0], [96.0, 80.0, 0.0]),
                    (3, 0, [58.0, 73.0, 0.0], [52.0, 79.0, 0.0]),
                    (3, 1, [52.0, 79.0, 0.0], [46.0, 73.0, 0.0]),
                ],
            }
        )

    def test_load_debian_svg(self):
        self._check_svg_file(
            filename='debian.svg',
            spads_count=1,
            cpads_count=1,
            total_segments=44,
            total_curves=76,
            spad_lengths=[(0, 44)],
            cpad_lengths=[(0, 76)],
            sample_points={
                'spad': [
                    (0, 0, 61.0, 53.0, 63.0, 52.0),
                    (0, 1, 63.0, 52.0, 59.0, 52.0),
                    (0, 2, 67.0, 50.0, 69.0, 46.0),
                ],
                'cpad': [
                    (0, 0, [59.0, 52.0, 0.0], [61.0, 53.0, 0.0]),
                    (0, 1, [68.0, 49.0, 0.0], [68.0, 47.0, 0.0]),
                    (0, 2, [68.0, 47.0, 0.0], [67.0, 50.0, 0.0]),
                ],
            }
        )

    def test_load_facebook_svg(self):
        self._check_svg_file(
            filename='facebook.svg',
            spads_count=3,
            cpads_count=1,
            total_segments=23,
            total_curves=2,
            spad_lengths=[(0, 15), (1, 4), (2, 4)],
            cpad_lengths=[(0, 2)],
            sample_points={
                'spad': [
                    (0, 0, 76.0, 94.0, 58.0, 94.0),
                    (0, 1, 58.0, 94.0, 58.0, 52.0),
                    (0, 2, 58.0, 52.0, 50.0, 52.0),
                    (1, 0, 0.0, 0.0, 100.0, 0.0),
                    (1, 1, 100.0, 0.0, 100.0, 100.0),
                    (2, 0, 5.0, 80.0, 95.0, 80.0),
                    (2, 1, 95.0, 80.0, 95.0, 95.0),
                ],
                'cpad': [
                    (0, 0, [58.0, 29.0, 0.0], [77.0, 10.0, 0.0]),
                    (0, 1, [82.0, 24.0, 0.0], [76.0, 30.0, 0.0]),
                ],
            }
        )

    def test_load_gnome2_svg(self):
        self._check_svg_file(
            filename='gnome2.svg',
            spads_count=1,
            cpads_count=1,
            total_segments=5,
            total_curves=14,
            spad_lengths=[(0, 5)],
            cpad_lengths=[(0, 14)],
            sample_points={
                'spad': [
                    (0, 0, 79.0, 0.0, 79.0, 0.0),
                    (0, 1, 41.0, 28.0, 41.0, 28.0),
                    (0, 2, 10.0, 44.0, 10.0, 44.0),
                ],
                'cpad': [
                    (0, 0, [79.0, 0.0, 0.0], [65.0, 32.0, 0.0]),
                    (0, 1, [65.0, 32.0, 0.0], [79.0, 0.0, 0.0]),
                    (0, 2, [41.0, 28.0, 0.0], [43.0, 4.0, 0.0]),
                ],
            }
        )

    def test_load_mars_svg(self):
        self._check_svg_file(
            filename='mars.svg',
            spads_count=1,
            cpads_count=2,
            total_segments=3,
            total_curves=4,
            spad_lengths=[(0, 3)],
            cpad_lengths=[(1, 4)],
            sample_points={
                'spad': [
                    (0, 0, 71.0, 8.0, 93.0, 8.0),    # M71,8h22
                    (0, 1, 93.0, 8.0, 93.0, 30.0),   # v22
                    (0, 2, 68.0, 33.0, 90.0, 11.0),  # M68,33l22-22
                ],
                'cpad': [
                    (1, 0, [77.0, 58.0, 0.0], [43.0, 92.0, 0.0]),  # circle cx="43" cy="58" r="34"  # noqa: E501
                    (1, 1, [43.0, 92.0, 0.0], [9.0, 58.0, 0.0]),
                    (1, 2, [9.0, 58.0, 0.0], [43.0, 24.0, 0.0]),
                    (1, 3, [43.0, 24.0, 0.0], [77.0, 58.0, 0.0]),
                ],
            }
        )

    @unittest.skip("smile.svg contains transforms (translate and matrix) which are not yet supported by the parser")  # noqa: E501
    def test_load_smile_svg(self):
        self._check_svg_file(
            filename='smile.svg',
            spads_count=2,
            cpads_count=4,
            total_segments=43,
            total_curves=12,
            spad_lengths=[(0, 39), (1, 4)],
            cpad_lengths=[(1, 4), (2, 4), (3, 4)],
            sample_points={
                'spad': [
                    (0, 0, 160.0, 304.0, 163.50682991124894, 306.7093364293326),  # Arc from path, scaled by 16  # noqa: E501
                    (1, 0, 8.0, 8.0, 472.0, 8.0),  # rect x='.5' y='.5' width='29' height='39', scaled by 16  # noqa: E501
                    (1, 1, 472.0, 8.0, 472.0, 632.0),
                ],
                'cpad': [
                    (1, 0, [400.0, 320.0, 0.0], [240.0, 480.0, 0.0]),  # circle cx='15' cy='15' r='10', translate(0,5), scale(16)  # noqa: E501
                    (2, 0, [216.0, 272.0, 0.0], [192.0, 296.0, 0.0]),  # circle cx='12' cy='12' r='1.5', translate(0,5), scale(16)  # noqa: E501
                    (3, 0, [296.0, 272.0, 0.0], [272.0, 296.0, 0.0]),  # circle cx='17' cy='12' r='1.5', translate(0,5), scale(16)  # noqa: E501
                ],
            }
        )

    def test_load_shapes_svg(self):
        self._check_svg_file(
            filename='shapes.svg',
            spads_count=6,
            cpads_count=3,
            total_segments=27,
            total_curves=10,
            spad_lengths=[(1, 4), (2, 4), (3, 1), (4, 8), (5, 10)],
            cpad_lengths=[(0, 2), (1, 4), (2, 4)],
            sample_points={
                'spad': [
                    (1, 0, 10.0, 10.0, 40.0, 10.0),
                    (1, 1, 40.0, 10.0, 40.0, 40.0),
                    (3, 0, 10.0, 110.0, 50.0, 150.0),
                    (4, 0, 60.0, 110.0, 65.0, 120.0),
                    (5, 0, 50.0, 160.0, 55.0, 180.0),
                ],
                'cpad': [
                    (0, 0, [20.0, 230.0, 0.0], [50.0, 230.0, 0.0]),
                    (0, 1, [50.0, 230.0, 0.0], [90.0, 230.0, 0.0]),
                    (1, 0, [45.0, 75.0, 0.0], [25.0, 95.0, 0.0]),
                    (2, 0, [95.0, 75.0, 0.0], [75.0, 80.0, 0.0]),
                ],
            }
        )


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

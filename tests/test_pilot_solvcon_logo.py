# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Test for drawing SOLVCON logo using World API.
"""

import os
import unittest
import xml.etree.ElementTree as ET

import solvcon
from solvcon.plot import svg
from solvcon.plot.logo import build_solvcon_logo_curvepads


def _points_close(p, q, tol=1e-9):
    return abs(p.x - q.x) < tol and abs(p.y - q.y) < tol


def _svg_path_ds_and_bbox(world):
    # Consecutive beziers whose p0 continues the previous p3 are the same
    # logo outline, so they are grouped into one SVG <path> "d" string.
    ds = []
    segs = None
    prev_p3 = None
    xs = []
    ys = []
    for i in range(world.nbezier):
        b = world.bezier(i)
        p0, p1, p2, p3 = b[0], b[1], b[2], b[3]
        for p in (p0, p1, p2, p3):
            xs.append(p.x)
            ys.append(p.y)
        if segs is None or not _points_close(p0, prev_p3):
            if segs is not None:
                ds.append(''.join(segs))
            segs = ['M%g,%g' % (p0.x, p0.y)]
        segs.append(' C%g,%g %g,%g %g,%g' % (
            p1.x, p1.y, p2.x, p2.y, p3.x, p3.y))
        prev_p3 = p3
    if segs is not None:
        ds.append(''.join(segs))
    bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
    return ds, bbox


def _write_svg(world, file_path):
    ds, (x0, y0, w, h) = _svg_path_ds_and_bbox(world)
    root = ET.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg',
        'viewBox': '%g %g %g %g' % (x0, y0, w, h),
    })
    for d in ds:
        ET.SubElement(root, 'path', {'d': d})
    ET.ElementTree(root).write(
        file_path, encoding='utf-8', xml_declaration=True)


class WorldSolvconLogoTC(unittest.TestCase):
    def setUp(self):
        self.w = solvcon.WorldFp64()
        self.Bezier = solvcon.Bezier3dFp64

    def test_add_solvcon_logo(self):
        # Prepare CurvePads for SOLVCON logo
        cps_logo = build_solvcon_logo_curvepads()
        shape_ids = [self.w.add_bezier_path_shape(cp) for cp in cps_logo]

        self.assertEqual(self.w.nbezier, 9 + 14 + 4)
        self.assertEqual(
            [self.w.shape_curve_count(sid) for sid in shape_ids],
            [9, 14, 4])

        svg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data', 'svg', 'solvcon_logo.svg')
        _write_svg(self.w, svg_path)

        parser = svg.SvgParser(file_path=svg_path)
        parser.parse()
        _, cpads = parser.get_pads()

        self.assertEqual([len(cpad) for cpad in cpads], [9, 14, 4])


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

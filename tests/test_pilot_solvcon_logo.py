# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Test for drawing SOLVCON logo using World API.
"""

import os
import unittest
import xml.etree.ElementTree as ET

import numpy as np

import solvcon
from solvcon.plot import svg


def _transform_points(points, matrix):
    # SVG matrix(a,b,c,d,e,f): x'=a*x+c*y+e, y'=b*x+d*y+f, via homogeneous
    # coordinates so the transform is a single matrix multiplication.
    a, b, c, d, e, f = matrix
    xform = np.array([[a, c, e], [b, d, f], [0, 0, 1]])
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    return (homogeneous @ xform.T)[:, :2]


def _curvepad_from_points(points, ndim=2):
    # `points` holds 4 control points (p0, p1, p2, p3) per cubic curve.
    cpad = solvcon.CurvePadFp64(ndim=ndim)
    Point = solvcon.Point3dFp64
    for i in range(0, len(points), 4):
        p0, p1, p2, p3 = (Point(x, y, 0) for x, y in points[i:i + 4])
        cpad.append(p0=p0, p1=p1, p2=p2, p3=p3)
    return cpad


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
        cps_logo = []

        # The first path (9 curves):
        # m 0,0
        # c 0.937,0.286 1.871,0.586 2.823,0.896
        # 6.974,2.278 14.406,9.987 17.156,15.722
        # 3.207,6.683 -0.017,15.278 -5.25,18.751
        # -7.975,5.296 -20.899,4.463 -29.856,8.531
        # -4.973,2.262 -9.177,5.491 -10.89,10.454
        # -3.234,9.339 7.091,19.777 16.614,26.11
        # C -25.813,73.642 -39.818,60.069 -45.07,44.049
        # -52.877,20.254 -38.421,0.076 -12.799,-1.019
        # -8.545,-1.197 -4.247,-0.833 0,0
        # transformation matrix=(1.3333333,0,0,-1.3333333,383.6328,493.4648)
        path1_points = np.array([
            [0.0, 0.0], [0.937, 0.286],
            [1.871, 0.586], [2.823, 0.896],  # 1
            [2.823, 0.896], [9.797, 3.174],
            [17.229, 10.883], [19.979, 16.618],  # 2
            [19.979, 16.618], [23.186, 23.301],
            [19.962, 31.896], [14.729, 35.369],  # 3
            [14.729, 35.369], [6.754, 40.665],
            [-6.17, 39.832], [-15.127, 43.9],  # 4
            [-15.127, 43.9], [-20.1, 46.162],
            [-24.304, 49.391], [-26.017, 54.354],  # 5
            [-26.017, 54.354], [-29.251, 63.693],
            [-18.926, 74.131], [-9.403, 80.464],  # 6
            [-9.403, 80.464], [-25.813, 73.642],
            [-39.818, 60.069], [-45.07, 44.049],  # 7
            [-45.07, 44.049], [-52.877, 20.254],
            [-38.421, 0.076], [-12.799, -1.019],  # 8
            [-12.799, -1.019], [-8.545, -1.197],
            [-4.247, -0.833], [0.0, 0.0],  # 9
        ], dtype=float)
        path1_matrix = (1.3333333, 0, 0, -1.3333333, 383.6328, 493.4648)
        cps_logo.append(_curvepad_from_points(
            _transform_points(path1_points, path1_matrix)))

        # The Second path (14 curves):
        # m 0,0
        # c -3.835,0 -7.784,-0.403 -11.754,-1.211
        # -1.807,-0.369 -3.629,-0.643 -5.431,-1.289
        # -0.985,-0.351 -1.878,-0.843 -2.768,-1.436
        # -0.692,-0.46 -1.354,-0.957 -2.015,-1.449
        # -1.34,-0.994 -2.633,-2.042 -3.876,-3.139
        # -2.384,-2.109 -4.593,-4.416 -6.443,-6.901
        # -2.542,-3.416 -4.426,-7.242 -4.682,-10.946
        # -0.284,-4.121 1.566,-7.499 4.377,-9.948
        # 13.204,-11.496 37.728,-5.124 50.474,-16.247
        # 2.955,-2.576 4.313,-5.847 4.455,-9.424
        # 0.238,-6.371 -3.395,-13.738 -8.833,-19.945
        # -1.841,-2.094 -4.057,-4.002 -6.418,-5.753
        # 14.548,8.985 26.213,22.937 30.623,38.618
        # C 45.336,-21.968 28.45,0 0,0
        # transformation matrix=(1.3333333,0,0,-1.3333333,412.026,373.6992)

        path2_points = np.array([
            [0.0, 0.0], [-3.835, 0.0],
            [-7.784, -0.403], [-11.754, -1.211],  # 1
            [-11.754, -1.211], [-13.561, -1.58],
            [-15.383, -1.854], [-17.185, -2.5],  # 2
            [-17.185, -2.5], [-18.17, -2.851],
            [-19.063, -3.343], [-19.953, -3.936],  # 3
            [-19.953, -3.936], [-20.645, -4.396],
            [-21.307, -4.893], [-21.968, -5.385],  # 4
            [-21.968, -5.385], [-23.308, -6.379],
            [-24.601, -7.427], [-25.844, -8.524],  # 5
            [-25.844, -8.524], [-28.228, -10.633],
            [-30.437, -12.94], [-32.287, -15.425],  # 6
            [-32.287, -15.425], [-34.829, -18.841],
            [-36.713, -22.667], [-36.969, -26.371],  # 7
            [-36.969, -26.371], [-37.253, -30.492],
            [-35.403, -33.87], [-32.592, -36.319],  # 8
            [-32.592, -36.319], [-19.388, -47.815],
            [5.136, -41.443], [17.882, -52.566],  # 9
            [17.882, -52.566], [20.837, -55.142],
            [22.195, -58.413], [22.337, -61.99],  # 10
            [22.337, -61.99], [22.575, -68.361],
            [18.942, -75.728], [13.504, -81.935],  # 11
            [13.504, -81.935], [11.663, -84.029],
            [9.447, -85.937], [7.086, -87.688],  # 12
            [7.086, -87.688], [21.634, -78.703],
            [33.299, -64.751], [37.709, -49.07],  # 13
            [37.709, -49.07], [45.336, -21.968],
            [28.45, 0.0], [0.0, 0.0],  # 14
        ], dtype=float)
        path2_matrix = (1.3333333, 0, 0, -1.3333333, 412.026, 373.6992)
        cps_logo.append(_curvepad_from_points(
            _transform_points(path2_points, path2_matrix)))

        # The third path (4 curves):
        # M 0,0
        # C 0,0 22.811,-6.204 18.708,-21.443
        # 16.74,-28.754 13.614,-35.336 1.31,-44.806
        # c 0,0 17.508,11.522 21.696,18.976
        # C 35.998,-2.705 0,0 0,0
        # transformation matrix(1.3333333,0,0,-1.3333333,411.97587,434.49613)
        path3_points = np.array([
            [0.0, 0.0], [0.0, 0.0],
            [22.811, -6.204], [18.708, -21.443],  # 1
            [18.708, -21.443], [16.74, -28.754],
            [13.614, -35.336], [1.31, -44.806],  # 2
            [1.31, -44.806], [1.31, -44.806],
            [18.818, -33.284], [23.006, -25.83],  # 3
            [23.006, -25.83], [35.998, -2.705],
            [0.0, 0.0], [0.0, 0.0],  # 4
        ], dtype=float)
        path3_matrix = (
            1.3333333, 0, 0, -1.3333333, 411.97587, 434.49613)
        cps_logo.append(_curvepad_from_points(
            _transform_points(path3_points, path3_matrix)))

        for cp in cps_logo:
            self.w.add_beziers(cp)

        self.assertEqual(self.w.nbezier, 9 + 14 + 4)

        svg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data', 'svg', 'solvcon_logo.svg')
        _write_svg(self.w, svg_path)

        parser = svg.SvgParser(file_path=svg_path)
        parser.parse()
        _, cpads = parser.get_pads()

        self.assertEqual([len(cpad) for cpad in cpads], [9, 14, 4])


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

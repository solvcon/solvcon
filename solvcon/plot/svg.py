# Copyright (c) 2025, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Input, output, and process SVG (scalleable vector graphic).
"""

import re
import xml.etree.ElementTree as ET
import math

import numpy as np

from .. import core

__all__ = [  # noqa: F822
    'SvgParser',
]


class SvgParser(object):
    """
    The SVG parser to extract SegmentPad and CurvePad from SVG file.

    Internally uses PathParser and ShapeParser to parse <path> and
    shape elements respectively.
    """
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.basic_shapes = []
        self.spads = []  # list of SegmentPad
        self.cpads = []  # list of CurvePad

    def parse(self):
        shapes = []
        SVG_NS = "{http://www.w3.org/2000/svg}"
        ELEMENT_TAGS = {'path': EPath,
                        'circle': ECircle,
                        'rect': ERectangle,
                        'ellipse': EEllipse,
                        'line': ELine,
                        'polyline': EPolyline,
                        'polygon': EPolygon}

        def collect_shapes(elem, chain):
            tag = elem.tag.replace(SVG_NS, '')
            local = self._parse_transform_attrib(elem.get('transform'))
            full_tranform = chain + local

            if tag in ELEMENT_TAGS:
                cls = ELEMENT_TAGS.get(tag)
                shapes.append(cls(elem.attrib, full_tranform))

            for child in elem:
                collect_shapes(child, full_tranform)

        tree = ET.parse(self.file_path)
        root = tree.getroot()
        transform_chain = []
        collect_shapes(root, transform_chain)
        self.basic_shapes = shapes

        # Collect all spads and cpads from shapes
        for sh in shapes:
            self.spads.extend(sh.spads)
            self.cpads.extend(sh.cpads)

    def _parse_transform_attrib(self, transform_attr):
        """
        Parse an SVG transform attribute string.

        :returns transform_chain: a list of transform operators
        """
        # Syntax of the SVG `transform` attribute
        # W3C spec: https://www.w3.org/TR/css-transforms-1/#svg-syntax
        WSP = r'[\x20\x09\x0D\x0A]'  # space, \t, \r, and \f
        NUMBER = r'[-+]?(?:\d+\.\d+|\.\d+|\d+)(?:[eE][-+]?\d+)?'
        COMMA_WSP = rf'(?:{WSP}+,?{WSP}*|,{WSP}*)'
        FUNC_GRAMMARS = {
            'matrix':    rf'^matrix{WSP}*\({WSP}*{NUMBER}(?:{COMMA_WSP}{NUMBER}){{5}}{WSP}*\)$',  # noqa: E501
            'translate': rf'^translate{WSP}*\({WSP}*{NUMBER}(?:{COMMA_WSP}{NUMBER})?{WSP}*\)$',  # noqa: E501
            'scale':     rf'^scale{WSP}*\({WSP}*{NUMBER}(?:{COMMA_WSP}{NUMBER})?{WSP}*\)$',  # noqa: E501
            'rotate':    rf'^rotate{WSP}*\({WSP}*{NUMBER}(?:{COMMA_WSP}{NUMBER}{COMMA_WSP}{NUMBER})?{WSP}*\)$',  # noqa: E501
            'skewX':     rf'^skewX{WSP}*\({WSP}*{NUMBER}{WSP}*\)$',
            'skewY':     rf'^skewY{WSP}*\({WSP}*{NUMBER}{WSP}*\)$',
        }

        transform_chain = []
        if not transform_attr:
            return transform_chain

        for m in re.finditer(r'[a-zA-Z]+\s*\([^)]*\)', transform_attr):
            block = m.group(0)

            name_match = re.match(r'([a-zA-Z]+)', block)
            name = name_match.group(1)

            if name not in FUNC_GRAMMARS:
                raise ValueError(f"Invalid or Unsupported function name for "
                                 f"SVG transform function: '{name}'")

            if not re.fullmatch(FUNC_GRAMMARS[name], block):
                raise ValueError(f"Invalid arguments for a '{name} "
                                 f"function: '{block}'")

            args = [float(n) for n in re.findall(NUMBER, block)]

            if name == 'translate':
                tx = args[0]
                ty = args[1] if len(args) > 1 else 0.0
                transform_chain.append(Translate(tx, ty))
            elif name == 'scale':
                sx = args[0]
                sy = args[1] if len(args) > 1 else sx
                transform_chain.append(Scale(sx, sy))
            elif name == 'rotate':
                angle = args[0]
                cx, cy = (args[1], args[2]) if len(args) == 3 else (0.0, 0.0)
                transform_chain.append(Rotate(angle, cx, cy))
            elif name == 'matrix':
                a, b, c, d, e, f = args
                transform_chain.append(MatrixTransform(a, b, c, d, e, f))
            elif name == 'skewX':
                transform_chain.append(skewX(args[0]))
            elif name == 'skewY':
                transform_chain.append(skewY(args[0]))

        return transform_chain

    def get_pads(self):
        return self.spads, self.cpads


class Transform(object):
    """
    Base class for a single SVG transform operation.
    """
    def matrix(self):
        raise NotImplementedError()


class MatrixTransform(Transform):
    def __init__(self, a, b, c, d, e, f):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def matrix(self):
        return np.array([[self.a, self.c, self.e],
                         [self.b, self.d, self.f],
                         [0, 0, 1]])


class Translate(Transform):
    def __init__(self, tx, ty=0.0):
        self.tx = tx
        self.ty = ty

    def matrix(self):
        return np.array([[1, 0, self.tx], [0, 1, self.ty], [0, 0, 1]])


class Rotate(Transform):
    def __init__(self, angle, cx=0.0, cy=0.0):
        self.angle = angle  # degree
        self.cx = cx
        self.cy = cy

    def matrix(self):
        angle = math.radians(self.angle)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rot = np.array([[cos_a, -sin_a, 0],
                        [sin_a, cos_a, 0],
                        [0, 0, 1]])
        if self.cx != 0.0 or self.cy != 0.0:
            # SVG defines rotate(a, cx, cy) as
            # translate(cx, cy) rotate(a) translate(-cx, -cy):
            # https://www.w3.org/TR/css-transforms-1/#svg-transform-functions
            to_center = np.array([[1, 0, self.cx],
                                  [0, 1, self.cy],
                                  [0, 0, 1]])
            from_center = np.array([[1, 0, -self.cx],
                                    [0, 1, -self.cy],
                                    [0, 0, 1]])
            mat = to_center @ rot @ from_center
        else:
            mat = rot

        return mat


class Scale(Transform):
    def __init__(self, sx, sy=None):
        self.sx = sx
        self.sy = sx if sy is None else sy

    def matrix(self):
        return np.array([[self.sx, 0, 0],
                         [0, self.sy, 0],
                         [0, 0, 1]])


class skewX(Transform):
    def __init__(self, angle):
        self.angle = angle

    def matrix(self):
        angle = math.radians(self.angle)
        return np.array([[1, math.tan(angle), 0],
                         [0, 1, 0],
                         [0, 0, 1]])


class skewY(Transform):
    def __init__(self, angle):
        self.angle = angle

    def matrix(self):
        angle = math.radians(self.angle)
        return np.array([[1, 0, 0],
                         [math.tan(angle), 1, 0],
                         [0, 0, 1]])


class EShapeBase(object):
    def __init__(self, attrib=None, transform_chain=None):
        self.attrib = attrib
        self.transform_chain = transform_chain

        self.spads = []  # list of SegmentPad
        self.cpads = []  # list of CurvePad

    def _calculate(self):
        raise NotImplementedError()

    @staticmethod
    def affine_transform(x, y, tm):
        """
        Apply affine transformation to give 2-dimentional point(s).

        :params x, y: SimpleArrayFloat64, points
        :params tm: (3, 3) ndarray, transformation matrix
        """
        x = np.asarray(x)
        y = np.asarray(y)

        new_x = tm[0, 0] * x + tm[0, 1] * y + tm[0, 2]
        new_y = tm[1, 0] * x + tm[1, 1] * y + tm[1, 2]
        return new_x, new_y

    def transformation_matrix(self):
        M = np.identity(3)
        for op in self.transform_chain:
            M = M @ op.matrix()
        return M

    def _apply_transformation(self):
        # Apply transformation to a shape if there is a transform_attr
        if len(self.transform_chain) != 0:
            tm = self.transformation_matrix()

            for i, spad in enumerate(self.spads):
                if len(spad) != 0:
                    x0, y0 = self.affine_transform(spad.x0, spad.y0, tm)
                    x1, y1 = self.affine_transform(spad.x1, spad.y1, tm)
                    new_spad = core.SegmentPadFp64(
                        x0=core.SimpleArrayFloat64(array=x0),
                        y0=core.SimpleArrayFloat64(array=y0),
                        x1=core.SimpleArrayFloat64(array=x1),
                        y1=core.SimpleArrayFloat64(array=y1),
                        clone=True
                    )
                    self.spads[i] = new_spad

            for i, cpad in enumerate(self.cpads):
                if len(cpad) != 0:
                    x0, y0 = self.affine_transform(cpad.x0, cpad.y0, tm)
                    x1, y1 = self.affine_transform(cpad.x1, cpad.y1, tm)
                    x2, y2 = self.affine_transform(cpad.x2, cpad.y2, tm)
                    x3, y3 = self.affine_transform(cpad.x3, cpad.y3, tm)

                    new_cpad = core.CurvePadFp64(ndim=2)
                    # [TODO] CurvePad has no SimpleArray-based constructor,
                    # unlike SegmentPad, so curves must be appended one by one.
                    for j in range(len(cpad)):
                        p0 = core.Point3dFp64(x0[j], y0[j], 0)
                        p1 = core.Point3dFp64(x1[j], y1[j], 0)
                        p2 = core.Point3dFp64(x2[j], y2[j], 0)
                        p3 = core.Point3dFp64(x3[j], y3[j], 0)
                        new_cpad.append(p0=p0, p1=p1, p2=p2, p3=p3)
                    self.cpads[i] = new_cpad


class EPath(EShapeBase):
    def __init__(self, attrib, transform_chain=[]):
        """
        :param closedPaths: list of closed paths in a <path>.
        """
        super().__init__(attrib, transform_chain)
        self.d_attr = attrib.get('d', '')
        self.path_cmds = None

        self._calculate()
        self._apply_transformation()

    def calc_arc2pnts(self, start_pt, end_pt, rx, ry, phi_deg, large_arc,
                      sweep, steps=40):
        """
        Populate points for an arc curve.

        :param start_pt: coordinates of starting point.
        :param end_pt: coordinates of ending point.
        :param rx: radius of the ellipse.
        :param ry: radius of the ellipse.
        :param phi_deg: rotation of the ellipse (in degree).
        :param large_arc: arc size.
        :type large_arc: boolean (1: larger, 0: smaller)
        :param sweep: arc direction.
        :type sweep: boolean (1: clockwise or 0: counterclockwise)
        :param steps: number of points used to approximate the arc.
        """

        x0, y0 = start_pt[0], start_pt[1]
        x1, y1 = end_pt[0], end_pt[1]

        # convert angle to radians
        phi = math.radians(phi_deg)

        # compute rotated midpoint (x1', y1')
        dx = (x0 - x1) / 2.0
        dy = (y0 - y1) / 2.0
        xp = math.cos(phi) * dx + math.sin(phi) * dy
        yp = -math.sin(phi) * dx + math.cos(phi) * dy

        # correct radii
        rx = abs(rx)
        ry = abs(ry)
        r_check = (xp**2) / (rx**2) + (yp**2) / (ry**2)
        if r_check > 1:
            rx *= math.sqrt(r_check)
            ry *= math.sqrt(r_check)

        # compute center (cx', cy')
        num = rx**2 * ry**2 - rx**2 * yp**2 - ry**2 * xp**2
        denom = rx**2 * yp**2 + ry**2 * xp**2
        factor = math.sqrt(max(0, num / denom))  # ensure non-negative

        if large_arc == sweep:
            factor = -factor

        cxp = factor * (rx * yp) / ry
        cyp = factor * -(ry * xp) / rx

        # compute (cx, cy)
        cx = math.cos(phi) * cxp - math.sin(phi) * cyp + (x0 + x1) / 2
        cy = math.sin(phi) * cxp + math.cos(phi) * cyp + (y0 + y1) / 2

        # compute angles (start, delta)
        def angle(u, v):
            dot = u[0] * v[0] + u[1] * v[1]
            det = u[0] * v[1] - u[1] * v[0]
            return math.atan2(det, dot)

        u = [(xp - cxp) / rx, (yp - cyp) / ry]
        v = [(-xp - cxp) / rx, (-yp - cyp) / ry]

        theta1 = angle([1, 0], u)
        delta_theta = angle(u, v)

        if not sweep and delta_theta > 0:
            delta_theta -= 2 * math.pi
        elif sweep and delta_theta < 0:
            delta_theta += 2 * math.pi

        # using parametric equations for the ellipse and the arc angles
        t = np.linspace(0, delta_theta, num=steps)
        x_arc = (cx +
                 rx * np.cos(theta1 + t) * math.cos(phi) -
                 ry * np.sin(theta1 + t) * math.sin(phi))
        y_arc = (cy +
                 rx * np.cos(theta1 + t) * math.sin(phi) +
                 ry * np.sin(theta1 + t) * math.cos(phi))

        return np.column_stack((x_arc, y_arc))

    def calc_vertices(self):
        """
        path commands for `d` attribute:
            https://developer.mozilla.org/en-US/docs/Web/SVG/Reference/Attribute/d
        """
        Point = core.Point3dFp64
        Segment = core.Segment3dFp64
        sp2d = core.SegmentPadFp64(ndim=2)
        cp2d = core.CurvePadFp64(ndim=2)

        commands = self.path_cmds
        start_pos = Point(0, 0, 0)
        current_pos = start_pos
        last_control = None
        last_cmd = None
        for idx, (cmd, coords) in enumerate(commands):
            i = 0
            if cmd in ('M', 'm'):
                # Move position:
                #   command: [M|m] (relative position in lowercase command)
                #   parameter: (dx, dy)+
                dx, dy = coords[i:i + 2]
                x_cur, y_cur = current_pos[0], current_pos[1]

                if cmd == 'M':
                    start_pos[0] = dx
                    start_pos[1] = dy
                else:
                    start_pos[0] = (x_cur + dx)
                    start_pos[1] = (y_cur + dy)

                current_pos = start_pos
                i += 2

                # implicit line-to command
                while i + 1 < len(coords):
                    dx, dy = coords[i:i + 2]
                    x_cur, y_cur = current_pos[0], current_pos[1]

                    x_end = dx
                    y_end = dy
                    if cmd == 'm':
                        x_end = (x_cur + dx)
                        y_end = (y_cur + dy)

                    end = Point(x_end, y_end, 0)
                    sp2d.append(Segment(current_pos, end))

                    current_pos = end
                    i += 2

                last_control = None
            elif cmd in ('L', 'l'):
                # Draw lines from current position
                #   command: [L|l] (relative position in lowercase command)
                #   parameter: (dx1, dy1)+
                while i + 1 < len(coords):
                    dx, dy = coords[i:i + 2]
                    x_cur, y_cur = current_pos[0], current_pos[1]

                    x_end = dx
                    y_end = dy
                    if cmd == 'l':
                        x_end = x_cur + dx
                        y_end = y_cur + dy

                    end = Point(x_end, y_end, 0)
                    sp2d.append(Segment(current_pos, end))

                    current_pos = end
                    last_control = None
                    i += 2
            elif cmd in ('H', 'h'):
                # Draws horizontal lines
                #   command: [H|h] (relative position in lowercase command)
                #   parameter: (dx)+
                while i < len(coords):
                    dx = coords[i]

                    new_x = dx
                    old_y = current_pos[1]
                    if cmd == 'h':
                        new_x = current_pos[0] + dx

                    end = Point(new_x, old_y, 0)
                    sp2d.append(Segment(current_pos, end))

                    current_pos = end
                    last_control = None
                    i += 1
            elif cmd in ('V', 'v'):
                # Draws a vertical line from current position
                #   command: [V|v] (relative position in lowercase command)
                #   parameter: (dy)+
                while i < len(coords):
                    dy = coords[i]
                    old_x = current_pos[0]

                    new_y = dy
                    if cmd == 'v':
                        new_y = current_pos[1] + dy

                    end = Point(old_x, new_y, 0)
                    sp2d.append(Segment(current_pos, end))

                    current_pos = end
                    last_control = None
                    i += 1
            elif cmd in ('C', 'c'):
                # Draw cubic bezier curves
                #   command: [C|c] (relative position in lowercase command)
                #   parameter: (dx1, dy1, dx2, dy2, dx3, dy3)+
                while i + 5 < len(coords):
                    dx1, dy1, dx2, dy2, dx3, dy3 = coords[i:i + 6]
                    x_cur, y_cur = current_pos[0], current_pos[1]

                    if cmd == 'C':
                        x1 = dx1
                        y1 = dy1
                        x2 = dx2
                        y2 = dy2
                        x3 = dx3
                        y3 = dy3
                    else:
                        x1 = x_cur + dx1
                        y1 = y_cur + dy1
                        x2 = x_cur + dx2
                        y2 = y_cur + dy2
                        x3 = x_cur + dx3
                        y3 = y_cur + dy3

                    p0 = current_pos
                    p1 = Point(x1, y1, 0)
                    p2 = Point(x2, y2, 0)
                    p3 = Point(x3, y3, 0)
                    cp2d.append(p0=p0, p1=p1, p2=p2, p3=p3)

                    current_pos = p3
                    last_control = p2
                    i += 6
            elif cmd in ('S', 's'):
                # draw a smooth curves
                # command: [S|s] (relative position in lowercase command)
                # parameter: (dx2, dy2, dx3, dy3)+
                while i + 3 < len(coords):
                    dx2, dy2, dx3, dy3 = coords[i:i + 4]
                    x_cur, y_cur = current_pos[0], current_pos[1]

                    if cmd == 'S':
                        x2 = dx2
                        y2 = dy2
                        x3 = dx3
                        y3 = dy3
                    else:
                        x2 = x_cur + dx2
                        y2 = y_cur + dy2
                        x3 = x_cur + dx3
                        y3 = y_cur + dy3

                    if (last_cmd in ('C', 'c', 'S', 's')
                            and last_control is not None):
                        x_lastc, y_lastc = last_control[0], last_control[1]
                        x1 = x_cur * 2 - x_lastc
                        y1 = y_cur * 2 - y_lastc
                    else:
                        x1, y1 = x_cur, y_cur

                    p0 = current_pos
                    p1 = Point(x1, y1, 0)
                    p2 = Point(x2, y2, 0)
                    p3 = Point(x3, y3, 0)
                    cp2d.append(p0=p0, p1=p1, p2=p2, p3=p3)

                    current_pos = p3
                    last_control = p2
                    i += 4
            elif cmd in ('Q', 'q'):
                # Draw quadratic bezier curves
                #   command: [Q|q] (relative position in lowercase command)
                #   parameter: (dx1, dy1, dx2, dy2)+
                while i + 3 < len(coords):
                    dx1, dy1, dx2, dy2 = coords[i:i + 4]
                    x_cur, y_cur = current_pos[0], current_pos[1]

                    if cmd == 'Q':
                        x1 = dx1
                        y1 = dy1
                        x2 = dx2
                        y2 = dy2
                    else:
                        x1 = x_cur + dx1
                        y1 = y_cur + dy1
                        x2 = x_cur + dx2
                        y2 = y_cur + dy2

                    p0 = current_pos
                    p1 = Point(x1, y1, 0)
                    p2 = Point(x2, y2, 0)
                    p3 = Point(x2, y2, 0)
                    cp2d.append(p0=p0, p1=p1, p2=p2, p3=p3)

                    current_pos = p3
                    last_control = p1
                    i += 4
            elif cmd in ('T', 't'):
                # Draw a smooth quadratic Bezier curve from the current point
                # to the end point specified by `dx2 dy2`.
                #   command: [T|t] (relative position in lowercase command)
                #   parameter: (dx2, dy2)+
                while i + 1 < len(coords):
                    dx2, dy2 = coords[i:i + 2]
                    x_cur, y_cur = current_pos[0], current_pos[1]

                    if cmd == 'T':
                        x2 = dx2
                        y2 = dy2
                    else:
                        x2 = x_cur + dx2
                        y2 = y_cur + dy2

                    if (last_cmd in ('Q', 'q', 'T', 't')
                            and last_control is not None):
                        x_lastc, y_lastc = last_control[0], last_control[1]
                        x1 = x_cur * 2 - x_lastc
                        y1 = y_cur * 2 - y_lastc
                    else:
                        x1, y1 = x_cur, y_cur

                    p0 = current_pos
                    p1 = Point(x1, y1, 0)
                    p2 = Point(x2, y2, 0)
                    p3 = Point(x2, y2, 0)
                    cp2d.append(p0=p0, p1=p1, p2=p2, p3=p3)

                    current_pos = p3
                    last_control = p1
                    i += 2
            elif cmd in ('A', 'a'):
                # Draw a elliptical arc curves
                #   command: [A|a] (relative position in lowercase command)
                #   parameter: (rx, ry, angle,
                #               large-arc-flag, sweep-flag, dx, dy)+
                while i + 6 < len(coords):
                    start_pt = current_pos
                    (rx, ry, rotation,
                     Flarge_arc, Fsweep, dx, dy) = coords[i:i + 7]
                    x_cur, y_cur = current_pos[0], current_pos[1]

                    x_end = dx
                    y_end = dy
                    if cmd == 'a':
                        x_end = x_cur + dx
                        y_end = y_cur + dy

                    end = Point(x_end, y_end, 0)
                    arc_pts = self.calc_arc2pnts(start_pt, end, rx, ry,
                                                 rotation, Flarge_arc, Fsweep)

                    for p in range(arc_pts.shape[0] - 1):
                        p_from = Point(arc_pts[p][0], arc_pts[p][1], 0)
                        p_to = Point(arc_pts[p + 1][0], arc_pts[p + 1][1], 0)
                        sp2d.append(Segment(p_from, p_to))

                    current_pos = end
                    i += 7
            elif cmd in ('Z', 'z'):
                # closed path:
                #   draw a straight line from the current position to
                #   the first point in the path.
                sp2d.append(Segment(current_pos, start_pos))
                current_pos = start_pos
                i += 1
            else:
                # [TODO] raise a value error
                i = len(coords)
            last_cmd = cmd

        if len(sp2d) != 0:
            self.spads.append(sp2d)
        if len(cp2d) != 0:
            self.cpads.append(cp2d)

    def parse_d_attrib(self):
        d_attr = self.d_attr
        tokens = re.findall(r'([MLCSHVAZQTmlcshvazqt])|(-?\d*\.?\d+)', d_attr)

        commands = []
        current_command = None
        current_coords = []

        for cmd, val in tokens:
            if cmd:
                if current_command:
                    commands.append((current_command, current_coords))
                current_command = cmd
                current_coords = []
            else:
                current_coords.append(float(val))

        if current_command:
            commands.append((current_command, current_coords))
        self.path_cmds = commands

    def _calculate(self):
        self.parse_d_attrib()
        self.calc_vertices()


class ECircle(EShapeBase):
    def __init__(self, attrib, transform_chain=[]):
        super().__init__(attrib, transform_chain)
        self.cx = float(attrib.get('cx', '0'))
        self.cy = float(attrib.get('cy', '0'))
        self.r = float(attrib.get('r', '0'))

        self._calculate()
        self._apply_transformation()

    def _calculate(self):
        # Use 4 cubic Bezier curves to represent the circle
        # Magic constant for circular arc approximation:
        # kappa = 4/3 * tan(pi/8)
        # This value minimizes the radial error when approximating
        # a circular arc with a cubic Bezier curve
        kappa = 0.5522847498

        cpad = core.CurvePadFp64(ndim=2)

        # Top-right quadrant (0 to 90 degrees)
        p0 = core.Point3dFp64(self.cx + self.r, self.cy, 0)
        p1 = core.Point3dFp64(self.cx + self.r,
                              self.cy + self.r * kappa, 0)
        p2 = core.Point3dFp64(self.cx + self.r * kappa,
                              self.cy + self.r, 0)
        p3 = core.Point3dFp64(self.cx, self.cy + self.r, 0)
        cpad.append(p0=p0, p1=p1, p2=p2, p3=p3)

        # Top-left quadrant (90 to 180 degrees)
        p0 = core.Point3dFp64(self.cx, self.cy + self.r, 0)
        p1 = core.Point3dFp64(self.cx - self.r * kappa,
                              self.cy + self.r, 0)
        p2 = core.Point3dFp64(self.cx - self.r,
                              self.cy + self.r * kappa, 0)
        p3 = core.Point3dFp64(self.cx - self.r, self.cy, 0)
        cpad.append(p0=p0, p1=p1, p2=p2, p3=p3)

        # Bottom-left quadrant (180 to 270 degrees)
        p0 = core.Point3dFp64(self.cx - self.r, self.cy, 0)
        p1 = core.Point3dFp64(self.cx - self.r,
                              self.cy - self.r * kappa, 0)
        p2 = core.Point3dFp64(self.cx - self.r * kappa,
                              self.cy - self.r, 0)
        p3 = core.Point3dFp64(self.cx, self.cy - self.r, 0)
        cpad.append(p0=p0, p1=p1, p2=p2, p3=p3)

        # Bottom-right quadrant (270 to 360 degrees)
        p0 = core.Point3dFp64(self.cx, self.cy - self.r, 0)
        p1 = core.Point3dFp64(self.cx + self.r * kappa,
                              self.cy - self.r, 0)
        p2 = core.Point3dFp64(self.cx + self.r,
                              self.cy - self.r * kappa, 0)
        p3 = core.Point3dFp64(self.cx + self.r, self.cy, 0)
        cpad.append(p0=p0, p1=p1, p2=p2, p3=p3)

        self.cpads.append(cpad)


class EEllipse(EShapeBase):
    def __init__(self, attrib, transform_chain=[]):
        super().__init__(attrib, transform_chain)
        self.cx = float(attrib.get('cx', '0'))
        self.cy = float(attrib.get('cy', '0'))
        self.rx = float(attrib.get('rx', '0'))
        self.ry = float(attrib.get('ry', '0'))

        self._calculate()
        self._apply_transformation()

    def _calculate(self):
        cpad = core.CurvePadFp64(ndim=2)

        # Top-right quadrant (0 to 90 degrees)
        p0 = core.Point3dFp64(self.cx + self.rx, self.cy, 0)
        p1 = core.Point3dFp64(self.cx + self.rx,
                              self.cy + self.ry * 0.5522847498, 0)
        p2 = core.Point3dFp64(self.cx + self.rx * 0.5522847498,
                              self.cy + self.ry, 0)
        p3 = core.Point3dFp64(self.cx, self.cy + self.ry, 0)
        cpad.append(p0=p0, p1=p1, p2=p2, p3=p3)

        # Top-left quadrant (90 to 180 degrees)
        p0 = core.Point3dFp64(self.cx, self.cy + self.ry, 0)
        p1 = core.Point3dFp64(self.cx - self.rx * 0.5522847498,
                              self.cy + self.ry, 0)
        p2 = core.Point3dFp64(self.cx - self.rx,
                              self.cy + self.ry * 0.5522847498, 0)
        p3 = core.Point3dFp64(self.cx - self.rx, self.cy, 0)
        cpad.append(p0=p0, p1=p1, p2=p2, p3=p3)

        # Bottom-left quadrant (180 to 270 degrees)
        p0 = core.Point3dFp64(self.cx - self.rx, self.cy, 0)
        p1 = core.Point3dFp64(self.cx - self.rx,
                              self.cy - self.ry * 0.5522847498, 0)
        p2 = core.Point3dFp64(self.cx - self.rx * 0.5522847498,
                              self.cy - self.ry, 0)
        p3 = core.Point3dFp64(self.cx, self.cy - self.ry, 0)
        cpad.append(p0=p0, p1=p1, p2=p2, p3=p3)

        # Bottom-right quadrant (270 to 360 degrees)
        p0 = core.Point3dFp64(self.cx, self.cy - self.ry, 0)
        p1 = core.Point3dFp64(self.cx + self.rx * 0.5522847498,
                              self.cy - self.ry, 0)
        p2 = core.Point3dFp64(self.cx + self.rx,
                              self.cy - self.ry * 0.5522847498, 0)
        p3 = core.Point3dFp64(self.cx + self.rx, self.cy, 0)
        cpad.append(p0=p0, p1=p1, p2=p2, p3=p3)

        self.cpads.append(cpad)


class ERectangle(EShapeBase):
    def __init__(self, attrib, transform_chain=[]):
        super().__init__(attrib, transform_chain)
        self.x = float(attrib.get('x', '0'))
        self.y = float(attrib.get('y', '0'))
        self.width = float(attrib.get('width', '0'))
        self.height = float(attrib.get('height', '0'))

        self._calculate()
        self._apply_transformation()

    def _calculate(self):
        p1 = core.Point3dFp64(self.x, self.y, 0)
        p2 = core.Point3dFp64(self.x + self.width, self.y, 0)
        p3 = core.Point3dFp64(self.x + self.width, self.y + self.height, 0)
        p4 = core.Point3dFp64(self.x, self.y + self.height, 0)

        spad = core.SegmentPadFp64(ndim=2)
        spad.append(core.Segment3dFp64(p1, p2))
        spad.append(core.Segment3dFp64(p2, p3))
        spad.append(core.Segment3dFp64(p3, p4))
        spad.append(core.Segment3dFp64(p4, p1))
        self.spads.append(spad)


class ELine(EShapeBase):
    def __init__(self, attrib, transform_chain=[]):
        super().__init__(attrib, transform_chain)
        self.x1 = float(attrib.get('x1', '0'))
        self.y1 = float(attrib.get('y1', '0'))
        self.x2 = float(attrib.get('x2', '0'))
        self.y2 = float(attrib.get('y2', '0'))

        self._calculate()
        self._apply_transformation()

    def _calculate(self):
        p1 = core.Point3dFp64(self.x1, self.y1, 0)
        p2 = core.Point3dFp64(self.x2, self.y2, 0)

        spad = core.SegmentPadFp64(ndim=2)
        spad.append(core.Segment3dFp64(p1, p2))
        self.spads.append(spad)


class EPolyline(EShapeBase):
    def __init__(self, attrib, transform_chain=[]):
        super().__init__(attrib, transform_chain)
        self.points_attr = attrib.get('points', '')
        self.points = []   # list of (x, y) tuples

        coords = re.split(r'[\s,]+', self.points_attr.strip())
        for i in range(0, len(coords), 2):
            self.points.append((float(coords[i]), float(coords[i + 1])))

        self._calculate()
        self._apply_transformation()

    def _calculate(self):
        spad = core.SegmentPadFp64(ndim=2)
        for i in range(len(self.points) - 1):
            x1, y1 = self.points[i]
            x2, y2 = self.points[i + 1]
            p1 = core.Point3dFp64(x1, y1, 0)
            p2 = core.Point3dFp64(x2, y2, 0)
            spad.append(core.Segment3dFp64(p1, p2))
        self.spads.append(spad)


class EPolygon(EShapeBase):
    def __init__(self, attrib, transform_chain=[]):
        super().__init__(attrib, transform_chain)
        self.points_attr = attrib.get('points', '')
        self.points = []   # list of (x, y) tuples

        coords = re.split(r'[\s,]+', self.points_attr.strip())
        for i in range(0, len(coords), 2):
            self.points.append((float(coords[i]), float(coords[i + 1])))

        self._calculate()
        self._apply_transformation()

    def _calculate(self):
        spad = core.SegmentPadFp64(ndim=2)
        num_points = len(self.points)
        for i in range(num_points):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % num_points]  # wrap around
            p1 = core.Point3dFp64(x1, y1, 0)
            p2 = core.Point3dFp64(x2, y2, 0)
            spad.append(core.Segment3dFp64(p1, p2))
        self.spads.append(spad)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

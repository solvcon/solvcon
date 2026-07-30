# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Stroke icons for the Painter draw tool selector.

The sources are inline strings rather than files or Qt resources, so the
selector needs no packaging step and an installed solvcon carries its icons.
Each body draws on the design's 16x16 grid and strokes ``currentColor``;
:func:`render` substitutes the wanted color, because Qt's SVG profile has no
CSS cascade to resolve that keyword on its own.
"""

from PySide6 import QtCore, QtGui, QtSvg

__all__ = [
    'ICONS',
    'render',
    'tool_icon',
    'placeholder_icon',
]

_DOCUMENT = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"'
             ' fill="none" stroke="currentColor" stroke-width="1.3">'
             '{body}</svg>')

#: Icon body per selector entry, keyed by draw-tool id where one exists.
ICONS = {
    # The select tool pans the view on empty space as well, but a cursor
    # arrow reads as selection where a hand would read as navigation only.
    "select": ('<path d="M4 2.5l8 5.2-3.6.9-1.6 3.4z"'
               ' stroke-linejoin="round"/>'),
    "line": ('<path d="M3 13L13 3"/><circle cx="3" cy="13" r="1.4"/>'
             '<circle cx="13" cy="3" r="1.4"/>'),
    "triangle": '<path d="M8 3l5 9.5H3z" stroke-linejoin="round"/>',
    "rectangle": '<rect x="2.8" y="4" width="10.4" height="8"/>',
    "ellipse": '<ellipse cx="8" cy="8" rx="6" ry="4.2"/>',
    "circle": '<circle cx="8" cy="8" r="5.2"/>',
    "text": '<path d="M3 4V2.8h10V4M8 2.8v10.4M6 13.2h4"/>',
    "grid": ('<path d="M2 2h12v12H2z"/>'
             '<path d="M2 6h12M6 2v12" opacity="0.6"/>'),
    "search": '<circle cx="7" cy="7" r="4.5"/><path d="M10.4 10.4L14 14"/>',
    "plus": '<path d="M8 3.5v9M3.5 8h9"/>',
    "minus": '<path d="M3.5 8h9"/>',
}


def render(name, color, size, ratio=1.0):
    """The icon named ``name`` as a ``size``-px pixmap stroked in ``color``.

    ``ratio`` is the device pixel ratio to rasterize for, so the strokes stay
    crisp on a scaled screen.
    """
    source = _DOCUMENT.format(body=ICONS[name])
    source = source.replace("currentColor", color.name())
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(source.encode("ascii")))
    pixmap = QtGui.QPixmap(round(size * ratio), round(size * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    # The target rectangle is in logical pixels, not the pixmap's own. A
    # painter on a scaled pixmap already carries the ratio in its transform,
    # while the viewport QSvgRenderer would default to is in device pixels, so
    # letting it default draws the icon ``ratio`` times too large and clips it.
    renderer.render(painter, QtCore.QRectF(0, 0, size, size))
    painter.end()
    return pixmap


def tool_icon(name, size, off, on, ratio=1.0):
    """The icon named ``name`` in the two appearances a tool entry shows.

    The entry strokes ``off`` while its tool is inactive and ``on`` over the
    accent pill while it is active. Qt picks between them by icon state, so
    one QIcon covers both and the entry needs no repaint logic of its own.
    """
    icon = QtGui.QIcon()
    icon.addPixmap(render(name, off, size, ratio),
                   QtGui.QIcon.Normal, QtGui.QIcon.Off)
    icon.addPixmap(render(name, on, size, ratio),
                   QtGui.QIcon.Normal, QtGui.QIcon.On)
    return icon


def placeholder_icon(name, size, disabled, ratio=1.0):
    """The icon named ``name`` in the one appearance a placeholder shows."""
    icon = QtGui.QIcon()
    icon.addPixmap(render(name, disabled, size, ratio), QtGui.QIcon.Disabled)
    return icon

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
The style rules the pilot's dock panels are drawn with.

A rule here says only what the palette does not already give the widget.
"""

from .._style import RuleCatalog, Shades

__all__ = [
    'Rules',
]

_TITLE_PX = 14

# How far a hovered section header sits from the panel.
_GRID_MIX = 0.12
_HOVER_MIX = 0.12


class Rules(RuleCatalog):
    """One method per role the pilot's dock panels draw."""

    @staticmethod
    def tree(shades):
        """The grid and the selection a profiling result is read against.

        The surface and the text are left to the palette a view already
        draws itself from, so the rules add only the grid the numbers are
        read against and the accent on the selected row. The grid is a border
        on each item, so its shade is mixed off that surface rather than off
        the panel around it.
        """
        grid = Shades.blend(shades.base, shades.on_base, _GRID_MIX)
        return f"""
            QTreeView {{
                border: 1px solid {shades.border.name()};
            }}
            QTreeView::item {{
                border: 1px solid {grid.name()};
                border-right: none;
                border-bottom: none;
            }}
            QTreeView::item:selected {{
                background-color: {shades.accent.name()};
                color: {shades.on_accent.name()};
            }}
            """

    @staticmethod
    def fold(shades):
        """The header a collapsible section folds from."""
        return f"""
            QPushButton {{
                border: none;
                font-weight: bold;
                font-size: {_TITLE_PX}px;
                padding: 4px 6px;
                text-align: left;
            }}
            QPushButton:hover {{
                background: {shades.raised(_HOVER_MIX).name()};
            }}
            """

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Shared scaffolding for the pilot dock trees (mesh and entity)."""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTreeWidget,
                               QTreeWidgetItem, QFrame)


class TreePanelBase(QWidget):
    """Base widget wrapping a single-column tree for a dock panel.

    The base owns the tree widget, the frameless single-column look, and
    the helpers that render ``(section, rows)`` groups and a placeholder
    row. A subclass fills the tree from its own source and may add header
    controls above the tree through :meth:`_build_header`.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tree = QTreeWidget()
        self._tree.setColumnCount(1)
        self._tree.setHeaderHidden(True)
        # Drop the tree frame so its scroll bar sits flush in the panel.
        self._tree.setFrameShape(QFrame.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._build_header(layout)
        layout.addWidget(self._tree)
        self.setLayout(layout)

    def _build_header(self, layout):
        """Add controls above the tree; the base panel adds none."""

    def _show_placeholder(self, text):
        """Clear the tree and show a single ``text`` row."""
        self._tree.clear()
        QTreeWidgetItem(self._tree, [text])

    def _render_sections(self, parent, sections):
        """Render ``(section, rows)`` groups under ``parent``.

        Each section is a foldable node holding ``prop: value`` rows.
        """
        for section, rows in sections:
            group = QTreeWidgetItem(parent, [section])
            for prop, value in rows:
                QTreeWidgetItem(group, [f"{prop}: {value}"])
            group.setExpanded(True)

    def _finalize_root(self, root):
        """Expand ``root`` and widen the column to fit the contents."""
        root.setExpanded(True)
        self._tree.resizeColumnToContents(0)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Painter toolbox for the 2D canvas: the draw tool selector and the inspector.
"""

from PySide6 import QtCore, QtGui, QtWidgets

from . import _icons
from ._canvas import CanvasPage
from ._design import DesignPage
from ._layers import LayersPage
from ._style import blend, rule, shade, PaletteStyled
from ..base import _gui_common
from .._pilot_core import draw_tool_names, default_draw_tool_name

__all__ = [
    'PainterPanel',
    'Painter',
]


def _tool_tip(action):
    """The design's hover tip for a tool: its name, then its shortcut key."""
    key = action.shortcut().toString(QtGui.QKeySequence.NativeText)
    return f"{action.text()}  {key}" if key else action.text()


class _SelectorEntry(QtWidgets.QToolButton):
    """One selector entry: a stroke icon over its short name.

    Qt re-reads a default action's text, icon, and tip whenever that action
    changes, and picking a tool changes one. The short name and the tip survive
    that as the action's own icon text and tool tip, but the tinted icon has
    nowhere on the action to live: the Canvas menu shows the same action and
    would then carry the icon too. So the entry keeps its icon and puts it back
    after Qt has taken it away.
    """

    def __init__(self, width, icon_px, parent=None):
        super().__init__(parent)
        self._icon = QtGui.QIcon()
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.setIconSize(QtCore.QSize(icon_px, icon_px))
        self.setFixedWidth(width)

    def set_entry_icon(self, icon):
        self._icon = icon
        self.setIcon(icon)

    def actionEvent(self, event):
        super().actionEvent(event)
        # setIcon always invalidates the layout, so only pay for it when Qt
        # has actually replaced the icon.
        if self.icon().cacheKey() != self._icon.cacheKey():
            self.setIcon(self._icon)


class _SelectorRule(QtWidgets.QWidget):
    """A hairline grouping divider in the draw tool selector.

    The design's breathing room above and below the line lives inside the
    divider's own height, so the column layout keeps one spacing for every
    child.
    """

    _WIDTH = 36
    _MARGIN = 6
    _MIX = 0.2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self._WIDTH, 2 * self._MARGIN + 1)

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.fillRect(0, self._MARGIN, self.width(), 1,
                         shade(self, self._MIX))


class _DrawToolSelector(PaletteStyled):
    """The tool column: flat entries on a shade of the inspector panel.

    The shade is painted rather than written into the widget's palette, because
    a palette color set from a palette-change handler is overwritten again as
    Qt finishes propagating the new palette into this widget, while painting
    always reads the color that is current. The entries lose their button bezel
    and the active tool wears the accent pill the design gives it. Every color,
    the icon strokes included, comes from the palette, so the column follows a
    light/dark switch.
    """

    # A column slot the model cannot back yet, as ``{name: (label, what it
    # waits for)}``: shown at its designed place and greyed out.
    _PLACEHOLDERS = {
        "text": ("Text", "the text shape"),
        "grid": ("Grid", "grid and snap options"),
    }

    # How far the column shade sits from the inspector panel color, and how
    # far a hovered entry sits from the column.
    _SHADE_MIX = 0.06
    _HOVER_MIX = 0.12

    # How far an entry's label moves back toward the column.
    _LABEL_MIX = 0.25
    _MUTED_MIX = 0.55

    # Column and entry geometry from the design, in device-independent pixels.
    _WIDTH = 64
    _ENTRY_WIDTH = 52
    _ICON_PX = 17
    _FONT_PX = 9
    _RADIUS = 7
    _GAP = 2
    _PAD_Y = 8

    def __init__(self, tool_actions, short_labels, parent=None):
        """
        :param tool_actions: The draw tools as ``{tool id: QAction}`` in
            column order; each becomes one entry's default action.
        :type tool_actions: dict
        :param short_labels: Short entry text per tool id, for the tools whose
            menu label does not fit the narrow column.
        :type short_labels: dict
        """
        super().__init__(parent)
        self.buttons = {}
        self.placeholders = {}
        self.setFixedWidth(self._WIDTH)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, self._PAD_Y, 0, self._PAD_Y)
        layout.setSpacing(self._GAP)
        for index, (tool, action) in enumerate(tool_actions.items()):
            # The registry lists the select tool first and the shape tools
            # after it, which is the grouping the design rules apart.
            if index == 1:
                self._add_rule(layout)
            self.buttons[tool] = self._add_tool(
                layout, tool, action, short_labels.get(tool))

        self._add_rule(layout)
        self.placeholders["text"] = self._add_placeholder(layout, "text")
        layout.addStretch(1)
        self.placeholders["grid"] = self._add_placeholder(layout, "grid")
        self._apply_style()

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), shade(self, self._SHADE_MIX))

    def _new_entry(self):
        return _SelectorEntry(self._ENTRY_WIDTH, self._ICON_PX, self)

    def _add_rule(self, layout):
        layout.addWidget(_SelectorRule(self), 0, QtCore.Qt.AlignHCenter)

    def _add_tool(self, layout, tool, action, short_label):
        """Add one entry as a view of the shared ``tool`` action."""
        entry = self._new_entry()
        if short_label:
            action.setIconText(short_label)
        action.setToolTip(_tool_tip(action))
        entry.setDefaultAction(action)
        entry.setCursor(QtCore.Qt.PointingHandCursor)
        layout.addWidget(entry, 0, QtCore.Qt.AlignHCenter)
        return entry

    def _add_placeholder(self, layout, name):
        label, waits_for = self._PLACEHOLDERS[name]
        entry = self._new_entry()
        entry.setText(label)
        entry.setToolTip(f"{label}  (needs {waits_for})")
        entry.setEnabled(False)
        layout.addWidget(entry, 0, QtCore.Qt.AlignHCenter)
        return entry

    def _apply_style(self):
        """Color the entries and their icons from the current palette."""
        palette = self.palette()
        text = palette.color(QtGui.QPalette.WindowText)
        panel = palette.color(QtGui.QPalette.Window)
        label = blend(text, panel, self._LABEL_MIX)
        muted = blend(text, panel, self._MUTED_MIX)
        on_accent = palette.color(QtGui.QPalette.HighlightedText)
        # The disabled rule comes last so a greyed entry keeps its color even
        # under the pointer, where the hover rule would otherwise light it up.
        self.setStyleSheet(f"""
            QToolButton {{
                border: none;
                border-radius: {self._RADIUS}px;
                padding: 7px 0 6px;
                font-size: {self._FONT_PX}px;
                background: transparent;
                color: {label.name()};
            }}
            QToolButton:hover {{
                background: {shade(self, self._HOVER_MIX).name()};
                color: {text.name()};
            }}
            QToolButton:checked {{
                background: {palette.color(QtGui.QPalette.Highlight).name()};
                color: {on_accent.name()};
            }}
            QToolButton:disabled {{
                background: transparent;
                color: {muted.name()};
            }}
            """)
        ratio = self.devicePixelRatioF()
        for name, entry in self.buttons.items():
            # A tool the icon set does not draw yet keeps its label alone.
            if name in _icons.ICONS:
                entry.set_entry_icon(_icons.tool_icon(
                    name, self._ICON_PX, label, on_accent, ratio))
        for name, entry in self.placeholders.items():
            entry.set_entry_icon(_icons.placeholder_icon(
                name, self._ICON_PX, muted, ratio))


class _SegmentedTabs(PaletteStyled):
    """The Design / Layers / Canvas selector as one segmented control.

    Qt has no segmented control, so the row is flat checkable buttons styled
    from the palette: the checked tab wears a neutral pill and the others stay
    transparent and dimmed. The pill is a shade of the panel rather than the
    accent color, which is what keeps it distinct from the accent pill the
    active tool wears.
    """

    #: The selected tab's name.
    selected = QtCore.Signal(str)

    # Row and segment geometry from the design, in device-independent pixels.
    _MARGINS = (10, 8, 10, 8)
    _GAP = 4
    _RADIUS = 5
    _FONT_PX = 11

    # How far the checked pill and the unchecked labels move toward the text
    # color: enough for the pill to read as raised and the rest as secondary.
    _PILL_MIX = 0.15
    _LABEL_MIX = 0.45

    def __init__(self, names, parent=None):
        super().__init__(parent)
        self.buttons = {}
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(*self._MARGINS)
        layout.setSpacing(self._GAP)
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)
        for index, name in enumerate(names):
            button = QtWidgets.QPushButton(name, self)
            button.setCheckable(True)
            button.setChecked(index == 0)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            group.addButton(button)
            button.clicked.connect(
                lambda _checked, at=name: self.selected.emit(at))
            layout.addWidget(button)
            self.buttons[name] = button
        self._apply_style()

    def _apply_style(self):
        """Color the segments from the current palette."""
        palette = self.palette()
        panel = palette.color(QtGui.QPalette.Window)
        text = palette.color(QtGui.QPalette.WindowText)
        pill = shade(self, self._PILL_MIX)
        self.setStyleSheet(f"""
            QPushButton {{
                border: none;
                border-radius: {self._RADIUS}px;
                padding: 5px 0;
                font-size: {self._FONT_PX}px;
                background: transparent;
                color: {blend(text, panel, self._LABEL_MIX).name()};
            }}
            QPushButton:hover {{
                color: {text.name()};
            }}
            QPushButton:checked {{
                background: {pill.name()};
                color: {text.name()};
                font-weight: 500;
            }}
            """)


class PainterPanel(QtWidgets.QWidget):
    """The Painter dock body: the draw tool selector left of the inspector.

    The selector holds one button per draw tool, each a view of the shared
    ``draw.tool`` action the Canvas menu also shows, plus the greyed-out Text
    and Grid slots the model cannot back yet. The inspector stacks the
    Design, Layers, and Canvas pages under a segmented tab row that selects
    among them, and hands the active canvas to every page.
    """

    # The design's inspector width, in device-independent pixels. It is the
    # floor rather than a fixed size: a dock the user widens hands the extra
    # room to the inspector instead of leaving bands beside the panel.
    _INSPECTOR_WIDTH = 264

    #: Inspector page names, in tab order.
    PAGES = ("Design", "Layers", "Canvas")

    def __init__(self, tool_actions, short_labels=None, parent=None):
        """
        :param tool_actions: The draw tools as ``{tool id: QAction}`` in
            selector order; each becomes one button's default action.
        :type tool_actions: dict
        :param short_labels: Short button text per tool id, for the tools
            whose menu label does not fit the narrow selector.
        :type short_labels: dict or None
        """
        super().__init__(parent)
        self._tool_actions = tool_actions
        self._source = None
        self._tools = _DrawToolSelector(tool_actions, short_labels or {}, self)
        self._stack = QtWidgets.QStackedWidget()
        self._design = DesignPage(self._stack)
        self._layers = LayersPage(self._stack)
        self._layers.picked.connect(self._on_picked)
        self._canvas = CanvasPage(self._stack)
        # In tab order, so the stack and the tab row cannot drift apart.
        self._pages = (self._design, self._layers, self._canvas)
        self._tabs = _SegmentedTabs(self.PAGES)
        self._tabs.selected.connect(self.show_page)
        self._inspector = self._build_inspector()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._tools)
        layout.addWidget(rule(QtWidgets.QFrame.VLine))
        layout.addWidget(self._inspector, 1)

    @property
    def tool_buttons(self):
        """The selector buttons as ``{tool id: QToolButton}``."""
        return self._tools.buttons

    @property
    def tool_placeholders(self):
        """The greyed-out entries as ``{slot name: QToolButton}``."""
        return self._tools.placeholders

    @property
    def tabs(self):
        """The inspector tab buttons as ``{page name: QPushButton}``."""
        return self._tabs.buttons

    @property
    def design(self):
        """The Design page."""
        return self._design

    @property
    def layers(self):
        """The Layers page."""
        return self._layers

    @property
    def canvas(self):
        """The Canvas page."""
        return self._canvas

    def set_canvas_source(self, source):
        """Tell the inspector how to reach the active 2D canvas."""
        self._source = source
        for page in self._pages:
            page.set_canvas_source(source)

    def _on_picked(self, shape_id):
        """Select on the canvas the shape a Layers row names."""
        widget = None if self._source is None else self._source()
        select = self._tool_actions.get("select")
        if widget is None or select is None:
            return
        world = widget.world
        # The list polls, so a row can lag the active canvas by one tick; ids
        # start over per world, and a stale row can name an unrelated live id.
        if (world is None
                or not self._layers.draws_world(world)
                or not world.shape_is_live(shape_id)):
            self._layers.refresh()
            return
        # Arm select through the shared action so the menu and rail follow;
        # the canvas paints no selection under a draw tool.
        select.trigger()
        widget.selectedShape = shape_id
        # Catch both pages up on this click rather than waiting on their polls.
        for page in (self._design, self._layers):
            page.refresh()

    def refresh(self):
        """Re-read the active canvas now, without waiting for a poll.

        Only the page on show is re-read; a hidden one reads the canvas as it
        comes back into view, so a canvas switch does not pay for pages the
        user is not looking at.
        """
        for page in self._pages:
            if page.isVisible():
                page.refresh()

    def _build_inspector(self):
        """Build the tab row over the stack of pages."""
        for page in self._pages:
            self._stack.addWidget(page)
        inspector = QtWidgets.QWidget(self)
        inspector.setMinimumWidth(self._INSPECTOR_WIDTH)
        layout = QtWidgets.QVBoxLayout(inspector)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._tabs)
        layout.addWidget(rule(QtWidgets.QFrame.HLine))
        layout.addWidget(self._stack, 1)
        return inspector

    def current_page(self):
        """The name of the page the inspector shows."""
        return self.PAGES[self._stack.currentIndex()]

    def show_page(self, name):
        """Show the page named ``name`` and check its tab."""
        self._stack.setCurrentIndex(self.PAGES.index(name))
        self.tabs[name].setChecked(True)


class Painter(_gui_common.PilotFeature):
    """
    Painter toolbox for drawing shapes on a 2D canvas, toggled from the View
    "Panels" submenu. The selected tool is held by the manager and applied
    to the focused 2D canvas, not bound to any one canvas.

    The dock body is a :class:`PainterPanel`; this feature owns the dock, the
    menu toggle, and the draw-tool actions the panel's selector shows.
    """

    # Button label for a tool id. The ids, their order, and the default
    # come from the C++ registry; this only supplies human-facing text.
    # A tool with no entry here falls back to its title-cased id.
    TOOL_LABELS = {
        "select": "Select",
        "line": "Line",
        "triangle": "Triangle",
        "rectangle": "Rectangle",
        "ellipse": "Ellipse",
        "circle": "Circle",
    }

    # Button text for a tool whose menu label overflows the 64px selector.
    # The names are the design's; a tool with no entry keeps its menu label.
    SHORT_LABELS = {
        "triangle": "Tri",
        "rectangle": "Rect",
    }

    def __init__(self, *args, **kw):
        super(Painter, self).__init__(*args, **kw)
        self._action = None
        self._dock = None
        self._panel = None
        self._tool_group = None
        self._tool_actions = {}

    @property
    def panel(self):
        """The dock body, or ``None`` before the dock is built."""
        return self._panel

    def populate_menu(self):
        """Add the Painter toggle and the draw-tool radio group."""
        self._action = self.add_action(
            "View/Panels", "Painter", "Toggle the Painter toolbox", None,
            id="panel.painter", weight=20, checkable=True)
        self._action.toggled.connect(self._on_toggled)
        self._build_tool_actions()

    def _build_tool_actions(self):
        """One exclusive checkable action per draw tool, the single source of
        truth shared by the Canvas/Draw tool radio items and the selector
        buttons. Each action routes its own trigger to the manager.

        Idempotent: the tools are declared once on the model, so a second
        Painter reuses the existing actions instead of duplicating them.
        """
        if self._tool_actions:
            return
        model = self._mgr.menu_model
        model.menu("Canvas/Draw tool", weight=10)
        # Held by the model under a group id so the selection is queryable.
        self._tool_group = model.group("draw.tool")
        self._tool_group.setExclusive(True)
        mgr = self._mgr
        weight = 10
        created = False
        for tool in draw_tool_names():
            act = model.action("draw.tool." + tool)
            if act is None:
                label = self.TOOL_LABELS.get(tool, tool.title())
                act = self.add_action(
                    "Canvas/Draw tool", label, f"Draw with the {label} tool",
                    lambda t=tool: mgr.setDrawTool(t),
                    id="draw.tool." + tool, weight=weight, checkable=True)
                self._tool_group.addAction(act)
                created = True
            self._tool_actions[tool] = act
            weight += 10
        if created:
            self._tool_actions[default_draw_tool_name()].setChecked(True)

    def _on_toggled(self, checked):
        """Show or hide the Painter dock from the menu toggle."""
        if checked:
            self._ensure_dock()
            self._dock.show()
        elif self._dock is not None:
            self._dock.hide()

    def _ensure_dock(self):
        """Create the dock once, its tool buttons views of the tool actions."""
        if self._dock is not None:
            return
        # A standalone Painter (used in tests) reaches the dock without
        # populate_menu, so make sure the tool actions exist first.
        self._build_tool_actions()
        dock = QtWidgets.QDockWidget("Painter", self._mainWindow)
        dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        self._panel = PainterPanel(self._tool_actions, self.SHORT_LABELS, dock)
        dock.setWidget(self._panel)
        self._mainWindow.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        # Keep the menu check in sync when the dock is closed by its button.
        if self._action is not None:
            dock.visibilityChanged.connect(self._action.setChecked)
        self._dock = dock
        self._panel.set_canvas_source(self._mgr.currentR2DWidget)
        mdi = self._mgr.mdiArea
        if mdi is not None:
            mdi.subWindowActivated.connect(self._on_subwindow_activated)

    def _on_subwindow_activated(self, _subwin):
        """Show the newly active canvas without waiting for the next poll."""
        if self._panel is not None:
            # The manager reports the active canvas once Qt has finished
            # switching, which is after this signal.
            QtCore.QTimer.singleShot(0, self._panel.refresh)

    def present(self):
        """Show the Painter dock and reset the focused canvas to the default
        tool; the action group updates every surface."""
        self._ensure_dock()
        self._mgr.setDrawTool(default_draw_tool_name())
        self._dock.show()
        self._dock.raise_()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

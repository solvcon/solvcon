# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Painter toolbox for the 2D canvas: the tool rail and the inspector.
"""

from PySide6 import QtCore, QtGui, QtWidgets

from ..base import _gui_common
from .._pilot_core import draw_tool_names, default_draw_tool_name

__all__ = [
    'PainterPanel',
    'Painter',
]


def _rule(shape):
    """A hairline divider between two areas of the panel."""
    line = QtWidgets.QFrame()
    line.setFrameShape(shape)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    return line


def _blend(color, other, ratio):
    """Mix ``ratio`` of ``other`` into ``color``."""
    keep = 1.0 - ratio
    return QtGui.QColor(
        round(color.red() * keep + other.red() * ratio),
        round(color.green() * keep + other.green() * ratio),
        round(color.blue() * keep + other.blue() * ratio))


def _shade(widget, ratio):
    """The panel color of ``widget`` moved ``ratio`` toward its text color.

    Blending toward the text is what makes a shade follow the theme: it steps
    darker under a light palette and lighter under a dark one, which is how the
    design separates the rail from the inspector and the checked tab from both.
    """
    palette = widget.palette()
    return _blend(palette.color(QtGui.QPalette.Window),
                  palette.color(QtGui.QPalette.WindowText), ratio)


class _PaletteStyled(QtWidgets.QWidget):
    """A widget whose child controls are styled from the palette.

    A subclass builds its style sheet in :meth:`_apply_style` and calls it once
    its children exist; the base re-applies it whenever the application palette
    changes, so the piece follows a light/dark switch.
    """

    # Both palette events matter. The theme manager pairs a new application
    # palette with a fresh global style sheet, whose re-polish arrives here as
    # PaletteChange; under the system look it sets the palette alone, and
    # ApplicationPaletteChange is what carries that.
    _RESTYLE_EVENTS = (QtCore.QEvent.PaletteChange,
                       QtCore.QEvent.ApplicationPaletteChange)

    def event(self, event):
        if event.type() in self._RESTYLE_EVENTS:
            self._apply_style()
            self.update()
        return super().event(event)

    def _apply_style(self):
        raise NotImplementedError


class _ToolRail(_PaletteStyled):
    """The tool column: flat entries on a shade of the inspector panel.

    The shade is painted rather than written into the widget's palette, because
    a palette color set from a palette-change handler is overwritten again as
    Qt finishes propagating the new palette into this widget, while painting
    always reads the color that is current. The entries lose their button
    bezel and the active tool wears the accent pill the design gives it; the
    icons and the 9px labels arrive with the tool box itself.
    """

    # How far the rail shade sits from the inspector panel color, and how far
    # a hovered entry sits from the rail.
    _SHADE_MIX = 0.06
    _HOVER_MIX = 0.12

    # Rail and entry geometry from the design, in device-independent pixels.
    _WIDTH = 64
    _ENTRY_WIDTH = 52
    _RADIUS = 7
    _GAP = 2
    _PAD_Y = 8

    def __init__(self, tool_actions, rail_labels, parent=None):
        """
        :param tool_actions: The draw tools as ``{tool id: QAction}`` in rail
            order; each becomes one entry's default action.
        :type tool_actions: dict
        :param rail_labels: Short rail text per tool id, for the tools whose
            menu label does not fit the narrow rail.
        :type rail_labels: dict
        """
        super().__init__(parent)
        self.buttons = {}
        self.setFixedWidth(self._WIDTH)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, self._PAD_Y, 0, self._PAD_Y)
        layout.setSpacing(self._GAP)
        for tool, action in tool_actions.items():
            button = QtWidgets.QToolButton(self)
            # The default action drives the button and reflects its checked
            # state, so a menu radio and a button stay one selection.
            button.setDefaultAction(action)
            # The action's menu text is too wide for the rail, so the button
            # shows the short rail name and keeps the menu text as its tip.
            button.setText(rail_labels.get(tool, action.text()))
            button.setFixedWidth(self._ENTRY_WIDTH)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            layout.addWidget(button, 0, QtCore.Qt.AlignHCenter)
            self.buttons[tool] = button

        layout.addStretch(1)
        self._apply_style()

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), _shade(self, self._SHADE_MIX))

    def _apply_style(self):
        """Color the entries from the current palette."""
        palette = self.palette()
        accent = palette.color(QtGui.QPalette.Highlight)
        on_accent = palette.color(QtGui.QPalette.HighlightedText)
        self.setStyleSheet(f"""
            QToolButton {{
                border: none;
                border-radius: {self._RADIUS}px;
                padding: 4px 0;
                background: transparent;
                color: {palette.color(QtGui.QPalette.WindowText).name()};
            }}
            QToolButton:hover {{
                background: {_shade(self, self._HOVER_MIX).name()};
            }}
            QToolButton:checked {{
                background: {accent.name()};
                color: {on_accent.name()};
            }}
            """)


class _SegmentedTabs(_PaletteStyled):
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
        pill = _shade(self, self._PILL_MIX)
        self.setStyleSheet(f"""
            QPushButton {{
                border: none;
                border-radius: {self._RADIUS}px;
                padding: 5px 0;
                font-size: {self._FONT_PX}px;
                background: transparent;
                color: {_blend(text, panel, self._LABEL_MIX).name()};
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
    """The Painter dock body: the tool rail left of the inspector.

    The rail holds one button per draw tool, each a view of the shared
    ``draw.tool`` action the Canvas menu also shows. The inspector stacks the
    Design, Layers, and Canvas pages under a segmented tab row that selects
    among them. The frame is at its designed widths; the rail icons and the
    page contents arrive with the later steps of the redesign, so every page
    is a greyed-out placeholder naming what it will hold.
    """

    # The design's inspector width, in device-independent pixels. It is the
    # floor rather than a fixed size: a dock the user widens hands the extra
    # room to the inspector instead of leaving bands beside the panel.
    _INSPECTOR_WIDTH = 264

    #: Inspector pages as ``(name, placeholder)``, in tab order.
    PAGES = (
        ("Design", "Selection, position, stroke, fill, and grid controls"),
        ("Layers", "Object list, filters, and visibility"),
        ("Canvas", "View, grid, axes, background, and units"),
    )

    def __init__(self, tool_actions, rail_labels=None, parent=None):
        """
        :param tool_actions: The draw tools as ``{tool id: QAction}`` in rail
            order; each becomes one rail button's default action.
        :type tool_actions: dict
        :param rail_labels: Short rail text per tool id, for the tools whose
            menu label does not fit the narrow rail.
        :type rail_labels: dict or None
        """
        super().__init__(parent)
        self._rail = _ToolRail(tool_actions, rail_labels or {}, self)
        self._stack = QtWidgets.QStackedWidget()
        self._tabs = _SegmentedTabs([name for name, _text in self.PAGES])
        self._tabs.selected.connect(self.show_page)
        self._inspector = self._build_inspector()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._rail)
        layout.addWidget(_rule(QtWidgets.QFrame.VLine))
        layout.addWidget(self._inspector, 1)

    @property
    def tool_buttons(self):
        """The rail buttons as ``{tool id: QToolButton}``."""
        return self._rail.buttons

    @property
    def tabs(self):
        """The inspector tab buttons as ``{page name: QPushButton}``."""
        return self._tabs.buttons

    def _build_inspector(self):
        """Build the tab row over the stack of pages."""
        for _name, placeholder in self.PAGES:
            self._stack.addWidget(self._build_page(placeholder))
        inspector = QtWidgets.QWidget(self)
        inspector.setMinimumWidth(self._INSPECTOR_WIDTH)
        layout = QtWidgets.QVBoxLayout(inspector)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._tabs)
        layout.addWidget(_rule(QtWidgets.QFrame.HLine))
        layout.addWidget(self._stack, 1)
        return inspector

    def _build_page(self, placeholder):
        """Build one inspector page, for now just its greyed-out summary."""
        page = QtWidgets.QWidget(self._stack)
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        label = QtWidgets.QLabel(placeholder, page)
        label.setWordWrap(True)
        label.setEnabled(False)
        layout.addWidget(label)
        layout.addStretch(1)
        return page

    def current_page(self):
        """The name of the page the inspector shows."""
        return self.PAGES[self._stack.currentIndex()][0]

    def show_page(self, name):
        """Show the page named ``name`` and check its tab."""
        names = [page for page, _text in self.PAGES]
        self._stack.setCurrentIndex(names.index(name))
        self.tabs[name].setChecked(True)


class Painter(_gui_common.PilotFeature):
    """
    Painter toolbox for drawing shapes on a 2D canvas, toggled from the View
    "Panels" submenu. The selected tool is held by the manager and applied
    to the focused 2D canvas, not bound to any one canvas.

    The dock body is a :class:`PainterPanel`; this feature owns the dock, the
    menu toggle, and the draw-tool actions the panel's rail shows.
    """

    # Button label for a tool id. The ids, their order, and the default
    # come from the C++ registry; this only supplies human-facing text.
    # A tool with no entry here falls back to its title-cased id.
    TOOL_LABELS = {
        "pan": "Pan / Move",
        "line": "Line",
        "triangle": "Triangle",
        "rectangle": "Rectangle",
        "ellipse": "Ellipse",
        "circle": "Circle",
    }

    # Rail text for a tool whose menu label overflows the 64px rail. The
    # names are the design's; a tool with no entry keeps its menu label.
    RAIL_LABELS = {
        "pan": "Pan",
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
        truth shared by the Canvas/Draw tool radio items and the rail
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
        """Create the dock once, its rail buttons views of the tool actions."""
        if self._dock is not None:
            return
        # A standalone Painter (used in tests) reaches the dock without
        # populate_menu, so make sure the tool actions exist first.
        self._build_tool_actions()
        dock = QtWidgets.QDockWidget("Painter", self._mainWindow)
        dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        self._panel = PainterPanel(self._tool_actions, self.RAIL_LABELS, dock)
        dock.setWidget(self._panel)
        self._mainWindow.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        # Keep the menu check in sync when the dock is closed by its button.
        if self._action is not None:
            dock.visibilityChanged.connect(self._action.setChecked)
        self._dock = dock

    def present(self):
        """Show the Painter dock and reset the focused canvas to the default
        tool; the action group updates every surface."""
        self._ensure_dock()
        self._mgr.setDrawTool(default_draw_tool_name())
        self._dock.show()
        self._dock.raise_()


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Screenshots and movies of the running pilot.
"""

import itertools
import os
import re

from PySide6.QtCore import (
    QCoreApplication, QEvent, QObject, QPoint, QRect, Qt, QTimer)
from PySide6.QtGui import QCursor, QGuiApplication, QPainter
from PySide6.QtWidgets import (
    QApplication, QDockWidget, QLabel, QRubberBand, QToolTip, QWidget)

from . import _gui_common
from .. import visual

__all__ = [
    'Recording',
    'ScreenCapture',
    'WidgetCapture',
    'grab_with_popups',
    'settle_events',
]


def settle_events(rounds=3):
    """Deliver the posted events so a grab shows what they change."""
    for _ in range(rounds):
        QCoreApplication.sendPostedEvents(None, 0)


def grab_with_popups(widget, region=None):
    """Grab ``widget``, or ``region`` of it, with the open popups and
    tooltips over it."""
    region = QRect() if region is None else QRect(region)
    pixmap = widget.grab(region) if region.isValid() else widget.grab()
    if pixmap.isNull():
        return pixmap
    origin = widget.mapToGlobal(
        region.topLeft() if region.isValid() else QPoint(0, 0))
    painter = None
    for top in QApplication.topLevelWidgets():
        if (top.isHidden() or top is widget.window()
                or top.windowType() not in (Qt.Popup, Qt.ToolTip)):
            continue
        if painter is None:
            painter = QPainter(pixmap)
        painter.drawPixmap(top.mapToGlobal(QPoint(0, 0)) - origin,
                           top.grab())
    if painter is not None:
        painter.end()
    return pixmap


class WidgetCapture(object):
    """Save what widgets show as PNG files in one folder.

    :ivar folder: Directory the images are written to, made on first use.
    """

    def __init__(self, folder):
        self.folder = folder

    def next_name(self, pattern, suffix='.png', **fields):
        """The first name from ``pattern`` that names no file yet.

        ``pattern`` formats ``fields`` and a counter ``n``.
        """
        def free(name):
            return not os.path.exists(self.path(name, suffix))

        if '{n' not in pattern:
            name = pattern.format(**fields)
            if free(name):
                return name
            pattern += '_{n}'
        for n in itertools.count(1):
            name = pattern.format(n=n, **fields)
            if free(name):
                return name

    def path(self, name, suffix='.png'):
        """Where the file ``name`` goes in the folder."""
        if not name or os.sep in name or (os.altsep and os.altsep in name):
            raise ValueError(f"'{name}' is not a file name")
        return os.path.join(self.folder, name + suffix)

    def shot(self, widget, name, region=None):
        """Save ``widget``, or ``region`` of it, as ``name``.png and return
        the path."""
        path = self.path(name)
        settle_events()
        if isinstance(widget, QWidget):
            pixmap = grab_with_popups(widget, region)
        else:
            pixmap = widget.grab()
        os.makedirs(self.folder, exist_ok=True)
        if pixmap.isNull() or not pixmap.save(path, 'PNG'):
            raise RuntimeError(f"the widget gave no image for '{name}'")
        return path


def _slug(text):
    """``text`` as a file-name stem."""
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_') or 'widget'


class ScreenCapture(_gui_common.PilotFeature):
    """Save PNG files and movies of pilot windows, docks, and widgets.

    A target is None or ``'main'``, ``'active'``, a dock title, or a widget.

    :ivar folder: Directory the files go to.
    :ivar recording: The :class:`Recording` running, or None.
    """

    MAIN = 'main'
    ACTIVE = 'active'
    PATTERN = '{target}_{n:02}'
    FRAME_MS = 80

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.folder = os.getcwd()
        self.recording = None
        self._record_action = None
        self._picker = None
        self._region = None
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.stop_recording)

    def populate_menu(self):
        self._mgr.menu_model.menu("File/Capture", weight=40)
        self.add_action(
            "File/Capture", "Main window",
            "Save a PNG of the whole pilot window",
            lambda: self._shoot_reporting(self.MAIN),
            id="file.capture.main", weight=10)
        self.add_action(
            "File/Capture", "Active sub-window",
            "Save a PNG of the active sub-window",
            lambda: self._shoot_reporting(self.ACTIVE),
            id="file.capture.subwindow", weight=20)
        self.add_action(
            "File/Capture", "Drag a region...",
            "Drag a rectangle over the window to save; Escape cancels",
            self.pick_region, id="file.capture.region", weight=30)
        self.add_action(
            "File/Capture", "Pick a widget...",
            "Click a widget to save a PNG of it; Escape cancels",
            self.pick, id="file.capture.pick", weight=40)
        self._record_action = self.add_action(
            "File/Capture", "Record movie",
            "Record the main window until chosen again",
            None, id="file.capture.record", weight=50, checkable=True,
            checked=False)
        self._record_action.triggered.connect(self._on_record_triggered)

    def shoot(self, target=None, region=None):
        """Save a PNG of ``target`` and return its path.

        ``region`` is a :class:`QRect` or ``(x, y, width, height)``.
        """
        widget, label = self._resolve(target)
        capture = WidgetCapture(self.folder)
        name = capture.next_name(self.PATTERN, target=label)
        if region is not None:
            region = QRect(*region) if isinstance(region, tuple) else region
        return capture.shot(widget, name, region)

    def record(self, target=None):
        """Record a movie of ``target`` and return its :class:`Recording`."""
        if self.recording is not None:
            raise ValueError("a recording is running; stop it first")
        widget, label = self._resolve(target)
        capture = WidgetCapture(self.folder)
        suffix = visual.default_movie_suffix()
        name = capture.next_name(self.PATTERN, suffix=suffix, target=label)
        recording = self.recording = Recording(
            self, widget, capture.path(name, suffix), self.FRAME_MS)
        self._sync_recording()
        recording.frame()
        return recording

    def stop_recording(self):
        """Write the running recording and return its path, or None."""
        if self.recording is None:
            return None
        return self.recording.stop()

    def pick(self):
        """Let the user click the widget to shoot."""
        if self._picking():
            return
        if self._picker is None:
            self._picker = _WidgetPicker(self._on_picked, parent=self)
        self._picker.start()

    def pick_region(self):
        """Let the user drag the rectangle of the main window to shoot."""
        if self._picking():
            return
        if self._region is None:
            self._region = _RegionPicker(self._mainWindow, self._on_region,
                                         parent=self)
        self._region.start()

    def _picking(self):
        return any(p is not None and p.active
                   for p in (self._picker, self._region))

    def _shoot_reporting(self, target, region=None):
        try:
            path = self.shoot(target, region=region)
        except (ValueError, RuntimeError, OSError) as exc:
            self._report(f"Capture: {exc}")
        else:
            self._report(f"Capture: saved {path}")

    def _on_picked(self, widget):
        if widget is None:
            self._report("Capture: pick cancelled")
        else:
            self._shoot_reporting(widget)

    def _on_region(self, rect):
        if rect is None:
            self._report("Capture: region cancelled")
        else:
            self._shoot_reporting(self.MAIN, region=rect)

    def _resolve(self, target):
        """``(widget, file label)`` for ``target``."""
        if isinstance(target, QWidget):
            return target, _slug(target.objectName()
                                 or type(target).__name__)
        if target is None or target == self.MAIN:
            return self._mainWindow, self.MAIN
        if target == self.ACTIVE:
            sub = self._mgr.mdiArea.activeSubWindow()
            if sub is None:
                raise ValueError("no sub-window is active")
            return sub, _slug(sub.windowTitle())
        for dock in self._mainWindow.findChildren(QDockWidget):
            if dock.windowTitle() == target and not dock.isHidden():
                return dock, _slug(target)
        raise ValueError(f"no window or open dock is called '{target}'")

    def _on_record_triggered(self, checked):
        if not checked:
            self.stop_recording()
            return
        try:
            recording = self.record(self.MAIN)
        except (ValueError, RuntimeError, OSError) as exc:
            self._report(f"Capture: {exc}")
        else:
            self._report(f"Capture: recording {recording.path}")
        self._sync_recording()

    def _recorded(self, recording, nframe, error):
        self.recording = None
        if error is None:
            self._report(f"Capture: wrote {nframe} frames to "
                         f"{recording.path}")
        else:
            self._report(f"Capture: wrote no movie: {error}")
        self._sync_recording()

    def _sync_recording(self):
        if self._record_action is not None:
            self._record_action.setChecked(self.recording is not None)

    def _report(self, message):
        self._pycon.writeToHistory(message + "\n")


class Recording(object):
    """A running movie of one widget; :meth:`stop` writes it.

    :ivar path: Where the movie is written.
    """

    def __init__(self, owner, widget, path, every_ms):
        self._owner = owner
        self._widget = widget
        self.path = path
        self._movie = visual.MovieRecorder(frame_ms=every_ms)
        self._timer = QTimer()
        self._timer.setInterval(every_ms)
        self._timer.timeout.connect(self.frame)
        self._timer.start()

    @property
    def nframe(self):
        """Frames held so far."""
        return 0 if self._movie is None else self._movie.nframe

    def frame(self):
        """Hold one frame; stop if the widget is gone."""
        if self._movie is None:
            return
        try:
            self._movie.capture(self._widget)
        except RuntimeError:
            self.stop()

    def stop(self):
        """Write the movie and return its path, or None."""
        if self._movie is None:
            return None
        self._timer.stop()
        try:
            self._movie.capture(self._widget)
        except RuntimeError:
            pass
        movie, self._movie = self._movie, None
        try:
            nframe = movie.write(self.path)
        except (OSError, ValueError) as exc:
            self._owner._recorded(self, 0, exc)
            return None
        finally:
            movie.close()
        self._owner._recorded(self, nframe, None)
        return self.path


class _Hint(QLabel):
    """A tooltip-like label that stays put."""

    def __init__(self):
        super().__init__(None, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setPalette(QToolTip.palette())
        self.setAutoFillBackground(True)
        self.setMargin(4)

    def show_by(self, rect, text):
        """Show ``text`` below the global ``rect``, or above it."""
        self.setText(text)
        self.adjustSize()
        spot = rect.bottomLeft() + QPoint(0, 4)
        screen = QGuiApplication.screenAt(rect.center())
        if (screen is not None and spot.y() + self.height()
                > screen.availableGeometry().bottom()):
            spot = rect.topLeft() - QPoint(0, self.height() + 4)
        self.move(spot)
        self.show()
        self.raise_()


class _WidgetPicker(QObject):
    """Outline the widget under the pointer until one is clicked."""

    POLL_MS = 50

    def __init__(self, done, parent=None):
        super().__init__(parent)
        self._done = done
        self._band = QRubberBand(QRubberBand.Rectangle)
        self._band.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._hint = _Hint()
        self._timer = QTimer(self)
        self._timer.setInterval(self.POLL_MS)
        self._timer.timeout.connect(self._follow)
        self._under = None
        self._chain = []
        self._level = 0
        self.active = False

    def start(self):
        if self.active:
            return
        self.active = True
        self._under = None
        QApplication.instance().installEventFilter(self)
        QApplication.setOverrideCursor(Qt.CrossCursor)
        self._timer.start()

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind == QEvent.Wheel:
            if self._chain:
                step = 1 if event.angleDelta().y() > 0 else -1
                self._level = min(max(self._level + step, 0),
                                  len(self._chain) - 1)
                self._outline()
            return True
        if kind == QEvent.MouseButtonPress:
            widget = self._current()
            self._finish(widget if event.button() == Qt.LeftButton
                         else None)
            return True
        if kind in (QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick):
            return True
        if kind == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            self._finish(None)
            return True
        return False

    def _follow(self):
        widget = QApplication.widgetAt(QCursor.pos())
        if widget is None or widget is self._under:
            return
        self._under = widget
        self._chain = []
        while widget is not None:
            self._chain.append(widget)
            widget = None if widget.isWindow() else widget.parentWidget()
        self._level = 0
        self._outline()

    def _current(self):
        return self._chain[self._level] if self._chain else None

    def _outline(self):
        widget = self._current()
        if widget is None:
            return
        rect = QRect(widget.mapToGlobal(QPoint(0, 0)), widget.size())
        self._band.setGeometry(rect)
        self._band.show()
        name = widget.objectName()
        self._hint.show_by(rect, f"{type(widget).__name__}"
                           + (f" '{name}'" if name else "")
                           + f"  {widget.width()} x {widget.height()}"
                           "\nwheel: parent / child, click: save, "
                           "Esc: cancel")

    def _finish(self, widget):
        self.active = False
        self._timer.stop()
        QApplication.instance().removeEventFilter(self)
        QApplication.restoreOverrideCursor()
        self._band.hide()
        self._hint.hide()
        self._done(widget)


class _RegionPicker(QObject):
    """Let the user drag a rectangle over a window; None if cancelled."""

    def __init__(self, window, done, parent=None):
        super().__init__(parent)
        self._window = window
        self._done = done
        self._band = QRubberBand(QRubberBand.Rectangle)
        self._band.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._hint = _Hint()
        self._origin = None
        self.active = False

    def start(self):
        if self.active:
            return
        self.active = True
        self._origin = None
        QApplication.instance().installEventFilter(self)
        QApplication.setOverrideCursor(Qt.CrossCursor)

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind == QEvent.MouseButtonPress:
            if event.button() != Qt.LeftButton:
                self._finish(None)
            else:
                self._origin = event.globalPosition().toPoint()
            return True
        if kind == QEvent.MouseMove and self._origin is not None:
            rect = self._rect(event.globalPosition().toPoint())
            self._band.setGeometry(rect)
            self._band.show()
            self._hint.show_by(rect, f"{rect.width()} x {rect.height()}")
            return True
        if kind == QEvent.MouseButtonRelease and self._origin is not None:
            rect = self._rect(event.globalPosition().toPoint())
            local = QRect(self._window.mapFromGlobal(rect.topLeft()),
                          rect.size()).intersected(self._window.rect())
            self._finish(local if local.width() > 2 and local.height() > 2
                         else None)
            return True
        if kind in (QEvent.MouseButtonRelease, QEvent.MouseButtonDblClick,
                    QEvent.Wheel):
            return True
        if kind == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            self._finish(None)
            return True
        return False

    def _rect(self, point):
        return QRect(self._origin, point).normalized()

    def _finish(self, rect):
        self.active = False
        QApplication.instance().removeEventFilter(self)
        QApplication.restoreOverrideCursor()
        self._band.hide()
        self._hint.hide()
        self._done(rect)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

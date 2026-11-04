# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Screenshots and movies of the running pilot.
"""

import itertools
import os
import re
import time

import numpy as np
from PySide6.QtCore import (
    QCoreApplication, QEvent, QObject, QPoint, QRect, Qt, QTimer, Signal)
from PySide6.QtGui import QCursor, QGuiApplication, QImage, QPainter
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
    'to_logical_size',
    'trim_edges',
]


def settle_events(rounds=3):
    """Deliver the posted events so a grab shows what they change."""
    for _ in range(rounds):
        QCoreApplication.sendPostedEvents(None, 0)


def to_logical_size(pixmap):
    """``pixmap`` at the size its widget is drawn, not its backing store.

    A high-DPI grab comes back at the display's pixel count, so one window
    would save at two sizes on two screens.  A null pixmap reports a ratio of
    one and passes through, leaving the caller its own grab to reject.
    """
    if pixmap.devicePixelRatio() <= 1:
        return pixmap
    pixmap = pixmap.scaled(pixmap.deviceIndependentSize().toSize(),
                           Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    # Or the PNG claims a resolution the frame no longer has.
    pixmap.setDevicePixelRatio(1)
    return pixmap


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


def trim_edges(pixmap):
    """``pixmap`` with its plain edges cut off.

    The bottom-left pixel gives the background color.
    """
    if pixmap.isNull():
        return pixmap
    image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
    width, height = image.width(), image.height()
    pixels = np.frombuffer(image.constBits(), dtype='uint8').reshape(
        height, image.bytesPerLine())[:, :width * 4].reshape(height, width, 4)
    plain = (pixels == pixels[-1, 0]).all(axis=2)
    rows = np.flatnonzero(~plain.all(axis=1))
    cols = np.flatnonzero(~plain.all(axis=0))
    if not len(rows):
        x0, y0, x1, y1 = 0, 0, 1, 1
    else:
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
    if (x0, y0, x1, y1) == (0, 0, width, height):
        return pixmap
    trimmed = pixmap.copy(x0, y0, x1 - x0, y1 - y0)
    trimmed.setDevicePixelRatio(pixmap.devicePixelRatio())
    return trimmed


class WidgetCapture(object):
    """Save what widgets show as PNG files in one folder.

    :ivar folder: Directory the images are written to, made on first use.
    :ivar logical: Whether to scale a grab down to the widget's own size.
    :ivar trim: Whether to cut the plain edges off.
    """

    def __init__(self, folder, logical=False, trim=False):
        self.folder = folder
        self.logical = logical
        self.trim = trim

    def next_name(self, pattern, suffix='.png', taken=(), **fields):
        """The first name from ``pattern`` that names no file yet.

        ``pattern`` formats ``fields`` and a counter ``n``.  ``taken`` holds
        paths reserved but not written.

        :raises ValueError: ``pattern`` does not format.
        """
        def fill(text, n):
            try:
                return text.format(n=n, **fields)
            except (KeyError, IndexError, AttributeError, TypeError) as exc:
                raise ValueError(f"the name pattern '{text}' does not "
                                 f"format: {exc}") from exc

        def free(name):
            path = self.path(name, suffix)
            return path not in taken and not os.path.exists(path)

        name = fill(pattern, 1)
        if name == fill(pattern, 2):
            if free(name):
                return name
            pattern += '_{n}'
        for n in itertools.count(1):
            name = fill(pattern, n)
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
        if self.trim:
            pixmap = trim_edges(pixmap)
        if self.logical:
            pixmap = to_logical_size(pixmap)
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
    :ivar pattern: Name pattern over ``{target}``, ``{n}`` and ``{time}``.
    :ivar logical: Whether to scale a high-DPI grab down to 1x.
    :ivar trim: Whether to cut the plain edges off.
    :ivar delay: Seconds a shot from the GUI waits.
    :ivar movie_suffix: Format of a movie, or None for the build's default.
    :ivar frame_ms: Milliseconds between the frames a timer records.
    :ivar shots: The files written this session, oldest first.
    :ivar recording: The :class:`Recording` running, or None.
    """

    #: A file was written: its path, a poster pixmap or None, and a detail.
    saved = Signal(str, object, str)
    #: A message for the user, and whether it reports success.
    reported = Signal(str, bool)
    #: The running :class:`Recording` changed, to it or to None.
    recording_changed = Signal(object)

    MAIN = 'main'
    ACTIVE = 'active'
    PICK = 'pick'
    REGION = 'region'
    PATTERN = '{target}_{n:02}'
    FRAME_MS = 80
    FRAME_MS_RANGE = (20, 2000)
    DELAY_RANGE = (0.0, 10.0)

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.folder = os.getcwd()
        self.pattern = self.PATTERN
        self.logical = False
        self.trim = False
        self.delay = 0.0
        self.movie_suffix = None
        self.frame_ms = self.FRAME_MS
        self.shots = []
        self.recording = None
        self._pending = set()
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
            lambda: self.shoot_reporting(self.MAIN),
            id="file.capture.main", weight=10)
        self.add_action(
            "File/Capture", "Active sub-window",
            "Save a PNG of the active sub-window",
            lambda: self.shoot_reporting(self.ACTIVE),
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

    def console_handles(self):
        """The console's capture helpers with their banner lines."""
        handles = {
            'capture': self.shoot,
            'record': self.record,
        }
        entries = [
            ('capture(target)', 'save a PNG of the window, a dock, or widget'),
            ('record(target)', 'record a movie; stop() writes it'),
        ]
        return handles, entries

    def shoot(self, target=None, name=None, *, region=None, delay=0,
              folder=None, logical=None, trim=None):
        """Save a PNG of ``target``.

        :param target: The target, as the class describes it.
        :param name: Name pattern over ``{target}``, ``{n}`` and ``{time}``.
            Defaults to :attr:`pattern`.
        :param region: The part to keep, a :class:`QRect` or
            ``(x, y, width, height)``.
        :param delay: Seconds to wait before writing the file.
        :param folder: Directory of the file.  Defaults to :attr:`folder`.
        :param logical: Whether to scale a high-DPI grab down to 1x.  Defaults
            to :attr:`logical`.
        :param trim: Whether to cut the plain edges off.  Defaults to
            :attr:`trim`.
        :return: The path of the file.
        """
        widget, label = self._resolve(target)
        capture = WidgetCapture(
            self.folder if folder is None else os.path.abspath(folder),
            logical=self.logical if logical is None else logical,
            trim=self.trim if trim is None else trim)
        name = capture.next_name(
            self.pattern if name is None else name, taken=self._pending,
            target=label, time=time.strftime('%Y%m%d_%H%M%S'))
        if region is not None:
            region = QRect(*region) if isinstance(region, tuple) else region
        if delay <= 0:
            return self._write(capture, widget, name, region)
        path = capture.path(name)
        self._pending.add(path)
        QTimer.singleShot(round(delay * 1000), lambda: self._write_later(
            capture, widget, name, region, path))
        return path

    def record(self, target=None, name=None, *, every_ms=None, suffix=None,
               folder=None):
        """Record a movie of ``target``.

        :param target: The target, as the class describes it.
        :param name: Name pattern over ``{target}``, ``{n}`` and ``{time}``.
            Defaults to :attr:`pattern`.
        :param every_ms: Milliseconds between the frames.  Defaults to
            :attr:`frame_ms`.
        :param suffix: Format of the movie.  Defaults to :attr:`movie_suffix`.
        :param folder: Directory of the movie.  Defaults to :attr:`folder`.
        :return: The running recording.
        :rtype: Recording
        """
        if self.recording is not None:
            raise ValueError("a recording is running; stop it first")
        every_ms = self.frame_ms if every_ms is None else every_ms
        if every_ms <= 0:
            raise ValueError(f"every_ms must be positive, not {every_ms}")
        suffix = suffix or self.movie_suffix or visual.default_movie_suffix()
        if suffix not in visual.movie_suffixes():
            raise ValueError(f"this build cannot write '{suffix}'; it "
                             f"writes {', '.join(visual.movie_suffixes())}")
        widget, label = self._resolve(target)
        capture = WidgetCapture(
            self.folder if folder is None else os.path.abspath(folder))
        name = capture.next_name(
            self.pattern if name is None else name, suffix=suffix,
            target=label, time=time.strftime('%Y%m%d_%H%M%S'))
        recording = self.recording = Recording(
            self, widget, capture.path(name, suffix), every_ms)
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

    def shoot_reporting(self, target=None, region=None):
        """:meth:`shoot` after :attr:`delay` and report instead of raising."""
        if target == self.PICK:
            self.pick()
            return None
        if target == self.REGION:
            self.pick_region()
            return None
        try:
            path = self.shoot(target, region=region, delay=self.delay)
        except (ValueError, RuntimeError, OSError) as exc:
            self._report(f"Capture: {exc}", ok=False)
            return None
        if path in self._pending:
            self._report(f"Capture: saving {os.path.basename(path)} in "
                         f"{self.delay:g} s", ok=True)
        else:
            self._report(f"Capture: saved {path}", ok=True)
        return path

    def record_reporting(self, target=None):
        """:meth:`record` and report instead of raising."""
        try:
            recording = self.record(target)
        except (ValueError, RuntimeError, OSError) as exc:
            self._report(f"Capture: {exc}", ok=False)
        else:
            # A first frame that fails has already ended it and reported why.
            if recording is self.recording:
                self._report(f"Capture: recording {recording.path}", ok=True)
        self._sync_recording()

    def _picking(self):
        return any(p is not None and p.active
                   for p in (self._picker, self._region))

    def _on_picked(self, widget):
        if widget is None:
            self._report("Capture: pick cancelled", ok=True)
        else:
            self.shoot_reporting(widget)

    def _on_region(self, rect):
        if rect is None:
            self._report("Capture: region cancelled", ok=True)
        else:
            self.shoot_reporting(self.MAIN, region=rect)

    def _resolve(self, target):
        """``(widget, file label)`` for ``target``."""
        if isinstance(target, QWidget):
            return target, _slug(target.objectName() or target.windowTitle()
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

    def _write(self, capture, widget, name, region):
        path = capture.shot(widget, name, region)
        self._add_file(path)
        return path

    def _write_later(self, capture, widget, name, region, path):
        self._pending.discard(path)
        try:
            self._write(capture, widget, name, region)
        except (ValueError, RuntimeError, OSError) as exc:
            self._report(f"Capture: {exc}", ok=False)
        else:
            self._report(f"Capture: saved {path}", ok=True)

    def _add_file(self, path, poster=None, detail=""):
        self.shots.append(path)
        self.saved.emit(path, poster, detail)

    def _on_record_triggered(self, checked):
        if checked:
            self.record_reporting(self.MAIN)
        else:
            self.stop_recording()

    def _recorded(self, recording, nframe, error):
        self.recording = None
        if error is None:
            self._add_file(recording.path, recording.poster,
                           f"{nframe} frames")
            self._report(f"Capture: wrote {nframe} frames to "
                         f"{recording.path}", ok=True)
        else:
            self._report(f"Capture: wrote no movie: {error}", ok=False)
        self._sync_recording()

    def _sync_recording(self):
        if self._record_action is not None:
            self._record_action.setChecked(self.recording is not None)
        self.recording_changed.emit(self.recording)

    def _report(self, message, ok):
        self._pycon.writeToHistory(message + "\n")
        self.reported.emit(message, ok)


class Recording(object):
    """A running movie of one widget; :meth:`stop` writes it.

    :ivar path: Where the movie is written.
    :ivar poster: The first frame.
    """

    def __init__(self, owner, widget, path, every_ms):
        self._owner = owner
        self._widget = widget
        self.path = path
        self.poster = to_logical_size(widget.grab())
        self._movie = visual.MovieRecorder(frame_ms=every_ms)
        self._started = time.monotonic()
        self._timer = QTimer()
        self._timer.setInterval(every_ms)
        self._timer.timeout.connect(self.frame)
        self._timer.start()

    @property
    def nframe(self):
        """Frames held so far."""
        return 0 if self._movie is None else self._movie.nframe

    @property
    def elapsed(self):
        """Seconds since the recording started."""
        return time.monotonic() - self._started

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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    def __repr__(self):
        state = 'stopped' if self._movie is None else 'recording'
        return f"<Recording {state} {self.path}>"


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

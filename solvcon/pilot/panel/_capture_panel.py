# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The Capture dock over the pilot's screen capture.
"""

import os
import subprocess
import sys

from PySide6.QtCore import QMimeData, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication, QIcon, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QRadioButton,
    QSpinBox, QToolButton, QVBoxLayout, QWidget)

from .. import base
from .. import visual

__all__ = [
    'CaptureWidget',
]


def _thumbnail(pixmap):
    return pixmap.scaled(192, 108, Qt.KeepAspectRatio,
                         Qt.SmoothTransformation)


class CaptureWidget(base.PilotFeature):
    """The Capture dock: settings, shot and record buttons, recent files.

    :ivar capture: The :class:`~solvcon.pilot.base.ScreenCapture` it drives.
    """

    def __init__(self, *args, capture, **kw):
        super().__init__(*args, **kw)
        self.capture = capture
        self._action = None
        self._dock = None
        self._column = None
        capture.saved.connect(self._on_saved)
        capture.reported.connect(self._on_reported)
        capture.recording_changed.connect(self._on_recording_changed)

    def populate_menu(self):
        self._action = self.add_action(
            "View/Panels", "Capture", "Open / Close the Capture panel",
            None, id="panel.capture", weight=45, checkable=True,
            checked=False)
        self._action.toggled.connect(self._on_toggled)

    def targets(self):
        """The dock's ``(text, target)`` choices."""
        capture = self.capture
        found = [("Main window", capture.MAIN),
                 ("Active sub-window", capture.ACTIVE)]
        for dock in self._mainWindow.findChildren(QDockWidget):
            if not dock.isHidden() and dock.windowTitle():
                found.append((f"Dock: {dock.windowTitle()}",
                              dock.windowTitle()))
        found.append(("Drag a region...", capture.REGION))
        found.append(("Pick a widget...", capture.PICK))
        return found

    def _on_saved(self, path, poster, detail):
        if self._column is not None:
            icon = None if poster is None else _thumbnail(poster)
            self._column.add_shot(path, icon, detail or None)

    def _on_reported(self, message, ok):
        if self._column is not None:
            self._column.set_status(message, ok)

    def _on_recording_changed(self, recording):
        if self._column is not None:
            self._column.sync_recording(recording)

    def _on_toggled(self, checked):
        if checked:
            self._ensure_dock()
            self._dock.show()
            self._dock.raise_()
        elif self._dock is not None:
            self._dock.hide()

    def _ensure_dock(self):
        if self._dock is not None:
            return
        self._column = _CaptureColumn(self)
        for path in self.capture.shots:
            self._column.add_shot(path)
        self._dock = QDockWidget("Capture")
        self._dock.setWidget(self._column)
        self._mainWindow.addDockWidget(Qt.RightDockWidgetArea, self._dock)
        # Floating keeps the dock out of main-window shots.
        self._dock.setFloating(True)
        self._dock.resize(self._dock.sizeHint())
        frame = self._mainWindow.frameGeometry()
        self._dock.move(frame.right() - self._dock.width(), frame.top())
        self._dock.visibilityChanged.connect(self._sync_action)

    def _sync_action(self, _visible):
        # The signal is also False for a dock tabbed behind another.
        self._action.setChecked(not self._dock.isHidden())


class _TargetCombo(QComboBox):
    """A combo that refreshes its entries each time it opens."""

    def __init__(self, entries, parent=None):
        super().__init__(parent)
        self._entries = entries
        self.refresh()

    def refresh(self):
        current = self.currentData()
        self.clear()
        for text, target in self._entries():
            self.addItem(text, target)
        index = self.findData(current)
        self.setCurrentIndex(max(index, 0))

    def showPopup(self):
        self.refresh()
        super().showPopup()


class _ShotList(QListWidget):
    """Recent files, newest first, draggable as files."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setIconSize(QSize(96, 54))
        self.setDragEnabled(True)

    def mimeData(self, items):
        data = QMimeData()
        data.setUrls([QUrl.fromLocalFile(it.data(Qt.UserRole))
                      for it in items])
        return data


class _CaptureColumn(QWidget):
    """The Capture dock's contents."""

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self._owner = owner
        capture = self._capture = owner.capture

        self._target = _TargetCombo(owner.targets)
        self._folder = QLineEdit(capture.folder)
        self._folder.editingFinished.connect(self._on_folder_edited)
        browse = QToolButton(text="...")
        browse.clicked.connect(self._on_browse)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self._folder)
        folder_row.addWidget(browse)
        self._pattern = QLineEdit(capture.pattern)
        self._pattern.setToolTip("{target}, {n}, and {time} are filled in")
        self._pattern.editingFinished.connect(self._on_pattern_edited)
        display, logical = QRadioButton("display"), QRadioButton("logical 1x")
        (logical if capture.logical else display).setChecked(True)
        logical.toggled.connect(self._on_logical_toggled)
        pixels = QButtonGroup(self)
        pixels.addButton(display)
        pixels.addButton(logical)
        pixels_row = QHBoxLayout()
        pixels_row.addWidget(display)
        pixels_row.addWidget(logical)
        self._trim = QCheckBox("trim plain edges")
        self._trim.setChecked(capture.trim)
        self._trim.toggled.connect(self._on_trim_toggled)
        self._delay = QDoubleSpinBox()
        self._delay.setRange(*capture.DELAY_RANGE)
        self._delay.setSingleStep(0.5)
        self._delay.setSuffix(" s")
        self._delay.setValue(capture.delay)
        self._delay.setToolTip("Wait before the shot, to open a menu")
        self._delay.valueChanged.connect(self._on_delay_changed)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.addRow("target", self._target)
        form.addRow("folder", folder_row)
        form.addRow("name", self._pattern)
        form.addRow("pixels", pixels_row)
        form.addRow("crop", self._trim)
        form.addRow("delay", self._delay)
        shoot = QPushButton("Capture")
        shoot.clicked.connect(self._on_shoot_clicked)

        self._format = QComboBox()
        for suffix in visual.movie_suffixes():
            self._format.addItem(suffix[1:].upper(), suffix)
        self._format.setCurrentIndex(
            max(self._format.findData(capture.movie_suffix), 0))
        self._format.currentIndexChanged.connect(self._on_format_changed)
        self._every = QSpinBox()
        self._every.setRange(*capture.FRAME_MS_RANGE)
        self._every.setSingleStep(20)
        self._every.setSuffix(" ms")
        self._every.setValue(capture.frame_ms)
        self._every.valueChanged.connect(self._on_every_changed)
        form.addRow(shoot)
        form.addRow(QLabel("Movie"))
        form.addRow("format", self._format)
        form.addRow("interval", self._every)
        self._record = QPushButton("Record")
        self._record.clicked.connect(self._on_record_clicked)
        form.addRow(self._record)
        self._ticker = QTimer(self)
        self._ticker.setInterval(200)
        self._ticker.timeout.connect(self._tick)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self._shots = _ShotList()
        buttons = QGridLayout()
        for i, (text, slot) in enumerate((
                ("Reveal", self._on_reveal),
                ("Copy path", self._on_copy_path),
                ("Copy image", self._on_copy_image),
                ("Delete", self._on_delete))):
            button = QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button, i // 2, i % 2)

        column = QVBoxLayout(self)
        column.addLayout(form)
        column.addWidget(self._status)
        column.addWidget(QLabel("Recent shots"))
        column.addWidget(self._shots, 1)
        column.addLayout(buttons)
        self.sync_recording(capture.recording)

    def add_shot(self, path, icon=None, detail=None):
        if icon is None:
            pixmap = QPixmap(path)
            icon = _thumbnail(pixmap) if not pixmap.isNull() else None
            if detail is None and not pixmap.isNull():
                detail = f"{pixmap.width()} x {pixmap.height()}"
        item = QListWidgetItem(os.path.basename(path)
                               + (f"\n{detail}" if detail else ""))
        if icon is not None:
            item.setIcon(QIcon(icon))
        item.setData(Qt.UserRole, path)
        item.setToolTip(path)
        self._shots.insertItem(0, item)
        self._shots.setCurrentItem(item)

    def set_status(self, message, ok):
        self._status.setText(message)
        self._status.setStyleSheet("" if ok else "color: #b3261e;")

    def sync_recording(self, recording):
        self._recording = recording
        running = recording is not None
        for widget in (self._format, self._every):
            widget.setEnabled(not running)
        if running:
            self._ticker.start()
        else:
            self._ticker.stop()
        self._tick()

    def _tick(self):
        recording = self._recording
        self._record.setText(
            "Record" if recording is None else
            f"Stop ({recording.elapsed:.1f} s, {recording.nframe} frames)")

    def _apply_fields(self):
        # A click on a button does not take the focus on macOS, so a field
        # being edited has not finished.
        self._on_folder_edited()
        self._on_pattern_edited()

    def _on_shoot_clicked(self):
        self._apply_fields()
        self._capture.shoot_reporting(self._target.currentData())

    def _on_record_clicked(self):
        capture = self._capture
        if self._recording is not None:
            capture.stop_recording()
            return
        target = self._target.currentData()
        if target in (capture.PICK, capture.REGION):
            self.set_status("Capture: choose a window or a dock to record",
                            ok=False)
            return
        self._apply_fields()
        capture.record_reporting(target)

    def _selected_path(self):
        item = self._shots.currentItem()
        return None if item is None else item.data(Qt.UserRole)

    def _on_folder_edited(self):
        self._capture.folder = os.path.abspath(
            os.path.expanduser(self._folder.text()))
        self._folder.setText(self._capture.folder)

    def _on_browse(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Folder for shots", self._capture.folder)
        if folder:
            self._folder.setText(folder)
            self._on_folder_edited()

    def _on_pattern_edited(self):
        self._capture.pattern = self._pattern.text() or self._capture.PATTERN
        self._pattern.setText(self._capture.pattern)

    def _on_logical_toggled(self, checked):
        self._capture.logical = checked

    def _on_trim_toggled(self, checked):
        self._capture.trim = checked

    def _on_delay_changed(self, value):
        self._capture.delay = value

    def _on_format_changed(self, _index):
        self._capture.movie_suffix = self._format.currentData()

    def _on_every_changed(self, value):
        self._capture.frame_ms = value

    def _on_reveal(self):
        path = self._selected_path()
        if path is None:
            return
        if sys.platform == 'darwin':
            subprocess.run(['open', '-R', path], check=False)
        else:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(os.path.dirname(path)))

    def _on_copy_path(self):
        path = self._selected_path()
        if path is not None:
            QGuiApplication.clipboard().setText(path)

    def _on_copy_image(self):
        """Copy the selected image, or a movie as a file."""
        path = self._selected_path()
        if path is None:
            return
        # A GIF or WebP movie would load as its first frame.
        pixmap = QPixmap(path) if path.endswith('.png') else QPixmap()
        if not pixmap.isNull():
            QGuiApplication.clipboard().setPixmap(pixmap)
            return
        data = QMimeData()
        data.setUrls([QUrl.fromLocalFile(path)])
        data.setText(path)
        QGuiApplication.clipboard().setMimeData(data)

    def _on_delete(self):
        item = self._shots.currentItem()
        if item is None:
            return
        path = item.data(Qt.UserRole)
        answer = QMessageBox.question(
            self, "Delete shot", f"Delete {os.path.basename(path)}?")
        if answer != QMessageBox.Yes:
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            self.set_status(f"Capture: {exc}", ok=False)
            return
        # A delayed shot may have landed while the question was open.
        self._shots.takeItem(self._shots.row(item))
        if path in self._capture.shots:
            self._capture.shots.remove(path)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

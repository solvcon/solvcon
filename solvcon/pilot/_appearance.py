# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QLabel,
                               QVBoxLayout, QHBoxLayout,
                               QGroupBox, QButtonGroup, QPushButton,
                               QRadioButton, QDialog)

from . import _gui_common


class AppearanceDialog(_gui_common.PilotFeature):
    """
    AppearanceDialog class for managing the general look
    and feel of Qt widgets.
    """
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.qApp = QApplication.instance()
        self.button_group = QButtonGroup()
        self.appearance_dialog = None

    def on_close_appearance(self):
        if self.appearance_dialog:
            self.appearance_dialog.close()

    def on_open_appearance(self):
        if self.appearance_dialog is None:
            self.appearance_dialog = self._build_dialog()
        self.appearance_dialog.show()

    def on_click_light_mode(self):
        self.qApp.styleHints().setColorScheme(Qt.ColorScheme.Light)

    def on_click_dark_mode(self):
        self.qApp.styleHints().setColorScheme(Qt.ColorScheme.Dark)

    def on_click_system_mode(self):
        self.qApp.styleHints().setColorScheme(Qt.ColorScheme.Unknown)

    def _build_dialog(self):
        appearance_dialog = QDialog(self._mainWindow)
        appearance_dialog.setWindowTitle("Appearance")
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Light/Dark Mode Toggle"))

        light_mode_button = QRadioButton("Light")
        dark_mode_button = QRadioButton("Dark")
        system_mode_button = QRadioButton("System")
        light_mode_button.clicked.connect(self.on_click_light_mode)
        dark_mode_button.clicked.connect(self.on_click_dark_mode)
        system_mode_button.clicked.connect(self.on_click_system_mode)
        light_mode_button.setFocusPolicy(Qt.NoFocus)
        dark_mode_button.setFocusPolicy(Qt.NoFocus)
        system_mode_button.setFocusPolicy(Qt.NoFocus)

        scheme = self.qApp.styleHints().colorScheme()
        if scheme == Qt.ColorScheme.Light:
            light_mode_button.setChecked(True)
        elif scheme == Qt.ColorScheme.Dark:
            dark_mode_button.setChecked(True)
        else:  # Qt.ColorScheme.Unknown -> "System"
            system_mode_button.setChecked(True)

        self.button_group.addButton(light_mode_button, 0)
        self.button_group.addButton(dark_mode_button, 1)
        self.button_group.addButton(system_mode_button, 2)

        hbox_1 = QHBoxLayout()
        hbox_1.addWidget(light_mode_button)
        hbox_1.addWidget(dark_mode_button)
        hbox_1.addWidget(system_mode_button)
        group_box_1 = QGroupBox()
        group_box_1.setLayout(hbox_1)
        layout.addWidget(group_box_1)

        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.on_close_appearance)
        layout.addWidget(ok_button)

        appearance_dialog.setLayout(layout)
        return appearance_dialog

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

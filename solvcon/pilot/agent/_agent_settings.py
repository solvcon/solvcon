# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Backend settings dialog for the Agent Console.

The dialog is built from whatever a backend advertises through
``settings_spec()``, so a backend that grows a knob needs no change here.
"""

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QComboBox, QLabel,
                               QLineEdit, QFormLayout, QVBoxLayout)

__all__ = [  # noqa: F822
    'AgentBackendSettingsDialog',
]


class AgentBackendSettingsDialog(QDialog):
    """Edit one backend's settings.  The editors start on the backend's current
    values and are written back only on accept, so cancelling leaves a running
    configuration untouched."""

    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self._backend = backend
        self._editors = {}
        self.setWindowTitle("Backend Settings")

        form = QFormLayout()
        spec = list(backend.settings_spec())
        for setting in spec:
            editor = self._build_editor(setting, backend.get_setting(
                setting.name))
            if setting.tooltip:
                editor.setToolTip(setting.tooltip)
            self._editors[setting.name] = editor
            form.addRow(setting.label, editor)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Backend: %s" % backend.name))
        if spec:
            layout.addLayout(form)
        else:
            layout.addWidget(QLabel("This backend has no settings yet."))

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _build_editor(setting, value):
        """A combo box for a fixed set of choices, a line edit otherwise."""
        if setting.choices:
            editor = QComboBox()
            editor.addItems(list(setting.choices))
            index = editor.findText(value)
            editor.setCurrentIndex(max(index, 0))
            return editor
        editor = QLineEdit()
        editor.setText(value or "")
        return editor

    @staticmethod
    def _editor_value(editor):
        return (editor.currentText() if isinstance(editor, QComboBox)
                else editor.text().strip())

    def accept(self):
        """Apply every editor, or none.  A knob the backend refuses would
        otherwise leave the earlier ones already committed while the dialog
        stays open, which is the opposite of what cancelling promises."""
        restore = self._backend.settings()
        try:
            for name, editor in self._editors.items():
                self._backend.set_setting(name, self._editor_value(editor))
        except (KeyError, ValueError):
            for name, value in restore.items():
                self._backend.set_setting(name, value)
            raise
        super().accept()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

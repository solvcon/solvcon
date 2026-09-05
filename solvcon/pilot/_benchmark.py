# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Control one isolated matmul worker from a reusable Qt widget."""

import json
import os

from PySide6 import QtCore, QtWidgets

from solvcon import system


class BenchmarkControl(QtWidgets.QWidget):
    """Show progress and emit a terminal signal after the worker exits.

    Call start with a MatmulSpec and an artifact path. A running control
    rejects another start. Stop and close kill the worker asynchronously.
    """

    completed = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    stopped = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.status = QtWidgets.QLabel('Idle', self)
        self.status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.elapsed = QtWidgets.QLabel('Elapsed: 0.0 s', self)
        self.stop_button = QtWidgets.QPushButton('Stop', self)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        layout = QtWidgets.QHBoxLayout(self)
        for widget in (self.status, self.elapsed, self.stop_button):
            layout.addWidget(widget)

        self._process = QtCore.QProcess(self)
        self._process.started.connect(self._send_request)
        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.readyReadStandardError.connect(self._read_stderr)
        self._process.errorOccurred.connect(self._process_error)
        self._process.finished.connect(self._finish)
        self._clock = QtCore.QElapsedTimer()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._update_elapsed)
        self._running = False
        self._closing = False

    @property
    def running(self):
        return self._running

    def start(self, specification, output_path):
        if self.running:
            raise RuntimeError('a benchmark is already running')
        request = {'spec': specification.to_dict(),
                   'output_path': os.fspath(output_path)}
        self._request = (json.dumps(request, allow_nan=False) + '\n').encode()
        self._kernels = specification.kernels + ('numpy',)
        self._result = None
        self._error = ''
        self._stderr = b''
        self._cancelled = False
        self._running = True
        self.status.setText('Preparing')
        self.stop_button.setEnabled(True)
        self._clock.start()
        self._update_elapsed()
        self._timer.start(100)
        command = system.python_command('-m', 'solvcon.benchmark.worker')
        self._process.start(command[0], command[1:])

    def stop(self):
        if self.running:
            self._cancelled = True
            self.status.setText('Stopping')
            self.stop_button.setEnabled(False)
            self._process.kill()

    def closeEvent(self, event):
        if self.running:
            self._closing = True
            self.stop()
            event.ignore()
        else:
            super().closeEvent(event)

    def _send_request(self):
        if self._cancelled:
            self._process.kill()
            return
        self._process.write(self._request)
        self._process.closeWriteChannel()

    def _update_elapsed(self):
        seconds = self._clock.elapsed() / 1000
        self.elapsed.setText(f'Elapsed: {seconds:.1f} s')

    def _read_stdout(self):
        while self._process.canReadLine():
            line = bytes(self._process.readLine())
            if self._cancelled or self._error:
                continue
            try:
                self._handle_event(json.loads(line))
            except ValueError as exc:
                self._fail(f'Worker protocol error: {exc}')

    def _handle_event(self, event):
        if not isinstance(event, dict) or self._result is not None:
            raise ValueError('unexpected worker event')
        kind = event.get('type')
        if kind == 'progress':
            phase, kernel = event.get('phase'), event.get('kernel')
            if (phase not in ('comparison', 'warmup', 'timing')
                    or kernel not in self._kernels):
                raise ValueError('invalid worker progress')
            self.status.setText(f'{phase.capitalize()}: {kernel}')
        elif kind == 'result':
            path = event.get('artifact_path')
            if not isinstance(path, str) or not path:
                raise ValueError('invalid worker artifact path')
            self._result = path
            self.status.setText('Finishing')
        elif kind == 'error':
            self._fail(str(event.get('message', 'worker failed')))
        else:
            raise ValueError('unknown worker event')

    def _read_stderr(self):
        data = bytes(self._process.readAllStandardError())
        self._stderr = (self._stderr + data)[-8192:]

    def _fail(self, message):
        if not self._error:
            self._error = message or 'Worker failed'
        self._process.kill()

    def _process_error(self, error):
        self._fail(self._process.errorString())
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            # Qt finishes its startup cleanup after errorOccurred returns.
            QtCore.QTimer.singleShot(
                0, self,
                lambda: self._finish(-1, QtCore.QProcess.ExitStatus.CrashExit))

    def _finish(self, exit_code, exit_status):
        if not self.running:
            return
        self._read_stdout()
        self._read_stderr()
        remaining = bytes(self._process.readAllStandardOutput())
        if remaining and not self._error:
            self._error = 'Worker protocol error: incomplete event'
        normal_exit = exit_status == QtCore.QProcess.ExitStatus.NormalExit
        if not self._error and (exit_code or not normal_exit):
            self._error = f'Worker exited with code {exit_code}'
        if not self._error and self._result is None:
            self._error = 'Worker exited without a result'

        self._running = False
        self._timer.stop()
        self._update_elapsed()
        self.stop_button.setEnabled(False)
        if self._closing:
            self._closing = False
            self.close()

        if self._cancelled:
            self.status.setText('Stopped')
            self.stopped.emit()
        elif self._error:
            message = self._error
            if self._stderr:
                message += '\n' + self._stderr.decode('utf8', errors='replace')
            self.status.setText(message)
            self.failed.emit(message)
        else:
            self.status.setText('Completed')
            self.completed.emit(self._result)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

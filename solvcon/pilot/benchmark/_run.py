# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Control one isolated benchmark worker from a reusable Qt widget."""

import json
import os

from PySide6 import QtCore, QtWidgets

from solvcon import system


class RunPanel(QtWidgets.QWidget):
    """Show progress and emit a terminal signal after the worker exits.

    Call start with a BenchmarkSpec and an artifact path. A running control
    rejects another start. Stop and close kill the worker asynchronously.
    Optional threads override only the new worker's BLAS/OpenMP environment.
    """

    completed = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    stopped = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setToolTip(
            'Completed warmup calls and timed repetition blocks, '
            'not an estimate of remaining time.'
        )
        self.status = QtWidgets.QLabel('Idle', self)
        self.status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.elapsed = QtWidgets.QLabel('Elapsed: 0.0 s', self)
        self.stop_button = QtWidgets.QPushButton('Stop', self)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.progress, 1)
        for widget in (self.status, self.elapsed, self.stop_button):
            layout.addWidget(widget)

        self._process = QtCore.QProcess(self)
        self._process.started.connect(self._send_request)
        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.readyReadStandardError.connect(self._read_stderr)
        self._process.errorOccurred.connect(self._process_error)
        self._process.finished.connect(self._finish)
        QtWidgets.QApplication.instance().installEventFilter(self)
        self._clock = QtCore.QElapsedTimer()
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._update_elapsed)
        self._running = False
        self._closing = False

    @property
    def running(self):
        """Whether a worker is starting, running, or stopping."""
        return self._running

    def start(self, specification, output_path, *, threads=None):
        """Start one worker without blocking the Qt event loop.

        :param specification: Validated benchmark specification.
        :param output_path: Destination for the completed JSON artifact.
        :param threads: Positive BLAS/OpenMP thread count for the worker,
            or ``None`` to inherit its environment.
        :raises RuntimeError: If a worker is already active.
        :raises ValueError: If the thread count is invalid.
        """
        if self.running:
            raise RuntimeError('a benchmark is already running')
        env = QtCore.QProcessEnvironment.systemEnvironment()
        if threads is not None:
            if (isinstance(threads, bool) or not isinstance(threads, int)
                    or threads < 1):
                raise ValueError('threads must be a positive integer')
            for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                         'MKL_NUM_THREADS', 'BLIS_NUM_THREADS',
                         'VECLIB_MAXIMUM_THREADS'):
                env.insert(name, str(threads))
        self._process.setProcessEnvironment(env)
        request = {'spec': specification.to_dict(),
                   'output_path': os.fspath(output_path)}
        self._request = (json.dumps(request, allow_nan=False) + '\n').encode()
        self._kernels = specification.kernels + ('numpy',)
        self._result = None
        self._completed = 0
        self._total = None
        self._error = ''
        self._stderr = b''
        self._cancelled = False
        self._running = True
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.status.setText('Preparing')
        self.stop_button.setEnabled(True)
        self._clock.start()
        self._update_elapsed()
        self._timer.start(100)
        command = system.python_command('-m', 'solvcon.benchmark.worker')
        self._process.start(command[0], command[1:])

    def stop(self):
        """Cancel an active run; emit ``stopped`` after the worker exits."""
        if self.running:
            self._cancelled = True
            self.status.setText('Stopping')
            self.progress.setTextVisible(False)
            self.stop_button.setEnabled(False)
            self._process.kill()

    def closeEvent(self, event):
        if self.running:
            self._closing = True
            self.stop()
            event.ignore()
        else:
            super().closeEvent(event)

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.Type.Quit:
            # Stop before closeEvent can defer and cancel QApplication.quit.
            self.stop()
            self._process.waitForFinished()
        return super().eventFilter(watched, event)

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
            self._show_progress(event)
        elif kind == 'result':
            path = event.get('artifact_path')
            if not isinstance(path, str) or not path:
                raise ValueError('invalid worker artifact path')
            self._result = path
            self.progress.setRange(0, 0)
            self.progress.setTextVisible(False)
            self.status.setText('Finishing')
        elif kind == 'error':
            self._fail(str(event.get('message', 'worker failed')))
        else:
            raise ValueError('unknown worker event')

    def _show_progress(self, event):
        phase, kernel = event.get('phase'), event.get('kernel')
        completed, total = event.get('completed'), event.get('total')
        phases = ('preparing', 'comparison', 'warmup', 'timing', 'finishing')
        has_kernel = phase in ('comparison', 'warmup', 'timing')
        kernels = self._kernels if has_kernel else (None,)
        if phase not in phases or kernel not in kernels:
            raise ValueError('invalid worker progress')

        if phase in ('warmup', 'timing'):
            self._show_counts(completed, total)
        elif completed is not None or total is not None:
            raise ValueError('invalid worker progress')
        else:
            self.progress.setRange(0, 0)
            self.progress.setTextVisible(False)
        label = phase.capitalize()
        self.status.setText(f'{label}: {kernel}' if kernel else label)

    def _show_counts(self, completed, total):
        if (
            type(completed) is not int or type(total) is not int
            or not 0 <= self._completed <= completed <= total
            or total <= 0 or self._total not in (None, total)
        ):
            raise ValueError('invalid worker progress counts')
        self._completed, self._total = completed, total
        self.progress.setRange(0, 100)
        self.progress.setValue(100 * completed // total)
        self.progress.setFormat(f'%p% ({completed}/{total} units)')
        self.progress.setTextVisible(True)

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
        self.progress.setRange(0, 1)
        success = not (self._cancelled or self._error)
        self.progress.setValue(int(success))
        self.progress.setFormat('%p%')
        self.progress.setTextVisible(success)
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
            self.status.setText(f'Failed: {message}')
            self.failed.emit(message)
        else:
            self.status.setText('Completed')
            self.completed.emit(self._result)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

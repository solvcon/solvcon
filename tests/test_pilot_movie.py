# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for the viewer movie recorder.

The recorder grabs whatever widget it is handed, so a stub widget covers the
frame bookkeeping and the movie assembly with no graphics surface at all.
One class hands it a real Qt widget, which is the only way to say that what
a live grab returns is what lands in the movie.
"""

import os
import platform
import tempfile
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.visual import _movie
    from PySide6.QtWidgets import QLabel
    from PIL import Image
except ImportError:
    pilot = None
    _movie = None
    Image = None

HAS_IMAGE = getattr(_movie, 'HAS_IMAGE', False)


class _StubPixmap(object):
    """Stand in for the pixmap a widget grab returns."""

    def __init__(self, size, color):
        self.size = size
        self.color = color

    def isNull(self):
        return self.size is None

    def save(self, path):
        if self.size is None:
            return False
        Image.new('RGB', self.size, self.color).save(path)
        return True


class _StubViewer(object):
    """Stand in for the viewer widget: grab a flat frame of a fixed size.

    ``ok`` turns the grab into the empty pixmap a widget hands back when it
    has nothing to show.
    """

    COLORS = ((255, 0, 0), (0, 255, 0), (0, 0, 255))

    def __init__(self, ok=True, size=(64, 32)):
        self.ok = ok
        self.size = size
        self.calls = []

    def grab(self):
        self.calls.append(self.size)
        color = self.COLORS[(len(self.calls) - 1) % len(self.COLORS)]
        return _StubPixmap(self.size if self.ok else None, color)


@unittest.skipUnless(solvcon.HAS_PILOT and HAS_IMAGE,
                     "Qt pilot is not built or PIL is missing")
class MovieRecorderTC(unittest.TestCase):
    """Frame bookkeeping and movie assembly, over a stub viewer."""

    def setUp(self):
        self.recorder = _movie.MovieRecorder(frame_ms=40, hold_ms=500)
        self.addCleanup(self.recorder.close)
        self.viewer = _StubViewer()

    def _capture(self, nframe):
        for _ in range(nframe):
            self.recorder.capture(self.viewer)

    def test_capture_grabs_the_widget_whole(self):
        path = self.recorder.capture(self.viewer)
        self.assertEqual(self.recorder.nframe, 1)
        self.assertTrue(os.path.exists(path))
        # The frame is the widget's own, not a size asked of it, so
        # nothing of what the widget shows is cropped away.
        self.assertEqual(self.viewer.size, Image.open(path).size)

    def test_capture_reports_a_viewer_that_grabs_nothing(self):
        # A viewer with nothing to show hands back an empty pixmap; holding
        # a frame that was never written would break the movie later.
        with self.assertRaises(RuntimeError):
            self.recorder.capture(_StubViewer(ok=False))
        self.assertEqual(self.recorder.nframe, 0)

    def test_either_suffix_writes_every_frame_into_a_loop(self):
        # The suffix is the whole of the format choice, and either format
        # carries every frame captured, at the shape it was grabbed in.
        self._capture(3)
        with tempfile.TemporaryDirectory() as folder:
            for suffix in _movie.MovieRecorder.SUFFIXES:
                with self.subTest(suffix=suffix):
                    path = os.path.join(folder, f"movie{suffix}")
                    self.assertEqual(self.recorder.write(path), 3)
                    movie = Image.open(path)
                    self.assertEqual(movie.format, suffix[1:].upper())
                    self.assertEqual(movie.n_frames, 3)
                    self.assertEqual(movie.size, self.viewer.size)
                    self.assertEqual(movie.info['loop'], 0)

    def test_webp_keeps_the_rendered_colors(self):
        # The point of WebP over GIF: no palette, so every frame keeps the
        # color it was rendered in rather than the nearest of 255.  The
        # encoder is lossy, so a flat frame comes back within a few counts.
        self._capture(3)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "movie.webp")
            self.recorder.write(path)
            movie = Image.open(path)
            colors = []
            for iframe in range(movie.n_frames):
                movie.seek(iframe)
                colors.append(movie.convert('RGB').getpixel((0, 0)))
        for got, rendered in zip(colors, _StubViewer.COLORS):
            for channel, want in zip(got, rendered):
                self.assertLessEqual(abs(channel - want), 8)

    def test_write_rejects_a_name_of_no_known_format(self):
        self._capture(1)
        with self.assertRaises(ValueError):
            self.recorder.write("movie.mp4")

    def test_write_holds_the_last_frame_longer(self):
        self._capture(2)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "movie.gif")
            self.recorder.write(path)
            movie = Image.open(path)
            durations = []
            for iframe in range(movie.n_frames):
                movie.seek(iframe)
                durations.append(movie.info['duration'])
        self.assertEqual(durations, [40, 500])

    def test_write_makes_the_output_folder(self):
        self._capture(1)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "made", "movie.gif")
            self.recorder.write(path)
            self.assertTrue(os.path.exists(path))

    def test_write_without_a_frame_raises(self):
        with self.assertRaises(ValueError):
            self.recorder.write("never-written.gif")

    def test_close_drops_the_frames(self):
        path = self.recorder.capture(self.viewer)
        self.recorder.close()
        self.assertEqual(self.recorder.nframe, 0)
        self.assertFalse(os.path.exists(path))
        # A second close is a no-op, so a caller need not track it.
        self.recorder.close()


@unittest.skipUnless(solvcon.HAS_PILOT and HAS_IMAGE,
                     "Qt pilot is not built or PIL is missing")
@unittest.skipIf("Windows" == platform.system(),
                 "offscreen grabbing is unreliable on Windows CI")
class WidgetMovieTC(unittest.TestCase):
    """The recorder over a real Qt widget, which is what it grabs."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_a_resized_widget_records_at_one_shape(self):
        widget = QLabel("wide")
        widget.resize(160, 90)
        recorder = _movie.MovieRecorder()
        self.addCleanup(recorder.close)
        recorder.capture(widget)
        shape = widget.grab().size()

        # The writer merges frames that come out identical, so the widget
        # has to show something else before the second capture.
        widget.resize(80, 120)
        widget.setText("tall")
        recorder.capture(widget)

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "widget.webp")
            self.assertEqual(recorder.write(path), 2)
            movie = Image.open(path)
            self.assertEqual(2, movie.n_frames)
            # A movie that changes size partway is not one a player can
            # show, so it keeps the shape of its first frame, whatever the
            # widget does afterwards.
            self.assertEqual((shape.width(), shape.height()), movie.size)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

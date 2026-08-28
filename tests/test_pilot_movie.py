# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Tests for the viewer movie recorder.

The recorder grabs whatever widget it is handed, so a stub widget covers the
frame bookkeeping and the movie assembly with no graphics surface at all.
One class hands it a real Qt widget, which is the only way to say that what
a live grab returns is what lands in the movie.

The MP4 tests need the Qt Multimedia encoder, which not every build has, so
they stand apart and skip themselves where there is none.

Every image read back here is closed before the folder holding it goes.
PIL keeps the file open for as long as the image lives, and Windows refuses
to remove a file that anything still holds open.
"""

import os
import platform
import tempfile
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.visual import _movie
    from PySide6.QtCore import QEventLoop, QSize, Qt, QTimer, QUrl
    from PySide6.QtGui import QImage, QPixmap
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

    def devicePixelRatio(self):
        return 1.0

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

    @classmethod
    def setUpClass(cls):
        # _can_write_mp4() answers for the application's media backend, so
        # without one here the build's own answer would depend on whether
        # another test module happened to run first.
        pilot.RManager.instance.setUp()

    def setUp(self):
        self.recorder = _movie.MovieRecorder(frame_ms=40, hold_ms=500)
        self.addCleanup(self.recorder.close)
        self.viewer = _StubViewer()

    def _capture(self, nframe):
        for _ in range(nframe):
            self.recorder.capture(self.viewer)

    def test_a_high_dpi_grab_comes_down_to_the_widget_size(self):
        # A grab off a high-DPI screen holds as many pixels as the display
        # does, so the same window recorded on one screen and then another
        # would give two movies of two sizes.
        pixmap = QPixmap(320, 180)
        pixmap.fill(Qt.GlobalColor.darkGreen)
        pixmap.setDevicePixelRatio(2.0)
        come_down = _movie.MovieRecorder._to_logical_size(pixmap)
        self.assertEqual(QSize(160, 90), come_down.size())
        # The ratio goes with the pixels, or the PNG still claims the
        # resolution the grab had.
        self.assertEqual(1.0, come_down.devicePixelRatio())

    def test_capture_grabs_the_widget_whole(self):
        path = self.recorder.capture(self.viewer)
        self.assertEqual(self.recorder.nframe, 1)
        self.assertTrue(os.path.exists(path))
        # The frame is the widget's own, not a size asked of it, so
        # nothing of what the widget shows is cropped away.
        with Image.open(path) as frame:
            self.assertEqual(self.viewer.size, frame.size)

    def test_capture_reports_a_viewer_that_grabs_nothing(self):
        # A viewer with nothing to show hands back an empty pixmap; holding
        # a frame that was never written would break the movie later.
        with self.assertRaises(RuntimeError):
            self.recorder.capture(_StubViewer(ok=False))
        self.assertEqual(self.recorder.nframe, 0)

    def test_either_image_suffix_writes_every_frame_into_a_loop(self):
        # The suffix is the whole of the format choice, and either format
        # carries every frame captured, at the shape it was grabbed in.
        self._capture(3)
        with tempfile.TemporaryDirectory() as folder:
            for suffix in _movie.MovieRecorder.IMAGE_SUFFIXES:
                with self.subTest(suffix=suffix):
                    path = os.path.join(folder, f"movie{suffix}")
                    self.assertEqual(self.recorder.write(path), 3)
                    with Image.open(path) as movie:
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
            colors = []
            with Image.open(path) as movie:
                for iframe in range(movie.n_frames):
                    movie.seek(iframe)
                    colors.append(movie.convert('RGB').getpixel((0, 0)))
        for got, rendered in zip(colors, _StubViewer.COLORS):
            for channel, want in zip(got, rendered):
                self.assertLessEqual(abs(channel - want), 8)

    def test_write_rejects_a_name_of_no_known_format(self):
        self._capture(1)
        with self.assertRaises(ValueError):
            self.recorder.write("movie.mkv")

    def test_the_default_suffix_names_a_format_write_takes(self):
        # Whichever encoder the build has, the name the panel opens on has
        # to be one write() will not turn down.
        self.assertIn(_movie._default_suffix(),
                      _movie.MovieRecorder.SUFFIXES)

    def test_a_flat_color_converts_to_its_standard_yuv(self):
        # The stream says BT.601 studio range, so what goes into the planes
        # has to be that.  Coefficients of any other kind would tag one
        # color space and carry another, shifting every color in the movie.
        self._capture(1)
        image = QImage(self.recorder._frames[0]).convertToFormat(
            QImage.Format.Format_RGB888)
        luma, blue_diff, red_diff = self.recorder._yuv420p(image)
        # The stub's first frame is pure red, which is (81, 90, 240) there,
        # and every pixel of it, so a plane laid down crooked shows up here.
        self.assertTrue((luma == 81).all())
        self.assertTrue((blue_diff == 90).all())
        self.assertTrue((red_diff == 240).all())
        # Chroma is subsampled by two, luma is not.
        self.assertEqual((32, 64), luma.shape)
        self.assertEqual((16, 32), blue_diff.shape)
        self.assertEqual((16, 32), red_diff.shape)

    def test_write_holds_the_last_frame_longer(self):
        self._capture(2)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "movie.gif")
            self.recorder.write(path)
            durations = []
            with Image.open(path) as movie:
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

    def test_mp4_is_a_known_name_wherever_it_is_asked_for(self):
        # A missing encoder (OSError) and a name of no known format
        # (ValueError) reach the control on different channels and read
        # differently there, so naming MP4 on a build that cannot encode
        # it must not come back as though the name were a typo.
        self._capture(1)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "movie.mp4")
            if _movie._can_write_mp4():
                self.assertEqual(self.recorder.write(path), 1)
            else:
                with self.assertRaises(OSError):
                    self.recorder.write(path)


@unittest.skipUnless(solvcon.HAS_PILOT and HAS_IMAGE,
                     "Qt pilot is not built or PIL is missing")
class Mp4MovieTC(unittest.TestCase):
    """The MP4 itself, on a build whose Qt Multimedia can encode one."""

    @classmethod
    def setUpClass(cls):
        # The backend that does the encoding is a plugin the application
        # loads, so what it can encode cannot be asked before there is one.
        pilot.RManager.instance.setUp()
        if not _movie._can_write_mp4():
            raise unittest.SkipTest("Qt Multimedia has no MP4 encoder")

    def _recorder(self, **kw):
        recorder = _movie.MovieRecorder(**kw)
        self.addCleanup(recorder.close)
        return recorder

    @staticmethod
    def _played_ms(path):
        """How long a player makes the movie at ``path`` run.

        QtMultimedia is imported here rather than at the top of the file:
        a build without it must skip these tests, not lose the rest.
        """
        from PySide6.QtMultimedia import QMediaPlayer

        player = QMediaPlayer()
        loop = QEventLoop()
        loaded = []
        player.mediaStatusChanged.connect(
            lambda status: (loaded.append(status), loop.quit())
            if QMediaPlayer.MediaStatus.LoadedMedia == status else None)
        QTimer.singleShot(10000, loop.quit)
        player.setSource(QUrl.fromLocalFile(path))
        loop.exec()
        played = player.duration()
        # Windows will not remove a file the player still holds open, and
        # the movie sits in a temporary directory that is about to go.
        player.setSource(QUrl())
        # A duration read off a player that never opened the file is zero,
        # which would read as a hold that never happened.
        assert loaded, "the player never loaded the movie"
        return played

    def test_every_frame_lands_in_an_mp4_container(self):
        # The frame is the size a viewer really is, not the stub's default.
        # The size is a viewer's, not the stub default, so the encoder
        # sees a picture of the shape a real recording gives it.
        recorder = self._recorder(frame_ms=40, hold_ms=500)
        viewer = _StubViewer(size=(800, 600))
        for _ in range(3):
            recorder.capture(viewer)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "movie.mp4")
            self.assertEqual(recorder.write(path), 3)
            with open(path, 'rb') as stream:
                head = stream.read(12)
        # An MP4 opens on a file-type box, which is what tells a player it
        # has an MP4 rather than whatever else was named one.
        self.assertEqual(b'ftyp', head[4:8])

    def test_the_last_frame_is_held_to_the_end(self):
        # An MP4 runs at one frame rate, so a longer duration on the last
        # frame is not a thing the stream can say; the hold has to come
        # from handing that frame over again.  Without it the movie stops
        # the instant the march does and never rests on the result.
        recorder = self._recorder(frame_ms=80, hold_ms=1500)
        viewer = _StubViewer(size=(320, 240))
        for _ in range(4):
            recorder.capture(viewer)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "held.mp4")
            self.assertEqual(4, recorder.write(path))
            played = self._played_ms(path)
        # Three frames at 80 ms and then 1500 ms on the last, give or take
        # the frame rate the muxer rounds the stream to.
        self.assertAlmostEqual(3 * 80 + 1500, played, delta=250)

    def test_the_frames_reach_the_encoder_as_yuv_4_2_0(self):
        # An encoder handed RGB keeps what it was given and puts the stream
        # on H.264's 4:4:4 profile, which a browser turns down; subsampling
        # to 4:2:0 here is what leaves an MP4 that plays where video plays.
        from PySide6.QtMultimedia import QVideoFrame, QVideoFrameFormat

        recorder = self._recorder()
        recorder.capture(_StubViewer(size=(64, 32)))
        frame = recorder._video_frame(0, recorder._movie_size())
        self.assertEqual(QVideoFrameFormat.PixelFormat.Format_YUV420P,
                         frame.surfaceFormat().pixelFormat())
        # Read the planes back off the frame, so what the encoder would
        # pull is checked rather than the format the frame was asked for.
        self.assertTrue(frame.map(QVideoFrame.MapMode.ReadOnly))
        try:
            self.assertEqual(81, frame.bits(0)[0])
            self.assertEqual(90, frame.bits(1)[0])
            self.assertEqual(240, frame.bits(2)[0])
        finally:
            frame.unmap()

    def test_an_odd_sized_grab_comes_down_to_even_sides(self):
        # H.264 subsamples chroma by two and has no half pixel to take it
        # from, so a grab of odd shape cannot go to the encoder as it is.
        recorder = self._recorder()
        recorder.capture(_StubViewer(size=(65, 33)))
        self.assertEqual(QSize(64, 32), recorder._movie_size())


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
        shape = widget.size()

        # The writer merges frames that come out identical, so the widget
        # has to show something else before the second capture.
        widget.resize(80, 120)
        widget.setText("tall")
        recorder.capture(widget)

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "widget.webp")
            self.assertEqual(recorder.write(path), 2)
            with Image.open(path) as movie:
                self.assertEqual(2, movie.n_frames)
                # A movie that changes size partway is not one a player can
                # show, so it keeps the shape of its first frame, whatever
                # the widget does afterwards.
                self.assertEqual((shape.width(), shape.height()), movie.size)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

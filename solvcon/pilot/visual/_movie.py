# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Record what a viewer widget shows into an animated movie.

:class:`MovieRecorder` grabs one frame per capture off a widget and
assembles the frames into a looping animation.  Grabbing the widget rather
than asking the 3D view to render itself takes in whatever is laid over it,
the color bar included, and takes it at the shape the widget has, where
rendering to a size of its own reframed the scene and cut the domain off at
the sides.

The frames stay on disk as PNG files in a temporary directory until they
are written out, so a long run does not pile up on the heap.

Two encoders sit behind the formats: Qt Multimedia's FFmpeg backend for
MP4, Pillow for GIF and WebP.  Either can be missing from a build, so
:func:`_default_suffix` names one the build can write.
"""

import os
import tempfile

import numpy as np

try:
    from PIL import Image

    HAS_IMAGE = True
except ImportError:
    Image = None
    HAS_IMAGE = False

from PySide6.QtCore import (QCoreApplication, QEventLoop, QSize, Qt,
                            QTimer, QUrl)
from PySide6.QtGui import QImage

try:
    from PySide6.QtMultimedia import (QMediaCaptureSession, QMediaFormat,
                                      QMediaRecorder, QVideoFrame,
                                      QVideoFrameFormat, QVideoFrameInput)

    HAS_VIDEO = True
except ImportError:
    HAS_VIDEO = False

# Hold the encoder to software: a GPU is the backend's first choice, and
# the drivers fail an opaque external-library error only once every frame
# is spent.  An empty pair turns the list off, where unsetting it reads as
# "no preference".  It has to be set before _can_write_mp4() makes the
# backend read it once and keep the answer.
os.environ.setdefault('QT_FFMPEG_ENCODING_HW_DEVICE_TYPES', ',')

__all__ = [
    'MovieRecorder',
]


def _can_write_mp4():
    """Whether this build can write an MP4.

    Only the FFmpeg backend encodes one; another plays an MP4 but cannot
    make one.  The backend is a plugin the application loads, so there is
    nothing to ask before one exists.
    """
    if not HAS_VIDEO or QCoreApplication.instance() is None:
        return False
    formats = QMediaFormat().supportedFileFormats(
        QMediaFormat.ConversionMode.Encode)
    return QMediaFormat.FileFormat.MPEG4 in formats


def _default_suffix():
    """The suffix a new movie is named with.

    MP4 leads where there is an encoder for it, playing where neither
    animation format does; without one, the WebP Pillow always writes.
    """
    return '.mp4' if _can_write_mp4() else '.webp'


class MovieRecorder(object):
    """Collect frames off a widget and write them out as an animation.

    The widget is anything carrying ``grab()``, which every Qt widget does.

    The movie takes the shape of its first frame, and a later frame of
    another shape is scaled to it, so resizing the window mid-run leaves an
    animation that still plays rather than one that changes size partway.
    That shape is the widget's own, not the screen's; see
    :meth:`_to_logical_size`.

    The output name picks the format.  An ``.mp4`` is the one a video
    player, a browser, and a slide deck all take, and the smallest of the
    three, but it needs the Qt Multimedia encoder :func:`_can_write_mp4`
    reports on; a ``.webp`` keeps the frames in true color, which a field
    of smooth colors needs; a ``.gif`` goes everywhere but is quantized to
    255 colors.

    :ivar frame_ms: Milliseconds each frame is shown.
    :ivar hold_ms: Milliseconds the last frame is held, so a loop ends on
        the result instead of snapping back.
    :ivar quality: WebP quality, 0 to 100.
    """

    #: Output suffixes that name a format :meth:`write` can write.
    SUFFIXES = ('.mp4', '.gif', '.webp')
    #: Of those, the ones Pillow assembles; the rest go through Qt.
    IMAGE_SUFFIXES = ('.gif', '.webp')
    #: Frames sampled across the movie to build its one GIF palette, and
    #: the factor each sample is shrunk by.
    PALETTE_FRAMES = 16
    PALETTE_SHRINK = 2
    #: WebP encoding effort, 0 (fastest) to 6 (smallest).
    WEBP_METHOD = 4
    #: Milliseconds the MP4 encoder is given before the write is called
    #: off; a stalled backend would hang the window it was called from.
    MP4_TIMEOUT_MS = 60000

    def __init__(self, frame_ms=80, hold_ms=1500, quality=80):
        self.frame_ms = frame_ms
        self.hold_ms = hold_ms
        self.quality = quality
        self._folder = tempfile.TemporaryDirectory(prefix='solvcon-movie-')
        self._frames = []

    @property
    def nframe(self):
        """Number of frames captured so far."""
        return len(self._frames)

    def capture(self, widget):
        """Grab one frame of what ``widget`` shows and hold it."""
        path = os.path.join(self._folder.name, f"frame{self.nframe:05d}.png")
        pixmap = self._to_logical_size(widget.grab())
        if pixmap.isNull() or not pixmap.save(path):
            raise RuntimeError("the viewer gave no frame")
        self._frames.append(path)
        return path

    @staticmethod
    def _to_logical_size(pixmap):
        """``pixmap`` at the size its widget is drawn, not its backing store.

        A high-DPI grab comes back at the display's pixel count, so one
        window would record at two sizes on two screens.  A null pixmap
        reports a ratio of one and passes through, leaving the caller its
        own grab to reject.
        """
        if pixmap.devicePixelRatio() <= 1:
            return pixmap
        pixmap = pixmap.scaled(pixmap.deviceIndependentSize().toSize(),
                               Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        # Or the PNG claims a resolution the frame no longer has.
        pixmap.setDevicePixelRatio(1)
        return pixmap

    def write(self, path):
        """Assemble the held frames into the movie at ``path``; returns how
        many were assembled.

        The suffix of ``path`` picks the format among :attr:`SUFFIXES`.
        """
        if not self._frames:
            raise ValueError("no frame is captured")
        suffix = os.path.splitext(path)[1].lower()
        if suffix not in self.SUFFIXES:
            raise ValueError(
                f"cannot write a movie named '{os.path.basename(path)}'; "
                f"name it {', '.join(self.SUFFIXES[:-1])} "
                f"or {self.SUFFIXES[-1]}")
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        if '.mp4' == suffix:
            return self._write_mp4(path)
        if not HAS_IMAGE:
            raise OSError("writing a movie needs PIL/Pillow installed")
        durations = ([self.frame_ms] * (len(self._frames) - 1)
                     + [self.hold_ms])
        if '.webp' == suffix:
            return self._write_webp(path, durations)
        return self._write_gif(path, durations)

    def _load(self):
        """The held frames, every one the shape of the first."""
        movie = [Image.open(fn).convert('RGB') for fn in self._frames]
        size = movie[0].size
        return [im if im.size == size
                else im.resize(size, Image.Resampling.LANCZOS)
                for im in movie]

    def _write_gif(self, path, durations):
        """Write the GIF at ``path``.

        Every frame is quantized against the one palette of :meth:`_palette`,
        so the movie does not flicker the way a per-frame palette makes it.
        The GIF writer merges identical neighboring frames, so the file can
        end up holding fewer than were assembled into it.
        """
        movie = self._load()
        palette = self._palette(movie)
        movie = [im.quantize(palette=palette, dither=Image.Dither.NONE)
                 for im in movie]
        movie[0].save(path, save_all=True, append_images=movie[1:],
                      duration=durations, loop=0, optimize=True)
        return len(movie)

    def _write_webp(self, path, durations):
        """Write the animated WebP at ``path``.

        The frames go in as they were rendered: WebP carries true color, so
        there is no palette to build and none of the banding one leaves in
        a smooth field.
        """
        movie = self._load()
        movie[0].save(path, save_all=True, append_images=movie[1:],
                      duration=durations, loop=0, quality=self.quality,
                      method=self.WEBP_METHOD)
        return len(movie)

    def _palette(self, movie):
        """Median-cut one palette over frames sampled across the movie.

        A palette read off a single frame misses the colors the rest of the
        movie needs, which flattens everything that frame does not show.
        The samples are shrunk with nearest-neighbor, keeping their colors
        exact while cutting the work.
        """
        stride = max(1, len(movie) // self.PALETTE_FRAMES)
        picks = movie[::stride]
        width = max(1, picks[0].width // self.PALETTE_SHRINK)
        height = max(1, picks[0].height // self.PALETTE_SHRINK)
        strip = Image.new('RGB', (width, height * len(picks)))
        for it, image in enumerate(picks):
            strip.paste(image.resize((width, height),
                                     Image.Resampling.NEAREST),
                        (0, it * height))
        return strip.quantize(colors=255)

    def _write_mp4(self, path):
        """Encode the held frames into the MP4 at ``path``.

        Qt Multimedia records off a live source, not a list, so the frames
        go through a ``QVideoFrameInput`` that :meth:`_pump` feeds.

        One frame rate leaves a frame no duration of its own, so the hold
        on the last is spelled by repeating it; H.264 codes a repeated
        picture into almost nothing.
        """
        if not _can_write_mp4():
            raise OSError("writing an MP4 needs Qt Multimedia built with "
                          "the FFmpeg backend")
        size = self._movie_size()
        held = max(0, round(self.hold_ms / self.frame_ms) - 1)

        session = QMediaCaptureSession()
        recorder = QMediaRecorder()
        # Left to take its format from the first frame: declaring one up
        # front never arms the input, and the write stalls having written
        # nothing.
        source = QVideoFrameInput()

        media = QMediaFormat(QMediaFormat.FileFormat.MPEG4)
        media.setVideoCodec(QMediaFormat.VideoCodec.H264)
        recorder.setMediaFormat(media)
        recorder.setOutputLocation(QUrl.fromLocalFile(os.path.abspath(path)))
        recorder.setVideoResolution(size)
        recorder.setVideoFrameRate(1000.0 / self.frame_ms)
        recorder.setQuality(QMediaRecorder.Quality.VeryHighQuality)
        session.setVideoFrameInput(source)
        session.setRecorder(recorder)

        try:
            self._pump(recorder, source, self.nframe + held,
                       lambda it: self._video_frame(min(it, self.nframe - 1),
                                                    size))
        except OSError:
            # What was muxed before it gave up plays, and would sit at
            # the named path looking like the recording that failed.
            if os.path.exists(path):
                os.remove(path)
            raise
        return self.nframe

    def _movie_size(self):
        """The shape every frame goes to the encoder in.

        The first frame sets it, rounded down to even sides: H.264 halves
        the chroma and has no half pixel to take it from.
        """
        size = QImage(self._frames[0]).size()
        return QSize(size.width() & ~1, size.height() & ~1)

    def _yuv420p(self, image):
        """The RGB888 ``image`` as the three planes of a YUV 4:2:0 frame.

        RGB handed over instead lands the stream on H.264's 4:4:4 profile,
        which browsers turn down: Qt picks the codec format nearest the
        source in bits per pixel, and no 4:2:0 format is near 32-bit RGB.
        Subsampling here is what leaves an MP4 that plays.

        BT.601 studio range, which :meth:`_video_frame` tags the frame with
        so no player has to guess.
        """
        width, height = image.width(), image.height()
        pixels = np.frombuffer(image.constBits(), dtype='uint8')
        pixels = pixels.reshape(height, image.bytesPerLine())
        pixels = pixels[:, :width * 3].reshape(height, width, 3)
        red, green, blue = (pixels[..., 0].astype('float32'),
                            pixels[..., 1].astype('float32'),
                            pixels[..., 2].astype('float32'))

        luma = 16 + (65.481 * red + 128.553 * green + 24.966 * blue) / 255
        blue_diff = 128 + (-37.797 * red - 74.203 * green
                           + 112.0 * blue) / 255
        red_diff = 128 + (112.0 * red - 93.786 * green
                          - 18.214 * blue) / 255
        return (self._to_byte_plane(luma),
                self._to_byte_plane(self._shrink(blue_diff)),
                self._to_byte_plane(self._shrink(red_diff)))

    @staticmethod
    def _shrink(plane):
        """Subsample a chroma plane, which 4:2:0 carries at half size.

        Both sides are even, which :meth:`_movie_size` guarantees.
        """
        height, width = plane.shape
        return plane.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))

    @staticmethod
    def _to_byte_plane(plane):
        return np.clip(np.rint(plane), 0, 255).astype('uint8')

    def _video_frame(self, index, size):
        """The held frame at ``index`` as a YUV 4:2:0 video frame of ``size``.

        Built one at a time, as the encoder takes them, so a recording is
        never held whole; the frames stay the PNG files :meth:`capture`
        wrote, as they do for the animation formats.
        """
        image = QImage(self._frames[index])
        if image.size() != size:
            image = image.scaled(size, Qt.IgnoreAspectRatio,
                                 Qt.SmoothTransformation)
        planes = self._yuv420p(
            image.convertToFormat(QImage.Format.Format_RGB888))

        video = QVideoFrameFormat(
            size, QVideoFrameFormat.PixelFormat.Format_YUV420P)
        video.setColorSpace(QVideoFrameFormat.ColorSpace.ColorSpace_BT601)
        video.setColorRange(QVideoFrameFormat.ColorRange.ColorRange_Video)
        video.setStreamFrameRate(1000.0 / self.frame_ms)
        frame = QVideoFrame(video)
        if not frame.map(QVideoFrame.MapMode.WriteOnly):
            raise OSError("a video frame would not open for writing")
        try:
            # Rows are padded to the stride the encoder chose.
            for it, plane in enumerate(planes):
                stride = frame.bytesPerLine(it)
                rows, columns = plane.shape
                padded = np.zeros((rows, stride), dtype='uint8')
                padded[:, :columns] = plane
                frame.bits(it)[:rows * stride] = padded.tobytes()
        finally:
            frame.unmap()
        return frame

    def _pump(self, recorder, source, total, build):
        """Feed ``total`` frames to ``source`` until the recorder has them.

        ``build`` makes the frame at an index, called as the encoder takes
        them so only the one in flight is held.  The recorder runs on the
        event loop, so the write borrows one until it stops.  A frame
        turned down means the encoder is behind, not that the frame is
        bad, so it is offered again on the next readiness signal.

        The loop takes no user input: closing the window mid-write would
        free the widgets the write returns into.
        """
        loop = QEventLoop()
        sent, pending, stopped, failure, late = 0, None, False, None, False

        def feed():
            nonlocal sent, pending
            while sent < total:
                if pending is None:
                    pending = build(sent)
                if not source.sendVideoFrame(pending):
                    return
                pending, sent = None, sent + 1
            if not stopped:
                recorder.stop()

        def on_state(state):
            nonlocal stopped
            if QMediaRecorder.RecorderState.StoppedState == state:
                stopped = True
                loop.quit()

        def on_error(_, message):
            nonlocal failure
            failure = message
            loop.quit()

        def on_late():
            nonlocal late
            late = True
            loop.quit()

        # A timer to stop and a flag to read: a watchdog outliving the
        # write would quit a later loop, and a shot spent before exec()
        # would leave the write with no timeout at all.
        watchdog = QTimer()
        watchdog.setSingleShot(True)
        watchdog.timeout.connect(on_late)

        source.readyToSendVideoFrame.connect(feed)
        recorder.recorderStateChanged.connect(on_state)
        recorder.errorOccurred.connect(on_error)
        try:
            watchdog.start(self.MP4_TIMEOUT_MS)
            recorder.record()
            # Qt does not carry a quit into the exec() after it, and a
            # backend can drain every frame while record() still runs.
            if not (stopped or late or failure):
                loop.exec(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        finally:
            watchdog.stop()

        # The recorder muxes what it was handed on its way to stopping, so
        # only the stop says the file is whole.
        if failure is not None:
            recorder.stop()
            raise OSError(f"the MP4 encoder failed: {failure}")
        if not stopped:
            recorder.stop()
            raise OSError(f"the MP4 encoder stalled at {sent} of "
                          f"{total} frames")

    def close(self):
        """Drop the held frames and the temporary directory."""
        self._frames = []
        if self._folder is not None:
            self._folder.cleanup()
            self._folder = None

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

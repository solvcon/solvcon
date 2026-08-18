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
"""

import os
import tempfile

try:
    from PIL import Image

    HAS_IMAGE = True
except ImportError:
    Image = None
    HAS_IMAGE = False

__all__ = [
    'MovieRecorder',
]


class MovieRecorder(object):
    """Collect frames off a widget and write them out as an animation.

    The widget is anything carrying ``grab()``, which every Qt widget does.

    The movie takes the shape of its first frame, and a later frame of
    another shape is scaled to it, so resizing the window mid-run leaves an
    animation that still plays rather than one that changes size partway.

    The output name picks the format.  A ``.webp`` keeps the frames in
    true color, which a field of smooth colors needs; a ``.gif`` goes
    everywhere but is quantized to 255 colors.

    :ivar frame_ms: Milliseconds each frame is shown.
    :ivar hold_ms: Milliseconds the last frame is held, so a loop ends on
        the result instead of snapping back.
    :ivar quality: WebP quality, 0 to 100.
    """

    #: Output suffixes that name a format :meth:`write` can write.
    SUFFIXES = ('.gif', '.webp')
    #: Frames sampled across the movie to build its one GIF palette, and
    #: the factor each sample is shrunk by.
    PALETTE_FRAMES = 16
    PALETTE_SHRINK = 2
    #: WebP encoding effort, 0 (fastest) to 6 (smallest).
    WEBP_METHOD = 4

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
        pixmap = widget.grab()
        if pixmap.isNull() or not pixmap.save(path):
            raise RuntimeError("the viewer gave no frame")
        self._frames.append(path)
        return path

    def write(self, path):
        """Assemble the held frames into the movie at ``path``; returns how
        many were assembled.

        The suffix of ``path`` picks the format among :attr:`SUFFIXES`.
        """
        if not HAS_IMAGE:
            raise OSError("writing a movie needs PIL/Pillow installed")
        if not self._frames:
            raise ValueError("no frame is captured")
        suffix = os.path.splitext(path)[1].lower()
        if suffix not in self.SUFFIXES:
            raise ValueError(
                f"cannot write a movie named '{os.path.basename(path)}'; "
                f"name it {' or '.join(self.SUFFIXES)}")
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
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

    def close(self):
        """Drop the held frames and the temporary directory."""
        self._frames = []
        if self._folder is not None:
            self._folder.cleanup()
            self._folder = None

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

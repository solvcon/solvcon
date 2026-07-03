# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for RDomainWidget, the pilot 3D viewer.

The widget renders through QRhi, so these tests exercise the offscreen capture
path (grabImage via saveImage). QRhi needs a real graphics surface; where one
is unavailable (e.g. the offscreen QPA platform on a headless macOS runner) the
render-dependent tests skip rather than fail. The Linux CI build job drives
them under Xvfb with the software rasterizer.
"""

import os
import platform
import tempfile
import unittest

import solvcon

try:
    from solvcon import pilot
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QWidget
except ImportError:
    pilot = None


def _make_2d_mesh():
    """Two triangles and one quadrilateral in the z = 0 plane."""
    core = solvcon.core
    T = core.StaticMesh.TRIANGLE
    Q = core.StaticMesh.QUADRILATERAL
    mh = core.StaticMesh(ndim=2, nnode=6, nface=0, ncell=3)
    mh.ndcrd.ndarray[:, :] = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1)]
    mh.cltpn.ndarray[:] = [T, T, Q]
    mh.clnds.ndarray[:, :5] = [(3, 0, 3, 2, -1), (3, 0, 1, 3, -1),
                               (4, 1, 4, 5, 3)]
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    return mh


def _make_3d_mesh():
    """A single tetrahedron."""
    core = solvcon.core
    mh = core.StaticMesh(ndim=3, nnode=4, nface=4, ncell=1)
    mh.ndcrd.ndarray[:, :] = [(0, 0, 0), (0, 1, 0), (-1, 1, 0), (0, 1, 1)]
    mh.cltpn.ndarray[:] = core.StaticMesh.TETRAHEDRON
    mh.clnds.ndarray[:, :5] = [(4, 0, 1, 2, 3)]
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    return mh


def _show_only(widget, name):
    """Draw only the named mesh style, hiding the other two."""
    for style in ("surface", "wireframe", "points"):
        widget.showMeshStyle(style, style == name)


def _grab_or_skip(widget):
    """Render the widget offscreen and return a QImage.

    Skip the calling test when an offscreen grab is unavailable or unreliable
    here: the headless Windows runner's debug software rasterizer stalls
    indefinitely creating a dedicated grab device, and the offscreen QPA
    platform on a headless macOS runner reports no QRhi support (saveImage
    then writes no file / a null image). Render correctness stays covered on
    Linux (Xvfb) and the platforms where grabbing works.
    """
    if platform.system() == "Windows":
        raise unittest.SkipTest("offscreen grabbing unreliable on Windows CI")
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "domain.png")
        widget.saveImage(path)
        if not os.path.exists(path):
            raise unittest.SkipTest("QRhi offscreen rendering is unavailable")
        image = QImage(path)
    if image.isNull():
        raise unittest.SkipTest("QRhi offscreen rendering is unavailable")
    return image


def _rgb_array(image):
    """Return the image as an (h, w, 3) uint8 array.

    Per-pixel QImage reads are far too slow in a debug build (minutes for a
    full frame), so the whole buffer is mapped through numpy at once. The
    bytes are copied out of the QImage immediately: the source image is a
    local that would otherwise be freed while the array still views it.
    """
    import numpy as np
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    width = converted.width()
    height = converted.height()
    buffer = converted.constBits()
    array = np.frombuffer(buffer, dtype=np.uint8,
                          count=converted.sizeInBytes()).copy()
    array = array.reshape(height, converted.bytesPerLine())[:, :width * 4]
    return array.reshape(height, width, 4)[:, :, :3]


def _count_foreground(image, threshold=60):
    """Count drawn (non-background) pixels.

    Foreground is whatever differs from the uniform background, where the
    background is the frame's most common color. The black wireframe stands
    out against the white clear, so this counts the wireframe; but keying on
    the difference (not on absolute darkness) keeps the count robust to a
    headless software rasterizer that reads an empty offscreen grab back as a
    uniformly dark frame instead of the white clear. A uniform frame, light
    or dark, has nothing that differs from its own background.
    """
    import numpy as np
    array = _rgb_array(image).astype('int16')
    flat = array.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    background = colors[counts.argmax()]
    diff = np.abs(array - background).max(axis=2)
    return int((diff > threshold).sum())


def _count_colored(image, threshold=240):
    """Count colored field pixels: those that differ from the white
    background in at least one channel."""
    array = _rgb_array(image)
    return int((array.min(axis=2) < threshold).sum())


def _count_reddish(image):
    """Count strongly red pixels (the first boundary set's highlight color);
    the white background and black wireframe both fail the low green/blue
    test."""
    array = _rgb_array(image)
    mask = ((array[:, :, 0] > 150) & (array[:, :, 1] < 120)
            & (array[:, :, 2] < 120))
    return int(mask.sum())


def _count_orange(image):
    """Count strongly orange pixels (the feature-edge overlay color); the
    white background, black wireframe, and saturated-red boundary highlight
    all fail the mid-green band."""
    array = _rgb_array(image)
    mask = ((array[:, :, 0] > 180) & (array[:, :, 1] > 70)
            & (array[:, :, 1] < 170) & (array[:, :, 2] < 90))
    return int(mask.sum())


def _count_green(image):
    """Count strongly green pixels (the face-normal arrow color); the white
    background and black wireframe both fail the low red/blue test."""
    array = _rgb_array(image)
    mask = ((array[:, :, 1] > 120) & (array[:, :, 0] < 120)
            & (array[:, :, 2] < 120))
    return int(mask.sum())


def _count_axis_pixels(image, channel):
    """Count saturated axis-guide pixels of one channel (red X, green Y,
    blue Z); the black wireframe and white background both fail these
    masks."""
    array = _rgb_array(image)
    red, green, blue = array[:, :, 0], array[:, :, 1], array[:, :, 2]
    if channel == "red":
        mask = (red > 150) & (green < 110) & (blue < 110)
    elif channel == "green":
        mask = (green > 150) & (red < 110) & (blue < 110)
    else:
        mask = (blue > 180) & (red < 130) & (green < 150)
    return int(mask.sum())


def _make_color_field():
    """A Gouraud-shaded quad (two triangles) with distinct corner colors."""
    import numpy as np
    vertices = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                        dtype='float32')
    colors = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)],
                      dtype='float32')
    indices = np.array([(0, 1, 2), (0, 2, 3)], dtype='uint32')
    return vertices, colors, indices


def _update_field(widget, vertices, colors, indices):
    """Wrap numpy tables in solvcon arrays and push them to the widget."""
    core = solvcon.core
    widget.updateColorField(
        core.SimpleArrayFloat32(array=vertices.astype('float32')),
        core.SimpleArrayFloat32(array=colors.astype('float32')),
        core.SimpleArrayUint32(array=indices.astype('uint32')))


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetFoundationTC(unittest.TestCase):
    """The render foundation and the Python control spine (step 1)."""

    @classmethod
    def setUpClass(cls):
        # The manager owns the QApplication the widget needs to exist.
        pilot.RManager.instance.setUp()

    def test_construct_from_python(self):
        """RDomainWidget is constructible directly from Python."""
        widget = pilot.RDomainWidget()
        self.assertIsNotNone(widget)

    def test_save_image_writes_png_file(self):
        """saveImage routes through grabImage and yields a valid frame whose
        pixel size matches the widget."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        image = _grab_or_skip(widget)
        self.assertGreater(image.width(), 0)
        self.assertGreater(image.height(), 0)

    def test_empty_scene_is_background(self):
        """With no mesh the frame is the uniform white clear color: nothing
        is drawn, so no pixel is a dark line."""
        widget = pilot.RDomainWidget()
        widget.resize(160, 120)
        image = _grab_or_skip(widget)
        self.assertEqual(_count_foreground(image), 0)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetMeshTC(unittest.TestCase):
    """Domain wireframe rendering for 2D and 3D meshes (step 2)."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_mesh_property_round_trip(self):
        """updateMesh exposes the mesh through the pybind11 widget."""
        widget = pilot.RDomainWidget()
        mh = _make_2d_mesh()
        widget.updateMesh(mh)
        self.assertIsNotNone(widget.mesh)
        self.assertEqual(widget.mesh.ncell, 3)

    def test_2d_mesh_draws_wireframe(self):
        """A 2D mesh renders a wireframe: some pixels are dark lines."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        image = _grab_or_skip(widget)
        self.assertGreater(_count_foreground(image), 0)

    def test_3d_mesh_draws_wireframe(self):
        """A 3D tetrahedron renders its edges without error."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_3d_mesh())
        image = _grab_or_skip(widget)
        self.assertGreater(_count_foreground(image), 0)

    def test_show_mesh_toggles_visibility(self):
        """showMesh(False) hides the wireframe; showMesh(True) restores it.

        Hiding removes most of the wireframe rather than every last pixel: a
        software rasterizer can leave a few stray edge pixels behind, so the
        check is relative (hidden is a small fraction of shown) instead of an
        exact zero.
        """
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        shown = _count_foreground(_grab_or_skip(widget))
        self.assertGreater(shown, 0)
        widget.showMesh(False)
        hidden = _count_foreground(_grab_or_skip(widget))
        self.assertLess(hidden, shown * 0.5)
        widget.showMesh(True)
        restored = _count_foreground(_grab_or_skip(widget))
        self.assertGreater(restored, hidden)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetStyleTC(unittest.TestCase):
    """Independently toggling the surface, wireframe, and points styles."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_2d_surface_renders_filled(self):
        """A 2D mesh drawn as a lit surface fills the cells with color, not
        just the black hairline the wireframe draws."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        _show_only(widget, "surface")
        self.assertGreater(_count_colored(_grab_or_skip(widget)), 0)

    def test_3d_surface_renders_filled(self):
        """A 3D mesh drawn as a lit surface shades its boundary faces."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_3d_mesh())
        _show_only(widget, "surface")
        widget.fitCameraToScene()
        self.assertGreater(_count_colored(_grab_or_skip(widget)), 0)

    def test_surface_differs_from_wireframe(self):
        """The surface fills the interior the wireframe leaves as background,
        so the two styles rasterize to different frames."""
        wire = pilot.RDomainWidget()
        wire.resize(320, 240)
        wire.updateMesh(_make_2d_mesh())
        wire_frame = _rgb_array(_grab_or_skip(wire))

        surf = pilot.RDomainWidget()
        surf.resize(320, 240)
        surf.updateMesh(_make_2d_mesh())
        _show_only(surf, "surface")
        surf_frame = _rgb_array(_grab_or_skip(surf))
        self.assertTrue((wire_frame != surf_frame).any())

    def test_points_render(self):
        """A 2D mesh drawn as points marks its nodes: some foreground pixels
        stand over the background."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        _show_only(widget, "points")
        self.assertGreater(_count_foreground(_grab_or_skip(widget)), 0)

    def test_show_mesh_hides_active_surface(self):
        """showMesh(False) hides whichever styles are shown, and
        showMesh(True) brings them back.

        The count keys on the difference from the frame's own background (as in
        the wireframe toggle test), so an empty scene that a software
        rasterizer reads back as a uniformly dark frame still counts as
        nothing drawn.
        """
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        _show_only(widget, "surface")
        shown = _count_foreground(_grab_or_skip(widget))
        self.assertGreater(shown, 0)
        widget.showMesh(False)
        hidden = _count_foreground(_grab_or_skip(widget))
        self.assertLess(hidden, shown * 0.5)
        widget.showMesh(True)
        self.assertGreater(_count_foreground(_grab_or_skip(widget)), hidden)

    def test_styles_toggle_independently(self):
        """Showing or hiding one style leaves the others as they were."""
        widget = pilot.RDomainWidget()
        self.assertTrue(widget.meshStyleShown("wireframe"))
        self.assertFalse(widget.meshStyleShown("surface"))
        self.assertFalse(widget.meshStyleShown("points"))
        widget.showMeshStyle("surface", True)
        self.assertTrue(widget.meshStyleShown("surface"))
        self.assertTrue(widget.meshStyleShown("wireframe"))
        widget.showMeshStyle("wireframe", False)
        self.assertFalse(widget.meshStyleShown("wireframe"))
        self.assertTrue(widget.meshStyleShown("surface"))

    def test_unknown_style_is_ignored(self):
        """An unknown style name toggles nothing and reads back false."""
        widget = pilot.RDomainWidget()
        before = [widget.meshStyleShown(n)
                  for n in ("surface", "wireframe", "points")]
        widget.showMeshStyle("bogus", True)
        after = [widget.meshStyleShown(n)
                 for n in ("surface", "wireframe", "points")]
        self.assertEqual(before, after)
        self.assertFalse(widget.meshStyleShown("bogus"))

    def test_wireframe_over_surface_overlays(self):
        """Adding the wireframe over the lit surface changes the frame: the
        black edges the surface alone does not draw now appear over the
        fill."""
        surf = pilot.RDomainWidget()
        surf.resize(320, 240)
        surf.updateMesh(_make_2d_mesh())
        _show_only(surf, "surface")
        surf_frame = _rgb_array(_grab_or_skip(surf))

        both = pilot.RDomainWidget()
        both.resize(320, 240)
        both.updateMesh(_make_2d_mesh())
        _show_only(both, "surface")
        both.showMeshStyle("wireframe", True)
        both_frame = _rgb_array(_grab_or_skip(both))
        self.assertTrue((surf_frame != both_frame).any())

    def test_all_styles_off_draws_nothing(self):
        """Turning every style off empties the scene."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        _show_only(widget, "surface")
        shown = _count_foreground(_grab_or_skip(widget))
        self.assertGreater(shown, 0)
        for name in ("surface", "wireframe", "points"):
            widget.showMeshStyle(name, False)
        self.assertLess(_count_foreground(_grab_or_skip(widget)), shown * 0.5)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetFieldTC(unittest.TestCase):
    """Field coloring and boundary highlight (step 3)."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_color_field_renders(self):
        """updateColorField draws per-vertex-colored triangles."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        vertices, colors, indices = _make_color_field()
        _update_field(widget, vertices, colors, indices)
        image = _grab_or_skip(widget)
        self.assertGreater(_count_colored(image), 0)

    def test_color_field_is_swappable(self):
        """A second updateColorField replaces the first and still renders."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        vertices, colors, indices = _make_color_field()
        _update_field(widget, vertices, colors, indices)
        # Swap in a dimmer field; the latest field must render. A single grab
        # after the swap keeps the offscreen capture deterministic.
        _update_field(widget, vertices, colors * 0.5, indices)
        self.assertGreater(_count_colored(_grab_or_skip(widget)), 0)

    def test_show_boundary_highlights_set(self):
        """showBoundary draws the set's colored ribbon and hides it again.

        Each state grabs a freshly configured widget: a single capture of a
        fully-set-up widget is exact, matching the live screenshot path.
        """
        base_widget = pilot.RDomainWidget()
        base_widget.resize(320, 240)
        base_widget.updateMesh(_make_2d_mesh())
        base = _count_reddish(_grab_or_skip(base_widget))

        shown_widget = pilot.RDomainWidget()
        shown_widget.resize(320, 240)
        shown_widget.updateMesh(_make_2d_mesh())
        shown_widget.showBoundary(0, True)
        shown = _count_reddish(_grab_or_skip(shown_widget))
        self.assertGreater(shown, base)

        hidden_widget = pilot.RDomainWidget()
        hidden_widget.resize(320, 240)
        hidden_widget.updateMesh(_make_2d_mesh())
        hidden_widget.showBoundary(0, True)
        hidden_widget.showBoundary(0, False)
        hidden = _count_reddish(_grab_or_skip(hidden_widget))
        self.assertLess(hidden, shown)

    def test_show_boundary_without_mesh_is_noop(self):
        """showBoundary on a widget with no mesh does nothing, not crash.

        A real highlight is hundreds of red pixels; the no-op leaves none
        beyond the odd stray edge pixel the software rasterizer emits.
        """
        widget = pilot.RDomainWidget()
        widget.resize(160, 120)
        widget.showBoundary(0, True)
        image = _grab_or_skip(widget)
        self.assertLess(_count_reddish(image), 5)

    def test_color_field_rejects_out_of_range_index(self):
        """A triangle index past the vertex count is rejected, not fed to
        the GPU as an out-of-bounds fetch."""
        import numpy as np
        widget = pilot.RDomainWidget()
        vertices = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0)],
                            dtype='float32')
        colors = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype='float32')
        indices = np.array([(0, 1, 9), (0, 1, 2)], dtype='uint32')
        with self.assertRaises(ValueError):
            _update_field(widget, vertices, colors, indices)


def _make_scalar_field():
    """A quad (two triangles) with a left-to-right scalar ramp."""
    import numpy as np
    vertices = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                        dtype='float32')
    scalars = np.array([0.0, 1.0, 1.0, 0.0], dtype='float32')
    indices = np.array([(0, 1, 2), (0, 2, 3)], dtype='uint32')
    return vertices, scalars, indices


def _update_scalar_field(widget, vertices, scalars, indices):
    """Wrap numpy tables in solvcon arrays and push them to the widget."""
    core = solvcon.core
    widget.updateScalarField(
        core.SimpleArrayFloat32(array=vertices.astype('float32')),
        core.SimpleArrayFloat32(array=scalars.astype('float32')),
        core.SimpleArrayUint32(array=indices.astype('uint32')))


def _count_saturated(image):
    """Count strongly saturated pixels: a wide channel spread rejects the
    white background, the black wireframe, and every gray."""
    array = _rgb_array(image).astype('int16')
    spread = array.max(axis=2) - array.min(axis=2)
    return int((spread > 60).sum())


def _count_dominant(image, channel):
    """Count pixels where one channel dominates muted others; catches the
    dark jet range ends ((0, 0, 0.5) and (0.5, 0, 0)) that the saturated
    masks miss."""
    array = _rgb_array(image)
    others = [c for c in range(3) if c != channel]
    mask = ((array[:, :, channel] > 100)
            & (array[:, :, others[0]] < 80)
            & (array[:, :, others[1]] < 80))
    return int(mask.sum())


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetScalarFieldTC(unittest.TestCase):
    """GPU LUT scalar coloring and the scalar bar."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_scalar_field_renders(self):
        """updateScalarField draws triangles colored through the LUT."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        vertices, scalars, indices = _make_scalar_field()
        _update_scalar_field(widget, vertices, scalars, indices)
        image = _grab_or_skip(widget)
        self.assertGreater(_count_colored(image), 0)

    def test_scalar_field_is_swappable(self):
        """A second updateScalarField replaces the first and still
        renders."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        vertices, scalars, indices = _make_scalar_field()
        _update_scalar_field(widget, vertices, scalars, indices)
        _update_scalar_field(widget, vertices, scalars * 0.5, indices)
        self.assertGreater(_count_colored(_grab_or_skip(widget)), 0)

    def test_colormap_round_trip(self):
        """The colormap defaults to viridis and switches by name."""
        widget = pilot.RDomainWidget()
        self.assertEqual(widget.colormap, "viridis")
        widget.colormap = "jet"
        self.assertEqual(widget.colormap, "jet")

    def test_colormap_rejects_unknown_name(self):
        """An unknown colormap name raises instead of rendering garbage."""
        widget = pilot.RDomainWidget()
        with self.assertRaises(ValueError):
            widget.colormap = "no-such-map"

    def test_lut_matches_named_map_at_range_ends(self):
        """With jet and a pinned [0, 2] range, a constant-0 field samples the
        LUT start (dark blue) and a constant-2 field the LUT end (red)."""
        import numpy as np
        vertices, _, indices = _make_scalar_field()

        low_widget = pilot.RDomainWidget()
        low_widget.resize(320, 240)
        low_widget.colormap = "jet"
        low_widget.setScalarRange(0.0, 2.0)
        _update_scalar_field(low_widget, vertices,
                             np.zeros(4, dtype='float32'), indices)
        low = _grab_or_skip(low_widget)
        self.assertGreater(_count_dominant(low, 2), 100)
        self.assertLess(_count_dominant(low, 0), 5)

        high_widget = pilot.RDomainWidget()
        high_widget.resize(320, 240)
        high_widget.colormap = "jet"
        high_widget.setScalarRange(0.0, 2.0)
        _update_scalar_field(high_widget, vertices,
                             np.full(4, 2.0, dtype='float32'), indices)
        high = _grab_or_skip(high_widget)
        self.assertGreater(_count_dominant(high, 0), 100)
        self.assertLess(_count_dominant(high, 2), 5)

    def test_grayscale_map_stays_gray(self):
        """The grayscale map colors the ramp without any saturated pixel,
        where a live swap to jet turns the quad saturated and red-ended.

        Each state grabs a freshly configured widget once: a second grab
        of a mutated widget is unreliable on the headless software
        rasterizer (see test_color_field_is_swappable).
        """
        vertices, scalars, indices = _make_scalar_field()

        gray_widget = pilot.RDomainWidget()
        gray_widget.resize(320, 240)
        gray_widget.colormap = "grayscale"
        _update_scalar_field(gray_widget, vertices, scalars, indices)
        gray_image = _grab_or_skip(gray_widget)
        self.assertGreater(_count_colored(gray_image), 0)
        self.assertLess(_count_saturated(gray_image), 5)

        # Swap the map after the field exists so the grab exercises the
        # live LUT re-upload; red-dominant pixels appear only under jet
        # (the default viridis has none).
        jet_widget = pilot.RDomainWidget()
        jet_widget.resize(320, 240)
        _update_scalar_field(jet_widget, vertices, scalars, indices)
        jet_widget.colormap = "jet"
        jet_image = _grab_or_skip(jet_widget)
        self.assertGreater(_count_saturated(jet_image), 100)
        self.assertGreater(_count_dominant(jet_image, 0), 50)

    def test_scalar_range_defaults_to_data_range(self):
        """Without setScalarRange the mapping spans the field data."""
        import numpy as np
        widget = pilot.RDomainWidget()
        vertices, _, indices = _make_scalar_field()
        scalars = np.array([2.0, 8.0, 8.0, 2.0], dtype='float32')
        _update_scalar_field(widget, vertices, scalars, indices)
        self.assertEqual(widget.scalarRange, (2.0, 8.0))

    def test_scalar_range_pins_across_updates(self):
        """setScalarRange overrides auto-ranging for later field updates."""
        widget = pilot.RDomainWidget()
        widget.setScalarRange(0.0, 10.0)
        vertices, scalars, indices = _make_scalar_field()
        _update_scalar_field(widget, vertices, scalars, indices)
        self.assertEqual(widget.scalarRange, (0.0, 10.0))

    def test_scalar_range_rejects_inverted_range(self):
        """A hi below lo raises instead of feeding a negative span."""
        widget = pilot.RDomainWidget()
        with self.assertRaises(ValueError):
            widget.setScalarRange(1.0, 0.0)

    def test_scalar_field_rejects_mismatched_scalars(self):
        """A scalar table shorter than the vertex table is rejected."""
        import numpy as np
        widget = pilot.RDomainWidget()
        vertices, _, indices = _make_scalar_field()
        with self.assertRaises(ValueError):
            _update_scalar_field(widget, vertices,
                                 np.zeros(3, dtype='float32'), indices)

    def test_scalar_bar_hidden_by_default(self):
        """Without showScalarBar an empty scene stays the white clear."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        image = _grab_or_skip(widget)
        self.assertEqual(_count_foreground(image), 0)

    def test_scalar_bar_renders_on_the_right(self):
        """showScalarBar draws the colormap strip along the right edge."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.colormap = "jet"
        widget.setScalarBarTitle("density")
        widget.showScalarBar(True)
        image = _grab_or_skip(widget)
        array = _rgb_array(image).astype('int16')
        spread = array.max(axis=2) - array.min(axis=2)
        left, right = spread[:, :160], spread[:, 160:]
        self.assertGreater(int((right > 60).sum()), 100)
        self.assertEqual(int((left > 60).sum()), 0)

    def test_scalar_bar_toggles_off(self):
        """showScalarBar(False) clears the overlay again.

        Each state grabs a fresh widget once (the double-grab caveat of
        test_grayscale_map_stays_gray), and saturated pixels stand in for
        the bar: an emptied frame can read back from the software
        rasterizer as uniformly dark, which a white-difference count
        would mistake for content."""
        shown_widget = pilot.RDomainWidget()
        shown_widget.resize(320, 240)
        shown_widget.colormap = "jet"
        shown_widget.showScalarBar(True)
        shown = _count_saturated(_grab_or_skip(shown_widget))
        self.assertGreater(shown, 100)

        hidden_widget = pilot.RDomainWidget()
        hidden_widget.resize(320, 240)
        hidden_widget.colormap = "jet"
        hidden_widget.showScalarBar(True)
        hidden_widget.showScalarBar(False)
        hidden = _count_saturated(_grab_or_skip(hidden_widget))
        self.assertLess(hidden, shown * 0.1)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetOpacityTC(unittest.TestCase):
    """Adjustable per-drawable opacity for the wireframe and the surface."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_field_opacity_fades_toward_background(self):
        """Lowering the field opacity blends the shaded surface toward the
        white background, so far fewer pixels stay strongly saturated."""
        vertices, colors, indices = _make_color_field()

        opaque = pilot.RDomainWidget()
        opaque.resize(320, 240)
        _update_field(opaque, vertices, colors, indices)
        strong_opaque = _count_colored(_grab_or_skip(opaque), 100)
        self.assertGreater(strong_opaque, 0)

        faded = pilot.RDomainWidget()
        faded.resize(320, 240)
        _update_field(faded, vertices, colors, indices)
        faded.setFieldOpacity(0.3)
        strong_faded = _count_colored(_grab_or_skip(faded), 100)
        self.assertLess(strong_faded, strong_opaque)

    def test_mesh_opacity_fades_wireframe(self):
        """A low mesh opacity fades the black wireframe toward white, so its
        lines fall below the foreground threshold."""
        opaque = pilot.RDomainWidget()
        opaque.resize(320, 240)
        opaque.updateMesh(_make_2d_mesh())
        shown = _count_foreground(_grab_or_skip(opaque))
        self.assertGreater(shown, 0)

        faded = pilot.RDomainWidget()
        faded.resize(320, 240)
        faded.updateMesh(_make_2d_mesh())
        faded.setMeshOpacity(0.2)
        self.assertLess(_count_foreground(_grab_or_skip(faded)), shown)

    def test_set_opacity_without_drawable_is_noop(self):
        """Setting opacity before a mesh or field exists is a harmless no-op,
        not a crash."""
        widget = pilot.RDomainWidget()
        widget.setMeshOpacity(0.5)
        widget.setFieldOpacity(0.5)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetOverlayTC(unittest.TestCase):
    """The feature-edge and face-normal overlays."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_feature_edges_draw_and_hide(self):
        """showFeatureEdges draws the bold orange boundary outline and hides
        it again. Each state grabs a freshly configured widget so the
        offscreen capture is deterministic."""
        base_widget = pilot.RDomainWidget()
        base_widget.resize(320, 240)
        base_widget.updateMesh(_make_2d_mesh())
        base = _count_orange(_grab_or_skip(base_widget))

        shown_widget = pilot.RDomainWidget()
        shown_widget.resize(320, 240)
        shown_widget.updateMesh(_make_2d_mesh())
        shown_widget.showFeatureEdges(True)
        shown = _count_orange(_grab_or_skip(shown_widget))
        self.assertGreater(shown, base)

        hidden_widget = pilot.RDomainWidget()
        hidden_widget.resize(320, 240)
        hidden_widget.updateMesh(_make_2d_mesh())
        hidden_widget.showFeatureEdges(True)
        hidden_widget.showFeatureEdges(False)
        hidden = _count_orange(_grab_or_skip(hidden_widget))
        self.assertLess(hidden, shown)

    def test_feature_edges_without_mesh_is_noop(self):
        """showFeatureEdges on a widget with no mesh does nothing, not
        crash."""
        widget = pilot.RDomainWidget()
        widget.resize(160, 120)
        widget.showFeatureEdges(True)
        image = _grab_or_skip(widget)
        self.assertLess(_count_orange(image), 5)

    def test_normals_draw_and_hide(self):
        """showNormals draws green face-normal arrows and hides them again."""
        base_widget = pilot.RDomainWidget()
        base_widget.resize(320, 240)
        base_widget.updateMesh(_make_2d_mesh())
        base = _count_green(_grab_or_skip(base_widget))

        shown_widget = pilot.RDomainWidget()
        shown_widget.resize(320, 240)
        shown_widget.updateMesh(_make_2d_mesh())
        shown_widget.showNormals(True)
        shown = _count_green(_grab_or_skip(shown_widget))
        self.assertGreater(shown, base)

        hidden_widget = pilot.RDomainWidget()
        hidden_widget.resize(320, 240)
        hidden_widget.updateMesh(_make_2d_mesh())
        hidden_widget.showNormals(True)
        hidden_widget.showNormals(False)
        hidden = _count_green(_grab_or_skip(hidden_widget))
        self.assertLess(hidden, shown)

    def test_normals_without_mesh_is_noop(self):
        """showNormals on a widget with no mesh does nothing, not crash."""
        widget = pilot.RDomainWidget()
        widget.resize(160, 120)
        widget.showNormals(True)
        image = _grab_or_skip(widget)
        self.assertLess(_count_green(image), 5)


def _distinct_field_colors(image):
    """Count distinct saturated (non-black, non-white) colors, quantized.

    The categorical coloring paints each cell category a distinct qualitative
    color; this counts how many show. Near-white background and near-black
    wireframe pixels are dropped so only the filled categories are counted.
    """
    import numpy as np
    array = _rgb_array(image).astype('int16')
    flat = array.reshape(-1, 3)
    lo = flat.min(axis=1)
    hi = flat.max(axis=1)
    colored = flat[(hi < 230) & (lo > 25) & ((hi - lo) > 25)]
    if colored.size == 0:
        return 0
    quant = colored // 40
    return len(np.unique(quant, axis=0))


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetCellColoringTC(unittest.TestCase):
    """Categorical coloring of the mesh by a cell attribute."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_color_by_cell_type_shows_distinct_categories(self):
        """The mixed 2D mesh (two triangles, one quad) colors its two element
        types in distinct colors."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        widget.showMeshStyle("wireframe", False)
        widget.colorByCellType()
        image = _grab_or_skip(widget)
        self.assertGreaterEqual(_distinct_field_colors(image), 2)

    def test_color_by_cell_group_renders(self):
        """Coloring by cell group fills the mesh with colored pixels."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        widget.showMeshStyle("wireframe", False)
        widget.colorByCellGroup()
        self.assertGreater(_count_colored(_grab_or_skip(widget)), 0)

    def test_color_by_boundary_renders(self):
        """Coloring by boundary set fills the mesh with colored pixels."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        widget.showMeshStyle("wireframe", False)
        widget.colorByBoundary()
        self.assertGreater(_count_colored(_grab_or_skip(widget)), 0)

    def test_clear_cell_coloring_removes_the_field(self):
        """clearCellColoring drops the categorical field, leaving far fewer
        colored pixels than while it was shown."""
        colored = pilot.RDomainWidget()
        colored.resize(320, 240)
        colored.updateMesh(_make_2d_mesh())
        colored.showMeshStyle("wireframe", False)
        colored.colorByCellType()
        shown = _count_colored(_grab_or_skip(colored))
        self.assertGreater(shown, 0)

        cleared = pilot.RDomainWidget()
        cleared.resize(320, 240)
        cleared.updateMesh(_make_2d_mesh())
        cleared.showMeshStyle("wireframe", False)
        cleared.colorByCellType()
        cleared.clearCellColoring()
        self.assertLess(_count_colored(_grab_or_skip(cleared)), shown)

    def test_color_by_cell_type_without_mesh_is_noop(self):
        """Coloring before a mesh loads is a harmless no-op, not a crash."""
        widget = pilot.RDomainWidget()
        widget.colorByCellType()
        widget.colorByCellGroup()
        widget.colorByBoundary()
        widget.clearCellColoring()


def _make_quality_mesh():
    """Two triangles, one well-shaped and one deliberately elongated, so the
    quality metrics have a clear spread.

    Cell 0 = (0,0),(1,0),(0,1): a unit right triangle, area 0.5, longest-to-
    shortest node distance sqrt(2). Cell 1 = (1,0),(5,0),(0,1): area 2.0, node
    distances 4, sqrt(2), sqrt(26), so aspect sqrt(26)/sqrt(2) = sqrt(13).
    """
    core = solvcon.core
    T = core.StaticMesh.TRIANGLE
    mh = core.StaticMesh(ndim=2, nnode=4, nface=0, ncell=2)
    mh.ndcrd.ndarray[:, :] = [(0, 0), (1, 0), (0, 1), (5, 0)]
    mh.cltpn.ndarray[:] = [T, T]
    mh.clnds.ndarray[:, :5] = [(3, 0, 1, 2, -1), (3, 1, 3, 2, -1)]
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    return mh


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetQualityTC(unittest.TestCase):
    """Per-cell quality metrics and their coloring."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_volume_range_matches_hand_computation(self):
        """qualityRange("volume") returns the two cell areas, 0.5 and 2.0."""
        import math
        widget = pilot.RDomainWidget()
        widget.updateMesh(_make_quality_mesh())
        lo, hi = widget.qualityRange("volume")
        self.assertTrue(math.isclose(lo, 0.5, rel_tol=1e-4))
        self.assertTrue(math.isclose(hi, 2.0, rel_tol=1e-4))

    def test_aspect_ratio_range_matches_hand_computation(self):
        """The elongated cell sets the aspect-ratio maximum to sqrt(13)."""
        import math
        widget = pilot.RDomainWidget()
        widget.updateMesh(_make_quality_mesh())
        lo, hi = widget.qualityRange("aspect_ratio")
        self.assertTrue(math.isclose(lo, math.sqrt(2.0), rel_tol=1e-4))
        self.assertTrue(math.isclose(hi, math.sqrt(13.0), rel_tol=1e-4))

    def test_color_by_quality_shows_a_gradient(self):
        """Coloring by aspect ratio paints the two cells distinct colors: the
        bad cell at the extreme of the map, the good one at the other end."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_quality_mesh())
        widget.showMeshStyle("wireframe", False)
        widget.colorByQuality("aspect_ratio")
        image = _grab_or_skip(widget)
        self.assertGreaterEqual(_distinct_field_colors(image), 2)

    def test_unknown_metric_raises(self):
        """An unknown metric name is rejected, not silently colored."""
        widget = pilot.RDomainWidget()
        widget.updateMesh(_make_quality_mesh())
        with self.assertRaises(ValueError):
            widget.qualityRange("bogus")
        with self.assertRaises(ValueError):
            widget.colorByQuality("bogus")

    def test_color_by_quality_without_mesh_is_noop(self):
        """Coloring by quality before a mesh loads does nothing, not crash."""
        widget = pilot.RDomainWidget()
        widget.colorByQuality("volume")
        self.assertEqual(widget.qualityRange("volume"), (0.0, 0.0))


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetViewPresetTC(unittest.TestCase):
    """Axis-aligned view presets and the projection toggle."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_projection_round_trip(self):
        """The projection defaults to auto and takes the forced values."""
        widget = pilot.RDomainWidget()
        self.assertEqual(widget.projection, "auto")
        widget.projection = "parallel"
        self.assertEqual(widget.projection, "parallel")
        widget.projection = "perspective"
        self.assertEqual(widget.projection, "perspective")
        widget.projection = "auto"
        self.assertEqual(widget.projection, "auto")

    def test_unknown_projection_is_ignored(self):
        """An unknown projection name leaves the current one untouched."""
        widget = pilot.RDomainWidget()
        widget.projection = "parallel"
        widget.projection = "bogus"
        self.assertEqual(widget.projection, "parallel")

    def test_view_preset_frames_3d_mesh(self):
        """Each preset frames a 3D mesh so it renders in view."""
        for name in ("front", "top", "right", "iso"):
            widget = pilot.RDomainWidget()
            widget.resize(320, 240)
            widget.updateMesh(_make_3d_mesh())
            widget.setView(name)
            self.assertGreater(
                _count_foreground(_grab_or_skip(widget)), 0,
                "preset %s drew nothing" % name)

    def test_presets_view_from_different_directions(self):
        """Front and top presets look from different directions, so they
        rasterize a 3D mesh to different frames."""
        front = pilot.RDomainWidget()
        front.resize(320, 240)
        front.updateMesh(_make_3d_mesh())
        front.setView("front")
        front_frame = _rgb_array(_grab_or_skip(front))

        top = pilot.RDomainWidget()
        top.resize(320, 240)
        top.updateMesh(_make_3d_mesh())
        top.setView("top")
        top_frame = _rgb_array(_grab_or_skip(top))
        self.assertTrue((front_frame != top_frame).any())

    def test_projection_toggle_changes_the_frame(self):
        """Forcing parallel versus perspective changes the rendered frame of a
        3D mesh."""
        para = pilot.RDomainWidget()
        para.resize(320, 240)
        para.updateMesh(_make_3d_mesh())
        para.setView("iso")
        para.projection = "parallel"
        para_frame = _rgb_array(_grab_or_skip(para))

        persp = pilot.RDomainWidget()
        persp.resize(320, 240)
        persp.updateMesh(_make_3d_mesh())
        persp.setView("iso")
        persp.projection = "perspective"
        persp_frame = _rgb_array(_grab_or_skip(persp))
        self.assertTrue((para_frame != persp_frame).any())


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetTrackballTC(unittest.TestCase):
    """Trackball orbit style and orbit pivot control."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_orbit_style_round_trip(self):
        """The orbit style defaults to turntable and switches to trackball."""
        widget = pilot.RDomainWidget()
        self.assertEqual(widget.orbitStyle, "turntable")
        widget.orbitStyle = "trackball"
        self.assertEqual(widget.orbitStyle, "trackball")
        widget.setOrbitStyle("turntable")
        self.assertEqual(widget.orbitStyle, "turntable")

    def test_turntable_keeps_the_horizon_level(self):
        """Turntable orbit holds the up axis fixed through a drag."""
        widget = pilot.RDomainWidget()
        widget.updateMesh(_make_3d_mesh())
        widget.cameraMode = "orbit"
        widget.fitCameraToScene()
        up0 = tuple(widget.cameraUp)
        widget.rotateCamera(40.0, 30.0)
        for a, b in zip(up0, tuple(widget.cameraUp)):
            self.assertAlmostEqual(a, b, places=4)

    def test_trackball_rolls_the_horizon(self):
        """Trackball orbit rolls the up axis with the drag, so the horizon is
        free to tilt (unlike the turntable)."""
        widget = pilot.RDomainWidget()
        widget.updateMesh(_make_3d_mesh())
        widget.cameraMode = "orbit"
        widget.fitCameraToScene()
        widget.orbitStyle = "trackball"
        up0 = tuple(widget.cameraUp)
        widget.rotateCamera(40.0, 30.0)
        up1 = tuple(widget.cameraUp)
        self.assertTrue(any(abs(a - b) > 1e-3 for a, b in zip(up0, up1)))

    def test_set_pivot_moves_the_center_of_rotation(self):
        """Setting the pivot moves the orbit target there, and orbiting then
        keeps the eye's distance to that new pivot."""
        import math
        widget = pilot.RDomainWidget()
        widget.updateMesh(_make_3d_mesh())
        widget.cameraMode = "orbit"
        widget.setPivot(1.0, 2.0, 3.0)
        self.assertEqual(tuple(widget.cameraTarget), (1.0, 2.0, 3.0))

        def radius():
            p = tuple(widget.cameraPosition)
            t = tuple(widget.cameraTarget)
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, t)))

        before = radius()
        widget.rotateCamera(20.0, 10.0)
        self.assertAlmostEqual(before, radius(), places=4)

    def test_frame_selected_frames_the_scene(self):
        """frameSelected recenters and frames the scene so it renders."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_3d_mesh())
        widget.frameSelected()
        self.assertGreater(_count_foreground(_grab_or_skip(widget)), 0)


def _make_single_quad_2d():
    """A single unit quad filling [0, 1] x [0, 1], so the framed view center
    lands squarely inside cell 0."""
    core = solvcon.core
    Q = core.StaticMesh.QUADRILATERAL
    mh = core.StaticMesh(ndim=2, nnode=4, nface=0, ncell=1)
    mh.ndcrd.ndarray[:, :] = [(0, 0), (1, 0), (1, 1), (0, 1)]
    mh.cltpn.ndarray[:] = [Q]
    mh.clnds.ndarray[:, :5] = [(4, 0, 1, 2, 3)]
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    return mh


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetPickTC(unittest.TestCase):
    """Picking a cell, node, or face and reporting its geometry.

    Picking back-projects the click through the same view-projection the
    renderer uses, entirely on the CPU, so these tests are exact and need no
    rendered frame.
    """

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_pick_cell_returns_id_and_geometry(self):
        """Clicking the framed view center of a unit quad picks cell 0 and
        reports its type, area, and centroid."""
        widget = pilot.RDomainWidget()
        widget.resize(200, 200)
        widget.updateMesh(_make_single_quad_2d())
        widget.fitCameraToScene()
        r = widget.pickCell(100, 100)
        self.assertIsNotNone(r)
        self.assertEqual(r["kind"], "cell")
        self.assertEqual(r["id"], 0)
        self.assertAlmostEqual(r["measure"], 1.0, places=4)
        cx, cy, _cz = r["centroid"]
        self.assertAlmostEqual(cx, 0.5, places=3)
        self.assertAlmostEqual(cy, 0.5, places=3)
        self.assertTrue(widget.hasSelection)

    def test_pick_outside_returns_none(self):
        """A click on the background (a corner, outside the framed quad) picks
        nothing."""
        widget = pilot.RDomainWidget()
        widget.resize(200, 200)
        widget.updateMesh(_make_single_quad_2d())
        widget.fitCameraToScene()
        self.assertIsNone(widget.pickCell(0, 0))

    def test_pick_without_mesh_returns_none(self):
        """Picking before a mesh loads returns None, not a crash."""
        widget = pilot.RDomainWidget()
        widget.resize(200, 200)
        self.assertIsNone(widget.pickCell(100, 100))
        self.assertIsNone(widget.pickNode(100, 100))
        self.assertIsNone(widget.pickFace(100, 100))

    def test_pick_node_returns_a_node(self):
        """Picking near the view center returns the nearest node."""
        widget = pilot.RDomainWidget()
        widget.resize(200, 200)
        widget.updateMesh(_make_single_quad_2d())
        widget.fitCameraToScene()
        r = widget.pickNode(100, 100)
        self.assertIsNotNone(r)
        self.assertEqual(r["kind"], "node")
        self.assertIn(r["id"], (0, 1, 2, 3))

    def test_pick_face_on_3d_mesh(self):
        """A ray that meets the 3D shell picks a boundary face with an area."""
        widget = pilot.RDomainWidget()
        widget.resize(200, 200)
        widget.updateMesh(_make_3d_mesh())
        widget.fitCameraToScene()
        hit = None
        for yy in range(20, 200, 15):
            for xx in range(20, 200, 15):
                r = widget.pickFace(xx, yy)
                if r is not None:
                    hit = r
                    break
            if hit is not None:
                break
        self.assertIsNotNone(hit, "no boundary face was picked over the shell")
        self.assertEqual(hit["kind"], "face")
        self.assertGreater(hit["measure"], 0.0)

    def test_clear_selection(self):
        """clearSelection drops the pick and its highlight."""
        widget = pilot.RDomainWidget()
        widget.resize(200, 200)
        widget.updateMesh(_make_single_quad_2d())
        widget.fitCameraToScene()
        widget.pickCell(100, 100)
        self.assertTrue(widget.hasSelection)
        widget.clearSelection()
        self.assertFalse(widget.hasSelection)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetNavMapTC(unittest.TestCase):
    """Blender-style navigation mapping, discrete steps, and sensitivity.

    The button-to-action routing lives in the C++ mouse handlers; a
    pybind-created widget is not a PySide QObject that ``sendEvent`` accepts,
    so the routing is exercised live, and the selectable mapping, the discrete
    orbit step, and the sensitivity scaling are covered here through the
    Python API.
    """

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_navigation_mapping_round_trip(self):
        """The mapping defaults to blender and switches to default."""
        widget = pilot.RDomainWidget()
        self.assertEqual(widget.navigationMapping, "blender")
        widget.navigationMapping = "default"
        self.assertEqual(widget.navigationMapping, "default")
        widget.setNavigationMapping("blender")
        self.assertEqual(widget.navigationMapping, "blender")

    def test_orbit_step_rotates_a_fixed_angle(self):
        """A discrete orbit step swings the eye about the target by a fixed
        angle, keeping the target and the eye-to-target distance."""
        import math
        widget = pilot.RDomainWidget()
        widget.updateMesh(_make_3d_mesh())
        widget.cameraMode = "orbit"

        def radius():
            p = tuple(widget.cameraPosition)
            t = tuple(widget.cameraTarget)
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, t)))

        t0 = tuple(widget.cameraTarget)
        p0 = tuple(widget.cameraPosition)
        r0 = radius()
        widget.orbitStep(30.0, 0.0)
        for a, b in zip(t0, tuple(widget.cameraTarget)):
            self.assertAlmostEqual(a, b, places=4)
        self.assertAlmostEqual(r0, radius(), places=4)
        self.assertTrue(
            any(abs(a - b) > 1e-3
                for a, b in zip(p0, tuple(widget.cameraPosition))))

    def test_orbit_sensitivity_scales_rotation(self):
        """A higher sensitivity sweeps the eye further for the same drag."""
        import math

        def swept(factor):
            widget = pilot.RDomainWidget()
            widget.updateMesh(_make_3d_mesh())
            widget.cameraMode = "orbit"
            widget.setOrbitSensitivity(factor)
            p0 = tuple(widget.cameraPosition)
            widget.rotateCamera(20.0, 0.0)
            p1 = tuple(widget.cameraPosition)
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(p0, p1)))

        self.assertGreater(swept(2.0), swept(1.0))


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetCaptureTC(unittest.TestCase):
    """High-resolution and transparent offscreen capture."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def _render_or_skip(self, widget, width, height, transparent=False):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "capture.png")
            ok = widget.renderToImage(path, width, height, transparent)
            if not ok or not os.path.exists(path):
                raise unittest.SkipTest("offscreen capture is unavailable")
            image = QImage(path)
        if image.isNull():
            raise unittest.SkipTest("offscreen capture is unavailable")
        return image

    def test_capture_size_is_independent_of_widget_size(self):
        """A capture is produced at the requested size regardless of the
        widget size."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        image = self._render_or_skip(widget, 512, 384)
        self.assertEqual(image.width(), 512)
        self.assertEqual(image.height(), 384)

    def test_transparent_capture_has_a_clear_corner(self):
        """A transparent capture leaves the background corner not fully
        opaque."""
        widget = pilot.RDomainWidget()
        widget.resize(200, 200)
        widget.updateMesh(_make_3d_mesh())
        image = self._render_or_skip(widget, 256, 256, transparent=True)
        self.assertLess(image.pixelColor(0, 0).alpha(), 255)

    def test_invalid_size_writes_nothing(self):
        """A non-positive size yields a null image, so the write fails and no
        file is produced (no crash)."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bad.png")
            self.assertFalse(widget.renderToImage(path, 0, 0))
            self.assertFalse(os.path.exists(path))


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetSceneObjectsTC(unittest.TestCase):
    """A scene of several named mesh objects with transforms."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_two_objects_register(self):
        """Two named meshes both register in the scene."""
        widget = pilot.RDomainWidget()
        widget.addObject("left", _make_2d_mesh())
        widget.addObject("right", _make_3d_mesh())
        names = sorted(widget.objectNames())
        self.assertEqual(names, ["left", "right"])

    def test_readding_replaces_object(self):
        """Adding a name that already exists replaces it, not duplicates."""
        widget = pilot.RDomainWidget()
        widget.addObject("m", _make_2d_mesh())
        widget.addObject("m", _make_3d_mesh())
        self.assertEqual(widget.objectNames(), ["m"])

    def test_transform_and_visibility_setters_by_name(self):
        """The transform, visibility, and opacity setters accept a known name
        and ignore an unknown one, without crashing."""
        widget = pilot.RDomainWidget()
        widget.addObject("a", _make_2d_mesh())
        widget.setObjectTransform("a", 3.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        widget.setObjectVisible("a", False)
        widget.setObjectOpacity("a", 0.5)
        widget.setObjectVisible("nope", True)  # unknown: no-op
        self.assertEqual(widget.objectNames(), ["a"])

    def test_two_objects_render_and_toggle(self):
        """Two objects with distinct transforms both render; hiding one drops
        drawn pixels.

        Each state grabs a freshly configured widget: a second grab of a
        mutated widget is unreliable on the headless software rasterizer.
        _count_foreground keys on the difference from the frame's own
        background, so a dark empty read-back still counts as nothing drawn.
        """
        both_widget = pilot.RDomainWidget()
        both_widget.resize(320, 240)
        both_widget.addObject("a", _make_2d_mesh())
        both_widget.addObject("b", _make_2d_mesh())
        both_widget.setObjectTransform("b", 4.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        both = _count_foreground(_grab_or_skip(both_widget))
        self.assertGreater(both, 0)

        one_widget = pilot.RDomainWidget()
        one_widget.resize(320, 240)
        one_widget.addObject("a", _make_2d_mesh())
        one_widget.addObject("b", _make_2d_mesh())
        one_widget.setObjectTransform("b", 4.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        one_widget.setObjectVisible("b", False)
        one = _count_foreground(_grab_or_skip(one_widget))
        self.assertLess(one, both)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetFilterTC(unittest.TestCase):
    """Geometric slice and clip of the mesh."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_clip_keeps_one_side(self):
        """A plane at x=1 keeps the two cells whose centroid is left of it and
        drops the right-hand quad."""
        widget = pilot.RDomainWidget()
        widget.updateMesh(_make_2d_mesh())
        kept = widget.addClip((1.0, 0.5, 0.0), (1.0, 0.0, 0.0))
        self.assertEqual(kept, 2)

    def test_clip_without_mesh_is_zero(self):
        """Clipping before a mesh loads keeps nothing, no crash."""
        widget = pilot.RDomainWidget()
        self.assertEqual(widget.addClip((0, 0, 0), (1, 0, 0)), 0)

    def test_slice_2d_cuts_cells(self):
        """A vertical plane cutting the 2D mesh draws cross-section lines."""
        widget = pilot.RDomainWidget()
        widget.updateMesh(_make_2d_mesh())
        self.assertGreater(
            widget.addSlice((0.9, 0.5, 0.0), (1.0, 0.0, 0.0)), 0)

    def test_slice_3d_cuts_the_tet(self):
        """A horizontal plane through the tetrahedron outlines the cut."""
        widget = pilot.RDomainWidget()
        widget.updateMesh(_make_3d_mesh())
        self.assertGreater(
            widget.addSlice((0.0, 0.0, 0.5), (0.0, 0.0, 1.0)), 0)

    def test_clear_filters_is_harmless(self):
        """Clearing after a slice and clip does not crash."""
        widget = pilot.RDomainWidget()
        widget.updateMesh(_make_2d_mesh())
        widget.addClip((1.0, 0.5, 0.0), (1.0, 0.0, 0.0))
        widget.addSlice((0.9, 0.5, 0.0), (1.0, 0.0, 0.0))
        widget.clearFilters()


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetCubeAxesTC(unittest.TestCase):
    """Cube-axes grid, tick values, and the figure title."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_cube_axes_ticks_match_extent(self):
        """The tick coordinates span the mesh extent per axis."""
        widget = pilot.RDomainWidget()
        widget.updateMesh(_make_2d_mesh())
        widget.showCubeAxes(True)
        tx = widget.cubeAxesTicks(0)
        ty = widget.cubeAxesTicks(1)
        self.assertEqual(len(tx), 5)
        self.assertAlmostEqual(tx[0], 0.0, places=4)
        self.assertAlmostEqual(tx[-1], 2.0, places=4)
        self.assertAlmostEqual(ty[0], 0.0, places=4)
        self.assertAlmostEqual(ty[-1], 1.0, places=4)

    def test_hiding_cube_axes_clears_ticks(self):
        """Hiding the cube axes drops the tick coordinates."""
        widget = pilot.RDomainWidget()
        widget.updateMesh(_make_2d_mesh())
        widget.showCubeAxes(True)
        widget.showCubeAxes(False)
        self.assertEqual(list(widget.cubeAxesTicks(0)), [])

    def test_title_round_trip(self):
        """The title is settable and readable, and clears to empty."""
        widget = pilot.RDomainWidget()
        self.assertEqual(widget.title, "")
        widget.title = "My Mesh"
        self.assertEqual(widget.title, "My Mesh")
        widget.setTitle("")
        self.assertEqual(widget.title, "")

    def test_cube_axes_draw_more_lines(self):
        """Showing the cube axes adds drawn pixels over the plain wireframe."""
        base_widget = pilot.RDomainWidget()
        base_widget.resize(320, 240)
        base_widget.updateMesh(_make_2d_mesh())
        base = _count_foreground(_grab_or_skip(base_widget))

        axes_widget = pilot.RDomainWidget()
        axes_widget.resize(320, 240)
        axes_widget.updateMesh(_make_2d_mesh())
        axes_widget.showCubeAxes(True)
        self.assertGreater(_count_foreground(_grab_or_skip(axes_widget)), base)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetZoomTC(unittest.TestCase):
    """Zoom to the selection and reset the camera."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_zoom_to_selection_frames_the_picked_cell(self):
        """Picking the right-hand quad then zooming frames it: the camera
        target moves to the picked cell centroid, off the scene center."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        widget.fitCameraToScene()
        r = widget.pickCell(240, 120)
        self.assertIsNotNone(r)
        widget.zoomToSelection()
        zoom_target = tuple(widget.cameraTarget)
        self.assertAlmostEqual(zoom_target[0], r["centroid"][0], places=3)
        widget.resetCamera()
        reset_target = tuple(widget.cameraTarget)
        self.assertNotAlmostEqual(zoom_target[0], reset_target[0], places=2)

    def test_zoom_without_selection_frames_the_scene(self):
        """With nothing selected, zoom-to-selection frames the whole scene
        (its bounding-box center)."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        widget.fitCameraToScene()
        widget.zoomToSelection()
        target = tuple(widget.cameraTarget)
        self.assertAlmostEqual(target[0], 1.0, places=3)
        self.assertAlmostEqual(target[1], 0.5, places=3)

    def test_reset_camera_restores_scene_framing(self):
        """After a zoom, reset returns the camera to the whole-scene center."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        widget.fitCameraToScene()
        widget.pickCell(240, 120)
        widget.zoomToSelection()
        widget.resetCamera()
        target = tuple(widget.cameraTarget)
        self.assertAlmostEqual(target[0], 1.0, places=3)
        self.assertAlmostEqual(target[1], 0.5, places=3)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetMeasureTC(unittest.TestCase):
    """Distance and angle measurement between world points."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_measure_distance(self):
        """A 3-4-5 right triangle gives distance 5."""
        widget = pilot.RDomainWidget()
        self.assertAlmostEqual(
            widget.measureDistance((0, 0, 0), (3, 4, 0)), 5.0, places=4)

    def test_measure_distance_between_mesh_nodes(self):
        """Measuring between two known mesh nodes returns their distance."""
        import math
        mesh = _make_3d_mesh()
        nd = mesh.ndcrd.ndarray
        p0 = tuple(float(v) for v in nd[0])
        p1 = tuple(float(v) for v in nd[3])
        expected = math.sqrt(sum((a - b) ** 2 for a, b in zip(p0, p1)))
        widget = pilot.RDomainWidget()
        widget.updateMesh(mesh)
        self.assertAlmostEqual(
            widget.measureDistance(p0, p1), expected, places=4)

    def test_measure_right_angle(self):
        """The angle at the origin between +x and +y is 90 degrees."""
        widget = pilot.RDomainWidget()
        self.assertAlmostEqual(
            widget.measureAngle((1, 0, 0), (0, 0, 0), (0, 1, 0)), 90.0,
            places=3)

    def test_measure_straight_angle(self):
        """The angle between opposite arms is 180 degrees."""
        widget = pilot.RDomainWidget()
        self.assertAlmostEqual(
            widget.measureAngle((1, 0, 0), (0, 0, 0), (-1, 0, 0)), 180.0,
            places=3)

    def test_clear_measurements_is_harmless(self):
        """Clearing the ruler after a measurement does not crash."""
        widget = pilot.RDomainWidget()
        widget.measureDistance((0, 0, 0), (1, 1, 1))
        widget.clearMeasurements()


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetSceneTC(unittest.TestCase):
    """Scene framing and the fit-to-scene camera (step 4)."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_fit_camera_keeps_2d_mesh_in_view(self):
        """fitCameraToScene frames a 2D mesh so its wireframe stays in view."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        widget.fitCameraToScene()
        self.assertGreater(_count_foreground(_grab_or_skip(widget)), 0)

    def test_fit_camera_frames_3d_mesh_in_perspective(self):
        """A 3D mesh is framed under the perspective projection and its
        edges render."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_3d_mesh())
        widget.fitCameraToScene()
        self.assertGreater(_count_foreground(_grab_or_skip(widget)), 0)

    def test_fit_camera_frames_3d_mesh_in_portrait(self):
        """A portrait viewport pulls the perspective camera back enough that
        the 3D domain is not clipped horizontally."""
        widget = pilot.RDomainWidget()
        widget.resize(240, 320)
        widget.updateMesh(_make_3d_mesh())
        widget.fitCameraToScene()
        self.assertGreater(_count_foreground(_grab_or_skip(widget)), 0)

    def test_fit_camera_without_scene_is_harmless(self):
        """fitCameraToScene on an empty widget does not crash or draw."""
        widget = pilot.RDomainWidget()
        widget.resize(160, 120)
        widget.fitCameraToScene()
        self.assertEqual(_count_foreground(_grab_or_skip(widget)), 0)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetCameraTC(unittest.TestCase):
    """Camera modes, programmatic pose, and interaction (step 5)."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_camera_mode_round_trip(self):
        """The camera defaults to orbit and switches between the modes."""
        widget = pilot.RDomainWidget()
        self.assertEqual(widget.cameraMode, "orbit")
        widget.cameraMode = "fps"
        self.assertEqual(widget.cameraMode, "fps")
        widget.cameraMode = "pan"
        self.assertEqual(widget.cameraMode, "pan")

    def test_3d_mesh_keeps_orbit_default(self):
        """A new widget defaults to orbit, and loading a 3D domain keeps it."""
        widget = pilot.RDomainWidget()
        self.assertEqual(widget.cameraMode, "orbit")
        widget.updateMesh(_make_3d_mesh())
        self.assertEqual(widget.cameraMode, "orbit")

    def test_2d_mesh_selects_pan_camera(self):
        """Loading a 2D domain selects pan/zoom, whose wheel zooms the
        orthographic view (the orbit dolly has no effect there)."""
        widget = pilot.RDomainWidget()
        widget.updateMesh(_make_2d_mesh())
        self.assertEqual(widget.cameraMode, "pan")

    def test_camera_pose_round_trip(self):
        """The camera pose is readable and settable from Python."""
        widget = pilot.RDomainWidget()
        widget.cameraPosition = (1.0, 2.0, 3.0)
        widget.cameraTarget = (0.5, 0.5, 0.0)
        widget.cameraUp = (0.0, 0.0, 1.0)
        self.assertEqual(tuple(widget.cameraPosition), (1.0, 2.0, 3.0))
        self.assertEqual(tuple(widget.cameraTarget), (0.5, 0.5, 0.0))
        self.assertEqual(tuple(widget.cameraUp), (0.0, 0.0, 1.0))

    def test_pan_alters_the_2d_view(self):
        """Panning the 2D camera shifts what is drawn."""
        before = pilot.RDomainWidget()
        before.resize(320, 240)
        before.updateMesh(_make_2d_mesh())
        frame_before = _rgb_array(_grab_or_skip(before))

        after = pilot.RDomainWidget()
        after.resize(320, 240)
        after.updateMesh(_make_2d_mesh())
        after.panCamera(60.0, 0.0)
        frame_after = _rgb_array(_grab_or_skip(after))
        self.assertTrue((frame_before != frame_after).any())

    def test_zoom_alters_the_view(self):
        """Zooming the camera changes the rendered frame."""
        before = pilot.RDomainWidget()
        before.resize(320, 240)
        before.updateMesh(_make_2d_mesh())
        frame_before = _rgb_array(_grab_or_skip(before))

        after = pilot.RDomainWidget()
        after.resize(320, 240)
        after.updateMesh(_make_2d_mesh())
        after.zoomCamera(6.0)
        frame_after = _rgb_array(_grab_or_skip(after))
        self.assertTrue((frame_before != frame_after).any())

    def test_first_person_rotation_alters_the_3d_view(self):
        """Looking around in first-person mode changes the 3D frame."""
        before = pilot.RDomainWidget()
        before.resize(320, 240)
        before.cameraMode = "fps"
        before.updateMesh(_make_3d_mesh())
        frame_before = _rgb_array(_grab_or_skip(before))

        after = pilot.RDomainWidget()
        after.resize(320, 240)
        after.cameraMode = "fps"
        after.updateMesh(_make_3d_mesh())
        after.rotateCamera(40.0, 15.0)
        frame_after = _rgb_array(_grab_or_skip(after))
        self.assertTrue((frame_before != frame_after).any())

    def test_first_person_extreme_pitch_stays_stable(self):
        """Pitching hard past vertical does not flip or break the view: the
        look direction is held off the up axis (no gimbal lock)."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.cameraMode = "fps"
        widget.updateMesh(_make_3d_mesh())
        for _ in range(10):
            widget.rotateCamera(0.0, 200.0)
        image = _grab_or_skip(widget)
        self.assertFalse(image.isNull())

    def test_orbit_mode_round_trips(self):
        """The camera mode switches to orbit and back to pan."""
        widget = pilot.RDomainWidget()
        widget.cameraMode = "orbit"
        self.assertEqual(widget.cameraMode, "orbit")
        widget.cameraMode = "pan"
        self.assertEqual(widget.cameraMode, "pan")

    def test_orbit_keeps_target_and_moves_eye(self):
        """Orbit swings the eye around a fixed target; fps instead holds the
        eye and swings the target."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_3d_mesh())
        widget.cameraMode = "orbit"
        target_before = tuple(widget.cameraTarget)
        pos_before = tuple(widget.cameraPosition)
        widget.rotateCamera(40.0, 15.0)
        for before, after in zip(target_before, tuple(widget.cameraTarget)):
            self.assertAlmostEqual(before, after, places=4)
        moved = sum((a - b) ** 2 for a, b
                    in zip(pos_before, tuple(widget.cameraPosition)))
        self.assertGreater(moved, 0.0)

    def test_orbit_preserves_distance_to_target(self):
        """Orbiting is a rotation about the target, so the eye-to-target
        distance is unchanged."""
        import math
        widget = pilot.RDomainWidget()
        widget.updateMesh(_make_3d_mesh())
        widget.cameraMode = "orbit"

        def radius():
            pos = tuple(widget.cameraPosition)
            tgt = tuple(widget.cameraTarget)
            return math.sqrt(sum((p - t) ** 2 for p, t in zip(pos, tgt)))

        before = radius()
        widget.rotateCamera(30.0, 20.0)
        self.assertAlmostEqual(before, radius(), places=4)

    def test_orbit_zoom_dollies_toward_target(self):
        """Orbit zoom shrinks the eye-to-target distance on a positive step
        without moving the target."""
        import math
        widget = pilot.RDomainWidget()
        widget.updateMesh(_make_3d_mesh())
        widget.cameraMode = "orbit"
        target_before = tuple(widget.cameraTarget)

        def radius():
            pos = tuple(widget.cameraPosition)
            tgt = tuple(widget.cameraTarget)
            return math.sqrt(sum((p - t) ** 2 for p, t in zip(pos, tgt)))

        before = radius()
        widget.zoomCamera(3.0)
        self.assertLess(radius(), before)
        for a, b in zip(target_before, tuple(widget.cameraTarget)):
            self.assertAlmostEqual(a, b, places=5)

    def test_orbit_extreme_pitch_stays_stable(self):
        """Orbiting hard past vertical does not flip or break the view: the
        eye-to-target direction is held off the up axis (no gimbal lock)."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_3d_mesh())
        widget.cameraMode = "orbit"
        for _ in range(10):
            widget.rotateCamera(0.0, 200.0)
        image = _grab_or_skip(widget)
        self.assertFalse(image.isNull())

    def test_pinch_zooms_the_orbit_camera(self):
        """A pinch scales the orbit eye-to-target distance inversely: a 2x
        spread halves it (zoom in), and a 0.5x pinch restores it (zoom out)."""
        import math
        widget = pilot.RDomainWidget()
        widget.updateMesh(_make_3d_mesh())  # orbit is the 3D default

        def radius():
            pos = tuple(widget.cameraPosition)
            tgt = tuple(widget.cameraTarget)
            return math.sqrt(sum((p - t) ** 2 for p, t in zip(pos, tgt)))

        start = radius()
        widget.pinchCamera(2.0)
        self.assertAlmostEqual(radius(), start / 2.0, places=4)
        widget.pinchCamera(0.5)
        self.assertAlmostEqual(radius(), start, places=4)

    def test_pinch_ignores_nonpositive_factor(self):
        """A non-positive pinch factor is ignored rather than applied."""
        import math
        widget = pilot.RDomainWidget()
        widget.updateMesh(_make_3d_mesh())

        def radius():
            pos = tuple(widget.cameraPosition)
            tgt = tuple(widget.cameraTarget)
            return math.sqrt(sum((p - t) ** 2 for p, t in zip(pos, tgt)))

        start = radius()
        widget.pinchCamera(0.0)
        self.assertEqual(radius(), start)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetManagerTC(unittest.TestCase):
    """The manager 3D-widget factory and screenshot path (step 7)."""

    def test_add3dwidget_returns_domain_widget(self):
        """RManager.add3DWidget hosts an RDomainWidget and exposes its mesh
        through the same currentR3DWidget accessor as before."""
        mgr = pilot.RManager.instance.setUp()
        widget = mgr.add3DWidget()
        self.assertIsInstance(widget, pilot.RDomainWidget)
        widget.updateMesh(_make_2d_mesh())
        current = mgr.currentR3DWidget()
        self.assertIsNotNone(current)
        self.assertEqual(current.mesh.ncell, 3)

    def test_factory_widget_screenshot(self):
        """The screenshot path of a factory-hosted widget routes through
        grabImage and renders the mesh."""
        mgr = pilot.RManager.instance.setUp()
        widget = mgr.add3DWidget()
        widget.updateMesh(_make_2d_mesh())
        image = _grab_or_skip(widget)
        self.assertGreater(_count_foreground(image), 0)

    def test_multiple_3d_widgets_coexist(self):
        """Several 3D viewers can be added at once. Each is a distinct
        RDomainWidget hosted in its own subwindow, and the accessor reaches
        the active viewer through its container wrapper. A bare QRhiWidget
        nested in a QMdiSubWindow fails to composite and a second one crashes
        the app; the wrapper is what keeps them independent."""
        mgr = pilot.RManager.instance.setUp()
        first = mgr.add3DWidget()
        first.updateMesh(_make_2d_mesh())
        second = mgr.add3DWidget()
        second.updateMesh(_make_3d_mesh())
        self.assertIsInstance(first, pilot.RDomainWidget)
        self.assertIsInstance(second, pilot.RDomainWidget)
        # The two viewers are independent objects with independent meshes.
        self.assertEqual(first.mesh.ncell, 3)
        self.assertEqual(second.mesh.ncell, 1)
        # currentR3DWidget resolves through the container to the active viewer
        # (the one just added), not the first.
        current = mgr.currentR3DWidget()
        self.assertIsNotNone(current)
        self.assertEqual(current.mesh.ncell, 1)

    def test_setup_primes_rhi_composition(self):
        """setUp parks a hidden RDomainWidget to fix the GUI-restart bug.

        The first QRhiWidget in the main window makes Qt rebuild the top-level
        native window so its backing store can flush through QRhi. On macOS
        that tears down every open sub-window and dock, so opening a mesh looks
        like the GUI restarts and other viewers vanish. The manager parks a
        hidden primer viewer in the MDI area at setUp, so the rebuild happens
        once up front and later viewers reuse the same native window.

        This test checks there is one and only one primer."""
        mgr = pilot.RManager.instance.setUp()
        mdi = mgr.mdiArea
        primers = [
            w for w in mgr.mainWindow.findChildren(QWidget)
            if w.metaObject().className().endswith("RDomainWidget")
            and w.parent() is mdi and not w.isVisible()]
        self.assertEqual(len(primers), 1)

    # Showing the main window with the QRhi primer through the MS WARP (Windows
    # Advanced Rasterization Platform) software rasterizer may fault the
    # headless Windows debug runner with an access violation.
    @unittest.skipIf(os.getenv('GITHUB_ACTIONS', False) and
                     platform.system() == "Windows",
                     "MS WARP may fault headless Windows debug CI run")
    def test_open_3d_keeps_native_window(self):
        """A 3D viewer opened after setUp reuses the primed native window.

        The native handle (winId) changing is the proxy for "the top-level was
        rebuilt"; with the primer in place it must stay put across add3DWidget.
        See :meth:`test_setup_primes_rhi_composition` for the mechanism."""
        mgr = pilot.RManager.instance.setUp()
        mw = mgr.mainWindow
        mw.show()
        before = int(mw.winId())
        if not before:
            raise unittest.SkipTest("no native window handle is available")
        mgr.add3DWidget()
        self.assertEqual(int(mw.winId()), before)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class RDomainWidgetAxisTC(unittest.TestCase):
    """The orientation-guide overlay (step 6)."""

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def test_axis_guide_hidden_by_default(self):
        """Without showAxis there is no colored triad over the black mesh."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        image = _grab_or_skip(widget)
        self.assertLess(_count_axis_pixels(image, "red"), 5)
        self.assertLess(_count_axis_pixels(image, "green"), 5)

    def test_axis_guide_2d_shows_two_axes(self):
        """A 2D domain shows the X (red) and Y (green) axes, no Z."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_2d_mesh())
        widget.showAxis(True)
        image = _grab_or_skip(widget)
        self.assertGreater(_count_axis_pixels(image, "red"), 0)
        self.assertGreater(_count_axis_pixels(image, "green"), 0)
        self.assertLess(_count_axis_pixels(image, "blue"), 5)

    def test_axis_guide_3d_shows_three_axes(self):
        """A 3D domain shows all three colored axes."""
        widget = pilot.RDomainWidget()
        widget.resize(320, 240)
        widget.updateMesh(_make_3d_mesh())
        widget.showAxis(True)
        image = _grab_or_skip(widget)
        self.assertGreater(_count_axis_pixels(image, "red"), 0)
        self.assertGreater(_count_axis_pixels(image, "green"), 0)
        self.assertGreater(_count_axis_pixels(image, "blue"), 0)


if __name__ == '__main__':
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

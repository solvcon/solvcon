# Domain Visualizer

The pilot's domain viewer ({cpp:class}`~solvcon::RDomainWidget`) shows 2D and
3D unstructured-mesh domains and fields, rendered through Qt's
[QRhi](https://doc.qt.io/qt-6/qrhi.html). It is hosted as a sub-window in the
pilot and is fully controllable from Python: populate the scene, navigate the
camera, toggle layers, and capture frames from code the same way the mouse
drives them.

## Showing a domain

- **Mesh**: `updateMesh(mesh)` draws the unstructured mesh (2D cells or the 3D
  boundary shell); `showMesh(on)` toggles it. Choose which styles are drawn
  with the mesh styles (see below).
- **Colored field**: `updateColorField(vertices, colors, indices)` draws
  per-vertex-colored triangles over the domain. The field is swappable at
  runtime: call it again to replace the previous one.
- **Boundary highlight**: `showBoundary(ibc, on)` highlights boundary set
  `ibc`.
- **Orientation guide**: `showAxis(on)` shows a small axis triad in the corner,
  two axes for a 2D domain and three for a 3D one, oriented by the camera. It
  is hidden by default.

## Mesh styles

The mesh draws in three styles that toggle **independently**, so any
combination shows at once, a wireframe over a lit surface for instance. Toggle
each from the **View > Mesh styles** submenu or the mesh panel's check boxes,
or from Python with `showMeshStyle(name, on)`; `meshStyleShown(name)` reads a
style back.

| Menu item | `name` | What it draws |
| --- | --- | --- |
| Surface (lit shaded) | `"surface"` | cell or boundary faces as a lit, shaded surface |
| Wireframe | `"wireframe"` | mesh edges as hairlines (on by default) |
| Points | `"points"` | mesh nodes as points |

Only the wireframe is on by default, so a freshly loaded mesh looks unchanged.
The surface is shaded by a camera-following headlight, so faces turned toward
the eye read brightest and the facets stay legible as the camera moves. A 2D
domain fills its cells in the plane; a 3D domain shades its boundary shell.
`showMesh(on)` hides or shows the mesh as a whole, on top of the per-style
toggles.

```python
viewer.updateMesh(mesh)                   # wireframe only, the default
viewer.showMeshStyle("surface", True)     # add the lit shaded surface
viewer.showMeshStyle("wireframe", False)  # drop the wireframe over it
viewer.meshStyleShown("points")           # False
```

## Layering and opacity

The wireframe and the colored field are separate layers drawn in the same
scene, so showing both draws the wireframe over the shaded surface as one
representation. Each layer takes an adjustable opacity, a fraction from `0`
(fully transparent) to `1` (fully opaque, the default):

- `setFieldOpacity(alpha)` fades the shaded surface. Lower it to read the
  wireframe and any structure behind the surface.
- `setMeshOpacity(alpha)` fades the wireframe.

```python
viewer.updateMesh(mesh)                    # wireframe layer
viewer.updateColorField(verts, cols, tris)  # shaded surface layer
viewer.setFieldOpacity(0.5)                # see the wireframe through it
```

The opacity is applied on the next frame, so it takes effect immediately in the
live viewer and in a captured frame.

## Camera navigation

The viewer has one camera with three modes; each suits a different domain.
Choose the mode from the **View > Camera** menu, or set `cameraMode` from
Python.

| Menu item (View > Camera) | `cameraMode` | Domain | Drag | Wheel / pinch |
| --- | --- | --- | --- | --- |
| Orbit camera (3D) | `"orbit"` | 3D | swing the eye around the center | dolly toward or away from the center |
| First-person camera (3D) | `"fps"` | 3D | look around in place | dolly along the view direction |
| Pan / zoom camera (2D) | `"pan"` | 2D | pan in the plane | zoom the orthographic view |

**Orbit is the default.** It is a turntable orbit: the up axis stays fixed, so
the horizon never rolls and a hard pitch past vertical eases off instead of
flipping the view. Loading a 2D domain switches to pan/zoom automatically,
because the orthographic projection a 2D domain uses ignores the orbit dolly
and wants the in-plane wheel zoom instead.

### Mouse, wheel, and pinch

- **Left-drag** rotates: orbit the center in orbit mode, look around in
  first-person mode, or pan in 2D pan/zoom mode.
- **Middle- or right-drag** pans in any mode.
- **Wheel** zooms; what it does depends on the mode (see the table above).
- **Pinch** on a trackpad or touchscreen zooms in any mode.

### Keyboard

- **W / A / S / D** or the **arrow keys** move the camera: forward and back,
  and strafe left and right.
- **Esc** reframes the whole domain (fit to scene).

The **View > Camera move** submenu offers the same nudges as clickable actions,
plus fixed-step yaw and pitch rotation and a reset.

### From Python

The camera exposes the same primitives the mouse and wheel drive, so a domain
navigates identically from code:

```python
from solvcon import pilot

mgr = pilot.RManager.instance.setUp()
viewer = mgr.add3DWidget()

viewer.updateMesh(mesh)             # show the wireframe
viewer.cameraMode = "orbit"         # orbit a 3D domain (the default)
viewer.fitCameraToScene()           # frame the whole domain

viewer.rotateCamera(40.0, 15.0)     # orbit by a pixel delta
viewer.panCamera(20.0, 0.0)         # pan by a pixel delta
viewer.zoomCamera(3.0)              # zoom by wheel notches
viewer.pinchCamera(1.5)             # zoom by a pinch factor (>1 zooms in)
```

The pose is readable and settable directly as `(x, y, z)` tuples through
`cameraPosition`, `cameraTarget`, and `cameraUp`, so a view can be saved and
restored, or set to an exact framing.

## Capturing frames

- `saveImage(path)` renders the current frame offscreen and writes it to an
  image file.
- `clipImage()` copies the current frame to the clipboard.

Both grab the frame deterministically inside the Qt event loop, which is what
the pytest screenshot tests rely on.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->

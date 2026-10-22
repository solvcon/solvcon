# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The visual pieces of the pilot: what draws meshes, fields, and plots."""

from .. import _pilot_core as _pcore

if _pcore.enable:
    from . import _colorizer
    from . import _mesh
    from . import _movie
    from . import _plot

    ColorBar = _colorizer.ColorBar
    FieldColorizer = _colorizer.FieldColorizer
    colormap = _colorizer.colormap
    MeshDrawer = _mesh.MeshDrawer
    SampleMesh = _mesh.SampleMesh
    SampleMeshFeature = _mesh.SampleMeshFeature
    MeshStyleStatus = _mesh.MeshStyleStatus
    GmshFileDialog = _mesh.GmshFileDialog
    MovieRecorder = _movie.MovieRecorder
    LinePlotWidget = _plot.LinePlotWidget
else:
    # Bind only the public names: a None module attribute would shadow the
    # real submodule import in no-GUI builds.
    ColorBar = None
    FieldColorizer = None
    colormap = None
    MeshDrawer = None
    SampleMesh = None
    SampleMeshFeature = None
    MeshStyleStatus = None
    GmshFileDialog = None
    MovieRecorder = None
    LinePlotWidget = None

__all__ = [
    'ColorBar',
    'FieldColorizer',
    'GmshFileDialog',
    'LinePlotWidget',
    'MeshDrawer',
    'MeshStyleStatus',
    'MovieRecorder',
    'SampleMesh',
    'SampleMeshFeature',
    'colormap',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

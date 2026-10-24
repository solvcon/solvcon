# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The visual pieces of the pilot: what draws meshes, fields, and plots."""

from .. import _pilot_core as _pcore

if _pcore.enable:
    from . import _colorizer
    from . import _mesh
    from . import _movie
    from . import _plot
    from . import _viewer

    ColorBar = _colorizer.ColorBar
    FieldColorizer = _colorizer.FieldColorizer
    colormap = _colorizer.colormap
    MeshDrawer = _mesh.MeshDrawer
    SampleMesh = _mesh.SampleMesh
    SampleMeshDialog = _mesh.SampleMeshDialog
    SampleMeshFeature = _mesh.SampleMeshFeature
    MeshStyleStatus = _mesh.MeshStyleStatus
    GmshFileDialog = _mesh.GmshFileDialog
    MovieRecorder = _movie.MovieRecorder
    default_movie_suffix = _movie.default_movie_suffix
    LinePlotWidget = _plot.LinePlotWidget
    plot_array = _plot.plot_array
    DomainViewer = _viewer.DomainViewer
else:
    # Bind only the public names: a None module attribute would shadow the
    # real submodule import in no-GUI builds.
    ColorBar = None
    FieldColorizer = None
    colormap = None
    MeshDrawer = None
    SampleMesh = None
    SampleMeshDialog = None
    SampleMeshFeature = None
    MeshStyleStatus = None
    GmshFileDialog = None
    MovieRecorder = None
    default_movie_suffix = None
    LinePlotWidget = None
    plot_array = None
    DomainViewer = None

__all__ = [
    'ColorBar',
    'DomainViewer',
    'FieldColorizer',
    'GmshFileDialog',
    'LinePlotWidget',
    'MeshDrawer',
    'MeshStyleStatus',
    'MovieRecorder',
    'SampleMesh',
    'SampleMeshDialog',
    'SampleMeshFeature',
    'colormap',
    'default_movie_suffix',
    'plot_array',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

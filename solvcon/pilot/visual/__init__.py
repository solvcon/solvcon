# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The visual pieces of the pilot: sample meshes, the Gmsh file dialog, the
mesh style status helpers, the viewer movie recorder, and the line plot
widget.
"""

from .. import _pilot_core as _pcore

if _pcore.enable:
    from . import _mesh
    from . import _movie
    from . import _plot

    SampleMesh = _mesh.SampleMesh
    SampleMeshFeature = _mesh.SampleMeshFeature
    MeshStyleStatus = _mesh.MeshStyleStatus
    GmshFileDialog = _mesh.GmshFileDialog
    MovieRecorder = _movie.MovieRecorder
    LinePlotWidget = _plot.LinePlotWidget
else:
    # Bind only the public names: a None module attribute would shadow the
    # real submodule import in no-GUI builds.
    SampleMesh = None
    SampleMeshFeature = None
    MeshStyleStatus = None
    GmshFileDialog = None
    MovieRecorder = None
    LinePlotWidget = None

__all__ = [
    'GmshFileDialog',
    'LinePlotWidget',
    'MeshStyleStatus',
    'MovieRecorder',
    'SampleMesh',
    'SampleMeshFeature',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

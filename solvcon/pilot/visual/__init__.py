# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The visual pieces of the pilot: what draws meshes, fields, and plots."""

from .. import _pilot_core as _pcore

__all__, __getattr__, __dir__ = _pcore.lazy_exports(__name__, {
    '_colorizer': ['ColorBar', 'FieldColorizer', 'colormap'],
    '_mesh': ['GmshFileDialog', 'MeshDrawer', 'MeshStyleStatus',
              'SampleMesh', 'SampleMeshDialog', 'SampleMeshFeature'],
    '_movie': ['MovieRecorder', 'default_movie_suffix'],
    '_plot': ['LinePlotWidget', 'plot_array'],
    '_viewer': ['DomainViewer'],
})

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

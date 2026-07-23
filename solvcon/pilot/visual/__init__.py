# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The mesh view: sample meshes, the Gmsh file dialog, and the mesh style
status helpers.
"""

from .. import _pilot_core as _pcore

if _pcore.enable:
    from . import _mesh

    SampleMesh = _mesh.SampleMesh
    SampleMeshFeature = _mesh.SampleMeshFeature
    MeshStyleStatus = _mesh.MeshStyleStatus
    GmshFileDialog = _mesh.GmshFileDialog
else:
    # Bind only the public names: a None module attribute would shadow the
    # real submodule import in no-GUI builds.
    SampleMesh = None
    SampleMeshFeature = None
    MeshStyleStatus = None
    GmshFileDialog = None

__all__ = [
    'GmshFileDialog',
    'MeshStyleStatus',
    'SampleMesh',
    'SampleMeshFeature',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Sample-mesh entries for the oblique-shock reflection.

:class:`ObliqueShockMesh` draws the mesh in a 3D widget and reports the
boundary groups (left / top / bottom / right edges) to the console.
"""

from .... import mesh as meshmod
from ... import visual
from ...base import _gui_common

__all__ = [  # noqa: F822
    'ObliqueShockMesh',
]


class ObliqueShockMesh(_gui_common.PilotFeature):
    """
    Draw the oblique-shock reflection mesh and its boundary groups.
    """

    def mesh_sample_dialog_entries(self):
        cat = "Oblique-shock reflection"
        return [
            (cat, "Quad mesh (2D)",
             "Draw the quad mesh for the oblique-shock reflection",
             self.draw_quad_mesh),
            (cat, "Triangle mesh (2D)",
             "Draw the triangle mesh for the oblique-shock reflection",
             self.draw_triangle_mesh),
            (cat, "Unstructured mesh (2D)",
             "Draw the unstructured (Delaunay) triangle mesh for the "
             "oblique-shock reflection",
             self.draw_unstructured_mesh),
        ]

    def draw_quad_mesh(self):
        return self._draw_mesh('quad')

    def draw_triangle_mesh(self):
        return self._draw_mesh('triangle')

    def draw_unstructured_mesh(self):
        return self._draw_mesh('unstructured')

    def _draw_mesh(self, cell_type):
        mh = meshmod.RectDomainMesher().make_mesh(cell_type=cell_type)
        notes = {
            'left': "red left inlet",
            'top': "blue top inlet",
            'bottom': "green bottom slip wall",
            'right': "amber right outflow",
        }
        return visual.MeshDrawer(self._mgr).draw(
            mh, f"oblique-shock {cell_type} mesh", notes=notes)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

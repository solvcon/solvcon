# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Example pilot apps for the oblique-shock reflection.

The mesh construction, boundary tagging, and solver driver live in
:mod:`solvcon.multidim.euler.oblique`.  :class:`ObliqueShockMesh` draws the
mesh in a 3D widget and reports the boundary classification (inlet / slip wall
/ outflow) to the console.  Running the solver and animating its solution
fields is handled by the interactive panel in :mod:`._solution_info`.
"""

from ...multidim.euler import oblique
from ..base import _gui_common

__all__ = [  # noqa: F822
    'ObliqueShockMesh',
]


class ObliqueShockMesh(_gui_common.PilotFeature):
    """
    Draw the oblique-shock reflection mesh and tag its boundary.
    """

    def mesh_sample_dialog_entries(self):
        cat = "Oblique-shock reflection"
        return [
            (cat, "Quad mesh (2D)",
             "Draw the quad wedge mesh for the oblique-shock reflection",
             self.draw_quad_mesh),
            (cat, "Triangle mesh (2D)",
             "Draw the triangle wedge mesh for the oblique-shock reflection",
             self.draw_triangle_mesh),
            (cat, "Unstructured mesh (2D)",
             "Draw the unstructured (Delaunay) triangle wedge mesh for the "
             "oblique-shock reflection",
             self.draw_unstructured_mesh),
        ]

    def draw_quad_mesh(self):
        self._draw_mesh('quad')

    def draw_triangle_mesh(self):
        self._draw_mesh('triangle')

    def draw_unstructured_mesh(self):
        self._draw_mesh('unstructured')

    def _draw_mesh(self, cell_type):
        mesher = oblique.ObliqueShockMesher()
        mh = mesher.make_mesh(cell_type=cell_type)
        inlet, walls, outflow = mesher.classify_boundary(mh)
        w = self._mgr.add3DWidget()
        w.updateMesh(mh)
        w.showAxis(True)
        self._pycon.writeToHistory(
            f"oblique-shock {cell_type} mesh: {mh.ncell} cells, "
            f"{mh.nedge} edges\n"
            f"boundary faces: {len(inlet)} inlet, {len(walls)} slip wall, "
            f"{len(outflow)} outflow\n")


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

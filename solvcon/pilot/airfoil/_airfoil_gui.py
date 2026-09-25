# Copyright (c) 2021, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
GUI for NACA airfoil shape
"""

from ... import core

from ..base import _gui_common
from . import _naca


class Naca4Airfoil(_gui_common.PilotFeature):
    """
    Provide pilot GUI control for the NACA 4-digit airfoil shape.
    """

    def mesh_sample_dialog_entries(self):
        return [
            ("Airfoil", "NACA 4-digit", "Draw a NACA 4-digit airfoil",
             self.sample_window),
        ]

    def sample_window(self):
        """
        A simple example for drawing a couple of cubic Bezier curves based on
        an airfoil.
        """
        w = core.WorldFp64()
        naca4 = _naca.Naca4(number='0012', open_trailing_edge=False,
                            cosine_spacing=False)
        sampler = _naca.Naca4Sampler(w, naca4)
        sampler.populate_points(npoint=101, fac=5.0, off_x=0.0, off_y=2.0)
        if False:
            sampler.draw_line()
        else:
            sampler.draw_cbc()
        # The airfoil curve is world geometry, so it renders in the 2D canvas.
        wid = self._mgr.add2DWidget()
        wid.updateWorld(w)
        wid.resetView()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

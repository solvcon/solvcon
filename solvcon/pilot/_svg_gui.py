# Copyright (c) 2025, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Show a SVG (scalleable vector graphic)
"""

import os
from PySide6 import QtCore, QtWidgets

from .. import core
from .. import apputil
from ..plot import svg
from . import _gui_common

__all__ = [  # noqa: F822
    'SVGFileDialog',
]


class SVGFileDialog(_gui_common.PilotFeature):
    """
    Download an example svg from: https://www.svgrepo.com/svg/530293/tree-2
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._diag = QtWidgets.QFileDialog()
        self._diag.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self._diag.setDirectory(self._get_initial_path())
        self._diag.setWindowTitle('Open SVG file')

    def run(self):
        self._diag.open(self, QtCore.SLOT('on_finished()'))

    def populate_menu(self):
        self.add_action(
            "File",
            text="Open SVG file",
            tip="Open SVG file",
            func=self.run,
            id="file.svg",
            weight=20,
        )

    @QtCore.Slot()
    def on_finished(self):
        filenames = []
        for path in self._diag.selectedFiles():
            filenames.append(path)
        self._load_svg_file(filename=filenames[0])

    @staticmethod
    def _get_initial_path():
        found = ''
        for dp in ('.', core.__file__):
            dp = os.path.dirname(os.path.abspath(dp))
            dp2 = os.path.dirname(dp)

            while dp != dp2:
                tp = os.path.join(dp, "tests", "data")
                fp = os.path.join(tp, "tree.svg")
                if os.path.exists(fp):
                    found = fp
                    break
                dp = dp2
                dp2 = os.path.dirname(dp)
            if found:
                break
        return found

    def _load_svg_file(self, filename):
        parser = svg.SvgParser(file_path=filename)
        parser.parse()
        spads, cpads = parser.get_pads()

        world = core.WorldFp64()

        for spad in spads:
            # Flip against the X axis for GUI coordinate system
            spad.mirror(axis='x')
            world.add_segments(pad=spad)

        for cpad in cpads:
            # Flip against the X axis for GUI coordinate system
            cpad.mirror(axis='x')
            world.add_beziers(pad=cpad)

        # World geometry (segments and curves) renders in the 2D canvas; the
        # 3D domain viewer is for meshes and fields.
        wid = self._mgr.add2DWidget()
        wid.updateWorld(world)
        wid.resetView()

        # Add the data objects to the appenv for command-line access.
        cae = apputil.get_current_appenv()
        cae.locals['parser'] = parser
        cae.locals['world'] = world
        cae.locals['widget'] = wid

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

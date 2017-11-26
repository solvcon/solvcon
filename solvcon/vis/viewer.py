# -*- coding: UTF-8 -*-
#
# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
#
# All rights reserved.


from traitlets import (Unicode, Int, Instance, Enum, List, Dict, Float,
                       Any, CFloat, Bool, This, CInt, TraitType)

import pythreejs as ptjs

from .surface import Surface


class Viewer(ptjs.Renderer):
    """
    Represent a window (canvas) for viewing SOLVCON results.
    """

    _view_module = Unicode('nbextensions/solvcon/solvcon', sync=True)
    _view_name = Unicode('ViewerView', sync=True)

    def __init__(self, *args, **kw):

        camera = kw.pop('camera', None)
        if not camera:
            position = kw.pop('position', [0,0,0])
            up = kw.pop('up', [0,1,0])
            camera = ptjs.PerspectiveCamera(position=position, up=up)
        kw['camera'] = camera

        controls = kw.pop('controls', None)
        if not controls:
            controls = ptjs.TrackballControls(controlling=kw['camera'])
        kw['controls'] = [controls]

        scene = kw.pop('scene', None)
        if not scene:
            scene = ptjs.Scene()
        kw['scene'] = scene

        super(Viewer, self).__init__(*args, **kw)

    def append(self, item):
        self.scene.children = self.scene.children + [item]

    def extend(self, items):
        self.scene.children = self.scene.children + items

    def __len__(self):
        return len(self.scene.children)

    def __getitem__(self, idx):
        return self.scene.children[idx]

    def __setitem__(self, idx, item):
        self.scene.children[idx] = item
        self.scene.children = list(self.scene.children)

    def __delitem__(self, idx):
        raise NotImplementedError("Need to study how to remove from scene")

    def insert(self, idx, item):
        self.scene.children = (
            self.scene.children[:idx] + [item] + self.scene.children[idx:])

# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The dock panels: the entity tree, the window manager, and the profiling
view.
"""

from .. import _pilot_core as _pcore

__all__, __getattr__, __dir__ = _pcore.lazy_exports(__name__, {
    '_profiling': ['Profiling', 'RunProfiling'],
    '_tree_panel': ['EntityTreeWidget', 'MeshInfoTree', 'TreePanel',
                    'TreePanelBase'],
    '_window_manager': ['WindowManager'],
})

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

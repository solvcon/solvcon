# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The dock panels: the entity tree, the window manager, and the profiling
view.
"""

from .. import _pilot_core as _pcore

if _pcore.enable:
    from . import _profiling
    from . import _tree_panel
    from . import _window_manager

    EntityTreeWidget = _tree_panel.EntityTreeWidget
    MeshInfoTree = _tree_panel.MeshInfoTree
    TreePanel = _tree_panel.TreePanel
    TreePanelBase = _tree_panel.TreePanelBase
    WindowManager = _window_manager.WindowManager
    Profiling = _profiling.Profiling
else:
    # Bind only the public names: a None module attribute would shadow the
    # real submodule import in no-GUI builds.
    EntityTreeWidget = None
    MeshInfoTree = None
    TreePanel = None
    TreePanelBase = None
    WindowManager = None
    Profiling = None

__all__ = [
    'EntityTreeWidget',
    'MeshInfoTree',
    'Profiling',
    'TreePanel',
    'TreePanelBase',
    'WindowManager',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The dock panels: the entity tree, the window manager, the profiling
view, screen capture, the Console panel, and the app config, with the
fold group of a config column, and the run strip, the window, and the small
widgets an app window and its control column are built of.
"""

from .. import _pilot_core as _pcore

__all__, __getattr__, __dir__ = _pcore.lazy_exports(__name__, {
    '_app_config': ['AppConfig'],
    '_app_widgets': ['ColumnGroup', 'ElidedLabel', 'Fitted', 'GLYPHS',
                     'Styled', 'form_label', 'guard_wheel', 'head_font',
                     'icon', 'message', 'reserve_width', 'say'],
    '_capture_panel': ['CaptureWidget'],
    '_console': ['ConsolePanel'],
    '_fold': ['FoldGroup'],
    '_profiling': ['Profiling', 'RunProfiling'],
    '_run_state': ['RUN_STATES', 'StripState', 'clock'],
    '_run_strip': ['RunStrip', 'RunWindow'],
    '_tree_panel': ['EntityTreeWidget', 'MeshInfoTree', 'TreePanel',
                    'TreePanelBase'],
    '_window_manager': ['WindowManager'],
}, ungated=('_run_state',))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

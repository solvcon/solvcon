# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The dock panels: the entity tree, the window manager, the profiling
view, screen capture, the Console panel, and the app config, with the
fold group of a config column, and what an app window and its control
column are built of: the run strip, the window, the run controller, the
results card, the setup groups, and the small widgets.
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
    '_readout': ['Cell', 'RUN_STATES', 'ResultRow', 'StripState', 'clock',
                 'format_error', 'format_number', 'format_sci',
                 'format_signed', 'tone_of'],
    '_results_card': ['ResultsCard'],
    '_run_control': ['RunController'],
    '_run_strip': ['RunStrip', 'RunWindow'],
    '_setup_group': ['NumberField', 'SetupGroup'],
    '_tree_panel': ['EntityTreeWidget', 'MeshInfoTree', 'TreePanel',
                    'TreePanelBase'],
    '_window_manager': ['WindowManager'],
}, ungated=('_readout',))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

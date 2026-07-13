# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


from . import _gui_common


class ThemeMenu(_gui_common.PilotFeature):
    """Build the View > Theme menu from Python over the C++ theme manager.

    The manager (implemented in C++) owns the palette, the platform backend,
    and the capability record; this feature only presents the choices. Each
    entry drives the manager through its scripting seam, and the manager keeps
    the radios in step when the theme changes from outside the menu.
    """

    def populate_menu(self):
        mgr = self._mgr

        self._build_group(
            "theme.mode",
            [
                ("system", "Follow system",
                 "Follow the operating system light or dark setting",
                 mgr.theme_can_follow_system),
                ("light", "Light", "Use the light palette",
                 mgr.theme_can_force_variant),
                ("dark", "Dark", "Use the dark palette",
                 mgr.theme_can_force_variant),
            ],
            base_weight=10, setter=mgr.set_theme, current=mgr.theme_mode)

        self._build_group(
            "theme.look",
            [
                ("system", "System colors",
                 "Show the platform's own colors through the native style",
                 mgr.theme_has_native_style),
                ("curated", "Curated colors",
                 "Apply the curated palette for a look that travels between"
                 " machines", True),
            ],
            base_weight=110, setter=mgr.set_look, current=mgr.theme_look)

    def _build_group(self, group_id, items, *, base_weight, setter, current):
        # The exclusive group lives in the C++ menu model under its id, so a
        # test and the manager's sync both reach the actions by that handle.
        group = self._mgr.menu_model.group(group_id)
        weight = base_weight
        for key, label, tip, enabled in items:
            # A choice the platform cannot honor is shown greyed with a note
            # rather than hidden, so the menu never offers a switch that does
            # nothing.
            full_tip = tip if enabled else \
                tip + " (not available on this platform)"
            action = _gui_common.build_action(
                self._mainWindow, label, full_tip,
                (lambda k=key: setter(k)),
                id="%s_%s" % (group_id, key),
                checkable=True, checked=(key == current))
            action.setEnabled(enabled)
            group.addAction(action)
            self._mgr.menu_model.place("View/Theme", action, weight)
            weight += 10

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

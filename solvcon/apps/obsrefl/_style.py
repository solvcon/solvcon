# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Style rules for the oblique-shock reflection controls."""

from ... import pilot

from . import _ink

__all__ = [
    'Rules',
]

#: The step from the panel toward the text of a rule between sections.
_RULE_MIX = 0.13


class Rules(pilot.AppRules):
    """Style roles used by the reflection controls, one method per role.

    The run buttons and the state pill are the roles of the pilot's
    :class:`~solvcon.pilot.AppRules`.  Every color is mixed from the palette
    of the widget the sheet is built for, through the pilot and
    :mod:`._ink`, so no color outlives a theme switch.
    """

    #: The buttons carry an icon on the left of the label.
    PADDING = ("4px 14px 4px 10px", "3px 10px 3px 8px")
    #: The tones the state pill takes, with 'bad' for a failed run.
    TONES = ('muted', 'accent', 'good', 'bad')
    #: The roles drawn on the base ground, which read :class:`_ink.Ink`
    #: where the others read :class:`pilot.Shades`.
    ON_BASE = ('card',)

    @classmethod
    def sheet(cls, widget, *roles):
        """Return the sheet a widget carries for the roles it draws.

        :param widget: The widget whose palette the colors are mixed from.
        :param roles: The names of the role methods the sheet is built of.
        """
        shades = pilot.Shades(widget)
        ink = _ink.Ink(widget) if set(roles) & set(cls.ON_BASE) else None
        return "".join(getattr(cls, role)(ink if role in cls.ON_BASE
                                          else shades) for role in roles)

    @staticmethod
    def strip(shades):
        """Style the run strip across the top of the window.

        :param shades: The :class:`~solvcon.pilot.Shades` of the widget.
        """
        failure = pilot.legible(shades, shades.bright_text, shades.panel)
        return f"""
            QFrame#runStrip {{
                background: {shades.panel.name()};
                border: none;
                border-bottom: 1px solid {shades.raised(_RULE_MIX).name()};
            }}
            QLabel#stripProblem {{
                color: {shades.text.name()};
                font-weight: 600;
            }}
            QLabel#stripCaption, QLabel#failure {{
                color: {shades.muted.name()};
                font-size: {pilot.SMALL_PX}px;
            }}
            QLabel#failure {{ color: {failure.name()}; }}
            """

    @staticmethod
    def progress(shades):
        """Style the thin determinate bar beside the time readout.

        :param shades: The :class:`~solvcon.pilot.Shades` of the widget.
        """
        track = shades.raised(pilot.FILL_MIX)
        chunk = pilot.legible(shades, shades.accent, track, shades.panel,
                              floor=pilot.GRAPHIC_CONTRAST)
        return f"""
            QProgressBar#progress {{
                border: none;
                border-radius: 2px;
                background: {track.name()};
                max-height: 4px; min-height: 4px;
            }}
            QProgressBar#progress::chunk {{
                border-radius: 2px;
                background: {chunk.name()};
            }}
            """

    @staticmethod
    def card(ink):
        """Style the results card at the head of the control column.

        :param ink: The :class:`~._ink.Ink` of the card.
        """
        rules = [f"""
            QFrame#card {{
                background: {ink.base.name()};
                border: 1px solid {ink.border.name()};
                border-radius: 8px;
            }}
            QFrame#card QLabel {{
                background: transparent;
                color: {ink.ink.name()};
            }}
            QFrame#card QLabel#metric, QFrame#card QLabel#caption,
            QFrame#card QLabel#cardHead {{
                color: {ink.muted.name()};
            }}
            QFrame#card QFrame#cardRule {{
                background: {ink.raised(_RULE_MIX).name()};
                border: none;
                max-height: 1px; min-height: 1px;
            }}
            """]
        for tone in ('good', 'bad', 'muted'):
            rules.append(f"""
            QFrame#card QLabel[tone="{tone}"] {{
                color: {ink.tone(tone).name()};
            }}
            """)
        return "".join(rules)

    @staticmethod
    def configuration_error(shades):
        return f"""
            QFrame#configurationError {{
                border: 1px solid {shades.bright_text.name()};
                border-radius: 6px;
            }}
            QLabel#configurationErrorIcon {{
                background-color: {shades.bright_text.name()};
                border-radius: 9px;
                color: {shades.on_base.name()};
                font-weight: bold;
            }}
            QLabel#configurationErrorTitle {{
                color: {shades.bright_text.name()};
                font-weight: bold;
            }}
            QLabel#configurationErrorDetail {{
                color: {shades.text.name()};
            }}
        """

    @staticmethod
    def error_outline(shades):
        return f"""
            QFrame#errorOutline {{
                border: 1px solid {shades.bright_text.name()};
                border-radius: 4px;
            }}
        """

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

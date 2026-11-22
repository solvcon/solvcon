# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Style rules for the oblique-shock reflection controls."""

from ... import pilot

from . import _ink

__all__ = [
    'Rules',
]

#: The size of a form label, in device-independent pixels.
_LABEL_PX = 12
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
    def fold(shades):
        """Style the head of a folded group of the control column.

        :param shades: The :class:`~solvcon.pilot.Shades` of the widget.
        """
        dirty = pilot.legible(shades, shades.accent, shades.panel)
        return f"""
            QPushButton#foldHead {{
                border: none;
                border-top: 1px solid {shades.raised(_RULE_MIX).name()};
                border-radius: 0px;
                padding: 5px 4px 5px 4px;
                text-align: left;
                color: {shades.muted.name()};
                background: transparent;
            }}
            QPushButton#foldHead:hover {{
                background: {shades.raised(pilot.HOVER_MIX).name()};
                color: {shades.text.name()};
            }}
            QLabel#foldSummary {{
                color: {shades.muted.name()};
                font-size: {pilot.SMALL_PX}px;
            }}
            QLabel#foldSummary[dirty="true"] {{ color: {dirty.name()}; }}
            """

    @staticmethod
    def form(shades):
        """Style the form grid of the groups: muted labels, uniform fields,
        and the outline of a field whose value is wrong.

        :param shades: The :class:`~solvcon.pilot.Shades` of the widget.
        """
        edge = pilot.legible(shades, shades.border, shades.base,
                             shades.panel, floor=pilot.GRAPHIC_CONTRAST)
        focus = pilot.legible(shades, shades.accent, shades.base,
                              shades.panel, floor=pilot.GRAPHIC_CONTRAST)
        hint = pilot.legible(shades, shades.accent, shades.panel)
        wrong = pilot.legible(shades, shades.bright_text, shades.base,
                              shades.panel)
        return f"""
            QLabel#formLabel {{
                color: {shades.muted.name()};
                font-size: {_LABEL_PX}px;
            }}
            QLabel#formLabel:disabled {{ color: {shades.greyed.name()}; }}
            QLabel#note, QLabel#hint, QLabel#fieldError {{
                color: {shades.muted.name()};
                font-size: {pilot.SMALL_PX}px;
            }}
            QLabel#hint {{ color: {hint.name()}; }}
            QLabel#fieldError {{ color: {wrong.name()}; }}
            QLineEdit#number {{
                border: 1px solid {edge.name()};
                border-radius: 5px;
                padding: 0px 5px;
                min-height: 20px; max-height: 20px;
                background: {shades.base.name()};
                color: {shades.on_base.name()};
                font-size: {_LABEL_PX}px;
            }}
            QLineEdit#number:focus {{
                border-color: {focus.name()};
            }}
            QLineEdit#number:disabled {{ color: {shades.greyed.name()}; }}
            QCheckBox {{ font-size: {_LABEL_PX}px; }}
            QLineEdit#number[invalid="true"] {{
                border-color: {wrong.name()};
            }}
            """

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

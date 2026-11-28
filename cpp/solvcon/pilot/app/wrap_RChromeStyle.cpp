/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/wrap_pilot.hpp> // Must be the first include.

#include <solvcon/pilot/app/RChromeStyle.hpp>

namespace solvcon::python
{

void wrap_chrome(pybind11::module & mod)
{
    namespace py = pybind11;

    mod.def(
        "chrome_enabled",
        &chrome_enabled,
        "Whether the pilot draws its own chrome: false when the environment "
        "variable SOLVCON_PILOT_CHROME is \"native\".");
    mod.def(
        "chrome_metrics",
        []()
        {
            py::dict out;
            out["bar_height"] = ChromeMetrics::BAR_HEIGHT;
            out["title_px"] = ChromeMetrics::TITLE_PX;
            out["button_side"] = ChromeMetrics::BUTTON_SIDE;
            out["button_gap"] = ChromeMetrics::BUTTON_GAP;
            out["bar_margin"] = ChromeMetrics::BAR_MARGIN;
            out["accent_width"] = ChromeMetrics::ACCENT_WIDTH;
            out["ground_active_mix"] = ChromeMetrics::GROUND_ACTIVE_MIX;
            out["ground_inactive_mix"] = ChromeMetrics::GROUND_INACTIVE_MIX;
            out["rule_mix"] = ChromeMetrics::RULE_MIX;
            out["hover_mix"] = ChromeMetrics::HOVER_MIX;
            out["press_mix"] = ChromeMetrics::PRESS_MIX;
            out["text_contrast"] = ChromeMetrics::TEXT_CONTRAST;
            out["graphic_contrast"] = ChromeMetrics::GRAPHIC_CONTRAST;
            return out;
        },
        "The geometry and the color mixes of the chrome, the one source the "
        "bar of a sub-window and the bar of a dock panel are drawn from.");
}

} /* end namespace solvcon::python */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

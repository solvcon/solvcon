#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <QObject>
#include <QProxyStyle>
#include <QRect>

class QMdiArea;
class QMdiSubWindow;

namespace solvcon
{

/**
 * The geometry and the color mixes of the chrome, in one place for both
 * languages: RChromeStyle paints the bar of a sub-window from them, and
 * the binding chrome_metrics() exposes them to Python.
 *
 * Lengths are in device-independent pixels. A ground or rule mix is the step
 * from the palette's Window color toward its WindowText; a hover or press mix
 * is the step from the ground toward WindowText.
 */
struct ChromeMetrics
{
    static constexpr int BAR_HEIGHT = 24;
    static constexpr int TITLE_PX = 12;
    static constexpr int BUTTON_SIDE = 18;
    static constexpr int BUTTON_GAP = 2;
    /// The room on either end of the bar.
    static constexpr int BAR_MARGIN = 6;
    /// The accent line under an active bar.
    static constexpr int ACCENT_WIDTH = 2;
    static constexpr double GROUND_ACTIVE_MIX = 0.07;
    static constexpr double GROUND_INACTIVE_MIX = 0.04;
    static constexpr double RULE_MIX = 0.16;
    static constexpr double HOVER_MIX = 0.12;
    static constexpr double PRESS_MIX = 0.22;
    /// The contrast floors of WCAG 2.1 for text (1.4.3) and for graphics (1.4.11).
    static constexpr double TEXT_CONTRAST = 4.5;
    static constexpr double GRAPHIC_CONTRAST = 3.0;
}; /* end struct ChromeMetrics */

/**
 * Whether the pilot draws its own chrome: true unless the environment
 * variable SOLVCON_PILOT_CHROME is "native".
 */
bool chrome_enabled();

/**
 * The title bar and frame of an MDI sub-window, drawn from the palette.
 *
 * The platform style draws the bar of a sub-window as a window of its own,
 * in the colors of the system appearance rather than of the palette, so it
 * stays light in the dark theme. This style paints only the bar and the
 * frame and leaves every behavior to QMdiSubWindow: move, resize, the
 * system menu, maximize and restore. The bar keeps its height when the
 * sub-window is maximized, on every platform.
 *
 * The bar holds the title on the left and the buttons on the right:
 * detach (when the sub-window has a child named "subWindowDetacher"),
 * maximize or restore, and close. The detach button is drawn here and its
 * click is delivered by RChromeFilter.
 */
class RChromeStyle
    : public QProxyStyle
{

public:

    explicit RChromeStyle(QStyle * base);

    int pixelMetric(PixelMetric metric, QStyleOption const * option, QWidget const * widget) const override;
    int styleHint(StyleHint hint, QStyleOption const * option, QWidget const * widget, QStyleHintReturn * ret) const override;
    QRect subControlRect(ComplexControl control, QStyleOptionComplex const * option, SubControl sub, QWidget const * widget) const override;
    SubControl hitTestComplexControl(ComplexControl control, QStyleOptionComplex const * option, QPoint const & pos, QWidget const * widget) const override;
    void drawComplexControl(ComplexControl control, QStyleOptionComplex const * option, QPainter * painter, QWidget const * widget) const override;
    void drawPrimitive(PrimitiveElement element, QStyleOption const * option, QPainter * painter, QWidget const * widget) const override;

    /// The rectangle of the detach button in the bar of @p subwin, or an empty one when it has no detacher.
    static QRect detachRect(QMdiSubWindow const * subwin);
    /// The rectangle of the detach button in the bar of a sub-window @p width pixels wide.
    static QRect detachRect(int width);

    /// Whether @p subwin has a detacher to show a button for.
    static bool hasDetacher(QWidget const * subwin);

}; /* end class RChromeStyle */

/**
 * Dress every sub-window of an area in RChromeStyle as it is polished, and
 * deliver a click on the detach button of a sub-window to its detacher, by
 * invoking the slot "toggle" of the child named "subWindowDetacher".
 */
class RChromeFilter
    : public QObject
{

public:

    RChromeFilter(QMdiArea * area, RChromeStyle * style);

protected:

    bool eventFilter(QObject * watched, QEvent * event) override;

private:

    void dress(QMdiSubWindow * subwin);
    bool onSubWindow(QMdiSubWindow * subwin, QEvent * event);

    RChromeStyle * m_style;

}; /* end class RChromeFilter */

/**
 * Draw the sub-windows of @p area in RChromeStyle when chrome_enabled(). The
 * style and the filter are children of the area.
 */
void install_chrome(QMdiArea * area);

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/app/RChromeStyle.hpp> // Must be the first include.

#include <algorithm>
#include <cmath>
#include <utility>

#include <QApplication>
#include <QChildEvent>
#include <QColor>
#include <QEvent>
#include <QFont>
#include <QFontMetrics>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QPainter>
#include <QPainterPath>
#include <QPalette>
#include <QPen>
#include <QStyleFactory>
#include <QStyleOptionTitleBar>
#include <QWidget>
#include <Qt>

namespace solvcon
{

namespace
{

char const * const DRESSED_PROPERTY = "solvconChrome";

/// The slots of the buttons, counted from the right end of the bar.
enum ButtonSlot : int
{
    SLOT_CLOSE = 0,
    SLOT_MAXIMIZE = 1,
    SLOT_COUNT = 2
}; /* end enum ButtonSlot */

QRect slot_rect(QRect const & bar, int slot)
{
    int const right = bar.right() - ChromeMetrics::BAR_MARGIN - slot * (ChromeMetrics::BUTTON_SIDE + ChromeMetrics::BUTTON_GAP);
    int const top = bar.top() + (bar.height() - ChromeMetrics::BUTTON_SIDE) / 2;
    return QRect(right - ChromeMetrics::BUTTON_SIDE + 1, top, ChromeMetrics::BUTTON_SIDE, ChromeMetrics::BUTTON_SIDE);
}

QColor blend(QColor const & one, QColor const & other, double ratio)
{
    return QColor::fromRgbF(
        static_cast<float>(one.redF() + (other.redF() - one.redF()) * ratio),
        static_cast<float>(one.greenF() + (other.greenF() - one.greenF()) * ratio),
        static_cast<float>(one.blueF() + (other.blueF() - one.blueF()) * ratio));
}

double luminance(QColor const & color)
{
    auto channel = [](double value)
    {
        return value <= 0.03928 ? value / 12.92 : std::pow((value + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * channel(color.redF()) + 0.7152 * channel(color.greenF()) + 0.0722 * channel(color.blueF());
}

double contrast(QColor const & one, QColor const & other)
{
    double const lone = luminance(one);
    double const lother = luminance(other);
    return (std::max(lone, lother) + 0.05) / (std::min(lone, lother) + 0.05);
}

/// Step @p color toward the end farther from @p ground until it reads on it at @p floor.
QColor legible(QColor const & color, QColor const & ground, double floor)
{
    QColor const end = luminance(ground) > 0.5 ? QColor(Qt::black) : QColor(Qt::white);
    QColor stepped = color;
    for (int step = 1; step <= 50 && contrast(stepped, ground) < floor; ++step)
    {
        stepped = blend(color, end, step / 50.0);
    }
    return stepped;
}

/// The colors of the bar, mixed from the palette of the sub-window.
struct BarInk
{
    QColor ground;
    QColor rule;
    QColor text;
    QColor glyph;
    QColor hover;
    QColor press;
    QColor accent;

    BarInk(QPalette const & palette, bool active)
    {
        QColor const window = palette.color(QPalette::Window);
        QColor const ink = palette.color(QPalette::WindowText);
        ground = blend(window, ink, active ? ChromeMetrics::GROUND_ACTIVE_MIX : ChromeMetrics::GROUND_INACTIVE_MIX);
        rule = blend(window, ink, ChromeMetrics::RULE_MIX);
        text = active ? ink : legible(blend(ink, ground, 0.45), ground, ChromeMetrics::TEXT_CONTRAST);
        glyph = legible(blend(ink, ground, active ? 0.15 : 0.45), ground, ChromeMetrics::GRAPHIC_CONTRAST);
        hover = blend(ground, ink, ChromeMetrics::HOVER_MIX);
        press = blend(ground, ink, ChromeMetrics::PRESS_MIX);
        accent = legible(palette.color(QPalette::Highlight), ground, ChromeMetrics::GRAPHIC_CONTRAST);
    }
}; /* end struct BarInk */

void draw_glyph(QPainter * painter, QRect const & box, int slot, bool restore, QColor const & color)
{
    QPointF const c = QRectF(box).center();
    double const half = 4.0;
    QPen pen(color, 1.3);
    pen.setCapStyle(Qt::RoundCap);
    pen.setJoinStyle(Qt::RoundJoin);
    painter->setPen(pen);
    painter->setBrush(Qt::NoBrush);
    if (SLOT_CLOSE == slot)
    {
        painter->drawLine(QPointF(c.x() - half, c.y() - half), QPointF(c.x() + half, c.y() + half));
        painter->drawLine(QPointF(c.x() - half, c.y() + half), QPointF(c.x() + half, c.y() - half));
    }
    else if (SLOT_MAXIMIZE == slot && !restore)
    {
        painter->drawRoundedRect(QRectF(c.x() - half, c.y() - half, 2 * half, 2 * half), 1.0, 1.0);
    }
    else
    {
        painter->drawRoundedRect(QRectF(c.x() - half, c.y() - half + 2, 2 * half - 2, 2 * half - 2), 1.0, 1.0);
        QPainterPath back;
        back.moveTo(c.x() - half + 2, c.y() - half + 2);
        back.lineTo(c.x() - half + 2, c.y() - half);
        back.lineTo(c.x() + half, c.y() - half);
        back.lineTo(c.x() + half, c.y() + half - 2);
        back.lineTo(c.x() + half - 2, c.y() + half - 2);
        painter->drawPath(back);
    }
}

} /* end namespace */

bool chrome_enabled()
{
    return qgetenv("SOLVCON_PILOT_CHROME") != "native";
}

RChromeStyle::RChromeStyle(QStyle * base)
    : QProxyStyle(base)
{
}

int RChromeStyle::pixelMetric(PixelMetric metric, QStyleOption const * option, QWidget const * widget) const
{
    if (qobject_cast<QMdiSubWindow const *>(widget) != nullptr)
    {
        if (PM_TitleBarHeight == metric)
        {
            return ChromeMetrics::BAR_HEIGHT;
        }
        if (PM_MdiSubWindowFrameWidth == metric)
        {
            return 1;
        }
    }
    return QProxyStyle::pixelMetric(metric, option, widget);
}

int RChromeStyle::styleHint(StyleHint hint, QStyleOption const * option, QWidget const * widget, QStyleHintReturn * ret) const
{
    if (qobject_cast<QMdiSubWindow const *>(widget) != nullptr)
    {
        // Keep the bar when maximized, on the platforms that would merge it into the menu bar.
        if (SH_Workspace_FillSpaceOnMaximize == hint)
        {
            return 1;
        }
        // The bar spans the sub-window rather than sitting inside a border of 4 pixels.
        if (SH_TitleBar_NoBorder == hint)
        {
            return 1;
        }
        if (SH_TitleBar_AutoRaise == hint)
        {
            return 1;
        }
        // The bar is square. A frame mask from the base (Fusion rounds the top corners) would clip it, and
        // QMdiSubWindow does not refresh the mask while hidden, so a stale one would also cut a resized window.
        if (SH_WindowFrame_Mask == hint)
        {
            return 0;
        }
    }
    return QProxyStyle::styleHint(hint, option, widget, ret);
}

QRect RChromeStyle::subControlRect(ComplexControl control, QStyleOptionComplex const * option, SubControl sub, QWidget const * widget) const
{
    auto const * bar = qstyleoption_cast<QStyleOptionTitleBar const *>(option);
    if (CC_TitleBar != control || bar == nullptr || qobject_cast<QMdiSubWindow const *>(widget) == nullptr)
    {
        return QProxyStyle::subControlRect(control, option, sub, widget);
    }
    QRect const rect = bar->rect;
    switch (sub)
    {
    case SC_TitleBarCloseButton:
        return (bar->titleBarFlags & Qt::WindowSystemMenuHint) ? slot_rect(rect, SLOT_CLOSE) : QRect();
    case SC_TitleBarMaxButton:
    case SC_TitleBarNormalButton:
        return (bar->titleBarFlags & Qt::WindowMaximizeButtonHint) ? slot_rect(rect, SLOT_MAXIMIZE) : QRect();
    case SC_TitleBarLabel:
    {
        QRect const leftmost = slot_rect(rect, SLOT_COUNT - 1);
        return QRect(rect.left() + ChromeMetrics::BAR_MARGIN + 2, rect.top(), std::max(0, leftmost.left() - rect.left() - 2 * ChromeMetrics::BAR_MARGIN), rect.height());
    }
    default:
        // No icon, no minimize, no shade, no help: the system menu opens on a right click of the bar.
        return QRect();
    }
}

QStyle::SubControl RChromeStyle::hitTestComplexControl(ComplexControl control, QStyleOptionComplex const * option, QPoint const & pos, QWidget const * widget) const
{
    auto const * bar = qstyleoption_cast<QStyleOptionTitleBar const *>(option);
    if (CC_TitleBar != control || bar == nullptr || qobject_cast<QMdiSubWindow const *>(widget) == nullptr)
    {
        return QProxyStyle::hitTestComplexControl(control, option, pos, widget);
    }
    if (!bar->rect.contains(pos))
    {
        return SC_None;
    }
    SubControl const maxsub = (bar->titleBarState & Qt::WindowMaximized) ? SC_TitleBarNormalButton : SC_TitleBarMaxButton;
    for (SubControl sub : {SC_TitleBarCloseButton, maxsub})
    {
        if (subControlRect(control, option, sub, widget).contains(pos))
        {
            return sub;
        }
    }
    return SC_TitleBarLabel;
}

void RChromeStyle::drawComplexControl(ComplexControl control, QStyleOptionComplex const * option, QPainter * painter, QWidget const * widget) const
{
    auto const * bar = qstyleoption_cast<QStyleOptionTitleBar const *>(option);
    if (CC_TitleBar != control || bar == nullptr || widget == nullptr || qobject_cast<QMdiSubWindow const *>(widget) == nullptr)
    {
        QProxyStyle::drawComplexControl(control, option, painter, widget);
        return;
    }
    // QMdiSubWindow fills the option's palette from title bar colors of its own, so read the widget's.
    bool const active = bar->state & State_Active;
    BarInk const ink(widget->palette(), active);
    QRect const rect = bar->rect;

    painter->save();
    painter->setRenderHint(QPainter::Antialiasing, false);
    painter->fillRect(rect, ink.ground);
    if (active)
    {
        painter->fillRect(QRect(rect.left(), rect.bottom() + 1 - ChromeMetrics::ACCENT_WIDTH, rect.width(), ChromeMetrics::ACCENT_WIDTH), ink.accent);
    }
    else
    {
        painter->fillRect(QRect(rect.left(), rect.bottom(), rect.width(), 1), ink.rule);
    }

    QFont font = widget->font();
    font.setPixelSize(ChromeMetrics::TITLE_PX);
    font.setWeight(active ? QFont::DemiBold : QFont::Normal);
    painter->setFont(font);
    painter->setPen(ink.text);
    QRect const label = subControlRect(control, option, SC_TitleBarLabel, widget);
    QString const text = QFontMetrics(font).elidedText(bar->text, Qt::ElideRight, label.width());
    painter->drawText(label, Qt::AlignVCenter | Qt::AlignLeft, text);

    painter->setRenderHint(QPainter::Antialiasing, true);
    bool const maximized = bar->titleBarState & Qt::WindowMaximized;
    auto draw_button = [&](QRect const & box, int slot, bool hovered, bool pressed)
    {
        if (box.isEmpty())
        {
            return;
        }
        if (pressed || hovered)
        {
            painter->setPen(Qt::NoPen);
            painter->setBrush(pressed ? ink.press : ink.hover);
            painter->drawRoundedRect(QRectF(box), 4.0, 4.0);
        }
        draw_glyph(painter, box, slot, maximized, ink.glyph);
    };
    bool const over = bar->state & State_MouseOver;
    bool const sunken = bar->state & State_Sunken;
    SubControl const maxsub = maximized ? SC_TitleBarNormalButton : SC_TitleBarMaxButton;
    for (auto const & [sub, slot] : {std::pair{SC_TitleBarCloseButton, SLOT_CLOSE}, std::pair{maxsub, SLOT_MAXIMIZE}})
    {
        bool const hit = bar->activeSubControls & sub;
        draw_button(subControlRect(control, option, sub, widget), slot, hit && over, hit && sunken);
    }
    painter->restore();
}

void RChromeStyle::drawPrimitive(PrimitiveElement element, QStyleOption const * option, QPainter * painter, QWidget const * widget) const
{
    if (PE_FrameWindow == element && qobject_cast<QMdiSubWindow const *>(widget) != nullptr)
    {
        BarInk const ink(widget->palette(), option->state & State_Active);
        painter->save();
        painter->setPen(QPen(ink.rule, 1));
        painter->setBrush(Qt::NoBrush);
        painter->drawRect(option->rect.adjusted(0, 0, -1, -1));
        painter->restore();
        return;
    }
    QProxyStyle::drawPrimitive(element, option, painter, widget);
}

RChromeFilter::RChromeFilter(QMdiArea * area, RChromeStyle * style)
    : QObject(area)
    , m_style(style)
{
    area->viewport()->installEventFilter(this);
}

void RChromeFilter::dress(QMdiSubWindow * subwin)
{
    if (subwin->property(DRESSED_PROPERTY).toBool())
    {
        return;
    }
    subwin->setProperty(DRESSED_PROPERTY, true);
    subwin->setStyle(m_style);
    subwin->setAttribute(Qt::WA_Hover, true);
}

bool RChromeFilter::eventFilter(QObject * watched, QEvent * event)
{
    if (QEvent::ChildPolished == event->type())
    {
        auto * child = static_cast<QChildEvent *>(event)->child();
        if (auto * subwin = qobject_cast<QMdiSubWindow *>(child))
        {
            dress(subwin);
        }
        return false;
    }
    return QObject::eventFilter(watched, event);
}

void install_chrome(QMdiArea * area)
{
    if (area == nullptr || !chrome_enabled())
    {
        return;
    }
    // A fresh base: a proxy owns the style it wraps.
    auto * style = new RChromeStyle(QStyleFactory::create(QApplication::style()->name()));
    style->setParent(area);
    new RChromeFilter(area, style);
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

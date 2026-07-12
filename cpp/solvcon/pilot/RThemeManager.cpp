/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RThemeManager.hpp> // Must be the first include.

#include <QApplication>
#include <QColor>
#include <QGuiApplication>
#include <QStyleFactory>
#include <QStyleHints>
#include <QString>
#include <QTimer>
#include <QWidget>
#include <Qt>
#include <QtGlobal>

namespace solvcon
{

RThemeManager::RThemeManager(QObject * parent)
    : QObject(parent)
    , m_backend(makeThemeBackend())
{
    // Track live operating-system color-scheme changes while in System mode.
    // The style-hints object is owned by the application and outlives the
    // manager, so the connection stays valid for the manager's lifetime.
    if (QStyleHints * hints = QGuiApplication::styleHints())
    {
        connect(
            hints,
            &QStyleHints::colorSchemeChanged,
            this,
            [this](Qt::ColorScheme)
            {
                // When this signal fires the old palette is still in effect, so
                // defer the re-apply to the next event-loop turn rather than
                // repainting against a palette about to change.
                if (m_mode == ThemeMode::System)
                {
                    QTimer::singleShot(0, this, [this]()
                                       { apply(); });
                }
            });
    }
}

RThemeManager::~RThemeManager() = default;

void RThemeManager::apply()
{
    // Install the platform style once; the default backend leaves the style
    // empty to keep the platform's own, and later calls only repaint.
    if (!m_style_installed)
    {
        std::string const style = m_backend->styleName();
        if (!style.empty())
        {
            QApplication::setStyle(
                QStyleFactory::create(QString::fromStdString(style)));
        }
        m_style_installed = true;
    }

    ThemeVariant const variant = currentVariant();
    QApplication::setPalette(
        buildPalette(themePaletteFor(m_backend->platform(), variant), variant));
    if (m_window != nullptr)
    {
        m_backend->applyNativeChrome(m_window, variant);
    }
    emit themeChanged(variant);
}

void RThemeManager::setWindow(QWidget * window)
{
    m_window = window;
    if (m_window != nullptr)
    {
        m_backend->applyNativeChrome(m_window, currentVariant());
    }
}

void RThemeManager::setMode(ThemeMode mode)
{
    m_mode = mode;
    syncOsColorScheme();
    apply();
}

void RThemeManager::setModeById(std::string const & id)
{
    setMode(themeModeFromId(id.c_str()));
}

ThemeVariant RThemeManager::currentVariant() const
{
    return resolveThemeVariant(m_mode, osPrefersDark());
}

PlatformId RThemeManager::platform() const
{
    return m_backend->platform();
}

ThemeCapabilities RThemeManager::capabilities() const
{
    return m_backend->capabilities();
}

std::string RThemeManager::modeId() const
{
    return themeModeId(m_mode);
}

std::string RThemeManager::variantId() const
{
    return currentVariant() == ThemeVariant::Dark ? "dark" : "light";
}

bool RThemeManager::osPrefersDark() const
{
    QStyleHints * hints = QGuiApplication::styleHints();
    return hints != nullptr && hints->colorScheme() == Qt::ColorScheme::Dark;
}

void RThemeManager::syncOsColorScheme()
{
#if QT_VERSION >= QT_VERSION_CHECK(6, 8, 0)
    QStyleHints * hints = QGuiApplication::styleHints();
    if (hints == nullptr)
    {
        return;
    }
    switch (m_mode)
    {
    case ThemeMode::Light:
        hints->setColorScheme(Qt::ColorScheme::Light);
        break;
    case ThemeMode::Dark:
        hints->setColorScheme(Qt::ColorScheme::Dark);
        break;
    case ThemeMode::System:
    default:
        hints->unsetColorScheme();
        break;
    }
#endif
}

QPalette RThemeManager::buildPalette(ThemePalette const & spec, ThemeVariant variant) const
{
    auto color = [](ThemeColor c)
    { return QColor(c.r, c.g, c.b); };

    // A native accent, when the backend reads one, overrides the curated
    // highlight so the pilot picks up the user's chosen system color.
    QColor highlight = color(spec.highlight);
    if (std::optional<ThemeColor> const accent = m_backend->accentColor(variant))
    {
        highlight = color(*accent);
    }

    QPalette pal;
    pal.setColor(QPalette::Window, color(spec.window));
    pal.setColor(QPalette::WindowText, color(spec.window_text));
    pal.setColor(QPalette::Base, color(spec.base));
    pal.setColor(QPalette::AlternateBase, color(spec.alternate_base));
    pal.setColor(QPalette::Text, color(spec.text));
    pal.setColor(QPalette::Button, color(spec.button));
    pal.setColor(QPalette::ButtonText, color(spec.button_text));
    pal.setColor(QPalette::BrightText, color(spec.bright_text));
    pal.setColor(QPalette::Highlight, highlight);
    pal.setColor(QPalette::HighlightedText, color(spec.highlighted_text));
    pal.setColor(QPalette::ToolTipBase, color(spec.tool_tip_base));
    pal.setColor(QPalette::ToolTipText, color(spec.tool_tip_text));
    pal.setColor(QPalette::PlaceholderText, color(spec.placeholder_text));
    pal.setColor(QPalette::Link, color(spec.link));
    pal.setColor(QPalette::LinkVisited, color(spec.link_visited));

    // The disabled group keeps greyed-out controls legible instead of letting
    // the style derive a washed-out shade from the enabled colors.
    pal.setColor(QPalette::Disabled, QPalette::Text, color(spec.disabled_text));
    pal.setColor(QPalette::Disabled, QPalette::WindowText, color(spec.disabled_window_text));
    pal.setColor(QPalette::Disabled, QPalette::ButtonText, color(spec.disabled_button_text));
    pal.setColor(QPalette::Disabled, QPalette::HighlightedText, color(spec.disabled_text));
    pal.setColor(QPalette::Disabled, QPalette::Highlight, color(spec.disabled_highlight));
    return pal;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/theme/RThemeManager.hpp> // Must be the first include.

#include <solvcon/pilot/theme/theme_qt.hpp>

#include <QApplication>
#include <QColor>
#include <QGuiApplication>
#include <QSettings>
#include <QStyle>
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
    // Start on the mode and look the last session left behind, so a chosen
    // theme carries across restarts.
    restorePersisted();

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
    if (m_look == ThemeLook::System)
    {
        // Let the platform's own colors show through the native style. The
        // color-scheme hint has already steered the style toward the requested
        // variant, so its standard palette carries the right light or dark set.
        QPalette const native = QApplication::style()->standardPalette();
        m_window_color = native.color(QPalette::Window);
        QApplication::setPalette(native);
        // Leave the native look untouched, including any style the platform set.
        setApplicationStyleSheet(QString());
    }
    else
    {
        QPalette const curated =
            buildPalette(themePaletteFor(m_backend->platform(), variant), variant);
        m_window_color = curated.color(QPalette::Window);
        QApplication::setPalette(curated);
        setApplicationStyleSheet(supplementalStyleSheet(curated));
    }
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
    persist();
    syncOsColorScheme();
    apply();
}

void RThemeManager::setModeById(std::string const & id)
{
    setMode(themeModeFromId(id.c_str()));
}

void RThemeManager::setLook(ThemeLook look)
{
    m_look = look;
    persist();
    apply();
}

void RThemeManager::setLookById(std::string const & id)
{
    setLook(themeLookFromId(id.c_str()));
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

Canvas2dPalette const & RThemeManager::canvas2dPalette() const
{
    // The canvas keeps its curated tables under either look: the system look
    // hands over the platform's widget colors, which say nothing about how a
    // drawing surface should read.
    return canvas2dPaletteFor(m_backend->platform(), currentVariant());
}

std::string RThemeManager::modeId() const
{
    return themeModeId(m_mode);
}

std::string RThemeManager::lookId() const
{
    return themeLookId(m_look);
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
    // A native accent, when the backend reads one, overrides the curated
    // highlight so the pilot picks up the user's chosen system color.
    QColor highlight = qcolor(spec.highlight);
    if (std::optional<ThemeColor> const accent = m_backend->accentColor(variant))
    {
        highlight = qcolor(*accent);
    }

    QPalette pal;
    pal.setColor(QPalette::Window, qcolor(spec.window));
    pal.setColor(QPalette::WindowText, qcolor(spec.window_text));
    pal.setColor(QPalette::Base, qcolor(spec.base));
    pal.setColor(QPalette::AlternateBase, qcolor(spec.alternate_base));
    pal.setColor(QPalette::Text, qcolor(spec.text));
    pal.setColor(QPalette::Button, qcolor(spec.button));
    pal.setColor(QPalette::ButtonText, qcolor(spec.button_text));
    pal.setColor(QPalette::BrightText, qcolor(spec.bright_text));
    pal.setColor(QPalette::Highlight, highlight);
#if QT_VERSION >= QT_VERSION_CHECK(6, 6, 0)
    // The dedicated Accent role (Qt 6.6+) lets widgets that want the platform
    // accent, distinct from a selection highlight, pick it up from the palette.
    pal.setColor(QPalette::Accent, highlight);
#endif
    pal.setColor(QPalette::HighlightedText, qcolor(spec.highlighted_text));
    pal.setColor(QPalette::ToolTipBase, qcolor(spec.tool_tip_base));
    pal.setColor(QPalette::ToolTipText, qcolor(spec.tool_tip_text));
    pal.setColor(QPalette::PlaceholderText, qcolor(spec.placeholder_text));
    pal.setColor(QPalette::Link, qcolor(spec.link));
    pal.setColor(QPalette::LinkVisited, qcolor(spec.link_visited));

    // The disabled group keeps greyed-out controls legible instead of letting
    // the style derive a washed-out shade from the enabled colors.
    pal.setColor(QPalette::Disabled, QPalette::Text, qcolor(spec.disabled_text));
    pal.setColor(QPalette::Disabled, QPalette::WindowText, qcolor(spec.disabled_window_text));
    pal.setColor(QPalette::Disabled, QPalette::ButtonText, qcolor(spec.disabled_button_text));
    pal.setColor(QPalette::Disabled, QPalette::HighlightedText, qcolor(spec.disabled_text));
    pal.setColor(QPalette::Disabled, QPalette::Highlight, qcolor(spec.disabled_highlight));
    return pal;
}

void RThemeManager::setApplicationStyleSheet(QString const & sheet)
{
    // An application-wide stylesheet change unpolishes and repolishes every
    // widget in the process, so pay for it only when the text really differs.
    if (sheet == m_style_sheet)
    {
        return;
    }

    m_style_sheet = sheet;
    qApp->setStyleSheet(sheet);
}

QString RThemeManager::supplementalStyleSheet(QPalette const & pal) const
{
    // A QPalette cannot draw a tooltip border or a focus ring, so add just
    // those two, colored from the theme, and nothing more, so the native style
    // keeps drawing everything else.
    QColor const border = pal.color(QPalette::WindowText);
    QColor const focus = pal.color(QPalette::Highlight);
    return QStringLiteral(
               "QToolTip { border: 1px solid %1; }\n"
               "QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus"
               " { border: 1px solid %2; }")
        .arg(border.name(), focus.name());
}

void RThemeManager::restorePersisted()
{
    QSettings settings(QStringLiteral("solvcon"), QStringLiteral("pilot"));
    m_mode = themeModeFromId(
        settings.value(QStringLiteral("theme/mode")).toString().toUtf8().constData());
    m_look = themeLookFromId(
        settings.value(QStringLiteral("theme/look")).toString().toUtf8().constData());
}

void RThemeManager::persist()
{
    // The store is only read when a session starts, so a write that repeats
    // what was already written buys nothing and touches the disk.
    if (m_persisted_mode == m_mode && m_persisted_look == m_look)
    {
        return;
    }

    QSettings settings(QStringLiteral("solvcon"), QStringLiteral("pilot"));
    settings.setValue(QStringLiteral("theme/mode"), QString::fromStdString(modeId()));
    settings.setValue(QStringLiteral("theme/look"), QString::fromStdString(lookId()));
    m_persisted_mode = m_mode;
    m_persisted_look = m_look;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

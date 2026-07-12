/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RThemeBackend.hpp>

#include <QtGlobal>

#if defined(Q_OS_MACOS)
#include <solvcon/pilot/RMacThemeBackend.hpp>
#endif

namespace solvcon
{

namespace
{

/**
 * @brief The fallback backend: it keeps the platform's own default style and
 * pulls no native levers.
 *
 * Every platform starts here and is replaced by its furnished room in a later
 * step. Keeping the default style already gives each platform its native
 * widgets; the room adds the accent, the title bar, and an explicit style
 * choice on top.
 */
class DefaultThemeBackend
    : public RThemeBackend
{

public:

    explicit DefaultThemeBackend(PlatformId platform)
        : m_platform(platform)
    {
    }

    PlatformId platform() const override { return m_platform; }

    std::string styleName() const override { return {}; }

    std::optional<ThemeColor> accentColor(ThemeVariant /*variant*/) const override
    {
        return std::nullopt;
    }

    void applyNativeChrome(QWidget * /*window*/, ThemeVariant /*variant*/) override {}

    ThemeCapabilities capabilities() const override
    {
        return themeCapabilitiesFor(m_platform);
    }

private:

    PlatformId m_platform;

}; /* end class DefaultThemeBackend */

} /* end namespace */

std::unique_ptr<RThemeBackend> makeThemeBackend()
{
#if defined(Q_OS_MACOS)
    return std::make_unique<RMacThemeBackend>();
#elif defined(Q_OS_WIN)
    return std::make_unique<DefaultThemeBackend>(PlatformId::Windows);
#else
    return std::make_unique<DefaultThemeBackend>(PlatformId::Linux);
#endif
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

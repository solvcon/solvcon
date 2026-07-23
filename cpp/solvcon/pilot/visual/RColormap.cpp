/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/visual/RColormap.hpp> // Must be the first include.

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace solvcon
{

namespace
{

// One RGB anchor triple per row, spaced uniformly on [0, 1].
// clang-format off

// The 20-anchor viridis palette (the widely published viridis(20) table).
constexpr float VIRIDIS[] = {
    0.267f, 0.005f, 0.329f,
    0.282f, 0.084f, 0.404f,
    0.282f, 0.150f, 0.467f,
    0.271f, 0.216f, 0.506f,
    0.251f, 0.278f, 0.533f,
    0.224f, 0.337f, 0.549f,
    0.200f, 0.388f, 0.553f,
    0.176f, 0.439f, 0.557f,
    0.157f, 0.490f, 0.557f,
    0.137f, 0.541f, 0.553f,
    0.122f, 0.588f, 0.545f,
    0.125f, 0.639f, 0.529f,
    0.161f, 0.686f, 0.498f,
    0.235f, 0.733f, 0.459f,
    0.333f, 0.776f, 0.404f,
    0.451f, 0.816f, 0.333f,
    0.584f, 0.847f, 0.251f,
    0.722f, 0.871f, 0.161f,
    0.863f, 0.890f, 0.098f,
    0.993f, 0.906f, 0.144f};

// Approximate anchors of Moreland's smooth diverging blue-white-red map.
constexpr float COOLWARM[] = {
    0.230f, 0.299f, 0.754f,
    0.484f, 0.620f, 0.974f,
    0.865f, 0.865f, 0.865f,
    0.926f, 0.502f, 0.412f,
    0.706f, 0.016f, 0.150f};

// The compact four-stop jet approximation; the anchors sit on the k/8
// breakpoints of its triangle-wave channels, so the piecewise-linear ramp
// reproduces the formula exactly.
constexpr float JET[] = {
    0.0f, 0.0f, 0.5f,
    0.0f, 0.0f, 1.0f,
    0.0f, 0.5f, 1.0f,
    0.0f, 1.0f, 1.0f,
    0.5f, 1.0f, 0.5f,
    1.0f, 1.0f, 0.0f,
    1.0f, 0.5f, 0.0f,
    1.0f, 0.0f, 0.0f,
    0.5f, 0.0f, 0.0f};

constexpr float GRAYSCALE[] = {
    0.0f, 0.0f, 0.0f,
    1.0f, 1.0f, 1.0f};

// A 12-color qualitative palette for categorical coloring: neighbors are
// picked far apart in hue and value so adjacent categories stay distinct.
constexpr float QUALITATIVE[] = {
    0.90f, 0.10f, 0.10f,
    0.12f, 0.47f, 0.71f,
    0.17f, 0.63f, 0.17f,
    1.00f, 0.50f, 0.05f,
    0.58f, 0.40f, 0.74f,
    0.55f, 0.34f, 0.29f,
    0.89f, 0.47f, 0.76f,
    0.50f, 0.50f, 0.50f,
    0.74f, 0.74f, 0.13f,
    0.09f, 0.75f, 0.81f,
    0.10f, 0.10f, 0.55f,
    0.60f, 0.85f, 0.20f};

// clang-format on

} /* end namespace */

RColormap::RColormap(std::string name, float const * anchors, size_t count)
    : m_name(std::move(name))
    , m_anchors(anchors)
    , m_count(count)
{
}

RColormap RColormap::named(std::string const & name)
{
    if ("viridis" == name)
    {
        return RColormap(name, VIRIDIS, sizeof(VIRIDIS) / sizeof(float) / 3);
    }
    if ("coolwarm" == name)
    {
        return RColormap(name, COOLWARM, sizeof(COOLWARM) / sizeof(float) / 3);
    }
    if ("jet" == name)
    {
        return RColormap(name, JET, sizeof(JET) / sizeof(float) / 3);
    }
    if ("grayscale" == name)
    {
        return RColormap(name, GRAYSCALE, sizeof(GRAYSCALE) / sizeof(float) / 3);
    }
    throw std::invalid_argument(
        "RColormap: unknown colormap \"" + name +
        "\"; pick one of \"viridis\", \"coolwarm\", \"jet\", \"grayscale\"");
}

RColormap RColormap::categorical()
{
    return RColormap("categorical", QUALITATIVE, sizeof(QUALITATIVE) / sizeof(float) / 3);
}

QVector3D RColormap::sample(float t) const
{
    t = std::clamp(t, 0.0f, 1.0f);
    float const pos = t * static_cast<float>(m_count - 1);
    size_t const lo = std::min(static_cast<size_t>(pos), m_count - 2);
    float const frac = pos - static_cast<float>(lo);
    float const * a = m_anchors + 3 * lo;
    float const * b = a + 3;
    return QVector3D(
        a[0] + frac * (b[0] - a[0]),
        a[1] + frac * (b[1] - a[1]),
        a[2] + frac * (b[2] - a[2]));
}

QImage RColormap::image(int width) const
{
    QImage img(width, 1, QImage::Format_RGBA8888);
    for (int i = 0; i < width; ++i)
    {
        float const t = (width > 1) ? static_cast<float>(i) / static_cast<float>(width - 1) : 0.0f;
        QVector3D const c = sample(t);
        img.setPixel(
            i,
            0,
            qRgba(
                static_cast<int>(std::lround(c.x() * 255.0f)),
                static_cast<int>(std::lround(c.y() * 255.0f)),
                static_cast<int>(std::lround(c.z() * 255.0f)),
                255));
    }
    return img;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

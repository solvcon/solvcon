/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/visual/RScalarBar.hpp> // Must be the first include.

#include <QColor>
#include <QFont>
#include <QGuiApplication>
#include <QPainter>
#include <QPalette>

#include <algorithm>
#include <utility>

namespace solvcon
{

namespace
{

// The bar image is painted at a fixed pixel size and scaled into an
// aspect-preserving viewport, so the layout below is in image pixels.
constexpr int BAR_WIDTH = 128;
constexpr int BAR_HEIGHT = 384;
constexpr int STRIP_LEFT = 14;
constexpr int STRIP_TOP = 56;
constexpr int STRIP_WIDTH = 30;
constexpr int STRIP_HEIGHT = 304;
constexpr float VIEWPORT_MARGIN = 8.0f;

} /* end namespace */

RScalarBar::RScalarBar()
    : m_colormap(RColormap::named("viridis"))
{
}

RScalarBar::~RScalarBar() = default;

void RScalarBar::setTitle(std::string const & title)
{
    m_title = QString::fromStdString(title);
    m_dirty = true;
}

void RScalarBar::setColormap(RColormap colormap)
{
    m_colormap = std::move(colormap);
    m_dirty = true;
}

void RScalarBar::setRange(float lo, float hi)
{
    if (lo != m_lo || hi != m_hi)
    {
        m_lo = lo;
        m_hi = hi;
        m_dirty = true;
    }
}

QImage RScalarBar::renderImage() const
{
    QImage image(BAR_WIDTH, BAR_HEIGHT, QImage::Format_RGBA8888);
    image.fill(Qt::transparent);
    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing, true);

    // Follow the application palette so the labels and the strip border stay
    // legible when the theme turns dark, instead of forcing black.
    QColor const ink = QGuiApplication::palette().color(QPalette::WindowText);

    QFont font;
    font.setPixelSize(15);
    font.setBold(true);
    painter.setFont(font);
    painter.setPen(ink);
    painter.drawText(QRect(0, 4, BAR_WIDTH, 20), Qt::AlignHCenter | Qt::AlignVCenter, m_title);

    font.setPixelSize(13);
    font.setBold(false);
    painter.setFont(font);
    painter.drawText(
        QRect(0, 26, BAR_WIDTH, 18),
        Qt::AlignHCenter | Qt::AlignVCenter,
        QString::fromStdString(m_colormap.name()));

    // The color strip, painted from the same table the GPU LUT is baked
    // from; the high end is at the top.
    QRect const strip(STRIP_LEFT, STRIP_TOP, STRIP_WIDTH, STRIP_HEIGHT);
    QImage const lut = m_colormap.image(STRIP_HEIGHT);
    painter.save();
    painter.translate(strip.left(), strip.top() + strip.height());
    painter.rotate(-90.0);
    painter.drawImage(QRect(0, 0, strip.height(), strip.width()), lut);
    painter.restore();
    painter.setPen(ink);
    painter.drawRect(strip.adjusted(0, 0, -1, -1));

    int const label_left = strip.right() + 7;
    int const label_width = BAR_WIDTH - label_left - 2;
    painter.drawText(
        QRect(label_left, strip.top() - 9, label_width, 18),
        Qt::AlignLeft | Qt::AlignVCenter,
        QString::number(m_hi, 'g', 4));
    painter.drawText(
        QRect(label_left, strip.bottom() - 8, label_width, 18),
        Qt::AlignLeft | Qt::AlignVCenter,
        QString::number(m_lo, 'g', 4));

    painter.end();
    return image;
}

void RScalarBar::prepare(
    QRhi * rhi, QRhiRenderPassDescriptor * rpdesc, int sample_count, QRhiResourceUpdateBatch * batch)
{
    // A single quad over the whole (aspect-preserving) viewport; the image
    // top row maps to the top of the viewport.
    // clang-format off
    float const quad[6 * 5] = {
        // x, y, z, u, v
        -1.0f, -1.0f, 0.0f, 0.0f, 1.0f,
        1.0f, -1.0f, 0.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 0.0f, 1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 0.0f, 1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    };
    // clang-format on
    m_vbuf.reset(rhi->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::VertexBuffer, sizeof(quad)));
    m_vbuf->create();
    batch->uploadStaticBuffer(m_vbuf.get(), quad);

    m_ubuf.reset(rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, 64 + 16));
    m_ubuf->create();

    m_sampler.reset(rhi->newSampler(
        QRhiSampler::Linear,
        QRhiSampler::Linear,
        QRhiSampler::None,
        QRhiSampler::ClampToEdge,
        QRhiSampler::ClampToEdge));
    m_sampler->create();

    m_texture.reset(rhi->newTexture(QRhiTexture::RGBA8, QSize(BAR_WIDTH, BAR_HEIGHT)));
    m_texture->create();
    batch->uploadTexture(m_texture.get(), renderImage());
    m_dirty = false;

    m_srb.reset(rhi->newShaderResourceBindings());
    m_srb->setBindings({
        QRhiShaderResourceBinding::uniformBuffer(
            0,
            QRhiShaderResourceBinding::VertexStage | QRhiShaderResourceBinding::FragmentStage,
            m_ubuf.get()),
        QRhiShaderResourceBinding::sampledTexture(
            1, QRhiShaderResourceBinding::FragmentStage, m_texture.get(), m_sampler.get()),
    });
    m_srb->create();

    QRhiVertexInputLayout layout;
    layout.setBindings({{5 * sizeof(float)}});
    layout.setAttributes({
        {0, 0, QRhiVertexInputAttribute::Float3, 0},
        {0, 1, QRhiVertexInputAttribute::Float2, 3 * sizeof(float)},
    });

    // The bar draws over the scene: depth test off, alpha blend on so only
    // the painted parts of the image cover the frame.
    m_material = std::make_unique<RMaterial>(RMaterial::Kind::Textured);
    m_pipeline.reset(m_material->buildPipeline(
        rhi, m_srb.get(), rpdesc, layout, QRhiGraphicsPipeline::Triangles, sample_count, false, true));

    m_ready = (nullptr != m_pipeline);
}

void RScalarBar::update(
    QRhi * rhi,
    QRhiRenderPassDescriptor * rpdesc,
    int sample_count,
    QSize pixel_size,
    QRhiResourceUpdateBatch * batch)
{
    m_drawable = false;
    if (!m_visible)
    {
        return;
    }
    if (!m_ready)
    {
        prepare(rhi, rpdesc, sample_count, batch);
        if (!m_ready)
        {
            return;
        }
    }

    if (m_dirty)
    {
        batch->uploadTexture(m_texture.get(), renderImage());
        m_dirty = false;
    }

    // An aspect-preserving viewport centered on the right edge.
    float const height = std::clamp(
        0.62f * static_cast<float>(pixel_size.height()),
        140.0f,
        static_cast<float>(BAR_HEIGHT));
    float const width = height * static_cast<float>(BAR_WIDTH) / static_cast<float>(BAR_HEIGHT);
    float const x = static_cast<float>(pixel_size.width()) - width - VIEWPORT_MARGIN;
    float const y = 0.5f * (static_cast<float>(pixel_size.height()) - height);
    m_viewport = QRhiViewport(x, y, width, height);

    QMatrix4x4 const mvp = rhi->clipSpaceCorrMatrix();
    batch->updateDynamicBuffer(m_ubuf.get(), 0, 64, mvp.constData());
    float const white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    batch->updateDynamicBuffer(m_ubuf.get(), 64, 16, white);

    m_drawable = true;
}

void RScalarBar::draw(QRhiCommandBuffer * cb)
{
    if (!m_visible || !m_drawable)
    {
        return;
    }

    cb->setViewport(m_viewport);
    cb->setGraphicsPipeline(m_pipeline.get());
    cb->setShaderResources(m_srb.get());
    QRhiCommandBuffer::VertexInput const quad_in(m_vbuf.get(), 0);
    cb->setVertexInput(0, 1, &quad_in);
    cb->draw(6);
}

void RScalarBar::release()
{
    m_pipeline.reset();
    m_srb.reset();
    m_texture.reset();
    m_sampler.reset();
    m_ubuf.reset();
    m_vbuf.reset();
    m_material.reset();
    m_ready = false;
    m_drawable = false;
    m_dirty = true;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

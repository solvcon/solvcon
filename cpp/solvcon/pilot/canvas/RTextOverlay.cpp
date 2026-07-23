/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/canvas/RTextOverlay.hpp> // Must be the first include.

#include <QColor>
#include <QFont>
#include <QGuiApplication>
#include <QPainter>
#include <QPalette>

#include <algorithm>

namespace solvcon
{

namespace
{

// The title image is painted at a fixed pixel size and scaled into an
// aspect-preserving viewport centered at the top.
constexpr int TEXT_WIDTH = 512;
constexpr int TEXT_HEIGHT = 48;
constexpr float VIEWPORT_MARGIN = 8.0f;

} /* end namespace */

RTextOverlay::RTextOverlay() = default;

RTextOverlay::~RTextOverlay() = default;

void RTextOverlay::setText(std::string const & text)
{
    m_text = QString::fromStdString(text);
    m_dirty = true;
}

QImage RTextOverlay::renderImage() const
{
    QImage image(TEXT_WIDTH, TEXT_HEIGHT, QImage::Format_RGBA8888);
    image.fill(Qt::transparent);
    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing, true);

    QFont font;
    font.setPixelSize(28);
    font.setBold(true);
    painter.setFont(font);
    // Follow the application palette so the label stays legible when the theme
    // turns dark, instead of forcing black.
    painter.setPen(QGuiApplication::palette().color(QPalette::WindowText));
    painter.drawText(image.rect(), Qt::AlignHCenter | Qt::AlignVCenter, m_text);
    painter.end();
    return image;
}

void RTextOverlay::prepare(
    QRhi * rhi, QRhiRenderPassDescriptor * rpdesc, int sample_count, QRhiResourceUpdateBatch * batch)
{
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

    m_texture.reset(rhi->newTexture(QRhiTexture::RGBA8, QSize(TEXT_WIDTH, TEXT_HEIGHT)));
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

    m_material = std::make_unique<RMaterial>(RMaterial::Kind::Textured);
    m_pipeline.reset(m_material->buildPipeline(
        rhi, m_srb.get(), rpdesc, layout, QRhiGraphicsPipeline::Triangles, sample_count, false, true));

    m_ready = (nullptr != m_pipeline);
}

void RTextOverlay::update(
    QRhi * rhi,
    QRhiRenderPassDescriptor * rpdesc,
    int sample_count,
    QSize pixel_size,
    QRhiResourceUpdateBatch * batch)
{
    m_drawable = false;
    if (!m_visible || m_text.isEmpty())
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

    // An aspect-preserving viewport centered at the top edge.
    float const width = std::clamp(
        0.6f * static_cast<float>(pixel_size.width()),
        160.0f,
        static_cast<float>(TEXT_WIDTH));
    float const height = width * static_cast<float>(TEXT_HEIGHT) / static_cast<float>(TEXT_WIDTH);
    float const x = 0.5f * (static_cast<float>(pixel_size.width()) - width);
    // QRhi viewport y is measured from the bottom, so a top margin sits high.
    float const y = static_cast<float>(pixel_size.height()) - height - VIEWPORT_MARGIN;
    m_viewport = QRhiViewport(x, y, width, height);

    QMatrix4x4 const mvp = rhi->clipSpaceCorrMatrix();
    batch->updateDynamicBuffer(m_ubuf.get(), 0, 64, mvp.constData());
    float const white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    batch->updateDynamicBuffer(m_ubuf.get(), 64, 16, white);

    m_drawable = true;
}

void RTextOverlay::draw(QRhiCommandBuffer * cb)
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

void RTextOverlay::release()
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

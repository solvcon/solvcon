#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * A screen-space text overlay (a figure title), painted with QPainter and
 * drawn as a textured quad over the scene.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/RMaterial.hpp>

#include <rhi/qrhi.h>

#include <QImage>
#include <QString>

#include <memory>
#include <string>

namespace solvcon
{

/**
 * @brief A title string painted into a texture and drawn as a top-center
 * overlay quad, following the scalar-bar overlay pattern.
 *
 * @ingroup group_domain
 */
class RTextOverlay
{

public:

    RTextOverlay();
    ~RTextOverlay();

    void setVisible(bool visible) { m_visible = visible; }
    bool isVisible() const { return m_visible; }

    void setText(std::string const & text);
    std::string text() const { return m_text.toStdString(); }

    void update(
        QRhi * rhi,
        QRhiRenderPassDescriptor * rpdesc,
        int sample_count,
        QSize pixel_size,
        QRhiResourceUpdateBatch * batch);
    void draw(QRhiCommandBuffer * cb);
    void release();

private:

    QImage renderImage() const;
    void prepare(QRhi * rhi, QRhiRenderPassDescriptor * rpdesc, int sample_count, QRhiResourceUpdateBatch * batch);

    QString m_text;
    bool m_visible = false;
    bool m_ready = false;
    bool m_drawable = false;
    bool m_dirty = true;

    QRhiViewport m_viewport;

    std::unique_ptr<RMaterial> m_material;
    std::unique_ptr<QRhiBuffer> m_vbuf;
    std::unique_ptr<QRhiBuffer> m_ubuf;
    std::unique_ptr<QRhiTexture> m_texture;
    std::unique_ptr<QRhiSampler> m_sampler;
    std::unique_ptr<QRhiShaderResourceBindings> m_srb;
    std::unique_ptr<QRhiGraphicsPipeline> m_pipeline;

}; /* end class RTextOverlay */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Edge-of-viewport scalar bar reporting the active colormap and range.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/visual/RColormap.hpp>
#include <solvcon/pilot/visual/RMaterial.hpp>

#include <rhi/qrhi.h>

#include <QImage>
#include <QSize>
#include <QString>

#include <memory>
#include <string>

namespace solvcon
{

/**
 * @brief An on-screen scalar bar reporting the active scalar-color mapping.
 *
 * Renders the title, the colormap name, the color strip (painted from the
 * same RColormap table the GPU LUT is baked from), and the range end labels
 * into one QImage, uploaded as a texture and drawn as a single overlay quad
 * along the right edge of the viewport with the depth test off. The bar
 * follows the gizmo pattern: resource updates happen in update() before the
 * render pass, draw calls in draw() inside the pass.
 *
 * @ingroup group_domain
 */
class RScalarBar
{

public:

    RScalarBar();
    ~RScalarBar();

    RScalarBar(RScalarBar const &) = delete;
    RScalarBar & operator=(RScalarBar const &) = delete;

    void setVisible(bool visible) { m_visible = visible; }
    bool isVisible() const { return m_visible; }

    void setTitle(std::string const & title);
    void setColormap(RColormap colormap);
    void setRange(float lo, float hi);

    /// Create or update the device resources. Call before the render pass.
    void update(
        QRhi * rhi,
        QRhiRenderPassDescriptor * rpdesc,
        int sample_count,
        QSize pixel_size,
        QRhiResourceUpdateBatch * batch);

    /// Record the draw calls into the active render pass.
    void draw(QRhiCommandBuffer * cb);

    /// Drop every device resource.
    void release();

private:

    QImage renderImage() const;
    void prepare(QRhi * rhi, QRhiRenderPassDescriptor * rpdesc, int sample_count, QRhiResourceUpdateBatch * batch);

    bool m_visible = false;
    bool m_ready = false;
    bool m_drawable = false; ///< update() succeeded for this frame.
    bool m_dirty = true; ///< The bar image needs a repaint and re-upload.

    QString m_title;
    RColormap m_colormap;
    float m_lo = 0.0f;
    float m_hi = 1.0f;

    QRhiViewport m_viewport;

    std::unique_ptr<RMaterial> m_material;
    std::unique_ptr<QRhiBuffer> m_vbuf;
    std::unique_ptr<QRhiBuffer> m_ubuf;
    std::unique_ptr<QRhiSampler> m_sampler;
    std::unique_ptr<QRhiTexture> m_texture;
    std::unique_ptr<QRhiShaderResourceBindings> m_srb;
    std::unique_ptr<QRhiGraphicsPipeline> m_pipeline;

}; /* end class RScalarBar */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Corner-of-viewport orientation gizmo that draws a colored axis triad.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/RMaterial.hpp>

#include <rhi/qrhi.h>

#include <QImage>
#include <QMatrix4x4>
#include <QSize>
#include <QString>
#include <QVector3D>

#include <array>
#include <memory>

namespace solvcon
{

/**
 * @brief A small orientation guide drawn in a corner of the viewport.
 *
 * Renders a 2- or 3-axis triad of colored arrows (hand-built cones on
 * shafts) with X/Y/Z labels (QPainter glyphs uploaded as textures and drawn
 * as camera-facing billboards). The guide has its own orthographic camera
 * whose orientation mirrors the main camera, so it always shows which way the
 * domain axes point. It is drawn over the scene in the same render pass with
 * the depth test off, confined to a corner viewport.
 *
 * Resource updates happen in update() before the render pass begins; the
 * draw calls happen in draw() inside the pass.
 *
 * @ingroup group_domain
 */
class RAxisGizmo
{

public:

    RAxisGizmo();
    ~RAxisGizmo();

    RAxisGizmo(RAxisGizmo const &) = delete;
    RAxisGizmo & operator=(RAxisGizmo const &) = delete;

    void setVisible(bool visible) { m_visible = visible; }
    bool isVisible() const { return m_visible; }

    /// Show 2 axes (X, Y) or 3 axes (X, Y, Z).
    void setAxisCount(int count) { m_axis_count = (count <= 2) ? 2 : 3; }

    /// Create or update the device resources and the per-frame transforms.
    /// Call before the render pass begins. @p forward and @p up come from the
    /// main camera so the triad mirrors the current view orientation.
    void update(
        QRhi * rhi,
        QRhiRenderPassDescriptor * rpdesc,
        int sample_count,
        QSize pixel_size,
        QVector3D const & forward,
        QVector3D const & up,
        QRhiResourceUpdateBatch * batch);

    /// Record the draw calls into the active render pass.
    void draw(QRhiCommandBuffer * cb);

    /// Drop every device resource.
    void release();

private:

    void buildGeometry();
    void prepare(QRhi * rhi, QRhiRenderPassDescriptor * rpdesc, int sample_count, QRhiResourceUpdateBatch * batch);
    static QImage makeLabelImage(QString const & text, QColor const & color);

    bool m_visible = false;
    int m_axis_count = 3;
    bool m_ready = false;
    bool m_drawable = false; ///< update() succeeded for this frame.

    QRhiViewport m_viewport;

    // Object-space geometry, interleaved [x, y, z, r, g, b]; built once.
    SimpleCollector<float> m_shaft_vertices;
    SimpleCollector<float> m_cone_vertices;

    std::unique_ptr<RMaterial> m_vcolor_material;
    std::unique_ptr<RMaterial> m_texture_material;

    std::unique_ptr<QRhiBuffer> m_shaft_vbuf;
    std::unique_ptr<QRhiBuffer> m_cone_vbuf;
    std::unique_ptr<QRhiBuffer> m_label_vbuf; ///< Dynamic billboard quads.
    std::unique_ptr<QRhiBuffer> m_ubuf;
    std::unique_ptr<QRhiShaderResourceBindings> m_srb; ///< Shafts and cones.
    std::unique_ptr<QRhiGraphicsPipeline> m_shaft_pipeline;
    std::unique_ptr<QRhiGraphicsPipeline> m_cone_pipeline;
    std::unique_ptr<QRhiGraphicsPipeline> m_label_pipeline;

    std::unique_ptr<QRhiSampler> m_sampler;
    std::array<std::unique_ptr<QRhiTexture>, 3> m_label_textures;
    std::array<std::unique_ptr<QRhiShaderResourceBindings>, 3> m_label_srb;

}; /* end class RAxisGizmo */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

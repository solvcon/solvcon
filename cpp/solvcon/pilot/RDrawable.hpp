#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Base class for a renderable object in the pilot domain scene.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/RMaterial.hpp>

#include <rhi/qrhi.h>

#include <QMatrix4x4>
#include <QVector3D>
#include <QVector4D>

#include <memory>
#include <string>
#include <vector>

namespace solvcon
{

/**
 * @brief Base class for a renderable object in the domain scene.
 *
 * A drawable owns its vertex (and optional index) buffer, a per-object
 * uniform buffer holding the model-view-projection matrix and a flat color,
 * the shader-resource bindings, and the graphics pipeline built from its
 * RMaterial. Subclasses supply the geometry and the pipeline configuration
 * (material variant, primitive topology, vertex input layout).
 *
 * The device-owned resources are created lazily through prepare() once the
 * QRhi and the render target are known, and dropped through release() when
 * the device is lost or replaced.
 *
 * @ingroup group_domain
 */
class RDrawable
{

public:

    virtual ~RDrawable();

    RDrawable(RDrawable const &) = delete;
    RDrawable & operator=(RDrawable const &) = delete;

    void setVisible(bool visible) { m_visible = visible; }
    bool isVisible() const { return m_visible; }

    void setColor(QVector4D const & color) { m_color = color; }
    QVector4D color() const { return m_color; }

    /// The model transform applied before the view-projection; identity by
    /// default, so a single object renders where its geometry already sits.
    void setModel(QMatrix4x4 const & model) { m_model = model; }
    QMatrix4x4 model() const { return m_model; }

    /// An optional name, used by the scene's named-object registry.
    void setName(std::string const & name) { m_name = name; }
    std::string const & name() const { return m_name; }

    /// World-space direction toward the light, read by the lit material and
    /// ignored by the others. The widget refreshes it as the camera moves.
    void setLightDir(QVector3D const & dir) { m_light_dir = dir; }

    /// Set the drawable opacity, a [0, 1] multiplier on the color alpha.
    /// 1 is fully opaque, 0 fully transparent. Clamped to the range.
    void setOpacity(float opacity)
    {
        m_opacity = (opacity < 0.0f) ? 0.0f : (opacity > 1.0f) ? 1.0f
                                                               : opacity;
    }
    float opacity() const { return m_opacity; }

    /// Create the device resources if they do not exist yet. Idempotent.
    void prepare(
        QRhi * rhi,
        QRhiRenderPassDescriptor * rpdesc,
        int sample_count,
        QRhiResourceUpdateBatch * batch);

    /// Drop all device resources (device lost or render target changed).
    virtual void release();

    /// Write the current model-view-projection and color into the uniform
    /// buffer. The model transform is identity for now.
    virtual void updateUniform(QRhiResourceUpdateBatch * batch, QMatrix4x4 const & view_proj);

    /// Record the draw call. Requires an active render pass and a viewport
    /// already set on the command buffer. A no-op while hidden or unprepared.
    void draw(QRhiCommandBuffer * cb);

protected:

    RDrawable() = default;

    /// std140 layout of the shared uniform block: mat4 mvp + vec4 color +
    /// vec4 lightDir. The trailing light direction is used only by the lit
    /// material; the other shaders declare a shorter block over the same buffer.
    static constexpr int UBUF_SIZE = 64 + 16 + 16;

    /// The shader variant this drawable renders with.
    virtual RMaterial::Kind materialKind() const = 0;

    /// The primitive topology to assemble.
    virtual QRhiGraphicsPipeline::Topology topology() const = 0;

    /// The vertex input layout matching the geometry buffers.
    virtual QRhiVertexInputLayout vertexInputLayout() const = 0;

    /// Create and fill m_vbuf (and m_ibuf when indexed), and set
    /// m_vertex_count / m_index_count. Called once from prepare().
    virtual void createGeometry(QRhi * rhi, QRhiResourceUpdateBatch * batch) = 0;

    /// Constant and slope-scaled depth bias for this drawable's pipeline. A
    /// filled surface returns a positive bias so a coplanar wireframe or point
    /// cloud drawn over it is not lost to the depth test; the default is none.
    virtual int depthBias() const { return 0; }
    virtual float slopeScaledDepthBias() const { return 0.0f; }

    /// Create and append shader resources past the shared uniform block
    /// (e.g. a sampled texture). Called from prepare(); adds nothing by
    /// default.
    virtual void extendBindings(
        QRhi * rhi,
        QRhiResourceUpdateBatch * batch,
        std::vector<QRhiShaderResourceBinding> & bindings);

    QVector4D m_color{1.0f, 1.0f, 1.0f, 1.0f};
    QVector3D m_light_dir{0.0f, 0.0f, 1.0f};
    QMatrix4x4 m_model; ///< Per-object model transform (identity by default).
    std::string m_name; ///< Optional name for the scene registry.
    float m_opacity = 1.0f;
    bool m_visible = true;

    std::unique_ptr<QRhiBuffer> m_vbuf;
    std::unique_ptr<QRhiBuffer> m_ibuf; ///< Null for non-indexed geometry.
    quint32 m_vertex_count = 0;
    quint32 m_index_count = 0;

private:

    std::unique_ptr<RMaterial> m_material;
    std::unique_ptr<QRhiBuffer> m_ubuf;
    std::unique_ptr<QRhiShaderResourceBindings> m_srb;
    std::unique_ptr<QRhiGraphicsPipeline> m_pipeline;
    bool m_ready = false;

}; /* end class RDrawable */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

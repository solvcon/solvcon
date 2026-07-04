#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Rendering material that pairs a baked GLSL shader set with a graphics
 * pipeline builder.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common_detail.hpp> // Must be the first include.

#include <rhi/qrhi.h>

#include <QString>

namespace solvcon
{

/**
 * @brief A rendering material: a baked GLSL shader pair plus the helper that
 * turns it into a QRhiGraphicsPipeline.
 *
 * The shader sources live next to this file under shaders/ and are baked to
 * .qsb by the qsb tool at build time (see cpp/solvcon/CMakeLists.txt). They
 * are embedded in the Qt resource system and loaded by resource path.
 *
 * @ingroup group_domain
 */
class RMaterial
{

public:

    /**
     * @brief The shader variant. More variants are added as the renderer
     * grows.
     *
     * @ingroup group_domain
     */
    enum class Kind
    {
        FlatColor, ///< One uniform color for the whole primitive.
        VertexColor, ///< A per-vertex color attribute.
        Textured, ///< A sampled texture tinted by the uniform color.
        Lit, ///< A per-vertex normal shaded by a directional light.
        ScalarColor, ///< A per-vertex scalar mapped through a LUT texture.
    };

    explicit RMaterial(Kind kind);

    Kind kind() const { return m_kind; }

    QShader const & vertexShader() const { return m_vert; }
    QShader const & fragmentShader() const { return m_frag; }

    bool isValid() const { return m_vert.isValid() && m_frag.isValid(); }

    /**
     * @brief Build a graphics pipeline for this material. Ownership of the
     * returned pipeline is transferred to the caller; nullptr on failure.
     */
    QRhiGraphicsPipeline * buildPipeline(
        QRhi * rhi,
        QRhiShaderResourceBindings * srb,
        QRhiRenderPassDescriptor * rpdesc,
        QRhiVertexInputLayout const & input_layout,
        QRhiGraphicsPipeline::Topology topology,
        int sample_count,
        bool depth_test = true,
        bool alpha_blend = false,
        int depth_bias = 0,
        float slope_scaled_depth_bias = 0.0f) const;

    /// Load a baked shader from the Qt resource system.
    static QShader loadShader(QString const & resource_path);

private:

    Kind m_kind;
    QShader m_vert;
    QShader m_frag;

}; /* end class RMaterial */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

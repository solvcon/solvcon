/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/visual/RMaterial.hpp> // Must be the first include.

#include <QFile>

namespace solvcon
{

QShader RMaterial::loadShader(QString const & resource_path)
{
    QFile file(resource_path);
    if (!file.open(QIODevice::ReadOnly))
    {
        qWarning("RMaterial: failed to open shader %s", qPrintable(resource_path));
        return QShader();
    }
    QByteArray const data = file.readAll();
    return QShader::fromSerialized(data);
}

RMaterial::RMaterial(Kind kind)
    : m_kind(kind)
{
    switch (kind)
    {
    case Kind::FlatColor:
        m_vert = loadShader(QStringLiteral(":/solvcon/pilot/shaders/color.vert.qsb"));
        m_frag = loadShader(QStringLiteral(":/solvcon/pilot/shaders/color.frag.qsb"));
        break;
    case Kind::VertexColor:
        // The fragment stage just passes the interpolated color through, so it
        // is shared with the flat-color variant.
        m_vert = loadShader(QStringLiteral(":/solvcon/pilot/shaders/vcolor.vert.qsb"));
        m_frag = loadShader(QStringLiteral(":/solvcon/pilot/shaders/color.frag.qsb"));
        break;
    case Kind::Textured:
        m_vert = loadShader(QStringLiteral(":/solvcon/pilot/shaders/texture.vert.qsb"));
        m_frag = loadShader(QStringLiteral(":/solvcon/pilot/shaders/texture.frag.qsb"));
        break;
    case Kind::Lit:
        m_vert = loadShader(QStringLiteral(":/solvcon/pilot/shaders/lit.vert.qsb"));
        m_frag = loadShader(QStringLiteral(":/solvcon/pilot/shaders/lit.frag.qsb"));
        break;
    case Kind::ScalarColor:
        m_vert = loadShader(QStringLiteral(":/solvcon/pilot/shaders/scalar.vert.qsb"));
        m_frag = loadShader(QStringLiteral(":/solvcon/pilot/shaders/scalar.frag.qsb"));
        break;
    }
}

QRhiGraphicsPipeline * RMaterial::buildPipeline(
    QRhi * rhi,
    QRhiShaderResourceBindings * srb,
    QRhiRenderPassDescriptor * rpdesc,
    QRhiVertexInputLayout const & input_layout,
    QRhiGraphicsPipeline::Topology topology,
    int sample_count,
    bool depth_test,
    bool alpha_blend,
    int depth_bias,
    float slope_scaled_depth_bias) const
{
    QRhiGraphicsPipeline * pipeline = rhi->newGraphicsPipeline();
    pipeline->setShaderStages({
        {QRhiShaderStage::Vertex, m_vert},
        {QRhiShaderStage::Fragment, m_frag},
    });
    pipeline->setVertexInputLayout(input_layout);
    pipeline->setShaderResourceBindings(srb);
    pipeline->setRenderPassDescriptor(rpdesc);
    pipeline->setTopology(topology);
    pipeline->setSampleCount(sample_count);
    pipeline->setDepthTest(depth_test);
    pipeline->setDepthWrite(depth_test);
    if (0 != depth_bias || 0.0f != slope_scaled_depth_bias)
    {
        // Push these fragments away from the eye so coplanar geometry drawn
        // later (a wireframe or points over a surface) wins the depth test.
        pipeline->setDepthBias(depth_bias);
        pipeline->setSlopeScaledDepthBias(slope_scaled_depth_bias);
    }
    if (alpha_blend)
    {
        QRhiGraphicsPipeline::TargetBlend blend;
        blend.enable = true;
        blend.srcColor = QRhiGraphicsPipeline::SrcAlpha;
        blend.dstColor = QRhiGraphicsPipeline::OneMinusSrcAlpha;
        blend.srcAlpha = QRhiGraphicsPipeline::One;
        blend.dstAlpha = QRhiGraphicsPipeline::OneMinusSrcAlpha;
        pipeline->setTargetBlends({blend});
    }
    if (!pipeline->create())
    {
        delete pipeline;
        return nullptr;
    }
    return pipeline;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

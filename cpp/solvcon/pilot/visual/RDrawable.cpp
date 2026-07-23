/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/visual/RDrawable.hpp> // Must be the first include.

namespace solvcon
{

RDrawable::~RDrawable() = default;

void RDrawable::prepare(
    QRhi * rhi,
    QRhiRenderPassDescriptor * rpdesc,
    int sample_count,
    QRhiResourceUpdateBatch * batch)
{
    if (m_ready)
    {
        return;
    }

    createGeometry(rhi, batch);
    if (!m_vbuf || 0 == m_vertex_count)
    {
        // Nothing to draw (e.g. an empty geometry); stay unprepared so a
        // later update with real data can build the pipeline.
        return;
    }

    m_ubuf.reset(rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, UBUF_SIZE));
    m_ubuf->create();

    std::vector<QRhiShaderResourceBinding> bindings;
    bindings.push_back(QRhiShaderResourceBinding::uniformBuffer(
        0,
        QRhiShaderResourceBinding::VertexStage | QRhiShaderResourceBinding::FragmentStage,
        m_ubuf.get()));
    extendBindings(rhi, batch, bindings);

    m_srb.reset(rhi->newShaderResourceBindings());
    m_srb->setBindings(bindings.begin(), bindings.end());
    m_srb->create();

    m_material = std::make_unique<RMaterial>(materialKind());
    m_pipeline.reset(m_material->buildPipeline(
        rhi, m_srb.get(), rpdesc, vertexInputLayout(), topology(), sample_count,
        /*depth_test*/ true,
        /*alpha_blend*/ true,
        depthBias(),
        slopeScaledDepthBias()));
    m_ready = (nullptr != m_pipeline);
}

void RDrawable::extendBindings(QRhi *, QRhiResourceUpdateBatch *, std::vector<QRhiShaderResourceBinding> &)
{
}

void RDrawable::release()
{
    m_pipeline.reset();
    m_srb.reset();
    m_ubuf.reset();
    m_ibuf.reset();
    m_vbuf.reset();
    m_material.reset();
    m_vertex_count = 0;
    m_index_count = 0;
    m_ready = false;
}

void RDrawable::updateUniform(QRhiResourceUpdateBatch * batch, QMatrix4x4 const & view_proj)
{
    if (!m_ready)
    {
        return;
    }
    QMatrix4x4 const mvp = view_proj * m_model;
    batch->updateDynamicBuffer(m_ubuf.get(), 0, 64, mvp.constData());
    float const color[4] = {
        m_color.x(), m_color.y(), m_color.z(), m_color.w() * m_opacity};
    batch->updateDynamicBuffer(m_ubuf.get(), 64, 16, color);
    float const light[4] = {m_light_dir.x(), m_light_dir.y(), m_light_dir.z(), 0.0f};
    batch->updateDynamicBuffer(m_ubuf.get(), 80, 16, light);
}

void RDrawable::draw(QRhiCommandBuffer * cb)
{
    if (!m_visible || !m_ready)
    {
        return;
    }
    cb->setGraphicsPipeline(m_pipeline.get());
    cb->setShaderResources();
    QRhiCommandBuffer::VertexInput const vbuf_binding(m_vbuf.get(), 0);
    if (m_ibuf && m_index_count > 0)
    {
        cb->setVertexInput(0, 1, &vbuf_binding, m_ibuf.get(), 0, QRhiCommandBuffer::IndexUInt32);
        cb->drawIndexed(m_index_count);
    }
    else
    {
        cb->setVertexInput(0, 1, &vbuf_binding);
        cb->draw(m_vertex_count);
    }
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

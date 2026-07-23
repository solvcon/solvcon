/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/visual/RAxisGizmo.hpp> // Must be the first include.

#include <QColor>
#include <QFont>
#include <QPainter>

#include <algorithm>
#include <array>
#include <cmath>

namespace solvcon
{

namespace
{

constexpr float SHAFT_LENGTH = 0.68f;
constexpr float TIP_LENGTH = 1.0f;
constexpr float CONE_RADIUS = 0.09f;
constexpr int CONE_SEGMENTS = 12;
constexpr int CONE_VERTS_PER_AXIS = CONE_SEGMENTS * 6; // sides + base cap
constexpr float LABEL_DISTANCE = 1.22f;
constexpr float LABEL_HALF = 0.18f;
constexpr float ORTHO_HALF = 1.5f;
constexpr float EYE_DISTANCE = 3.0f;
constexpr int LABEL_TEXTURE_SIZE = 64;

std::array<QVector3D, 3> const AXIS_DIRECTIONS{{
    QVector3D(1.0f, 0.0f, 0.0f),
    QVector3D(0.0f, 1.0f, 0.0f),
    QVector3D(0.0f, 0.0f, 1.0f),
}};

std::array<QVector3D, 3> const AXIS_COLORS{{
    QVector3D(1.0f, 0.30f, 0.30f), // X red
    QVector3D(0.30f, 0.90f, 0.30f), // Y green
    QVector3D(0.40f, 0.55f, 1.0f), // Z blue
}};

std::array<QString, 3> const AXIS_LABELS{{QStringLiteral("X"), QStringLiteral("Y"), QStringLiteral("Z")}};

// An orthonormal vector perpendicular to @p dir.
QVector3D perpendicular(QVector3D const & dir)
{
    QVector3D const seed = (std::abs(dir.x()) < 0.9f) ? QVector3D(1.0f, 0.0f, 0.0f) : QVector3D(0.0f, 1.0f, 0.0f);
    return QVector3D::crossProduct(dir, seed).normalized();
}

} /* end namespace */

RAxisGizmo::RAxisGizmo()
{
    buildGeometry();
}

RAxisGizmo::~RAxisGizmo() = default;

void RAxisGizmo::buildGeometry()
{
    auto push = [](SimpleCollector<float> & out, QVector3D const & p, QVector3D const & c)
    {
        out.push_back(p.x());
        out.push_back(p.y());
        out.push_back(p.z());
        out.push_back(c.x());
        out.push_back(c.y());
        out.push_back(c.z());
    };

    for (int axis = 0; axis < 3; ++axis)
    {
        QVector3D const dir = AXIS_DIRECTIONS[axis];
        QVector3D const color = AXIS_COLORS[axis];

        // Shaft: a line from the origin to the cone base.
        push(m_shaft_vertices, QVector3D(0.0f, 0.0f, 0.0f), color);
        push(m_shaft_vertices, dir * SHAFT_LENGTH, color);

        // Cone arrowhead: a fan of side triangles plus a base cap.
        QVector3D const apex = dir * TIP_LENGTH;
        QVector3D const base_center = dir * SHAFT_LENGTH;
        QVector3D const u = perpendicular(dir);
        QVector3D const v = QVector3D::crossProduct(dir, u).normalized();
        for (int seg = 0; seg < CONE_SEGMENTS; ++seg)
        {
            float const a0 = 2.0f * float(M_PI) * static_cast<float>(seg) / CONE_SEGMENTS;
            float const a1 = 2.0f * float(M_PI) * static_cast<float>(seg + 1) / CONE_SEGMENTS;
            QVector3D const p0 = base_center + CONE_RADIUS * (std::cos(a0) * u + std::sin(a0) * v);
            QVector3D const p1 = base_center + CONE_RADIUS * (std::cos(a1) * u + std::sin(a1) * v);
            // Side triangle.
            push(m_cone_vertices, apex, color);
            push(m_cone_vertices, p0, color);
            push(m_cone_vertices, p1, color);
            // Base cap triangle (reverse winding).
            push(m_cone_vertices, base_center, color);
            push(m_cone_vertices, p1, color);
            push(m_cone_vertices, p0, color);
        }
    }
}

QImage RAxisGizmo::makeLabelImage(QString const & text, QColor const & color)
{
    QImage image(LABEL_TEXTURE_SIZE, LABEL_TEXTURE_SIZE, QImage::Format_RGBA8888);
    image.fill(Qt::transparent);
    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing, true);
    QFont font;
    font.setPixelSize(LABEL_TEXTURE_SIZE - 16);
    font.setBold(true);
    painter.setFont(font);
    painter.setPen(color);
    painter.drawText(image.rect(), Qt::AlignCenter, text);
    painter.end();
    return image;
}

void RAxisGizmo::prepare(
    QRhi * rhi, QRhiRenderPassDescriptor * rpdesc, int sample_count, QRhiResourceUpdateBatch * batch)
{
    quint32 const shaft_bytes = static_cast<quint32>(m_shaft_vertices.size() * sizeof(float));
    m_shaft_vbuf.reset(rhi->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::VertexBuffer, shaft_bytes));
    m_shaft_vbuf->create();
    batch->uploadStaticBuffer(m_shaft_vbuf.get(), m_shaft_vertices.data());

    quint32 const cone_bytes = static_cast<quint32>(m_cone_vertices.size() * sizeof(float));
    m_cone_vbuf.reset(rhi->newBuffer(QRhiBuffer::Immutable, QRhiBuffer::VertexBuffer, cone_bytes));
    m_cone_vbuf->create();
    batch->uploadStaticBuffer(m_cone_vbuf.get(), m_cone_vertices.data());

    // Dynamic billboard quads: 3 labels * 6 vertices * (pos3 + uv2).
    m_label_vbuf.reset(rhi->newBuffer(
        QRhiBuffer::Dynamic, QRhiBuffer::VertexBuffer, 3 * 6 * 5 * sizeof(float)));
    m_label_vbuf->create();

    m_ubuf.reset(rhi->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, 64 + 16));
    m_ubuf->create();

    m_srb.reset(rhi->newShaderResourceBindings());
    m_srb->setBindings({
        QRhiShaderResourceBinding::uniformBuffer(
            0,
            QRhiShaderResourceBinding::VertexStage | QRhiShaderResourceBinding::FragmentStage,
            m_ubuf.get()),
    });
    m_srb->create();

    m_sampler.reset(rhi->newSampler(
        QRhiSampler::Linear,
        QRhiSampler::Linear,
        QRhiSampler::None,
        QRhiSampler::ClampToEdge,
        QRhiSampler::ClampToEdge));
    m_sampler->create();

    for (int i = 0; i < 3; ++i)
    {
        QImage const image = makeLabelImage(AXIS_LABELS[i], QColor::fromRgbF(AXIS_COLORS[i].x(), AXIS_COLORS[i].y(), AXIS_COLORS[i].z()));
        m_label_textures[i].reset(rhi->newTexture(
            QRhiTexture::RGBA8, QSize(LABEL_TEXTURE_SIZE, LABEL_TEXTURE_SIZE)));
        m_label_textures[i]->create();
        batch->uploadTexture(m_label_textures[i].get(), image);

        m_label_srb[i].reset(rhi->newShaderResourceBindings());
        m_label_srb[i]->setBindings({
            QRhiShaderResourceBinding::uniformBuffer(
                0,
                QRhiShaderResourceBinding::VertexStage | QRhiShaderResourceBinding::FragmentStage,
                m_ubuf.get()),
            QRhiShaderResourceBinding::sampledTexture(
                1, QRhiShaderResourceBinding::FragmentStage, m_label_textures[i].get(), m_sampler.get()),
        });
        m_label_srb[i]->create();
    }

    m_vcolor_material = std::make_unique<RMaterial>(RMaterial::Kind::VertexColor);
    m_texture_material = std::make_unique<RMaterial>(RMaterial::Kind::Textured);

    QRhiVertexInputLayout color_layout;
    color_layout.setBindings({{6 * sizeof(float)}});
    color_layout.setAttributes({
        {0, 0, QRhiVertexInputAttribute::Float3, 0},
        {0, 1, QRhiVertexInputAttribute::Float3, 3 * sizeof(float)},
    });

    QRhiVertexInputLayout texture_layout;
    texture_layout.setBindings({{5 * sizeof(float)}});
    texture_layout.setAttributes({
        {0, 0, QRhiVertexInputAttribute::Float3, 0},
        {0, 1, QRhiVertexInputAttribute::Float2, 3 * sizeof(float)},
    });

    // The guide draws over the scene, so the depth test is off.
    m_shaft_pipeline.reset(m_vcolor_material->buildPipeline(
        rhi, m_srb.get(), rpdesc, color_layout, QRhiGraphicsPipeline::Lines, sample_count, false, false));
    m_cone_pipeline.reset(m_vcolor_material->buildPipeline(
        rhi, m_srb.get(), rpdesc, color_layout, QRhiGraphicsPipeline::Triangles, sample_count, false, false));
    m_label_pipeline.reset(m_texture_material->buildPipeline(
        rhi, m_label_srb[0].get(), rpdesc, texture_layout, QRhiGraphicsPipeline::Triangles, sample_count, false, true));

    m_ready = (nullptr != m_shaft_pipeline && nullptr != m_cone_pipeline && nullptr != m_label_pipeline);
}

void RAxisGizmo::update(
    QRhi * rhi,
    QRhiRenderPassDescriptor * rpdesc,
    int sample_count,
    QSize pixel_size,
    QVector3D const & forward,
    QVector3D const & up,
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

    // A square corner viewport in the lower-left, sized to the widget.
    float const side = std::clamp(
        0.28f * static_cast<float>(std::min(pixel_size.width(), pixel_size.height())), 80.0f, 160.0f);
    constexpr float margin = 10.0f;
    m_viewport = QRhiViewport(margin, margin, side, side);

    QVector3D fwd = forward;
    if (fwd.lengthSquared() <= 0.0f)
    {
        fwd = QVector3D(0.0f, 0.0f, -1.0f);
    }
    fwd.normalize();
    QVector3D camera_up = up;
    if (camera_up.lengthSquared() <= 0.0f)
    {
        camera_up = QVector3D(0.0f, 1.0f, 0.0f);
    }

    // Mirror the main camera orientation: look at the origin from along the
    // main view direction.
    QMatrix4x4 view;
    view.lookAt(-fwd * EYE_DISTANCE, QVector3D(0.0f, 0.0f, 0.0f), camera_up);
    QMatrix4x4 proj;
    proj.ortho(-ORTHO_HALF, ORTHO_HALF, -ORTHO_HALF, ORTHO_HALF, 0.1f, 2.0f * EYE_DISTANCE);
    QMatrix4x4 const mvp = rhi->clipSpaceCorrMatrix() * proj * view;

    batch->updateDynamicBuffer(m_ubuf.get(), 0, 64, mvp.constData());
    float const white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    batch->updateDynamicBuffer(m_ubuf.get(), 64, 16, white);

    // Camera-facing billboard quads for the labels, rebuilt for this view.
    QVector3D const right = QVector3D::crossProduct(fwd, camera_up).normalized();
    QVector3D const screen_up = QVector3D::crossProduct(right, fwd).normalized();
    std::array<float, 3 * 6 * 5> label_data{};
    size_t at = 0;
    auto put = [&label_data, &at](QVector3D const & p, float s, float t)
    {
        label_data[at++] = p.x();
        label_data[at++] = p.y();
        label_data[at++] = p.z();
        label_data[at++] = s;
        label_data[at++] = t;
    };
    for (int axis = 0; axis < 3; ++axis)
    {
        QVector3D const center = AXIS_DIRECTIONS[axis] * LABEL_DISTANCE;
        QVector3D const bl = center - right * LABEL_HALF - screen_up * LABEL_HALF;
        QVector3D const br = center + right * LABEL_HALF - screen_up * LABEL_HALF;
        QVector3D const tr = center + right * LABEL_HALF + screen_up * LABEL_HALF;
        QVector3D const tl = center - right * LABEL_HALF + screen_up * LABEL_HALF;
        put(bl, 0.0f, 1.0f);
        put(br, 1.0f, 1.0f);
        put(tr, 1.0f, 0.0f);
        put(bl, 0.0f, 1.0f);
        put(tr, 1.0f, 0.0f);
        put(tl, 0.0f, 0.0f);
    }
    batch->updateDynamicBuffer(
        m_label_vbuf.get(), 0, static_cast<quint32>(label_data.size() * sizeof(float)), label_data.data());

    m_drawable = true;
}

void RAxisGizmo::draw(QRhiCommandBuffer * cb)
{
    if (!m_visible || !m_drawable)
    {
        return;
    }

    cb->setViewport(m_viewport);

    quint32 const shaft_count = static_cast<quint32>(m_axis_count) * 2;
    quint32 const cone_count = static_cast<quint32>(m_axis_count) * CONE_VERTS_PER_AXIS;

    QRhiCommandBuffer::VertexInput const shaft_in(m_shaft_vbuf.get(), 0);
    cb->setGraphicsPipeline(m_shaft_pipeline.get());
    cb->setShaderResources(m_srb.get());
    cb->setVertexInput(0, 1, &shaft_in);
    cb->draw(shaft_count);

    QRhiCommandBuffer::VertexInput const cone_in(m_cone_vbuf.get(), 0);
    cb->setGraphicsPipeline(m_cone_pipeline.get());
    cb->setShaderResources(m_srb.get());
    cb->setVertexInput(0, 1, &cone_in);
    cb->draw(cone_count);

    cb->setGraphicsPipeline(m_label_pipeline.get());
    for (int axis = 0; axis < m_axis_count; ++axis)
    {
        quint32 const offset = static_cast<quint32>(axis) * 6 * 5 * sizeof(float);
        QRhiCommandBuffer::VertexInput const label_in(m_label_vbuf.get(), offset);
        cb->setShaderResources(m_label_srb[axis].get());
        cb->setVertexInput(0, 1, &label_in);
        cb->draw(6);
    }
}

void RAxisGizmo::release()
{
    m_shaft_pipeline.reset();
    m_cone_pipeline.reset();
    m_label_pipeline.reset();
    m_srb.reset();
    for (auto & srb : m_label_srb)
    {
        srb.reset();
    }
    for (auto & tex : m_label_textures)
    {
        tex.reset();
    }
    m_sampler.reset();
    m_shaft_vbuf.reset();
    m_cone_vbuf.reset();
    m_label_vbuf.reset();
    m_ubuf.reset();
    m_vcolor_material.reset();
    m_texture_material.reset();
    m_ready = false;
    m_drawable = false;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

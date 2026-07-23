/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/visual/RDomainWidget.hpp> // Must be the first include.

#include <solvcon/pilot/visual/RBoundary.hpp>
#include <solvcon/pilot/visual/RFeatureEdges.hpp>
#include <solvcon/pilot/visual/RField.hpp>
#include <solvcon/pilot/visual/RMeshFrame.hpp>
#include <solvcon/pilot/visual/RNormals.hpp>
#include <solvcon/pilot/visual/RScalarField.hpp>
#include <solvcon/pilot/visual/RSegments.hpp>

#include <QGestureEvent>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QNativeGestureEvent>
#include <QPinchGesture>
#include <QVector4D>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace solvcon
{

namespace
{

constexpr float PI = 3.14159265358979323846f;

/// Per-cell value of the named geometric quality metric, over the SimpleArray
/// mesh path. The angle-based metrics read the cell node polygon, so they are
/// meaningful for 2D cells; "volume" and "aspect_ratio" work in any dimension.
/// Throws std::invalid_argument for an unknown metric name.
std::vector<float> cell_quality(StaticMesh const & mh, std::string const & metric)
{
    static char const * const known[] = {
        "volume", "aspect_ratio", "skewness", "min_angle", "max_angle"};
    if (std::none_of(std::begin(known), std::end(known), [&metric](char const * n)
                     { return metric == n; }))
    {
        throw std::invalid_argument(
            "RDomainWidget: unknown quality metric \"" + metric +
            "\"; pick one of volume, aspect_ratio, skewness, min_angle, "
            "max_angle");
    }

    uint32_t const ndim = mh.ndim();
    uint32_t const ncell = mh.ncell();
    auto coord = [&mh, ndim](int32_t ind, uint32_t d) -> float
    { return (d < ndim) ? static_cast<float>(mh.ndcrd(ind, d)) : 0.0f; };

    std::vector<float> out(ncell, 0.0f);
    for (uint32_t icl = 0; icl < ncell; ++icl)
    {
        if ("volume" == metric)
        {
            out[icl] = static_cast<float>(mh.clvol(icl));
            continue;
        }

        int32_t const nnd = mh.clnds(icl, 0);
        std::vector<QVector3D> p(static_cast<size_t>(nnd));
        for (int32_t k = 0; k < nnd; ++k)
        {
            int32_t const ind = mh.clnds(icl, k + 1);
            p[k] = QVector3D(coord(ind, 0), coord(ind, 1), coord(ind, 2));
        }

        if ("aspect_ratio" == metric)
        {
            float dmin = std::numeric_limits<float>::max();
            float dmax = 0.0f;
            for (int32_t a = 0; a < nnd; ++a)
            {
                for (int32_t b = a + 1; b < nnd; ++b)
                {
                    float const d = (p[a] - p[b]).length();
                    dmin = std::min(dmin, d);
                    dmax = std::max(dmax, d);
                }
            }
            out[icl] = (dmin > 0.0f) ? dmax / dmin : 0.0f;
            continue;
        }

        // The remaining metrics read the interior angles of the node polygon.
        float amin = 180.0f;
        float amax = 0.0f;
        for (int32_t k = 0; k < nnd; ++k)
        {
            QVector3D const a = p[(k + nnd - 1) % nnd] - p[k];
            QVector3D const b = p[(k + 1) % nnd] - p[k];
            float const la = a.length();
            float const lb = b.length();
            if (la <= 0.0f || lb <= 0.0f)
            {
                continue;
            }
            float const c = std::clamp(
                QVector3D::dotProduct(a, b) / (la * lb), -1.0f, 1.0f);
            float const ang = std::acos(c) * 180.0f / PI;
            amin = std::min(amin, ang);
            amax = std::max(amax, ang);
        }

        if ("min_angle" == metric)
        {
            out[icl] = amin;
        }
        else if ("max_angle" == metric)
        {
            out[icl] = amax;
        }
        else // skewness: the equiangle skew against the regular-polygon angle.
        {
            float const te = 180.0f * static_cast<float>(nnd - 2) / static_cast<float>(nnd);
            float const skew = std::max(
                (amax - te) / std::max(180.0f - te, 1.0e-3f),
                (te - amin) / std::max(te, 1.0e-3f));
            out[icl] = std::clamp(skew, 0.0f, 1.0f);
        }
    }
    return out;
}

} /* end namespace */

RDomainWidget::RDomainWidget(QWidget * parent)
    : QRhiWidget(parent)
{
    // Accept keyboard focus so the first-person movement keys reach the
    // widget, and track the mouse for drag-based navigation.
    setFocusPolicy(Qt::StrongFocus);
    // Receive touchscreen pinch gestures (trackpad pinches arrive as native
    // gesture events, which need no grab).
    grabGesture(Qt::PinchGesture);
}

float RDomainWidget::viewportAspect() const
{
    int const h = height();
    return (h > 0) ? static_cast<float>(width()) / static_cast<float>(h) : 1.0f;
}

RDomainWidget::~RDomainWidget() = default;

QImage RDomainWidget::grabImage()
{
    return grabFramebuffer();
}

QImage RDomainWidget::renderToImage(int width, int height, bool transparent)
{
    if (width <= 0 || height <= 0)
    {
        return QImage();
    }
    // Render at the requested size by resizing the (possibly hidden) widget,
    // grabbing, and restoring; the transparent flag clears the frame's alpha.
    QSize const old_size = size();
    bool const old_transparent = m_transparent_capture;
    m_transparent_capture = transparent;
    resize(width, height);
    QImage image = grabFramebuffer();
    resize(old_size.width(), old_size.height());
    m_transparent_capture = old_transparent;

    // grabFramebuffer() returns physical pixels (logical size scaled by the
    // screen's device pixel ratio), so on a HiDPI display the grab is larger
    // than requested. Normalize to the requested pixel dimensions.
    if (image.size() != QSize(width, height))
    {
        image = image.scaled(
            width, height, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    }
    image.setDevicePixelRatio(1.0);
    return image;
}

void RDomainWidget::updateMesh(std::shared_ptr<StaticMesh> const & mesh)
{
    // Drop the previous mesh drawables and replace them; a new mesh redefines
    // the framing, so the bounding box is recomputed from scratch.
    m_scene.removeDrawable(m_mesh_surface);
    m_scene.removeDrawable(m_mesh_frame);
    m_scene.removeDrawable(m_mesh_points);
    m_mesh_surface = nullptr;
    m_mesh_frame = nullptr;
    m_mesh_points = nullptr;

    m_mesh = mesh;

    // Build one drawable per representation and switch between them by
    // visibility; rebuilding on every toggle would be wasteful.
    auto surface = std::make_unique<RMeshFrame>(mesh, RMeshFrame::Style::Surface);
    m_mesh_surface = surface.get();
    m_scene.addDrawable(std::move(surface));
    auto frame = std::make_unique<RMeshFrame>(mesh, RMeshFrame::Style::Wireframe);
    m_mesh_frame = frame.get();
    m_scene.addDrawable(std::move(frame));
    auto points = std::make_unique<RMeshFrame>(mesh, RMeshFrame::Style::Points);
    m_mesh_points = points.get();
    m_scene.addDrawable(std::move(points));
    applyMeshVisibility();

    StaticMesh const & mh = *mesh;
    m_scene.setDimension(mh.ndim());
    // For now, limit 2D domain to the in-plane pan/zoom whose wheel scales the
    // orthographic box.
    if (2 == mh.ndim())
    {
        m_scene.camera().setMode(RCameraController::Mode::PanZoom);
    }
    QVector3D lo(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max());
    QVector3D hi(
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest());
    bool const is_3d = (3 == mh.ndim());
    for (uint32_t ind = 0; ind < mh.nnode(); ++ind)
    {
        float const x = static_cast<float>(mh.ndcrd(ind, 0));
        float const y = static_cast<float>(mh.ndcrd(ind, 1));
        float const z = is_3d ? static_cast<float>(mh.ndcrd(ind, 2)) : 0.0f;
        lo = QVector3D(std::min(lo.x(), x), std::min(lo.y(), y), std::min(lo.z(), z));
        hi = QVector3D(std::max(hi.x(), x), std::max(hi.y(), y), std::max(hi.z(), z));
    }
    m_scene.resetBoundingBox();
    if (mh.nnode() > 0)
    {
        m_scene.extendBoundingBox(lo, hi);
    }
    // A field set earlier still draws, so keep it inside the framed box.
    if (m_has_field_bbox)
    {
        m_scene.extendBoundingBox(m_field_lo, m_field_hi);
    }

    m_scene.fitCameraToScene(viewportAspect());
    update();
}

namespace
{

/// The axis-aligned node bounds of a mesh.
void mesh_bounds(StaticMesh const & mh, QVector3D & lo, QVector3D & hi)
{
    bool const is_3d = (3 == mh.ndim());
    lo = QVector3D(
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max());
    hi = QVector3D(
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest());
    for (uint32_t ind = 0; ind < mh.nnode(); ++ind)
    {
        float const x = static_cast<float>(mh.ndcrd(ind, 0));
        float const y = static_cast<float>(mh.ndcrd(ind, 1));
        float const z = is_3d ? static_cast<float>(mh.ndcrd(ind, 2)) : 0.0f;
        lo = QVector3D(std::min(lo.x(), x), std::min(lo.y(), y), std::min(lo.z(), z));
        hi = QVector3D(std::max(hi.x(), x), std::max(hi.y(), y), std::max(hi.z(), z));
    }
}

} /* end namespace */

void RDomainWidget::addObject(
    std::string const & name, std::shared_ptr<StaticMesh> const & mesh)
{
    auto const it = m_objects.find(name);
    if (it != m_objects.end())
    {
        m_scene.removeDrawable(it->second.drawable);
        m_objects.erase(it);
    }

    auto surface = std::make_unique<RMeshFrame>(mesh, RMeshFrame::Style::Surface);
    surface->setName(name);
    RDrawable * const drawable = surface.get();
    m_scene.addDrawable(std::move(surface));
    m_objects[name] = ObjectEntry{drawable, mesh};

    m_scene.setDimension(mesh->ndim());
    QVector3D lo;
    QVector3D hi;
    mesh_bounds(*mesh, lo, hi);
    if (mesh->nnode() > 0)
    {
        m_scene.extendBoundingBox(lo, hi);
    }
    m_scene.fitCameraToScene(viewportAspect());
    update();
}

void RDomainWidget::setObjectTransform(
    std::string const & name,
    float tx,
    float ty,
    float tz,
    float sx,
    float sy,
    float sz)
{
    auto const it = m_objects.find(name);
    if (it == m_objects.end())
    {
        return;
    }
    QMatrix4x4 model;
    model.translate(tx, ty, tz);
    model.scale(sx, sy, sz);
    it->second.drawable->setModel(model);

    // Grow the framing box to include the transformed object corners.
    QVector3D lo;
    QVector3D hi;
    mesh_bounds(*it->second.mesh, lo, hi);
    for (int i = 0; i < 8; ++i)
    {
        QVector3D const corner(
            (i & 1) ? hi.x() : lo.x(),
            (i & 2) ? hi.y() : lo.y(),
            (i & 4) ? hi.z() : lo.z());
        QVector3D const mapped = model.map(corner);
        m_scene.extendBoundingBox(mapped, mapped);
    }
    m_scene.fitCameraToScene(viewportAspect());
    update();
}

void RDomainWidget::setObjectVisible(std::string const & name, bool visible)
{
    auto const it = m_objects.find(name);
    if (it != m_objects.end())
    {
        it->second.drawable->setVisible(visible);
        update();
    }
}

void RDomainWidget::setObjectOpacity(std::string const & name, float opacity)
{
    auto const it = m_objects.find(name);
    if (it != m_objects.end())
    {
        it->second.drawable->setOpacity(opacity);
        update();
    }
}

std::vector<std::string> RDomainWidget::objectNames() const
{
    std::vector<std::string> names;
    names.reserve(m_objects.size());
    for (auto const & entry : m_objects)
    {
        names.push_back(entry.first);
    }
    return names;
}

void RDomainWidget::showMesh(bool show)
{
    m_mesh_shown = show;
    applyMeshVisibility();
    update();
}

void RDomainWidget::showMeshStyle(std::string const & name, bool show)
{
    if ("surface" == name)
    {
        m_show_surface = show;
    }
    else if ("wireframe" == name)
    {
        m_show_wireframe = show;
    }
    else if ("points" == name)
    {
        m_show_points = show;
    }
    else
    {
        return; // Ignore an unknown name.
    }
    applyMeshVisibility();
    update();
}

bool RDomainWidget::meshStyleShown(std::string const & name) const
{
    if ("surface" == name)
    {
        return m_show_surface;
    }
    if ("wireframe" == name)
    {
        return m_show_wireframe;
    }
    if ("points" == name)
    {
        return m_show_points;
    }
    return false;
}

void RDomainWidget::applyMeshVisibility()
{
    if (nullptr != m_mesh_surface)
    {
        m_mesh_surface->setVisible(m_mesh_shown && m_show_surface);
    }
    if (nullptr != m_mesh_frame)
    {
        m_mesh_frame->setVisible(m_mesh_shown && m_show_wireframe);
    }
    if (nullptr != m_mesh_points)
    {
        m_mesh_points->setVisible(m_mesh_shown && m_show_points);
    }
}

void RDomainWidget::setMeshOpacity(float opacity)
{
    if (nullptr != m_mesh_frame)
    {
        m_mesh_frame->setOpacity(opacity);
        update();
    }
}

void RDomainWidget::setFieldOpacity(float opacity)
{
    if (nullptr != m_field)
    {
        m_field->setOpacity(opacity);
        update();
    }
}

void RDomainWidget::updateColorField(
    SimpleArray<float> const & vertices,
    SimpleArray<float> const & colors,
    SimpleArray<uint32_t> const & indices)
{
    auto field = std::make_unique<RField>(vertices, colors, indices);
    installField(std::move(field));
}

void RDomainWidget::updateScalarField(
    SimpleArray<float> const & vertices,
    SimpleArray<float> const & scalars,
    SimpleArray<uint32_t> const & indices)
{
    auto field = std::make_unique<RScalarField>(vertices, scalars, indices, m_colormap);
    if (m_range_pinned)
    {
        field->setScalarRange(m_range_lo, m_range_hi);
    }
    m_scalar_bar.setRange(field->rangeLo(), field->rangeHi());
    RScalarField * scalar_field = field.get();
    installField(std::move(field));
    m_scalar_field = (m_field == scalar_field) ? scalar_field : nullptr;
}

void RDomainWidget::setColormap(std::string const & name)
{
    m_colormap = RColormap::named(name);
    if (nullptr != m_scalar_field)
    {
        m_scalar_field->setColormap(m_colormap);
    }
    m_scalar_bar.setColormap(m_colormap);
    update();
}

void RDomainWidget::setScalarRange(float lo, float hi)
{
    if (hi < lo)
    {
        throw std::invalid_argument("RDomainWidget: scalar range must have hi >= lo");
    }
    m_range_pinned = true;
    m_range_lo = lo;
    m_range_hi = hi;
    if (nullptr != m_scalar_field)
    {
        m_scalar_field->setScalarRange(lo, hi);
    }
    m_scalar_bar.setRange(lo, hi);
    update();
}

std::pair<float, float> RDomainWidget::scalarRange() const
{
    if (nullptr != m_scalar_field)
    {
        return {m_scalar_field->rangeLo(), m_scalar_field->rangeHi()};
    }
    return {m_range_lo, m_range_hi};
}

void RDomainWidget::showScalarBar(bool show)
{
    m_scalar_bar.setVisible(show);
    update();
}

void RDomainWidget::setScalarBarTitle(std::string const & title)
{
    m_scalar_bar.setTitle(title);
    update();
}

void RDomainWidget::colorByCellType()
{
    if (nullptr == m_mesh)
    {
        return;
    }
    StaticMesh const & mh = *m_mesh;
    std::vector<int32_t> category;
    if (3 == mh.ndim())
    {
        SimpleArray<int32_t> const & bndfcs = mh.bndfcs();
        category.reserve(bndfcs.shape(0));
        for (size_t ibnd = 0; ibnd < bndfcs.shape(0); ++ibnd)
        {
            category.push_back(mh.cltpn(mh.fcicl(bndfcs(ibnd, 0))));
        }
    }
    else
    {
        category.reserve(mh.ncell());
        for (uint32_t icl = 0; icl < mh.ncell(); ++icl)
        {
            category.push_back(mh.cltpn(icl));
        }
    }
    installCategoryField(category, "cell type");
}

void RDomainWidget::colorByCellGroup()
{
    if (nullptr == m_mesh)
    {
        return;
    }
    StaticMesh const & mh = *m_mesh;
    std::vector<int32_t> category;
    if (3 == mh.ndim())
    {
        SimpleArray<int32_t> const & bndfcs = mh.bndfcs();
        category.reserve(bndfcs.shape(0));
        for (size_t ibnd = 0; ibnd < bndfcs.shape(0); ++ibnd)
        {
            category.push_back(mh.clgrp(mh.fcicl(bndfcs(ibnd, 0))));
        }
    }
    else
    {
        category.reserve(mh.ncell());
        for (uint32_t icl = 0; icl < mh.ncell(); ++icl)
        {
            category.push_back(mh.clgrp(icl));
        }
    }
    installCategoryField(category, "cell group");
}

void RDomainWidget::colorByBoundary()
{
    if (nullptr == m_mesh)
    {
        return;
    }
    StaticMesh const & mh = *m_mesh;
    SimpleArray<int32_t> const & bndfcs = mh.bndfcs();
    std::vector<int32_t> category;
    if (3 == mh.ndim())
    {
        // The 3D surface is the boundary shell, one primitive per boundary
        // face, so the boundary set is the face's own set.
        category.reserve(bndfcs.shape(0));
        for (size_t ibnd = 0; ibnd < bndfcs.shape(0); ++ibnd)
        {
            category.push_back(bndfcs(ibnd, 1));
        }
    }
    else
    {
        // The 2D surface is the cells, so color each cell by a boundary set it
        // owns; interior cells share one extra "none" category past the top set.
        int32_t max_bc = -1;
        for (size_t ibnd = 0; ibnd < bndfcs.shape(0); ++ibnd)
        {
            max_bc = std::max(max_bc, bndfcs(ibnd, 1));
        }
        category.assign(mh.ncell(), max_bc + 1);
        for (size_t ibnd = 0; ibnd < bndfcs.shape(0); ++ibnd)
        {
            int32_t const icl = mh.fcicl(bndfcs(ibnd, 0));
            if (icl >= 0 && static_cast<size_t>(icl) < category.size())
            {
                category[icl] = bndfcs(ibnd, 1);
            }
        }
    }
    installCategoryField(category, "boundary set");
}

void RDomainWidget::clearCellColoring()
{
    m_scene.removeDrawable(m_field);
    m_field = nullptr;
    m_scalar_field = nullptr;
    m_has_field_bbox = false;
    m_range_pinned = false;
    m_scalar_bar.setVisible(false);
    update();
}

void RDomainWidget::colorByQuality(std::string const & metric)
{
    if (nullptr == m_mesh)
    {
        return;
    }
    StaticMesh const & mh = *m_mesh;
    std::vector<float> const cellval = cell_quality(mh, metric);

    // The 3D surface is the boundary shell, so a boundary face takes its owning
    // cell's metric; a 2D cell takes its own.
    std::vector<float> primitive;
    if (3 == mh.ndim())
    {
        SimpleArray<int32_t> const & bndfcs = mh.bndfcs();
        primitive.reserve(bndfcs.shape(0));
        for (size_t ibnd = 0; ibnd < bndfcs.shape(0); ++ibnd)
        {
            primitive.push_back(cellval[mh.fcicl(bndfcs(ibnd, 0))]);
        }
    }
    else
    {
        primitive = cellval;
    }
    installMetricField(primitive, metric);
}

std::pair<float, float> RDomainWidget::qualityRange(std::string const & metric) const
{
    if (nullptr == m_mesh)
    {
        return {0.0f, 0.0f};
    }
    std::vector<float> const v = cell_quality(*m_mesh, metric);
    if (v.empty())
    {
        return {0.0f, 0.0f};
    }
    auto const mm = std::minmax_element(v.begin(), v.end());
    return {*mm.first, *mm.second};
}

void RDomainWidget::collectSurfaceScalars(
    std::vector<float> const & primitive_scalar,
    std::vector<float> & verts,
    std::vector<float> & scals,
    std::vector<uint32_t> & tris) const
{
    StaticMesh const & mh = *m_mesh;
    uint32_t const ndim = mh.ndim();

    auto node_coord = [&mh, ndim](int32_t ind, uint32_t dim) -> float
    { return (dim < ndim) ? static_cast<float>(mh.ndcrd(ind, dim)) : 0.0f; };

    // Fan-triangulate one surface polygon, giving every emitted vertex the
    // primitive's scalar so the whole face reads one value.
    auto add_polygon = [&](auto node, int32_t nnd, float scalar)
    {
        for (int32_t k = 1; k + 1 < nnd; ++k)
        {
            int32_t const tri[3] = {node(0), node(k), node(k + 1)};
            uint32_t const base = static_cast<uint32_t>(verts.size() / 3);
            for (int32_t const ind : tri)
            {
                verts.push_back(node_coord(ind, 0));
                verts.push_back(node_coord(ind, 1));
                verts.push_back(node_coord(ind, 2));
                scals.push_back(scalar);
            }
            tris.push_back(base + 0);
            tris.push_back(base + 1);
            tris.push_back(base + 2);
        }
    };

    if (3 == ndim)
    {
        SimpleArray<int32_t> const & bndfcs = mh.bndfcs();
        size_t const nprim = std::min(static_cast<size_t>(bndfcs.shape(0)), primitive_scalar.size());
        for (size_t ibnd = 0; ibnd < nprim; ++ibnd)
        {
            int32_t const ifc = bndfcs(ibnd, 0);
            add_polygon(
                [&mh, ifc](int32_t k)
                { return mh.fcnds(ifc, k + 1); },
                mh.fcnds(ifc, 0),
                primitive_scalar[ibnd]);
        }
    }
    else
    {
        size_t const nprim = std::min(static_cast<size_t>(mh.ncell()), primitive_scalar.size());
        for (uint32_t icl = 0; icl < nprim; ++icl)
        {
            add_polygon(
                [&mh, icl](int32_t k)
                { return mh.clnds(icl, k + 1); },
                mh.clnds(icl, 0),
                primitive_scalar[icl]);
        }
    }
}

namespace
{

/// Pack the collected interleaved arrays into the (nvert, 3), (nvert,), and
/// (ntri, 3) SimpleArrays the scalar-field drawable expects.
void pack_scalar_arrays(
    std::vector<float> const & verts,
    std::vector<float> const & scals,
    std::vector<uint32_t> const & tris,
    SimpleArray<float> & va,
    SimpleArray<float> & sa,
    SimpleArray<uint32_t> & ia)
{
    size_t const nvert = verts.size() / 3;
    size_t const ntri = tris.size() / 3;
    va = SimpleArray<float>(small_vector<ssize_t>{static_cast<ssize_t>(nvert), 3});
    sa = SimpleArray<float>(small_vector<ssize_t>{static_cast<ssize_t>(nvert)});
    ia = SimpleArray<uint32_t>(small_vector<ssize_t>{static_cast<ssize_t>(ntri), 3});
    for (size_t i = 0; i < nvert; ++i)
    {
        va(i, 0) = verts[3 * i + 0];
        va(i, 1) = verts[3 * i + 1];
        va(i, 2) = verts[3 * i + 2];
        sa(i) = scals[i];
    }
    for (size_t t = 0; t < ntri; ++t)
    {
        ia(t, 0) = tris[3 * t + 0];
        ia(t, 1) = tris[3 * t + 1];
        ia(t, 2) = tris[3 * t + 2];
    }
}

} /* end namespace */

void RDomainWidget::installCategoryField(
    std::vector<int32_t> const & primitive_category, std::string const & title)
{
    if (nullptr == m_mesh)
    {
        return;
    }

    // Dense-index the distinct categories so the colors pack from the palette
    // start and the legend runs 0..ncat-1.
    std::vector<int32_t> distinct(primitive_category);
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    if (distinct.empty())
    {
        return;
    }
    int const ncat = static_cast<int>(distinct.size());

    std::vector<float> primitive_scalar;
    primitive_scalar.reserve(primitive_category.size());
    for (int32_t const v : primitive_category)
    {
        primitive_scalar.push_back(static_cast<float>(
            std::lower_bound(distinct.begin(), distinct.end(), v) - distinct.begin()));
    }

    std::vector<float> verts;
    std::vector<float> scals;
    std::vector<uint32_t> tris;
    collectSurfaceScalars(primitive_scalar, verts, scals, tris);
    if (verts.empty())
    {
        return;
    }

    SimpleArray<float> va;
    SimpleArray<float> sa;
    SimpleArray<uint32_t> ia;
    pack_scalar_arrays(verts, scals, tris, va, sa, ia);

    m_colormap = RColormap::categorical();
    float const hi = (ncat > 1) ? static_cast<float>(ncat - 1) : 1.0f;
    auto field = std::make_unique<RScalarField>(va, sa, ia, m_colormap);
    field->setScalarRange(0.0f, hi);
    m_range_pinned = true;
    m_range_lo = 0.0f;
    m_range_hi = hi;
    m_scalar_bar.setColormap(m_colormap);
    m_scalar_bar.setRange(0.0f, hi);
    m_scalar_bar.setTitle(title);
    m_scalar_bar.setVisible(true);
    RScalarField * scalar_field = field.get();
    installField(std::move(field));
    m_scalar_field = (m_field == scalar_field) ? scalar_field : nullptr;
}

void RDomainWidget::installMetricField(
    std::vector<float> const & primitive_value, std::string const & title)
{
    if (nullptr == m_mesh)
    {
        return;
    }

    std::vector<float> verts;
    std::vector<float> scals;
    std::vector<uint32_t> tris;
    collectSurfaceScalars(primitive_value, verts, scals, tris);
    if (verts.empty())
    {
        return;
    }

    SimpleArray<float> va;
    SimpleArray<float> sa;
    SimpleArray<uint32_t> ia;
    pack_scalar_arrays(verts, scals, tris, va, sa, ia);

    // A metric is continuous, so drop any categorical map left from a cell
    // coloring and let the field auto-range over the metric values.
    if ("categorical" == m_colormap.name())
    {
        m_colormap = RColormap::named("viridis");
    }
    m_range_pinned = false;
    auto field = std::make_unique<RScalarField>(va, sa, ia, m_colormap);
    m_range_lo = field->rangeLo();
    m_range_hi = field->rangeHi();
    m_scalar_bar.setColormap(m_colormap);
    m_scalar_bar.setRange(m_range_lo, m_range_hi);
    m_scalar_bar.setTitle(title);
    m_scalar_bar.setVisible(true);
    RScalarField * scalar_field = field.get();
    installField(std::move(field));
    m_scalar_field = (m_field == scalar_field) ? scalar_field : nullptr;
}

void RDomainWidget::showBoundary(int ibc, bool show)
{
    // Remove an existing highlight for this set so a re-show stays single and
    // a hide leaves none behind.
    m_scene.removeDrawableIf(
        [ibc](RDrawable const * d)
        {
            auto const * boundary = dynamic_cast<RBoundary const *>(d);
            return nullptr != boundary && boundary->ibc() == ibc;
        });

    if (show && nullptr != m_mesh)
    {
        auto boundary = std::make_unique<RBoundary>(m_mesh, ibc);
        if (boundary->hasGeometry())
        {
            m_scene.addDrawable(std::move(boundary));
        }
    }

    update();
}

void RDomainWidget::showFeatureEdges(bool show)
{
    // Drop any existing overlay so a re-show stays single and a hide leaves
    // none behind.
    m_scene.removeDrawableIf(
        [](RDrawable const * d)
        { return nullptr != dynamic_cast<RFeatureEdges const *>(d); });

    if (show && nullptr != m_mesh)
    {
        auto edges = std::make_unique<RFeatureEdges>(m_mesh);
        if (edges->hasGeometry())
        {
            m_scene.addDrawable(std::move(edges));
        }
    }

    update();
}

void RDomainWidget::showNormals(bool show)
{
    m_scene.removeDrawableIf(
        [](RDrawable const * d)
        { return nullptr != dynamic_cast<RNormals const *>(d); });

    if (show && nullptr != m_mesh)
    {
        auto normals = std::make_unique<RNormals>(m_mesh);
        if (normals->hasGeometry())
        {
            m_scene.addDrawable(std::move(normals));
        }
    }

    update();
}

namespace
{

/// Moeller-Trumbore ray/triangle intersection. On a hit, @p t is the ray
/// parameter of the front-facing or back-facing intersection.
bool ray_triangle(
    QVector3D const & o,
    QVector3D const & d,
    QVector3D const & a,
    QVector3D const & b,
    QVector3D const & c,
    float & t)
{
    QVector3D const e1 = b - a;
    QVector3D const e2 = c - a;
    QVector3D const pvec = QVector3D::crossProduct(d, e2);
    float const det = QVector3D::dotProduct(e1, pvec);
    if (std::abs(det) < 1.0e-12f)
    {
        return false;
    }
    float const inv = 1.0f / det;
    QVector3D const tvec = o - a;
    float const u = QVector3D::dotProduct(tvec, pvec) * inv;
    if (u < 0.0f || u > 1.0f)
    {
        return false;
    }
    QVector3D const qvec = QVector3D::crossProduct(tvec, e1);
    float const v = QVector3D::dotProduct(d, qvec) * inv;
    if (v < 0.0f || u + v > 1.0f)
    {
        return false;
    }
    t = QVector3D::dotProduct(e2, qvec) * inv;
    return t > 0.0f;
}

/// Even-odd point-in-polygon test in the xy plane.
bool point_in_polygon(std::vector<QVector3D> const & poly, float px, float py)
{
    bool inside = false;
    size_t const n = poly.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++)
    {
        float const yi = poly[i].y();
        float const yj = poly[j].y();
        if ((yi > py) != (yj > py))
        {
            float const xcross =
                (poly[j].x() - poly[i].x()) * (py - yi) / (yj - yi) + poly[i].x();
            if (px < xcross)
            {
                inside = !inside;
            }
        }
    }
    return inside;
}

} /* end namespace */

bool RDomainWidget::computePickRay(int x, int y, QVector3D & origin, QVector3D & dir) const
{
    int const w = width();
    int const h = height();
    if (w <= 0 || h <= 0)
    {
        return false;
    }
    QMatrix4x4 const vp = m_scene.viewProjection(QSize(w, h), nullptr);
    bool ok = false;
    QMatrix4x4 const inv = vp.inverted(&ok);
    if (!ok)
    {
        return false;
    }
    float const ndc_x = 2.0f * static_cast<float>(x) / static_cast<float>(w) - 1.0f;
    float const ndc_y = 1.0f - 2.0f * static_cast<float>(y) / static_cast<float>(h);
    QVector4D near_h = inv * QVector4D(ndc_x, ndc_y, -1.0f, 1.0f);
    QVector4D far_h = inv * QVector4D(ndc_x, ndc_y, 1.0f, 1.0f);
    if (qFuzzyIsNull(near_h.w()) || qFuzzyIsNull(far_h.w()))
    {
        return false;
    }
    origin = near_h.toVector3DAffine();
    dir = (far_h.toVector3DAffine() - origin).normalized();
    return true;
}

RDomainWidget::PickResult RDomainWidget::pickCell(int x, int y)
{
    PickResult result;
    if (nullptr == m_mesh)
    {
        return result;
    }
    QVector3D origin;
    QVector3D dir;
    if (!computePickRay(x, y, origin, dir))
    {
        return result;
    }
    StaticMesh const & mh = *m_mesh;
    uint32_t const ndim = mh.ndim();
    auto coord = [&mh, ndim](int32_t ind, uint32_t dm) -> float
    { return (dm < ndim) ? static_cast<float>(mh.ndcrd(ind, dm)) : 0.0f; };

    int32_t hit_cell = -1;
    if (3 == ndim)
    {
        SimpleArray<int32_t> const & bndfcs = mh.bndfcs();
        float best = std::numeric_limits<float>::max();
        for (size_t ibnd = 0; ibnd < bndfcs.shape(0); ++ibnd)
        {
            int32_t const ifc = bndfcs(ibnd, 0);
            int32_t const nnd = mh.fcnds(ifc, 0);
            QVector3D const a(
                coord(mh.fcnds(ifc, 1), 0), coord(mh.fcnds(ifc, 1), 1), coord(mh.fcnds(ifc, 1), 2));
            for (int32_t k = 1; k + 1 < nnd; ++k)
            {
                int32_t const nb = mh.fcnds(ifc, k + 1);
                int32_t const nc = mh.fcnds(ifc, k + 2);
                QVector3D const b(coord(nb, 0), coord(nb, 1), coord(nb, 2));
                QVector3D const c(coord(nc, 0), coord(nc, 1), coord(nc, 2));
                float t = 0.0f;
                if (ray_triangle(origin, dir, a, b, c, t) && t < best)
                {
                    best = t;
                    hit_cell = mh.fcicl(ifc);
                }
            }
        }
    }
    else
    {
        if (std::abs(dir.z()) > 1.0e-9f)
        {
            float const t = -origin.z() / dir.z();
            if (t >= 0.0f)
            {
                QVector3D const hit = origin + dir * t;
                for (uint32_t icl = 0; icl < mh.ncell() && hit_cell < 0; ++icl)
                {
                    int32_t const nnd = mh.clnds(icl, 0);
                    std::vector<QVector3D> poly(static_cast<size_t>(nnd));
                    for (int32_t k = 0; k < nnd; ++k)
                    {
                        int32_t const nd = mh.clnds(icl, k + 1);
                        poly[k] = QVector3D(coord(nd, 0), coord(nd, 1), 0.0f);
                    }
                    if (point_in_polygon(poly, hit.x(), hit.y()))
                    {
                        hit_cell = static_cast<int32_t>(icl);
                    }
                }
            }
        }
    }

    if (hit_cell < 0)
    {
        return result;
    }
    result.kind = "cell";
    result.id = hit_cell;
    result.type = mh.cltpn(hit_cell);
    result.measure = static_cast<double>(mh.clvol(hit_cell));
    result.centroid = QVector3D(
        static_cast<float>(mh.clcnd(hit_cell, 0)),
        static_cast<float>(mh.clcnd(hit_cell, 1)),
        (3 == ndim) ? static_cast<float>(mh.clcnd(hit_cell, 2)) : 0.0f);

    // The picked cell's bounding box, for framing and the highlight.
    QVector3D lo(result.centroid);
    QVector3D hi(result.centroid);
    int32_t const nnd = mh.clnds(hit_cell, 0);
    for (int32_t k = 0; k < nnd; ++k)
    {
        int32_t const nd = mh.clnds(hit_cell, k + 1);
        QVector3D const p(coord(nd, 0), coord(nd, 1), coord(nd, 2));
        lo = QVector3D(std::min(lo.x(), p.x()), std::min(lo.y(), p.y()), std::min(lo.z(), p.z()));
        hi = QVector3D(std::max(hi.x(), p.x()), std::max(hi.y(), p.y()), std::max(hi.z(), p.z()));
    }
    setSelection("cell", hit_cell, lo, hi);
    return result;
}

RDomainWidget::PickResult RDomainWidget::pickNode(int x, int y)
{
    PickResult result;
    if (nullptr == m_mesh)
    {
        return result;
    }
    QVector3D origin;
    QVector3D dir;
    if (!computePickRay(x, y, origin, dir))
    {
        return result;
    }
    StaticMesh const & mh = *m_mesh;
    uint32_t const ndim = mh.ndim();

    float best = std::numeric_limits<float>::max();
    int32_t hit = -1;
    QVector3D hit_pos;
    for (uint32_t ind = 0; ind < mh.nnode(); ++ind)
    {
        QVector3D const p(
            static_cast<float>(mh.ndcrd(ind, 0)),
            static_cast<float>(mh.ndcrd(ind, 1)),
            (3 == ndim) ? static_cast<float>(mh.ndcrd(ind, 2)) : 0.0f);
        // Perpendicular distance from the node to the pick ray.
        QVector3D const w = p - origin;
        QVector3D const closest = origin + dir * QVector3D::dotProduct(w, dir);
        float const dist = (p - closest).length();
        if (dist < best)
        {
            best = dist;
            hit = static_cast<int32_t>(ind);
            hit_pos = p;
        }
    }
    if (hit < 0)
    {
        return result;
    }
    result.kind = "node";
    result.id = hit;
    result.centroid = hit_pos;
    setSelection("node", hit, hit_pos, hit_pos);
    return result;
}

RDomainWidget::PickResult RDomainWidget::pickFace(int x, int y)
{
    PickResult result;
    if (nullptr == m_mesh)
    {
        return result;
    }
    QVector3D origin;
    QVector3D dir;
    if (!computePickRay(x, y, origin, dir))
    {
        return result;
    }
    StaticMesh const & mh = *m_mesh;
    if (3 != mh.ndim())
    {
        // A 2D domain's faces are edges, not a ray-castable surface.
        return result;
    }
    auto coord = [&mh](int32_t ind, uint32_t dm) -> float
    { return static_cast<float>(mh.ndcrd(ind, dm)); };

    SimpleArray<int32_t> const & bndfcs = mh.bndfcs();
    float best = std::numeric_limits<float>::max();
    int32_t hit_face = -1;
    for (size_t ibnd = 0; ibnd < bndfcs.shape(0); ++ibnd)
    {
        int32_t const ifc = bndfcs(ibnd, 0);
        int32_t const nnd = mh.fcnds(ifc, 0);
        QVector3D const a(
            coord(mh.fcnds(ifc, 1), 0), coord(mh.fcnds(ifc, 1), 1), coord(mh.fcnds(ifc, 1), 2));
        for (int32_t k = 1; k + 1 < nnd; ++k)
        {
            int32_t const nb = mh.fcnds(ifc, k + 1);
            int32_t const nc = mh.fcnds(ifc, k + 2);
            QVector3D const b(coord(nb, 0), coord(nb, 1), coord(nb, 2));
            QVector3D const c(coord(nc, 0), coord(nc, 1), coord(nc, 2));
            float t = 0.0f;
            if (ray_triangle(origin, dir, a, b, c, t) && t < best)
            {
                best = t;
                hit_face = ifc;
            }
        }
    }
    if (hit_face < 0)
    {
        return result;
    }
    result.kind = "face";
    result.id = hit_face;
    result.measure = static_cast<double>(mh.fcara(hit_face));
    result.centroid = QVector3D(
        static_cast<float>(mh.fccnd(hit_face, 0)),
        static_cast<float>(mh.fccnd(hit_face, 1)),
        static_cast<float>(mh.fccnd(hit_face, 2)));
    setSelection("face", hit_face, result.centroid, result.centroid);
    return result;
}

void RDomainWidget::setSelection(
    std::string const & kind, int id, QVector3D const & lo, QVector3D const & hi)
{
    m_scene.removeDrawable(m_selection);
    m_selection = nullptr;
    m_selection_kind = kind;
    m_selection_id = id;
    m_selection_lo = lo;
    m_selection_hi = hi;
    m_has_selection = true;

    // A picked cell gets a bright surface highlight; a node or face records the
    // selection point without one.
    if ("cell" == kind && nullptr != m_mesh)
    {
        StaticMesh const & mh = *m_mesh;
        uint32_t const ndim = mh.ndim();
        auto coord = [&mh, ndim](int32_t ind, uint32_t dm) -> float
        { return (dm < ndim) ? static_cast<float>(mh.ndcrd(ind, dm)) : 0.0f; };

        std::vector<float> verts;
        std::vector<uint32_t> tris;
        auto add_polygon = [&](auto node, int32_t nnd)
        {
            for (int32_t k = 1; k + 1 < nnd; ++k)
            {
                int32_t const tri[3] = {node(0), node(k), node(k + 1)};
                uint32_t const base = static_cast<uint32_t>(verts.size() / 3);
                for (int32_t const ind : tri)
                {
                    verts.push_back(coord(ind, 0));
                    verts.push_back(coord(ind, 1));
                    verts.push_back(coord(ind, 2));
                }
                tris.push_back(base + 0);
                tris.push_back(base + 1);
                tris.push_back(base + 2);
            }
        };
        if (3 == ndim)
        {
            SimpleArray<int32_t> const & bndfcs = mh.bndfcs();
            for (size_t ibnd = 0; ibnd < bndfcs.shape(0); ++ibnd)
            {
                int32_t const ifc = bndfcs(ibnd, 0);
                if (mh.fcicl(ifc) == id)
                {
                    add_polygon(
                        [&mh, ifc](int32_t k)
                        { return mh.fcnds(ifc, k + 1); },
                        mh.fcnds(ifc, 0));
                }
            }
        }
        else
        {
            add_polygon(
                [&mh, id](int32_t k)
                { return mh.clnds(id, k + 1); },
                mh.clnds(id, 0));
        }

        if (!verts.empty())
        {
            size_t const nvert = verts.size() / 3;
            size_t const ntri = tris.size() / 3;
            SimpleArray<float> va(small_vector<ssize_t>{static_cast<ssize_t>(nvert), 3});
            SimpleArray<float> ca(small_vector<ssize_t>{static_cast<ssize_t>(nvert), 3});
            SimpleArray<uint32_t> ia(small_vector<ssize_t>{static_cast<ssize_t>(ntri), 3});
            for (size_t i = 0; i < nvert; ++i)
            {
                va(i, 0) = verts[3 * i + 0];
                va(i, 1) = verts[3 * i + 1];
                va(i, 2) = verts[3 * i + 2];
                ca(i, 0) = 1.0f;
                ca(i, 1) = 0.9f;
                ca(i, 2) = 0.1f;
            }
            for (size_t t = 0; t < ntri; ++t)
            {
                ia(t, 0) = tris[3 * t + 0];
                ia(t, 1) = tris[3 * t + 1];
                ia(t, 2) = tris[3 * t + 2];
            }
            auto highlight = std::make_unique<RField>(va, ca, ia);
            if (highlight->hasGeometry())
            {
                m_selection = highlight.get();
                m_scene.addDrawable(std::move(highlight));
            }
        }
    }
    update();
}

void RDomainWidget::clearSelection()
{
    m_scene.removeDrawable(m_selection);
    m_selection = nullptr;
    m_has_selection = false;
    m_selection_kind = "none";
    m_selection_id = -1;
    update();
}

double RDomainWidget::measureDistance(QVector3D const & p0, QVector3D const & p1)
{
    m_scene.removeDrawable(m_ruler);
    m_ruler = nullptr;
    auto ruler = std::make_unique<RSegments>(std::vector<QVector3D>{p0, p1});
    if (ruler->hasGeometry())
    {
        m_ruler = ruler.get();
        m_scene.addDrawable(std::move(ruler));
    }
    update();
    return static_cast<double>((p1 - p0).length());
}

double RDomainWidget::measureAngle(
    QVector3D const & p0, QVector3D const & p1, QVector3D const & p2)
{
    m_scene.removeDrawable(m_ruler);
    m_ruler = nullptr;
    // Two arms out of the vertex p1.
    auto ruler = std::make_unique<RSegments>(
        std::vector<QVector3D>{p1, p0, p1, p2});
    if (ruler->hasGeometry())
    {
        m_ruler = ruler.get();
        m_scene.addDrawable(std::move(ruler));
    }
    update();

    QVector3D const a = p0 - p1;
    QVector3D const b = p2 - p1;
    float const la = a.length();
    float const lb = b.length();
    if (la <= 0.0f || lb <= 0.0f)
    {
        return 0.0;
    }
    float const c = std::clamp(QVector3D::dotProduct(a, b) / (la * lb), -1.0f, 1.0f);
    return static_cast<double>(std::acos(c) * 180.0f / PI);
}

void RDomainWidget::clearMeasurements()
{
    m_scene.removeDrawable(m_ruler);
    m_ruler = nullptr;
    update();
}

int RDomainWidget::addClip(QVector3D const & origin, QVector3D const & normal)
{
    m_scene.removeDrawable(m_clip);
    m_clip = nullptr;
    if (nullptr == m_mesh)
    {
        return 0;
    }
    StaticMesh const & mh = *m_mesh;
    uint32_t const ndim = mh.ndim();
    QVector3D const n = normal.normalized();
    auto coord = [&mh, ndim](int32_t ind, uint32_t dm) -> float
    { return (dm < ndim) ? static_cast<float>(mh.ndcrd(ind, dm)) : 0.0f; };
    auto side = [&](QVector3D const & p) -> float
    { return QVector3D::dotProduct(p - origin, n); };

    std::vector<float> verts;
    std::vector<float> cols;
    std::vector<uint32_t> tris;
    auto add_polygon = [&](auto node, int32_t nnd)
    {
        for (int32_t k = 1; k + 1 < nnd; ++k)
        {
            int32_t const tri[3] = {node(0), node(k), node(k + 1)};
            uint32_t const base = static_cast<uint32_t>(verts.size() / 3);
            for (int32_t const ind : tri)
            {
                verts.push_back(coord(ind, 0));
                verts.push_back(coord(ind, 1));
                verts.push_back(coord(ind, 2));
                cols.push_back(0.55f);
                cols.push_back(0.62f);
                cols.push_back(0.75f);
            }
            tris.push_back(base + 0);
            tris.push_back(base + 1);
            tris.push_back(base + 2);
        }
    };

    int kept = 0;
    if (3 == ndim)
    {
        SimpleArray<int32_t> const & bndfcs = mh.bndfcs();
        for (size_t ibnd = 0; ibnd < bndfcs.shape(0); ++ibnd)
        {
            int32_t const ifc = bndfcs(ibnd, 0);
            QVector3D const centroid(
                static_cast<float>(mh.fccnd(ifc, 0)),
                static_cast<float>(mh.fccnd(ifc, 1)),
                static_cast<float>(mh.fccnd(ifc, 2)));
            if (side(centroid) <= 0.0f)
            {
                add_polygon(
                    [&mh, ifc](int32_t k)
                    { return mh.fcnds(ifc, k + 1); },
                    mh.fcnds(ifc, 0));
                ++kept;
            }
        }
    }
    else
    {
        for (uint32_t icl = 0; icl < mh.ncell(); ++icl)
        {
            QVector3D const centroid(
                static_cast<float>(mh.clcnd(icl, 0)),
                static_cast<float>(mh.clcnd(icl, 1)),
                0.0f);
            if (side(centroid) <= 0.0f)
            {
                add_polygon(
                    [&mh, icl](int32_t k)
                    { return mh.clnds(icl, k + 1); },
                    mh.clnds(icl, 0));
                ++kept;
            }
        }
    }

    if (!verts.empty())
    {
        size_t const nvert = verts.size() / 3;
        size_t const ntri = tris.size() / 3;
        SimpleArray<float> va(small_vector<ssize_t>{static_cast<ssize_t>(nvert), 3});
        SimpleArray<float> ca(small_vector<ssize_t>{static_cast<ssize_t>(nvert), 3});
        SimpleArray<uint32_t> ia(small_vector<ssize_t>{static_cast<ssize_t>(ntri), 3});
        for (size_t i = 0; i < nvert; ++i)
        {
            va(i, 0) = verts[3 * i + 0];
            va(i, 1) = verts[3 * i + 1];
            va(i, 2) = verts[3 * i + 2];
            ca(i, 0) = cols[3 * i + 0];
            ca(i, 1) = cols[3 * i + 1];
            ca(i, 2) = cols[3 * i + 2];
        }
        for (size_t t = 0; t < ntri; ++t)
        {
            ia(t, 0) = tris[3 * t + 0];
            ia(t, 1) = tris[3 * t + 1];
            ia(t, 2) = tris[3 * t + 2];
        }
        auto clip = std::make_unique<RField>(va, ca, ia);
        if (clip->hasGeometry())
        {
            m_clip = clip.get();
            m_scene.addDrawable(std::move(clip));
        }
    }
    update();
    return kept;
}

int RDomainWidget::addSlice(QVector3D const & origin, QVector3D const & normal)
{
    m_scene.removeDrawable(m_slice);
    m_slice = nullptr;
    if (nullptr == m_mesh)
    {
        return 0;
    }
    StaticMesh const & mh = *m_mesh;
    uint32_t const ndim = mh.ndim();
    QVector3D const n = normal.normalized();
    auto pos = [&mh, ndim](int32_t ind) -> QVector3D
    {
        return QVector3D(
            static_cast<float>(mh.ndcrd(ind, 0)),
            static_cast<float>(mh.ndcrd(ind, 1)),
            (3 == ndim) ? static_cast<float>(mh.ndcrd(ind, 2)) : 0.0f);
    };
    auto side = [&](QVector3D const & p) -> float
    { return QVector3D::dotProduct(p - origin, n); };

    std::vector<QVector3D> segments;
    for (uint32_t icl = 0; icl < mh.ncell(); ++icl)
    {
        int32_t const nnd = mh.clnds(icl, 0);
        std::vector<int32_t> nd(static_cast<size_t>(nnd));
        for (int32_t k = 0; k < nnd; ++k)
        {
            nd[k] = mh.clnds(icl, k + 1);
        }

        // Candidate edges: the polygon rim in 2D, every node pair in 3D (exact
        // for a simplex, an over-set for a hex but the crossings stay on real
        // faces).
        std::vector<QVector3D> cuts;
        auto try_edge = [&](int32_t ia, int32_t ib)
        {
            QVector3D const pa = pos(ia);
            QVector3D const pb = pos(ib);
            float const sa = side(pa);
            float const sb = side(pb);
            if (sa * sb < 0.0f)
            {
                float const t = sa / (sa - sb);
                cuts.push_back(pa + t * (pb - pa));
            }
        };
        if (3 == ndim)
        {
            for (int32_t a = 0; a < nnd; ++a)
            {
                for (int32_t b = a + 1; b < nnd; ++b)
                {
                    try_edge(nd[a], nd[b]);
                }
            }
        }
        else
        {
            for (int32_t k = 0; k < nnd; ++k)
            {
                try_edge(nd[k], nd[(k + 1) % nnd]);
            }
        }

        if (2 == cuts.size())
        {
            segments.push_back(cuts[0]);
            segments.push_back(cuts[1]);
        }
        else if (cuts.size() > 2)
        {
            // Close the crossing points into a loop to outline the cut face.
            for (size_t k = 0; k < cuts.size(); ++k)
            {
                segments.push_back(cuts[k]);
                segments.push_back(cuts[(k + 1) % cuts.size()]);
            }
        }
    }

    auto slice = std::make_unique<RSegments>(segments);
    slice->setColor(QVector4D(0.10f, 0.10f, 0.10f, 1.0f));
    if (slice->hasGeometry())
    {
        m_slice = slice.get();
        m_scene.addDrawable(std::move(slice));
    }
    update();
    return static_cast<int>(segments.size() / 2);
}

void RDomainWidget::clearFilters()
{
    m_scene.removeDrawable(m_clip);
    m_scene.removeDrawable(m_slice);
    m_clip = nullptr;
    m_slice = nullptr;
    update();
}

void RDomainWidget::fitCameraToScene()
{
    m_scene.fitCameraToScene(viewportAspect());
    update();
}

void RDomainWidget::setView(std::string const & name)
{
    m_scene.setViewPreset(name, viewportAspect());
    update();
}

void RDomainWidget::setProjection(std::string const & name)
{
    m_scene.setProjection(name);
    update();
}

std::string RDomainWidget::projection() const
{
    return m_scene.projectionName();
}

void RDomainWidget::setOrbitStyle(std::string const & name)
{
    m_scene.camera().setOrbitStyle(RCameraController::orbitStyleFromName(name));
    update();
}

std::string RDomainWidget::orbitStyle() const
{
    return RCameraController::orbitStyleName(m_scene.camera().orbitStyle());
}

void RDomainWidget::setPivot(float x, float y, float z)
{
    m_scene.camera().setPivot(QVector3D(x, y, z));
    update();
}

void RDomainWidget::frameSelected()
{
    zoomToSelection();
}

void RDomainWidget::zoomToSelection()
{
    if (m_has_selection)
    {
        m_scene.frameBox(m_selection_lo, m_selection_hi, viewportAspect());
    }
    else
    {
        m_scene.fitCameraToScene(viewportAspect());
    }
    update();
}

void RDomainWidget::resetCamera()
{
    m_scene.setProjection("auto");
    m_scene.camera().setOrbitStyle(RCameraController::OrbitStyle::Turntable);
    m_scene.fitCameraToScene(viewportAspect());
    update();
}

void RDomainWidget::setNavigationMapping(std::string const & name)
{
    if ("default" == name || "blender" == name)
    {
        m_nav_mapping = name;
    }
}

std::string RDomainWidget::navigationMapping() const
{
    return m_nav_mapping;
}

void RDomainWidget::setOrbitSensitivity(float factor)
{
    m_scene.camera().setOrbitSensitivity(factor);
}

void RDomainWidget::orbitStep(float yaw_deg, float pitch_deg)
{
    m_scene.camera().orbitStep(yaw_deg, pitch_deg);
    update();
}

void RDomainWidget::setCameraMode(std::string const & name)
{
    m_scene.camera().setMode(RCameraController::modeFromName(name));
    update();
}

std::string RDomainWidget::cameraMode() const
{
    return RCameraController::modeName(m_scene.camera().mode());
}

QVector3D RDomainWidget::cameraPosition() const
{
    return m_scene.camera().position();
}

void RDomainWidget::setCameraPosition(QVector3D const & position)
{
    m_scene.camera().setPosition(position);
    update();
}

QVector3D RDomainWidget::cameraTarget() const
{
    return m_scene.camera().target();
}

void RDomainWidget::setCameraTarget(QVector3D const & target)
{
    m_scene.camera().setTarget(target);
    update();
}

QVector3D RDomainWidget::cameraUp() const
{
    return m_scene.camera().up();
}

void RDomainWidget::setCameraUp(QVector3D const & up)
{
    m_scene.camera().setUp(up);
    update();
}

void RDomainWidget::rotateCamera(float dx, float dy)
{
    m_scene.camera().rotate(dx, dy);
    update();
}

void RDomainWidget::panCamera(float dx, float dy)
{
    m_scene.camera().pan(dx, dy);
    update();
}

void RDomainWidget::zoomCamera(float steps)
{
    m_scene.camera().zoom(steps);
    update();
}

void RDomainWidget::pinchCamera(float factor)
{
    m_scene.camera().pinch(factor);
    update();
}

void RDomainWidget::mousePressEvent(QMouseEvent * event)
{
    m_last_mouse_pos = event->position().toPoint();

    Qt::MouseButton const btn = event->button();
    Qt::KeyboardModifiers const mods = event->modifiers();

    if ("blender" == m_nav_mapping)
    {
        // Middle drives navigation; Alt+left aliases it for a trackpad with no
        // middle button. Shift pans, Ctrl zooms, Alt recenters the pivot, and
        // a plain middle drag orbits. Left alone still orbits (a convenience
        // over Blender's select), right pans.
        bool const navigate = (Qt::MiddleButton == btn) ||
                              (Qt::LeftButton == btn && (mods & Qt::AltModifier));
        if (navigate)
        {
            if (mods & Qt::ShiftModifier)
            {
                m_drag_action = DragAction::Pan;
            }
            else if (mods & Qt::ControlModifier)
            {
                m_drag_action = DragAction::Zoom;
            }
            else if ((mods & Qt::AltModifier) && Qt::MiddleButton == btn)
            {
                // Recenter the orbit pivot on the scene, then orbit about it.
                m_scene.camera().setPivot(m_scene.boundingBoxCenter());
                m_drag_action = DragAction::Rotate;
            }
            else
            {
                m_drag_action = DragAction::Rotate;
            }
        }
        else if (Qt::LeftButton == btn)
        {
            m_drag_action = DragAction::Rotate;
        }
        else
        {
            m_drag_action = DragAction::Pan;
        }
    }
    else
    {
        // Default mapping: left rotates, other buttons pan.
        m_drag_action = (Qt::LeftButton == btn) ? DragAction::Rotate
                                                : DragAction::Pan;
    }
    update();
}

void RDomainWidget::mouseMoveEvent(QMouseEvent * event)
{
    if (event->buttons() == Qt::NoButton)
    {
        return;
    }
    QPoint const pos = event->position().toPoint();
    float const dx = static_cast<float>(pos.x() - m_last_mouse_pos.x());
    float const dy = static_cast<float>(pos.y() - m_last_mouse_pos.y());
    m_last_mouse_pos = pos;
    switch (m_drag_action)
    {
    case DragAction::Pan:
        m_scene.camera().pan(dx, dy);
        break;
    case DragAction::Zoom:
        // Dragging up zooms in; scale the pixel delta to wheel-notch units.
        m_scene.camera().zoom(-dy * 0.05f);
        break;
    default:
        m_scene.camera().rotate(dx, dy);
        break;
    }
    update();
}

void RDomainWidget::mouseReleaseEvent(QMouseEvent *)
{
    m_drag_action = DragAction::Rotate;
}

void RDomainWidget::wheelEvent(QWheelEvent * event)
{
    // One wheel notch is 120 eighths of a degree.
    float const steps = static_cast<float>(event->angleDelta().y()) / 120.0f;
    m_scene.camera().zoom(steps);
    update();
}

bool RDomainWidget::event(QEvent * event)
{
    // A trackpad pinch arrives as a native zoom gesture whose value() is the
    // incremental magnification (positive spreads, negative pinches); a
    // touchscreen pinch arrives as a QPinchGesture whose scaleFactor() is the
    // incremental multiplier. Feed both to pinchCamera as a scale around 1.
    if (QEvent::NativeGesture == event->type())
    {
        auto * gesture = static_cast<QNativeGestureEvent *>(event);
        if (Qt::ZoomNativeGesture == gesture->gestureType())
        {
            pinchCamera(1.0f + static_cast<float>(gesture->value()));
            return true;
        }
    }
    else if (QEvent::Gesture == event->type())
    {
        auto * gesture = static_cast<QGestureEvent *>(event);
        if (auto * pinch = static_cast<QPinchGesture *>(gesture->gesture(Qt::PinchGesture)))
        {
            pinchCamera(static_cast<float>(pinch->scaleFactor()));
            return true;
        }
    }
    return QRhiWidget::event(event);
}

void RDomainWidget::keyPressEvent(QKeyEvent * event)
{
    // First-person movement; a step is a tenth of the scene size.
    constexpr float step = 0.1f;
    switch (event->key())
    {
    case Qt::Key_W:
    case Qt::Key_Up:
        m_scene.camera().moveForward(step);
        break;
    case Qt::Key_S:
    case Qt::Key_Down:
        m_scene.camera().moveForward(-step);
        break;
    case Qt::Key_D:
    case Qt::Key_Right:
        m_scene.camera().moveRight(step);
        break;
    case Qt::Key_A:
    case Qt::Key_Left:
        m_scene.camera().moveRight(-step);
        break;
    default:
        QRhiWidget::keyPressEvent(event);
        return;
    }
    update();
}

void RDomainWidget::showAxis(bool show)
{
    m_gizmo.setVisible(show);
    update();
}

void RDomainWidget::showCubeAxes(bool show)
{
    m_scene.removeDrawable(m_cube_axes);
    m_cube_axes = nullptr;
    m_ticks_x.clear();
    m_ticks_y.clear();
    m_ticks_z.clear();
    if (!show || !m_scene.hasBoundingBox())
    {
        update();
        return;
    }

    QVector3D const lo = m_scene.boundingBoxLo();
    QVector3D const hi = m_scene.boundingBoxHi();
    constexpr int kTicks = 5;
    auto ticks = [](float a, float b)
    {
        std::vector<float> t;
        for (int i = 0; i < kTicks; ++i)
        {
            t.push_back(a + (b - a) * static_cast<float>(i) / static_cast<float>(kTicks - 1));
        }
        return t;
    };
    m_ticks_x = ticks(lo.x(), hi.x());
    m_ticks_y = ticks(lo.y(), hi.y());
    m_ticks_z = ticks(lo.z(), hi.z());

    std::vector<QVector3D> pts;
    QVector3D corner[8];
    for (int i = 0; i < 8; ++i)
    {
        corner[i] = QVector3D(
            (i & 1) ? hi.x() : lo.x(),
            (i & 2) ? hi.y() : lo.y(),
            (i & 4) ? hi.z() : lo.z());
    }
    // The 12 box edges connect corners that differ in exactly one axis bit.
    for (int i = 0; i < 8; ++i)
    {
        for (int b = 0; b < 3; ++b)
        {
            int const j = i ^ (1 << b);
            if (i < j)
            {
                pts.push_back(corner[i]);
                pts.push_back(corner[j]);
            }
        }
    }
    // Short tick marks along the axes at the low corner.
    float const step = 0.03f * (hi - lo).length();
    for (float xv : m_ticks_x)
    {
        pts.push_back(QVector3D(xv, lo.y(), lo.z()));
        pts.push_back(QVector3D(xv, lo.y() + step, lo.z()));
    }
    for (float yv : m_ticks_y)
    {
        pts.push_back(QVector3D(lo.x(), yv, lo.z()));
        pts.push_back(QVector3D(lo.x() + step, yv, lo.z()));
    }
    if (3 == m_scene.dimension())
    {
        for (float zv : m_ticks_z)
        {
            pts.push_back(QVector3D(lo.x(), lo.y(), zv));
            pts.push_back(QVector3D(lo.x() + step, lo.y(), zv));
        }
    }

    auto axes = std::make_unique<RSegments>(pts);
    axes->setColor(QVector4D(0.35f, 0.35f, 0.35f, 1.0f));
    if (axes->hasGeometry())
    {
        m_cube_axes = axes.get();
        m_scene.addDrawable(std::move(axes));
    }
    update();
}

std::vector<float> RDomainWidget::cubeAxesTicks(int axis) const
{
    if (0 == axis)
    {
        return m_ticks_x;
    }
    if (1 == axis)
    {
        return m_ticks_y;
    }
    return m_ticks_z;
}

void RDomainWidget::setTitle(std::string const & text)
{
    m_title.setText(text);
    m_title.setVisible(!text.empty());
    update();
}

std::string RDomainWidget::title() const
{
    return m_title.text();
}

void RDomainWidget::initialize(QRhiCommandBuffer *)
{
    QRhiRenderPassDescriptor * const rpdesc = renderTarget()->renderPassDescriptor();
    if (m_rhi != rhi() || m_rpdesc != rpdesc || m_sample_count != sampleCount())
    {
        // The graphics device, render target, or sample count changed; drop
        // every device resource so the drawables rebuild against the new one
        // (the pipelines are tied to the render-pass descriptor).
        m_scene.releaseAll();
        m_gizmo.release();
        m_scalar_bar.release();
        m_rhi = rhi();
        m_rpdesc = rpdesc;
        m_sample_count = sampleCount();
    }
}

void RDomainWidget::render(QRhiCommandBuffer * cb)
{
    QRhiResourceUpdateBatch * batch = m_rhi->nextResourceUpdateBatch();

    QSize const pixel_size = renderTarget()->pixelSize();
    QMatrix4x4 const view_proj = m_scene.viewProjection(pixel_size, m_rhi);
    QRhiRenderPassDescriptor * const rpdesc = renderTarget()->renderPassDescriptor();

    // A camera-following headlight: the lit surface faces toward the eye read
    // brightest. Non-lit drawables ignore the direction.
    QVector3D light_dir = m_scene.camera().position() - m_scene.camera().target();
    if (light_dir.lengthSquared() <= 0.0f)
    {
        light_dir = QVector3D(0.0f, 0.0f, 1.0f);
    }
    light_dir.normalize();

    for (std::unique_ptr<RDrawable> const & drawable : m_scene.drawables())
    {
        drawable->prepare(m_rhi, rpdesc, sampleCount(), batch);
        drawable->setLightDir(light_dir);
        drawable->updateUniform(batch, view_proj);
    }

    // The orientation guide shows two axes for a 2D domain and three for 3D,
    // oriented by the main camera. Its resources update before the pass.
    m_gizmo.setAxisCount((2 == m_scene.dimension()) ? 2 : 3);
    QVector3D const camera_forward = m_scene.camera().target() - m_scene.camera().position();
    m_gizmo.update(
        m_rhi, rpdesc, sampleCount(), pixel_size, camera_forward, m_scene.camera().up(), batch);
    m_scalar_bar.update(m_rhi, rpdesc, sampleCount(), pixel_size, batch);
    m_title.update(m_rhi, rpdesc, sampleCount(), pixel_size, batch);

    QColor const clear_color = m_transparent_capture
                                   ? QColor::fromRgbF(1.0f, 1.0f, 1.0f, 0.0f)
                                   : QColor::fromRgbF(1.0f, 1.0f, 1.0f, 1.0f);
    QRhiDepthStencilClearValue const ds_clear(1.0f, 0);

    cb->beginPass(renderTarget(), clear_color, ds_clear, batch);
    cb->setViewport(QRhiViewport(
        0, 0, float(pixel_size.width()), float(pixel_size.height())));
    for (std::unique_ptr<RDrawable> const & drawable : m_scene.drawables())
    {
        drawable->draw(cb);
    }
    m_gizmo.draw(cb);
    m_scalar_bar.draw(cb);
    m_title.draw(cb);
    cb->endPass();
}

void RDomainWidget::releaseResources()
{
    m_scene.releaseAll();
    m_gizmo.release();
    m_scalar_bar.release();
    m_title.release();
    m_rhi = nullptr;
    m_rpdesc = nullptr;
    m_sample_count = 0;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RDomainWidget.hpp> // Must be the first include.

#include <solvcon/pilot/RBoundary.hpp>
#include <solvcon/pilot/RFeatureEdges.hpp>
#include <solvcon/pilot/RField.hpp>
#include <solvcon/pilot/RMeshFrame.hpp>
#include <solvcon/pilot/RNormals.hpp>
#include <solvcon/pilot/RScalarField.hpp>

#include <QGestureEvent>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QNativeGestureEvent>
#include <QPinchGesture>
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
        size_t const nprim = std::min(bndfcs.shape(0), primitive_scalar.size());
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
    va = SimpleArray<float>(std::vector<size_t>{nvert, 3});
    sa = SimpleArray<float>(std::vector<size_t>{nvert});
    ia = SimpleArray<uint32_t>(std::vector<size_t>{ntri, 3});
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

void RDomainWidget::fitCameraToScene()
{
    m_scene.fitCameraToScene(viewportAspect());
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
    m_panning = (event->button() != Qt::LeftButton);
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
    if (m_panning)
    {
        m_scene.camera().pan(dx, dy);
    }
    else
    {
        m_scene.camera().rotate(dx, dy);
    }
    update();
}

void RDomainWidget::mouseReleaseEvent(QMouseEvent *)
{
    m_panning = false;
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

    QColor const clear_color = QColor::fromRgbF(1.0f, 1.0f, 1.0f, 1.0f);
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
    cb->endPass();
}

void RDomainWidget::releaseResources()
{
    m_scene.releaseAll();
    m_gizmo.release();
    m_scalar_bar.release();
    m_rhi = nullptr;
    m_rpdesc = nullptr;
    m_sample_count = 0;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

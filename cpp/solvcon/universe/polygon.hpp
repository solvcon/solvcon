#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Polygons and their containers in 2 and 3 dimensional spaces.
 *
 * @ingroup group_geometry
 */

#include <solvcon/base.hpp>
#include <solvcon/buffer/SimpleCollector.hpp>
#include <solvcon/buffer/buffer.hpp>
#include <solvcon/universe/bezier.hpp>
#include <solvcon/universe/coord.hpp>
#include <solvcon/universe/rtree.hpp>

#include <solvcon/universe/polygon_boolean.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <vector>

namespace solvcon
{

namespace detail
{

template <typename T>
struct Triangle3dNamed
{
    T x0, x1, x2, y0, y1, y2, z0, z1, z2;
}; /* end struct Triangle3dNamed */

template <typename T>
union Triangle3dData
{
    T v[9]; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    Triangle3dNamed<T> f;
}; /* end union Triangle3dData */

template <typename T>
struct Trapezoid3dNamed
{
    T x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
}; /* end struct Trapezoid3dNamed */

template <typename T>
union Trapezoid3dData
{
    T v[12]; // NOLINT(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    Trapezoid3dNamed<T> f;
}; /* end union Trapezoid3dData */

} /* end namespace detail */

/**
 * Trapezoid in three-dimensional space.
 *
 * Stores the four corner points (p0, p1, p2, p3) as twelve scalar
 * coordinates in a flat buffer.
 *
 * @tparam T floating-point type
 * @ingroup group_geometry
 */
template <typename T>
class Trapezoid3d
    : public NumberBase<int32_t, T>
{

public:

    static_assert(std::is_arithmetic_v<T>, "T in Trapezoid3d<T> must be arithmetic type");

    using point_type = Point3d<T>;
    using value_type = typename point_type::value_type;

    Trapezoid3d(point_type const & p0, point_type const & p1, point_type const & p2, point_type const & p3)
        : m_data{p0.x(), p1.x(), p2.x(), p3.x(), p0.y(), p1.y(), p2.y(), p3.y(), p0.z(), p1.z(), p2.z(), p3.z()}
    {
    }

    Trapezoid3d() = default;
    Trapezoid3d(Trapezoid3d const &) = default;
    Trapezoid3d & operator=(Trapezoid3d const &) = default;
    Trapezoid3d(Trapezoid3d &&) = default;
    Trapezoid3d & operator=(Trapezoid3d &&) = default;
    ~Trapezoid3d() = default;

    point_type p0() const { return point_type(m_data.f.x0, m_data.f.y0, m_data.f.z0); }
    void set_p0(point_type const & p)
    {
        m_data.f.x0 = p.x();
        m_data.f.y0 = p.y();
        m_data.f.z0 = p.z();
    }
    point_type p1() const { return point_type(m_data.f.x1, m_data.f.y1, m_data.f.z1); }
    void set_p1(point_type const & p)
    {
        m_data.f.x1 = p.x();
        m_data.f.y1 = p.y();
        m_data.f.z1 = p.z();
    }
    point_type p2() const { return point_type(m_data.f.x2, m_data.f.y2, m_data.f.z2); }
    void set_p2(point_type const & p)
    {
        m_data.f.x2 = p.x();
        m_data.f.y2 = p.y();
        m_data.f.z2 = p.z();
    }
    point_type p3() const { return point_type(m_data.f.x3, m_data.f.y3, m_data.f.z3); }
    void set_p3(point_type const & p)
    {
        m_data.f.x3 = p.x();
        m_data.f.y3 = p.y();
        m_data.f.z3 = p.z();
    }

    value_type x0() const { return m_data.f.x0; }
    value_type & x0() { return m_data.f.x0; }
    void set_x0(value_type v) { m_data.f.x0 = v; }

    value_type y0() const { return m_data.f.y0; }
    value_type & y0() { return m_data.f.y0; }
    void set_y0(value_type v) { m_data.f.y0 = v; }

    value_type z0() const { return m_data.f.z0; }
    value_type & z0() { return m_data.f.z0; }
    void set_z0(value_type v) { m_data.f.z0 = v; }

    value_type x1() const { return m_data.f.x1; }
    value_type & x1() { return m_data.f.x1; }
    void set_x1(value_type v) { m_data.f.x1 = v; }

    value_type y1() const { return m_data.f.y1; }
    value_type & y1() { return m_data.f.y1; }
    void set_y1(value_type v) { m_data.f.y1 = v; }

    value_type z1() const { return m_data.f.z1; }
    value_type & z1() { return m_data.f.z1; }
    void set_z1(value_type v) { m_data.f.z1 = v; }

    value_type x2() const { return m_data.f.x2; }
    value_type & x2() { return m_data.f.x2; }
    void set_x2(value_type v) { m_data.f.x2 = v; }

    value_type y2() const { return m_data.f.y2; }
    value_type & y2() { return m_data.f.y2; }
    void set_y2(value_type v) { m_data.f.y2 = v; }

    value_type z2() const { return m_data.f.z2; }
    value_type & z2() { return m_data.f.z2; }
    void set_z2(value_type v) { m_data.f.z2 = v; }

    value_type x3() const { return m_data.f.x3; }
    value_type & x3() { return m_data.f.x3; }
    void set_x3(value_type v) { m_data.f.x3 = v; }

    value_type y3() const { return m_data.f.y3; }
    value_type & y3() { return m_data.f.y3; }
    void set_y3(value_type v) { m_data.f.y3 = v; }

    value_type z3() const { return m_data.f.z3; }
    value_type & z3() { return m_data.f.z3; }
    void set_z3(value_type v) { m_data.f.z3 = v; }

    point_type operator[](size_t i) const { return point_type(m_data.v[i], m_data.v[i + 4], m_data.v[i + 8]); }

    point_type at(size_t i) const
    {
        check_size(i, 4);
        return (*this)[i];
    }

    bool operator==(Trapezoid3d const & other) const
    {
        return m_data.v[0] == other.m_data.v[0] &&
               m_data.v[1] == other.m_data.v[1] &&
               m_data.v[2] == other.m_data.v[2] &&
               m_data.v[3] == other.m_data.v[3] &&
               m_data.v[4] == other.m_data.v[4] &&
               m_data.v[5] == other.m_data.v[5] &&
               m_data.v[6] == other.m_data.v[6] &&
               m_data.v[7] == other.m_data.v[7] &&
               m_data.v[8] == other.m_data.v[8] &&
               m_data.v[9] == other.m_data.v[9] &&
               m_data.v[10] == other.m_data.v[10] &&
               m_data.v[11] == other.m_data.v[11];
    }

    bool operator!=(Trapezoid3d const & other) const
    {
        return m_data.v[0] != other.m_data.v[0] ||
               m_data.v[1] != other.m_data.v[1] ||
               m_data.v[2] != other.m_data.v[2] ||
               m_data.v[3] != other.m_data.v[3] ||
               m_data.v[4] != other.m_data.v[4] ||
               m_data.v[5] != other.m_data.v[5] ||
               m_data.v[6] != other.m_data.v[6] ||
               m_data.v[7] != other.m_data.v[7] ||
               m_data.v[8] != other.m_data.v[8] ||
               m_data.v[9] != other.m_data.v[9] ||
               m_data.v[10] != other.m_data.v[10] ||
               m_data.v[11] != other.m_data.v[11];
    }

    size_t size() const { return 4; }

    void mirror_x()
    {
        m_data.f.y0 = -m_data.f.y0;
        m_data.f.y1 = -m_data.f.y1;
        m_data.f.y2 = -m_data.f.y2;
        m_data.f.y3 = -m_data.f.y3;
        m_data.f.z0 = -m_data.f.z0;
        m_data.f.z1 = -m_data.f.z1;
        m_data.f.z2 = -m_data.f.z2;
        m_data.f.z3 = -m_data.f.z3;
    }

    void mirror_y()
    {
        m_data.f.x0 = -m_data.f.x0;
        m_data.f.x1 = -m_data.f.x1;
        m_data.f.x2 = -m_data.f.x2;
        m_data.f.x3 = -m_data.f.x3;
        m_data.f.z0 = -m_data.f.z0;
        m_data.f.z1 = -m_data.f.z1;
        m_data.f.z2 = -m_data.f.z2;
        m_data.f.z3 = -m_data.f.z3;
    }

    void mirror_z()
    {
        m_data.f.x0 = -m_data.f.x0;
        m_data.f.x1 = -m_data.f.x1;
        m_data.f.x2 = -m_data.f.x2;
        m_data.f.x3 = -m_data.f.x3;
        m_data.f.y0 = -m_data.f.y0;
        m_data.f.y1 = -m_data.f.y1;
        m_data.f.y2 = -m_data.f.y2;
        m_data.f.y3 = -m_data.f.y3;
    }

    void mirror(Axis axis)
    {
        switch (axis)
        {
        case Axis::X: mirror_x(); break;
        case Axis::Y: mirror_y(); break;
        case Axis::Z: mirror_z(); break;
        default: throw std::invalid_argument("Trapezoid3d::mirror: invalid axis"); break;
        }
    }

private:

    void check_size(size_t i, size_t s) const
    {
        if (i >= s)
        {
            throw std::out_of_range(std::format("Trapezoid3d: i {} >= size {}", i, s));
        }
    }

    detail::Trapezoid3dData<T> m_data;

}; /* end class Trapezoid3d */

using Trapezoid3dFp32 = Trapezoid3d<float>;
using Trapezoid3dFp64 = Trapezoid3d<double>;

/**
 * Container for many trapezoids stored as four parallel point columns.
 *
 * The four corner points (p0, p1, p2, p3) of every trapezoid are kept in four
 * separate PointPad instances, so trapezoid i is read by gathering element i
 * from each column. The pad works for both 2 and 3 dimensional points.
 *
 * @tparam T floating-point type
 * @ingroup group_geometry
 */
template <typename T>
class TrapezoidPad
    : public NumberBase<int32_t, T>
    , public std::enable_shared_from_this<TrapezoidPad<T>>
{

private:

    struct ctor_passkey
    {
    };

public:

    using real_type = T;
    using value_type = T;
    using point_type = Point3d<T>;
    using trapezoid_type = Trapezoid3d<T>;
    using point_pad_type = PointPad<T>;

    template <typename... Args>
    static std::shared_ptr<TrapezoidPad<T>> construct(Args &&... args)
    {
        return std::make_shared<TrapezoidPad<T>>(std::forward<Args>(args)..., ctor_passkey());
    }

    TrapezoidPad(uint8_t ndim, ctor_passkey const &)
        : m_p0(point_pad_type::construct(ndim))
        , m_p1(point_pad_type::construct(ndim))
        , m_p2(point_pad_type::construct(ndim))
        , m_p3(point_pad_type::construct(ndim))
    {
    }

    TrapezoidPad(uint8_t ndim, size_t nelem, ctor_passkey const &)
        : m_p0(point_pad_type::construct(ndim, nelem))
        , m_p1(point_pad_type::construct(ndim, nelem))
        , m_p2(point_pad_type::construct(ndim, nelem))
        , m_p3(point_pad_type::construct(ndim, nelem))
    {
    }

    TrapezoidPad(
        SimpleArray<T> const & x0,
        SimpleArray<T> const & y0,
        SimpleArray<T> const & x1,
        SimpleArray<T> const & y1,
        SimpleArray<T> const & x2,
        SimpleArray<T> const & y2,
        SimpleArray<T> const & x3,
        SimpleArray<T> const & y3,
        ctor_passkey const &)
        : m_p0(point_pad_type::construct(x0, y0))
        , m_p1(point_pad_type::construct(x1, y1))
        , m_p2(point_pad_type::construct(x2, y2))
        , m_p3(point_pad_type::construct(x3, y3))
    {
        check_constructor_point_size(*m_p0, *m_p1, *m_p2, *m_p3);
    }

    TrapezoidPad(
        SimpleArray<T> const & x0,
        SimpleArray<T> const & y0,
        SimpleArray<T> const & z0,
        SimpleArray<T> const & x1,
        SimpleArray<T> const & y1,
        SimpleArray<T> const & z1,
        SimpleArray<T> const & x2,
        SimpleArray<T> const & y2,
        SimpleArray<T> const & z2,
        SimpleArray<T> const & x3,
        SimpleArray<T> const & y3,
        SimpleArray<T> const & z3,
        ctor_passkey const &)
        : m_p0(point_pad_type::construct(x0, y0, z0))
        , m_p1(point_pad_type::construct(x1, y1, z1))
        , m_p2(point_pad_type::construct(x2, y2, z2))
        , m_p3(point_pad_type::construct(x3, y3, z3))
    {
        check_constructor_point_size(*m_p0, *m_p1, *m_p2, *m_p3);
    }

    TrapezoidPad(
        SimpleArray<T> & x0,
        SimpleArray<T> & y0,
        SimpleArray<T> & x1,
        SimpleArray<T> & y1,
        SimpleArray<T> & x2,
        SimpleArray<T> & y2,
        SimpleArray<T> & x3,
        SimpleArray<T> & y3,
        bool clone,
        ctor_passkey const &)
        : m_p0(point_pad_type::construct(x0, y0, clone))
        , m_p1(point_pad_type::construct(x1, y1, clone))
        , m_p2(point_pad_type::construct(x2, y2, clone))
        , m_p3(point_pad_type::construct(x3, y3, clone))
    {
        check_constructor_point_size(*m_p0, *m_p1, *m_p2, *m_p3);
    }

    TrapezoidPad(
        SimpleArray<T> & x0,
        SimpleArray<T> & y0,
        SimpleArray<T> & z0,
        SimpleArray<T> & x1,
        SimpleArray<T> & y1,
        SimpleArray<T> & z1,
        SimpleArray<T> & x2,
        SimpleArray<T> & y2,
        SimpleArray<T> & z2,
        SimpleArray<T> & x3,
        SimpleArray<T> & y3,
        SimpleArray<T> & z3,
        bool clone,
        ctor_passkey const &)
        : m_p0(point_pad_type::construct(x0, y0, z0, clone))
        , m_p1(point_pad_type::construct(x1, y1, z1, clone))
        , m_p2(point_pad_type::construct(x2, y2, z2, clone))
        , m_p3(point_pad_type::construct(x3, y3, z3, clone))
    {
        check_constructor_point_size(*m_p0, *m_p1, *m_p2, *m_p3);
    }

    std::shared_ptr<TrapezoidPad<T>> clone()
    {
        if (ndim() == 2)
        {
            return TrapezoidPad<T>::construct(x0(), y0(), x1(), y1(), x2(), y2(), x3(), y3());
        }
        else
        {
            return TrapezoidPad<T>::construct(x0(), y0(), z0(), x1(), y1(), z1(), x2(), y2(), z2(), x3(), y3(), z3());
        }
    }

    TrapezoidPad() = delete;
    TrapezoidPad(TrapezoidPad const &) = delete;
    TrapezoidPad(TrapezoidPad &&) = delete;
    TrapezoidPad & operator=(TrapezoidPad const &) = delete;
    TrapezoidPad & operator=(TrapezoidPad &&) = delete;

    ~TrapezoidPad() = default;

    void append(trapezoid_type const & t)
    {
        if (ndim() == 2)
        {
            m_p0->append(t.x0(), t.y0());
            m_p1->append(t.x1(), t.y1());
            m_p2->append(t.x2(), t.y2());
            m_p3->append(t.x3(), t.y3());
        }
        else
        {
            m_p0->append(t.x0(), t.y0(), t.z0());
            m_p1->append(t.x1(), t.y1(), t.z1());
            m_p2->append(t.x2(), t.y2(), t.z2());
            m_p3->append(t.x3(), t.y3(), t.z3());
        }
    }

    void append(point_type const & p0, point_type const & p1, point_type const & p2, point_type const & p3)
    {
        if (ndim() == 2)
        {
            m_p0->append(p0.x(), p0.y());
            m_p1->append(p1.x(), p1.y());
            m_p2->append(p2.x(), p2.y());
            m_p3->append(p3.x(), p3.y());
        }
        else
        {
            m_p0->append(p0.x(), p0.y(), p0.z());
            m_p1->append(p1.x(), p1.y(), p1.z());
            m_p2->append(p2.x(), p2.y(), p2.z());
            m_p3->append(p3.x(), p3.y(), p3.z());
        }
    }

    void append(T x0, T y0, T x1, T y1, T x2, T y2, T x3, T y3)
    {
        m_p0->append(x0, y0);
        m_p1->append(x1, y1);
        m_p2->append(x2, y2);
        m_p3->append(x3, y3);
    }

    void append(T x0, T y0, T z0, T x1, T y1, T z1, T x2, T y2, T z2, T x3, T y3, T z3)
    {
        m_p0->append(x0, y0, z0);
        m_p1->append(x1, y1, z1);
        m_p2->append(x2, y2, z2);
        m_p3->append(x3, y3, z3);
    }

    void extend_with(TrapezoidPad<T> const & other)
    {
        size_t const ntrap = other.size();
        for (size_t i = 0; i < ntrap; ++i)
        {
            append(other.get(i));
        }
    }

    uint8_t ndim() const { return m_p0->ndim(); }

    size_t size() const { return m_p0->size(); }

    SimpleArray<T> pack_array() const
    {
        using shape_type = typename SimpleArray<T>::shape_type;
        SimpleArray<T> ret(shape_type{static_cast<ssize_t>(m_p0->size()), ndim() * 4});
        if (ndim() == 3)
        {
            for (size_t i = 0; i < m_p0->size(); ++i)
            {
                ret(i, 0) = m_p0->x(i);
                ret(i, 1) = m_p0->y(i);
                ret(i, 2) = m_p0->z(i);
                ret(i, 3) = m_p1->x(i);
                ret(i, 4) = m_p1->y(i);
                ret(i, 5) = m_p1->z(i);
                ret(i, 6) = m_p2->x(i);
                ret(i, 7) = m_p2->y(i);
                ret(i, 8) = m_p2->z(i);
                ret(i, 9) = m_p3->x(i);
                ret(i, 10) = m_p3->y(i);
                ret(i, 11) = m_p3->z(i);
            }
        }
        else
        {
            for (size_t i = 0; i < m_p0->size(); ++i)
            {
                ret(i, 0) = m_p0->x(i);
                ret(i, 1) = m_p0->y(i);
                ret(i, 2) = m_p1->x(i);
                ret(i, 3) = m_p1->y(i);
                ret(i, 4) = m_p2->x(i);
                ret(i, 5) = m_p2->y(i);
                ret(i, 6) = m_p3->x(i);
                ret(i, 7) = m_p3->y(i);
            }
        }
        return ret;
    }

    void expand(size_t length)
    {
        m_p0->expand(length);
        m_p1->expand(length);
        m_p2->expand(length);
        m_p3->expand(length);
    }

    real_type x0_at(size_t i) const { return m_p0->x_at(i); }
    real_type y0_at(size_t i) const { return m_p0->y_at(i); }
    real_type z0_at(size_t i) const { return m_p0->z_at(i); }
    real_type x1_at(size_t i) const { return m_p1->x_at(i); }
    real_type y1_at(size_t i) const { return m_p1->y_at(i); }
    real_type z1_at(size_t i) const { return m_p1->z_at(i); }
    real_type x2_at(size_t i) const { return m_p2->x_at(i); }
    real_type y2_at(size_t i) const { return m_p2->y_at(i); }
    real_type z2_at(size_t i) const { return m_p2->z_at(i); }
    real_type x3_at(size_t i) const { return m_p3->x_at(i); }
    real_type y3_at(size_t i) const { return m_p3->y_at(i); }
    real_type z3_at(size_t i) const { return m_p3->z_at(i); }
    real_type & x0_at(size_t i) { return m_p0->x_at(i); }
    real_type & y0_at(size_t i) { return m_p0->y_at(i); }
    real_type & z0_at(size_t i) { return m_p0->z_at(i); }
    real_type & x1_at(size_t i) { return m_p1->x_at(i); }
    real_type & y1_at(size_t i) { return m_p1->y_at(i); }
    real_type & z1_at(size_t i) { return m_p1->z_at(i); }
    real_type & x2_at(size_t i) { return m_p2->x_at(i); }
    real_type & y2_at(size_t i) { return m_p2->y_at(i); }
    real_type & z2_at(size_t i) { return m_p2->z_at(i); }
    real_type & x3_at(size_t i) { return m_p3->x_at(i); }
    real_type & y3_at(size_t i) { return m_p3->y_at(i); }
    real_type & z3_at(size_t i) { return m_p3->z_at(i); }

    real_type x0(size_t i) const { return m_p0->x(i); }
    real_type y0(size_t i) const { return m_p0->y(i); }
    real_type z0(size_t i) const { return m_p0->z(i); }
    real_type x1(size_t i) const { return m_p1->x(i); }
    real_type y1(size_t i) const { return m_p1->y(i); }
    real_type z1(size_t i) const { return m_p1->z(i); }
    real_type x2(size_t i) const { return m_p2->x(i); }
    real_type y2(size_t i) const { return m_p2->y(i); }
    real_type z2(size_t i) const { return m_p2->z(i); }
    real_type x3(size_t i) const { return m_p3->x(i); }
    real_type y3(size_t i) const { return m_p3->y(i); }
    real_type z3(size_t i) const { return m_p3->z(i); }
    real_type & x0(size_t i) { return m_p0->x(i); }
    real_type & y0(size_t i) { return m_p0->y(i); }
    real_type & z0(size_t i) { return m_p0->z(i); }
    real_type & x1(size_t i) { return m_p1->x(i); }
    real_type & y1(size_t i) { return m_p1->y(i); }
    real_type & z1(size_t i) { return m_p1->z(i); }
    real_type & x2(size_t i) { return m_p2->x(i); }
    real_type & y2(size_t i) { return m_p2->y(i); }
    real_type & z2(size_t i) { return m_p2->z(i); }
    real_type & x3(size_t i) { return m_p3->x(i); }
    real_type & y3(size_t i) { return m_p3->y(i); }
    real_type & z3(size_t i) { return m_p3->z(i); }

    point_type p0_at(size_t i) const { return m_p0->get_at(i); }
    point_type p1_at(size_t i) const { return m_p1->get_at(i); }
    point_type p2_at(size_t i) const { return m_p2->get_at(i); }
    point_type p3_at(size_t i) const { return m_p3->get_at(i); }
    void set_p0_at(size_t i, point_type const & p) { m_p0->set_at(i, p); }
    void set_p1_at(size_t i, point_type const & p) { m_p1->set_at(i, p); }
    void set_p2_at(size_t i, point_type const & p) { m_p2->set_at(i, p); }
    void set_p3_at(size_t i, point_type const & p) { m_p3->set_at(i, p); }

    point_type p0(size_t i) const { return m_p0->get(i); }
    point_type p1(size_t i) const { return m_p1->get(i); }
    point_type p2(size_t i) const { return m_p2->get(i); }
    point_type p3(size_t i) const { return m_p3->get(i); }
    void set_p0(size_t i, point_type const & p) { m_p0->set(i, p); }
    void set_p1(size_t i, point_type const & p) { m_p1->set(i, p); }
    void set_p2(size_t i, point_type const & p) { m_p2->set(i, p); }
    void set_p3(size_t i, point_type const & p) { m_p3->set(i, p); }

    SimpleArray<value_type> x0() { return m_p0->x(); }
    SimpleArray<value_type> y0() { return m_p0->y(); }
    SimpleArray<value_type> z0() { return m_p0->z(); }
    SimpleArray<value_type> x1() { return m_p1->x(); }
    SimpleArray<value_type> y1() { return m_p1->y(); }
    SimpleArray<value_type> z1() { return m_p1->z(); }
    SimpleArray<value_type> x2() { return m_p2->x(); }
    SimpleArray<value_type> y2() { return m_p2->y(); }
    SimpleArray<value_type> z2() { return m_p2->z(); }
    SimpleArray<value_type> x3() { return m_p3->x(); }
    SimpleArray<value_type> y3() { return m_p3->y(); }
    SimpleArray<value_type> z3() { return m_p3->z(); }

    std::shared_ptr<point_pad_type> p0() { return m_p0; }
    std::shared_ptr<point_pad_type> p1() { return m_p1; }
    std::shared_ptr<point_pad_type> p2() { return m_p2; }
    std::shared_ptr<point_pad_type> p3() { return m_p3; }

    trapezoid_type get_at(size_t i) const
    {
        if (ndim() == 3)
        {
            return trapezoid_type(point_type(x0_at(i), y0_at(i), z0_at(i)),
                                  point_type(x1_at(i), y1_at(i), z1_at(i)),
                                  point_type(x2_at(i), y2_at(i), z2_at(i)),
                                  point_type(x3_at(i), y3_at(i), z3_at(i)));
        }
        else
        {
            return trapezoid_type(point_type(x0_at(i), y0_at(i), 0.0),
                                  point_type(x1_at(i), y1_at(i), 0.0),
                                  point_type(x2_at(i), y2_at(i), 0.0),
                                  point_type(x3_at(i), y3_at(i), 0.0));
        }
    }
    void set_at(size_t i, trapezoid_type const & t)
    {
        x0_at(i) = t.x0();
        y0_at(i) = t.y0();
        x1_at(i) = t.x1();
        y1_at(i) = t.y1();
        x2_at(i) = t.x2();
        y2_at(i) = t.y2();
        x3_at(i) = t.x3();
        y3_at(i) = t.y3();
        if (ndim() == 3)
        {
            z0_at(i) = t.z0();
            z1_at(i) = t.z1();
            z2_at(i) = t.z2();
            z3_at(i) = t.z3();
        }
    }
    void set_at(size_t i, point_type const & p0, point_type const & p1, point_type const & p2, point_type const & p3)
    {
        x0_at(i) = p0.x();
        y0_at(i) = p0.y();
        x1_at(i) = p1.x();
        y1_at(i) = p1.y();
        x2_at(i) = p2.x();
        y2_at(i) = p2.y();
        x3_at(i) = p3.x();
        y3_at(i) = p3.y();
        if (ndim() == 3)
        {
            z0_at(i) = p0.z();
            z1_at(i) = p1.z();
            z2_at(i) = p2.z();
            z3_at(i) = p3.z();
        }
    }
    // clang-format off
    void set_at(size_t i, value_type x0, value_type y0,
                value_type x1, value_type y1,
                value_type x2, value_type y2,
                value_type x3, value_type y3)
    // clang-format on
    {
        x0_at(i) = x0;
        y0_at(i) = y0;
        x1_at(i) = x1;
        y1_at(i) = y1;
        x2_at(i) = x2;
        y2_at(i) = y2;
        x3_at(i) = x3;
        y3_at(i) = y3;
    }
    // clang-format off
    void set_at(size_t i, value_type x0, value_type y0, value_type z0,
                value_type x1, value_type y1, value_type z1,
                value_type x2, value_type y2, value_type z2,
                value_type x3, value_type y3, value_type z3)
    // clang-format on
    {
        x0_at(i) = x0;
        y0_at(i) = y0;
        x1_at(i) = x1;
        y1_at(i) = y1;
        x2_at(i) = x2;
        y2_at(i) = y2;
        x3_at(i) = x3;
        y3_at(i) = y3;
        if (ndim() == 3)
        {
            z0_at(i) = z0;
            z1_at(i) = z1;
            z2_at(i) = z2;
            z3_at(i) = z3;
        }
    }

    trapezoid_type get(size_t i) const
    {
        if (ndim() == 3)
        {
            return trapezoid_type(point_type(x0(i), y0(i), z0(i)),
                                  point_type(x1(i), y1(i), z1(i)),
                                  point_type(x2(i), y2(i), z2(i)),
                                  point_type(x3(i), y3(i), z3(i)));
        }
        else
        {
            return trapezoid_type(point_type(x0(i), y0(i), 0.0),
                                  point_type(x1(i), y1(i), 0.0),
                                  point_type(x2(i), y2(i), 0.0),
                                  point_type(x3(i), y3(i), 0.0));
        }
    }
    void set(size_t i, trapezoid_type const & t)
    {
        x0(i) = t.x0();
        y0(i) = t.y0();
        x1(i) = t.x1();
        y1(i) = t.y1();
        x2(i) = t.x2();
        y2(i) = t.y2();
        x3(i) = t.x3();
        y3(i) = t.y3();
        if (ndim() == 3)
        {
            z0(i) = t.z0();
            z1(i) = t.z1();
            z2(i) = t.z2();
            z3(i) = t.z3();
        }
    }
    void set(size_t i, point_type const & p0, point_type const & p1, point_type const & p2, point_type const & p3)
    {
        x0(i) = p0.x();
        y0(i) = p0.y();
        x1(i) = p1.x();
        y1(i) = p1.y();
        x2(i) = p2.x();
        y2(i) = p2.y();
        x3(i) = p3.x();
        y3(i) = p3.y();
        if (ndim() == 3)
        {
            z0(i) = p0.z();
            z1(i) = p1.z();
            z2(i) = p2.z();
            z3(i) = p3.z();
        }
    }
    // clang-format off
    void set(size_t i, value_type x0_value, value_type y0_value,
             value_type x1_value, value_type y1_value,
             value_type x2_value, value_type y2_value,
             value_type x3_value, value_type y3_value)
    // clang-format on
    {
        x0(i) = x0_value;
        y0(i) = y0_value;
        x1(i) = x1_value;
        y1(i) = y1_value;
        x2(i) = x2_value;
        y2(i) = y2_value;
        x3(i) = x3_value;
        y3(i) = y3_value;
    }
    // clang-format off
    void set(size_t i, value_type x0_value, value_type y0_value, value_type z0_value,
             value_type x1_value, value_type y1_value, value_type z1_value,
             value_type x2_value, value_type y2_value, value_type z2_value,
             value_type x3_value, value_type y3_value, value_type z3_value)
    // clang-format on
    {
        x0(i) = x0_value;
        y0(i) = y0_value;
        x1(i) = x1_value;
        y1(i) = y1_value;
        x2(i) = x2_value;
        y2(i) = y2_value;
        x3(i) = x3_value;
        y3(i) = y3_value;
        if (ndim() == 3)
        {
            z0(i) = z0_value;
            z1(i) = z1_value;
            z2(i) = z2_value;
            z3(i) = z3_value;
        }
    }

    void mirror_x()
    {
        size_t const ntrap = size();
        for (size_t i = 0; i < ntrap; ++i)
        {
            y0(i) = -y0(i);
            y1(i) = -y1(i);
            y2(i) = -y2(i);
            y3(i) = -y3(i);
            if (ndim() == 3)
            {
                z0(i) = -z0(i);
                z1(i) = -z1(i);
                z2(i) = -z2(i);
                z3(i) = -z3(i);
            }
        }
    }

    void mirror_y()
    {
        size_t const ntrap = size();
        for (size_t i = 0; i < ntrap; ++i)
        {
            x0(i) = -x0(i);
            x1(i) = -x1(i);
            x2(i) = -x2(i);
            x3(i) = -x3(i);
            if (ndim() == 3)
            {
                z0(i) = -z0(i);
                z1(i) = -z1(i);
                z2(i) = -z2(i);
                z3(i) = -z3(i);
            }
        }
    }

    void mirror_z()
    {
        if (ndim() != 3)
        {
            throw std::out_of_range(
                std::format("TrapezoidPad::mirror_z: cannot mirror Z axis for ndim {}", int(ndim())));
        }
        size_t const ntrap = size();
        for (size_t i = 0; i < ntrap; ++i)
        {
            x0(i) = -x0(i);
            x1(i) = -x1(i);
            x2(i) = -x2(i);
            x3(i) = -x3(i);
            y0(i) = -y0(i);
            y1(i) = -y1(i);
            y2(i) = -y2(i);
            y3(i) = -y3(i);
        }
    }

    void mirror(Axis axis)
    {
        switch (axis)
        {
        case Axis::X: mirror_x(); break;
        case Axis::Y: mirror_y(); break;
        case Axis::Z: mirror_z(); break;
        default: throw std::invalid_argument("TrapezoidPad::mirror: invalid axis"); break;
        }
    }

private:

    void check_constructor_point_size(point_pad_type const & p0, point_pad_type const & p1, point_pad_type const & p2, point_pad_type const & p3)
    {
        if (m_p0->size() != m_p1->size() || m_p0->size() != m_p2->size() || m_p0->size() != m_p3->size())
        {
            throw std::invalid_argument(
                std::format("TrapezoidPad::TrapezoidPad: p0.size() {} p1.size() {} p2.size() {} p3.size() {} are not the same",
                            p0.size(),
                            p1.size(),
                            p2.size(),
                            p3.size()));
        }
    }

    std::shared_ptr<point_pad_type> m_p0;
    std::shared_ptr<point_pad_type> m_p1;
    std::shared_ptr<point_pad_type> m_p2;
    std::shared_ptr<point_pad_type> m_p3;

}; /* end class TrapezoidPad */

using TrapezoidPadFp32 = TrapezoidPad<float>;
using TrapezoidPadFp64 = TrapezoidPad<double>;

/**
 * Forward declaration of Polygon3d for use in helper classes.
 */
template <typename T>
class Polygon3d;

/**
 * Forward declaration of PolygonPad for use in Polygon3d handle class.
 */
template <typename T>
class PolygonPad;

/**
 * Helper class for trapezoidal decomposition of polygons.
 *
 * This class implements the sweep line algorithm to decompose polygons into
 * trapezoids. The decomposition is used for polygon boolean operations.
 *
 * @tparam T floating-point type
 * @ingroup group_geometry
 */
template <typename T>
class TrapezoidalDecomposer
{

public:

    using value_type = T;
    using point_type = Point3d<T>;
    using trapezoid_pad_type = TrapezoidPad<T>;
    using ssize_type = ssize_t;

private:
    /**
     * Internal structure representing a trapezoid during decomposition for the sweep line algorithm.
     * The trap has bottom and top lines perpendicular to the Y-axis, defined by lower_y and upper_y.
     */
    struct YTrap
    {
        value_type lower_y, upper_y;
        value_type lower_x0, lower_x1, upper_x0, upper_x1;
        value_type lower_z0, lower_z1, upper_z0, upper_z1; // TODO: currently unused for 2D decomposition
        size_t source_polygon;
    }; /* end struct YTrap */

    struct Edge
    {
        value_type lower_y, upper_y;
        value_type dxdy; // delta x / delta y
        value_type dzdy; // delta z / delta y
        value_type x_at_lower_y;
        value_type z_at_lower_y; // TODO: currently unused for 2D decomposition
    }; /* end struct Edge */

    using SweepEvent = std::vector<Edge>;

public:

    explicit TrapezoidalDecomposer(uint8_t ndim)
        : m_trapezoids(trapezoid_pad_type::construct(ndim))
    {
    }

    TrapezoidalDecomposer() = delete;
    TrapezoidalDecomposer(TrapezoidalDecomposer const &) = delete;
    TrapezoidalDecomposer(TrapezoidalDecomposer &&) = delete;
    TrapezoidalDecomposer & operator=(TrapezoidalDecomposer const &) = delete;
    TrapezoidalDecomposer & operator=(TrapezoidalDecomposer &&) = delete;
    ~TrapezoidalDecomposer() = default;

    /**
     * Decompose a polygon into trapezoids using vertical sweep line algorithm.
     *
     * @param polygon_id ID of the polygon to decompose
     * @param points Vector of polygon vertices in order
     * @return Pair of begin and end indices into the trapezoid pad, where the begin index is inclusive and the end index is exclusive
     */
    std::pair<size_t, size_t> decompose(size_t polygon_id, std::vector<point_type> const & points);

    size_t num_trapezoids(size_t polygon_id) const
    {
        if (polygon_id >= m_begins.size() || m_begins[polygon_id] == -1)
        {
            return 0;
        }
        return static_cast<size_t>(m_ends[polygon_id] - m_begins[polygon_id]);
    }

    std::shared_ptr<trapezoid_pad_type> trapezoids() { return m_trapezoids; }
    std::shared_ptr<trapezoid_pad_type const> trapezoids() const { return m_trapezoids; }

    void clear()
    {
        m_begins.clear();
        m_ends.clear();
        m_trapezoids = trapezoid_pad_type::construct(m_trapezoids->ndim());
    }

private:

    void build_edges_and_events(std::vector<point_type> const & points,
                                std::map<value_type, SweepEvent> & events,
                                std::vector<value_type> & y_values);

    std::shared_ptr<trapezoid_pad_type> m_trapezoids;
    std::vector<ssize_type> m_begins;
    std::vector<ssize_type> m_ends;

}; /* end class TrapezoidalDecomposer */

using TrapezoidalDecomposerFp32 = TrapezoidalDecomposer<float>;
using TrapezoidalDecomposerFp64 = TrapezoidalDecomposer<double>;

template <typename T>
void TrapezoidalDecomposer<T>::build_edges_and_events(std::vector<point_type> const & points,
                                                      std::map<value_type, SweepEvent> & events,
                                                      std::vector<value_type> & y_values)
{
    size_t const npoints = points.size();
    for (size_t i = 0; i < npoints; ++i)
    {
        point_type const & p1 = points[i];
        point_type const & p2 = points[(i + 1) % npoints];

        if (p1.y() == p2.y())
        {
            continue;
        }

        Edge edge; // FIXME: NOLINT(cppcoreguidelines-pro-type-member-init)
        if (p1.y() < p2.y())
        {
            edge.lower_y = p1.y();
            edge.upper_y = p2.y();
            edge.x_at_lower_y = p1.x();
            edge.dxdy = (p2.x() - p1.x()) / (p2.y() - p1.y());
            if (m_trapezoids->ndim() == 3)
            {
                // TODO: currently unused for 2D decomposition
                throw std::runtime_error("TrapezoidalDecomposer::build_edges_and_events: 3D edge dzdy not implemented");
            }
        }
        else
        {
            edge.lower_y = p2.y();
            edge.upper_y = p1.y();
            edge.x_at_lower_y = p2.x();
            edge.dxdy = (p1.x() - p2.x()) / (p1.y() - p2.y());
            if (m_trapezoids->ndim() == 3)
            {
                // TODO: currently unused for 2D decomposition
                throw std::runtime_error("TrapezoidalDecomposer::build_edges_and_events: 3D edge dzdy not implemented");
            }
        }

        events[edge.lower_y].push_back(edge);
    }

    // TODO: optimize the collection of unique y-values
    std::set<value_type> unique_y_values;
    for (const auto & [_, edges] : events)
    {
        for (const Edge & e : edges)
        {
            unique_y_values.insert(e.lower_y);
            unique_y_values.insert(e.upper_y);
        }
    }
    y_values.assign(unique_y_values.begin(), unique_y_values.end());
}

template <typename T>
std::pair<size_t, size_t> TrapezoidalDecomposer<T>::decompose(size_t polygon_id, std::vector<point_type> const & points)
{
    if (polygon_id >= m_begins.size())
    {
        m_begins.reserve(polygon_id + 1);
        m_ends.reserve(polygon_id + 1);
        while (m_begins.size() <= polygon_id)
        {
            m_begins.push_back(-1);
            m_ends.push_back(-1);
        }
    }

    if (m_begins[polygon_id] != -1)
    {
        return {static_cast<size_t>(m_begins[polygon_id]), static_cast<size_t>(m_ends[polygon_id])};
    }

    size_t const begin_index = m_trapezoids->size();

    std::map<value_type, SweepEvent> events; // y-value to events starting at that y
    std::vector<value_type> y_values;
    build_edges_and_events(points, events, y_values);

    if (y_values.size() < 2)
    {
        // Not enough y-values to form trapezoids
        m_begins[polygon_id] = static_cast<ssize_type>(begin_index);
        m_ends[polygon_id] = static_cast<ssize_type>(begin_index);
        return {begin_index, begin_index};
    }

    std::vector<Edge> active_edges;

    // Sweep from bottom to top
    for (size_t i = 0; i < y_values.size() - 1; ++i)
    {
        value_type const y_current = y_values[i];
        value_type const y_next = y_values[i + 1];

        // Remove ending edges from the active list
        // TODO: be aware of O(n^2) complexity
        active_edges.erase(std::remove_if(active_edges.begin(), active_edges.end(), [y_current](const Edge & e)
                                          { return e.upper_y == y_current; }),
                           active_edges.end());

        // Add starting edges to the active list
        for (const Edge & e : events[y_current])
        {
            active_edges.push_back(e);
        }

        if (active_edges.size() % 2 != 0)
        {
            throw std::runtime_error("TrapezoidalDecomposer::decompose: odd number of active edges at y = " + std::to_string(y_current));
        }

        // Sort active edges by x at current y
        std::sort(active_edges.begin(), active_edges.end(), [y_current](const Edge & a, const Edge & b)
                  {
                      value_type const ax = a.x_at_lower_y + a.dxdy * (y_current - a.lower_y);
                      value_type const bx = b.x_at_lower_y + b.dxdy * (y_current - b.lower_y);
                      if (ax != bx)
                      {
                          return ax < bx;
                      }
                      // If x values are equal, sort by dxdy to ensure consistent ordering of edges
                      return a.dxdy < b.dxdy; });

        // Create trapezoids between pairs of active edges
        for (size_t j = 0; j < active_edges.size() - 1; j += 2)
        {
            const Edge & left_edge = active_edges[j];
            const Edge & right_edge = active_edges[j + 1];

            value_type const lower_x0 = left_edge.x_at_lower_y + left_edge.dxdy * (y_current - left_edge.lower_y);
            value_type const lower_x1 = right_edge.x_at_lower_y + right_edge.dxdy * (y_current - right_edge.lower_y);
            value_type const upper_x0 = left_edge.x_at_lower_y + left_edge.dxdy * (y_next - left_edge.lower_y);
            value_type const upper_x1 = right_edge.x_at_lower_y + right_edge.dxdy * (y_next - right_edge.lower_y);

            if (m_trapezoids->ndim() == 3)
            {
                // TODO: currently unused for 2D decomposition
                throw std::runtime_error("TrapezoidalDecomposer::decompose: 3D trapezoid creation not implemented");
            }
            else
            {
                m_trapezoids->append(lower_x0, y_current, lower_x1, y_current, upper_x1, y_next, upper_x0, y_next);
            }
        }
    }

    size_t const end_index = m_trapezoids->size();

    m_begins[polygon_id] = static_cast<ssize_type>(begin_index);
    m_ends[polygon_id] = static_cast<ssize_type>(end_index);

    return {begin_index, end_index};
}

/**
 * This class implements the union algorithm for two polygons by decomposing them
 * into trapezoids and merging overlapping regions.
 *
 * @tparam T floating-point type
 * @ingroup group_geometry
 */
template <typename T>
class AreaBooleanUnion
{

public:

    using value_type = T;
    using point_type = Point3d<T>;
    using polygon_type = Polygon3d<T>;
    using polygon_pad_type = PolygonPad<T>;

    AreaBooleanUnion() = default;
    AreaBooleanUnion(AreaBooleanUnion const &) = delete;
    AreaBooleanUnion(AreaBooleanUnion &&) = delete;
    AreaBooleanUnion & operator=(AreaBooleanUnion const &) = delete;
    AreaBooleanUnion & operator=(AreaBooleanUnion &&) = delete;
    ~AreaBooleanUnion() = default;

    /**
     * Compute union of two polygons.
     *
     * @param pad Polygon container holding both polygons
     * @param polygon_id1 ID of first polygon
     * @param polygon_id2 ID of second polygon
     * @return polygon pad containing the union of the two polygons
     */
    std::shared_ptr<polygon_pad_type> compute(const std::shared_ptr<polygon_pad_type> & pad, size_t polygon_id1, size_t polygon_id2);

}; /* end class AreaBooleanUnion */

using AreaBooleanUnionFp32 = AreaBooleanUnion<float>;
using AreaBooleanUnionFp64 = AreaBooleanUnion<double>;

/**
 * This class implements the intersection algorithm for two polygons by decomposing them
 * into trapezoids and finding overlapping regions.
 *
 * @tparam T floating-point type
 * @ingroup group_geometry
 */
template <typename T>
class AreaBooleanIntersection
{

public:

    using value_type = T;
    using point_type = Point3d<T>;
    using polygon_type = Polygon3d<T>;
    using polygon_pad_type = PolygonPad<T>;

    AreaBooleanIntersection() = default;
    AreaBooleanIntersection(AreaBooleanIntersection const &) = delete;
    AreaBooleanIntersection(AreaBooleanIntersection &&) = delete;
    AreaBooleanIntersection & operator=(AreaBooleanIntersection const &) = delete;
    AreaBooleanIntersection & operator=(AreaBooleanIntersection &&) = delete;
    ~AreaBooleanIntersection() = default;

    /**
     * Compute intersection of two polygons.
     *
     * @param pad Polygon container holding both polygons
     * @param polygon_id1 ID of first polygon
     * @param polygon_id2 ID of second polygon
     * @return polygon pad containing polygons forming the intersection
     */
    std::shared_ptr<polygon_pad_type> compute(const std::shared_ptr<polygon_pad_type> & pad, size_t polygon_id1, size_t polygon_id2);

}; /* end class AreaBooleanIntersection */

using AreaBooleanIntersectionFp32 = AreaBooleanIntersection<float>;
using AreaBooleanIntersectionFp64 = AreaBooleanIntersection<double>;

/**
 * This class implements the difference algorithm for two polygons (p1 - p2) by decomposing
 * them into trapezoids and removing overlapping regions.
 *
 * @tparam T floating-point type
 * @ingroup group_geometry
 */
template <typename T>
class AreaBooleanDifference
{

public:

    using value_type = T;
    using point_type = Point3d<T>;
    using polygon_type = Polygon3d<T>;
    using polygon_pad_type = PolygonPad<T>;

    AreaBooleanDifference() = default;
    AreaBooleanDifference(AreaBooleanDifference const &) = delete;
    AreaBooleanDifference(AreaBooleanDifference &&) = delete;
    AreaBooleanDifference & operator=(AreaBooleanDifference const &) = delete;
    AreaBooleanDifference & operator=(AreaBooleanDifference &&) = delete;
    ~AreaBooleanDifference() = default;

    /**
     * Compute difference of two polygons (p1 - p2).
     *
     * @param pad Polygon container holding both polygons
     * @param polygon_id1 ID of first polygon
     * @param polygon_id2 ID of second polygon to subtract
     * @return polygon pad containing polygons forming the difference
     */
    std::shared_ptr<polygon_pad_type> compute(const std::shared_ptr<polygon_pad_type> & pad, size_t polygon_id1, size_t polygon_id2);

}; /* end class AreaBooleanDifference */

using AreaBooleanDifferenceFp32 = AreaBooleanDifference<float>;
using AreaBooleanDifferenceFp64 = AreaBooleanDifference<double>;

/**
 * Triangle in three-dimensional space.
 *
 * Stores the three corner points (p0, p1, p2) as nine scalar coordinates in a
 * flat buffer.
 *
 * @tparam T floating-point type
 * @ingroup group_geometry
 */
template <typename T>
class Triangle3d
    : public NumberBase<int32_t, T>
{

public:

    static_assert(std::is_arithmetic_v<T>, "T in Triangle3d<T> must be arithmetic type");

    using point_type = Point3d<T>;
    using value_type = typename point_type::value_type;

    Triangle3d(point_type const & p0, point_type const & p1, point_type const & p2)
        : m_data{p0.x(), p1.x(), p2.x(), p0.y(), p1.y(), p2.y(), p0.z(), p1.z(), p2.z()}
    {
    }

    Triangle3d() = default;
    Triangle3d(Triangle3d const &) = default;
    Triangle3d & operator=(Triangle3d const &) = default;
    Triangle3d(Triangle3d &&) = default;
    Triangle3d & operator=(Triangle3d &&) = default;
    ~Triangle3d() = default;

    point_type p0() const { return point_type(m_data.f.x0, m_data.f.y0, m_data.f.z0); }
    void set_p0(point_type const & p)
    {
        m_data.f.x0 = p.x();
        m_data.f.y0 = p.y();
        m_data.f.z0 = p.z();
    }
    point_type p1() const { return point_type(m_data.f.x1, m_data.f.y1, m_data.f.z1); }
    void set_p1(point_type const & p)
    {
        m_data.f.x1 = p.x();
        m_data.f.y1 = p.y();
        m_data.f.z1 = p.z();
    }
    point_type p2() const { return point_type(m_data.f.x2, m_data.f.y2, m_data.f.z2); }
    void set_p2(point_type const & p)
    {
        m_data.f.x2 = p.x();
        m_data.f.y2 = p.y();
        m_data.f.z2 = p.z();
    }

    value_type x0() const { return m_data.f.x0; }
    value_type & x0() { return m_data.f.x0; }
    void set_x0(value_type v) { m_data.f.x0 = v; }

    value_type y0() const { return m_data.f.y0; }
    value_type & y0() { return m_data.f.y0; }
    void set_y0(value_type v) { m_data.f.y0 = v; }

    value_type z0() const { return m_data.f.z0; }
    value_type & z0() { return m_data.f.z0; }
    void set_z0(value_type v) { m_data.f.z0 = v; }

    value_type x1() const { return m_data.f.x1; }
    value_type & x1() { return m_data.f.x1; }
    void set_x1(value_type v) { m_data.f.x1 = v; }

    value_type y1() const { return m_data.f.y1; }
    value_type & y1() { return m_data.f.y1; }
    void set_y1(value_type v) { m_data.f.y1 = v; }

    value_type z1() const { return m_data.f.z1; }
    value_type & z1() { return m_data.f.z1; }
    void set_z1(value_type v) { m_data.f.z1 = v; }

    value_type x2() const { return m_data.f.x2; }
    value_type & x2() { return m_data.f.x2; }
    void set_x2(value_type v) { m_data.f.x2 = v; }

    value_type y2() const { return m_data.f.y2; }
    value_type & y2() { return m_data.f.y2; }
    void set_y2(value_type v) { m_data.f.y2 = v; }

    value_type z2() const { return m_data.f.z2; }
    value_type & z2() { return m_data.f.z2; }
    void set_z2(value_type v) { m_data.f.z2 = v; }

    point_type operator[](size_t i) const { return point_type(m_data.v[i], m_data.v[i + 3], m_data.v[i + 6]); }

    point_type at(size_t i) const
    {
        check_size(i, 3);
        return (*this)[i];
    }

    bool operator==(Triangle3d const & other) const
    {
        return m_data.v[0] == other.m_data.v[0] &&
               m_data.v[1] == other.m_data.v[1] &&
               m_data.v[2] == other.m_data.v[2] &&
               m_data.v[3] == other.m_data.v[3] &&
               m_data.v[4] == other.m_data.v[4] &&
               m_data.v[5] == other.m_data.v[5] &&
               m_data.v[6] == other.m_data.v[6] &&
               m_data.v[7] == other.m_data.v[7] &&
               m_data.v[8] == other.m_data.v[8];
    }

    bool operator!=(Triangle3d const & other) const
    {
        return m_data.v[0] != other.m_data.v[0] ||
               m_data.v[1] != other.m_data.v[1] ||
               m_data.v[2] != other.m_data.v[2] ||
               m_data.v[3] != other.m_data.v[3] ||
               m_data.v[4] != other.m_data.v[4] ||
               m_data.v[5] != other.m_data.v[5] ||
               m_data.v[6] != other.m_data.v[6] ||
               m_data.v[7] != other.m_data.v[7] ||
               m_data.v[8] != other.m_data.v[8];
    }

    size_t size() const { return 3; }

    /**
     * Mirror the triangle with respect to the X axis.
     * This negates Y and Z coordinates, keeping X unchanged.
     */
    void mirror_x()
    {
        m_data.f.y0 = -m_data.f.y0;
        m_data.f.y1 = -m_data.f.y1;
        m_data.f.y2 = -m_data.f.y2;
        m_data.f.z0 = -m_data.f.z0;
        m_data.f.z1 = -m_data.f.z1;
        m_data.f.z2 = -m_data.f.z2;
    }

    /**
     * Mirror the triangle with respect to the Y axis.
     * This negates X and Z coordinates, keeping Y unchanged.
     */
    void mirror_y()
    {
        m_data.f.x0 = -m_data.f.x0;
        m_data.f.x1 = -m_data.f.x1;
        m_data.f.x2 = -m_data.f.x2;
        m_data.f.z0 = -m_data.f.z0;
        m_data.f.z1 = -m_data.f.z1;
        m_data.f.z2 = -m_data.f.z2;
    }

    /**
     * Mirror the triangle with respect to the Z axis.
     * This negates X and Y coordinates, keeping Z unchanged.
     */
    void mirror_z()
    {
        m_data.f.x0 = -m_data.f.x0;
        m_data.f.x1 = -m_data.f.x1;
        m_data.f.x2 = -m_data.f.x2;
        m_data.f.y0 = -m_data.f.y0;
        m_data.f.y1 = -m_data.f.y1;
        m_data.f.y2 = -m_data.f.y2;
    }

    void mirror(Axis axis)
    {
        switch (axis)
        {
        case Axis::X: mirror_x(); break;
        case Axis::Y: mirror_y(); break;
        case Axis::Z: mirror_z(); break;
        default: throw std::invalid_argument("Triangle3d::mirror: invalid axis"); break;
        }
    }

private:

    void check_size(size_t i, size_t s) const
    {
        if (i >= s)
        {
            throw std::out_of_range(std::format("Triangle3d: i {} >= size {}", i, s));
        }
    }

    detail::Triangle3dData<T> m_data;

}; /* end class Triangle3d */

using Triangle3dFp32 = Triangle3d<float>;
using Triangle3dFp64 = Triangle3d<double>;

/**
 * Container for many triangles stored as three parallel point columns.
 *
 * The three corner points (p0, p1, p2) of every triangle are kept in three
 * separate PointPad instances, so triangle i is read by gathering element i
 * from each column. The pad works for both 2 and 3 dimensional points.
 *
 * @tparam T floating-point type
 * @ingroup group_geometry
 */
template <typename T>
class TrianglePad
    : public NumberBase<int32_t, T>
    , public std::enable_shared_from_this<TrianglePad<T>>
{

private:

    struct ctor_passkey
    {
    };

public:

    using real_type = T;
    using value_type = T;
    using point_type = Point3d<T>;
    using triangle_type = Triangle3d<T>;
    using point_pad_type = PointPad<T>;

    template <typename... Args>
    static std::shared_ptr<TrianglePad<T>> construct(Args &&... args)
    {
        return std::make_shared<TrianglePad<T>>(std::forward<Args>(args)..., ctor_passkey());
    }

    TrianglePad(uint8_t ndim, ctor_passkey const &)
        : m_p0(point_pad_type::construct(ndim))
        , m_p1(point_pad_type::construct(ndim))
        , m_p2(point_pad_type::construct(ndim))
    {
    }

    TrianglePad(uint8_t ndim, size_t nelem, ctor_passkey const &)
        : m_p0(point_pad_type::construct(ndim, nelem))
        , m_p1(point_pad_type::construct(ndim, nelem))
        , m_p2(point_pad_type::construct(ndim, nelem))
    {
    }

    TrianglePad(
        SimpleArray<T> const & x0,
        SimpleArray<T> const & y0,
        SimpleArray<T> const & x1,
        SimpleArray<T> const & y1,
        SimpleArray<T> const & x2,
        SimpleArray<T> const & y2,
        ctor_passkey const &)
        : m_p0(point_pad_type::construct(x0, y0))
        , m_p1(point_pad_type::construct(x1, y1))
        , m_p2(point_pad_type::construct(x2, y2))
    {
        check_constructor_point_size(*m_p0, *m_p1, *m_p2);
    }

    TrianglePad(
        SimpleArray<T> const & x0,
        SimpleArray<T> const & y0,
        SimpleArray<T> const & z0,
        SimpleArray<T> const & x1,
        SimpleArray<T> const & y1,
        SimpleArray<T> const & z1,
        SimpleArray<T> const & x2,
        SimpleArray<T> const & y2,
        SimpleArray<T> const & z2,
        ctor_passkey const &)
        : m_p0(point_pad_type::construct(x0, y0, z0))
        , m_p1(point_pad_type::construct(x1, y1, z1))
        , m_p2(point_pad_type::construct(x2, y2, z2))
    {
        check_constructor_point_size(*m_p0, *m_p1, *m_p2);
    }

    TrianglePad(
        SimpleArray<T> & x0,
        SimpleArray<T> & y0,
        SimpleArray<T> & x1,
        SimpleArray<T> & y1,
        SimpleArray<T> & x2,
        SimpleArray<T> & y2,
        bool clone,
        ctor_passkey const &)
        : m_p0(point_pad_type::construct(x0, y0, clone))
        , m_p1(point_pad_type::construct(x1, y1, clone))
        , m_p2(point_pad_type::construct(x2, y2, clone))
    {
        check_constructor_point_size(*m_p0, *m_p1, *m_p2);
    }

    TrianglePad(
        SimpleArray<T> & x0,
        SimpleArray<T> & y0,
        SimpleArray<T> & z0,
        SimpleArray<T> & x1,
        SimpleArray<T> & y1,
        SimpleArray<T> & z1,
        SimpleArray<T> & x2,
        SimpleArray<T> & y2,
        SimpleArray<T> & z2,
        bool clone,
        ctor_passkey const &)
        : m_p0(point_pad_type::construct(x0, y0, z0, clone))
        , m_p1(point_pad_type::construct(x1, y1, z1, clone))
        , m_p2(point_pad_type::construct(x2, y2, z2, clone))
    {
        check_constructor_point_size(*m_p0, *m_p1, *m_p2);
    }

    std::shared_ptr<TrianglePad<T>> clone()
    {
        if (ndim() == 2)
        {
            return TrianglePad<T>::construct(x0(), y0(), x1(), y1(), x2(), y2());
        }
        else
        {
            return TrianglePad<T>::construct(x0(), y0(), z0(), x1(), y1(), z1(), x2(), y2(), z2());
        }
    }

    TrianglePad() = delete;
    TrianglePad(TrianglePad const &) = delete;
    TrianglePad(TrianglePad &&) = delete;
    TrianglePad & operator=(TrianglePad const &) = delete;
    TrianglePad & operator=(TrianglePad &&) = delete;

    ~TrianglePad() = default;

    void append(triangle_type const & t)
    {
        if (ndim() == 2)
        {
            m_p0->append(t.x0(), t.y0());
            m_p1->append(t.x1(), t.y1());
            m_p2->append(t.x2(), t.y2());
        }
        else
        {
            m_p0->append(t.x0(), t.y0(), t.z0());
            m_p1->append(t.x1(), t.y1(), t.z1());
            m_p2->append(t.x2(), t.y2(), t.z2());
        }
    }

    void append(point_type const & p0, point_type const & p1, point_type const & p2)
    {
        if (ndim() == 2)
        {
            m_p0->append(p0.x(), p0.y());
            m_p1->append(p1.x(), p1.y());
            m_p2->append(p2.x(), p2.y());
        }
        else
        {
            m_p0->append(p0.x(), p0.y(), p0.z());
            m_p1->append(p1.x(), p1.y(), p1.z());
            m_p2->append(p2.x(), p2.y(), p2.z());
        }
    }

    void append(T x0, T y0, T x1, T y1, T x2, T y2)
    {
        m_p0->append(x0, y0);
        m_p1->append(x1, y1);
        m_p2->append(x2, y2);
    }

    void append(T x0, T y0, T z0, T x1, T y1, T z1, T x2, T y2, T z2)
    {
        m_p0->append(x0, y0, z0);
        m_p1->append(x1, y1, z1);
        m_p2->append(x2, y2, z2);
    }

    void extend_with(TrianglePad<T> const & other)
    {
        size_t const ntri = other.size();
        for (size_t i = 0; i < ntri; ++i)
        {
            append(other.get(i));
        }
    }

    uint8_t ndim() const { return m_p0->ndim(); }

    size_t size() const { return m_p0->size(); }

    SimpleArray<T> pack_array() const
    {
        using shape_type = typename SimpleArray<T>::shape_type;
        SimpleArray<T> ret(shape_type{static_cast<ssize_t>(m_p0->size()), ndim() * 3});
        if (ndim() == 3)
        {
            for (size_t i = 0; i < m_p0->size(); ++i)
            {
                ret(i, 0) = m_p0->x(i);
                ret(i, 1) = m_p0->y(i);
                ret(i, 2) = m_p0->z(i);
                ret(i, 3) = m_p1->x(i);
                ret(i, 4) = m_p1->y(i);
                ret(i, 5) = m_p1->z(i);
                ret(i, 6) = m_p2->x(i);
                ret(i, 7) = m_p2->y(i);
                ret(i, 8) = m_p2->z(i);
            }
        }
        else
        {
            for (size_t i = 0; i < m_p0->size(); ++i)
            {
                ret(i, 0) = m_p0->x(i);
                ret(i, 1) = m_p0->y(i);
                ret(i, 2) = m_p1->x(i);
                ret(i, 3) = m_p1->y(i);
                ret(i, 4) = m_p2->x(i);
                ret(i, 5) = m_p2->y(i);
            }
        }
        return ret;
    }

    void expand(size_t length)
    {
        m_p0->expand(length);
        m_p1->expand(length);
        m_p2->expand(length);
    }

    real_type x0_at(size_t i) const { return m_p0->x_at(i); }
    real_type y0_at(size_t i) const { return m_p0->y_at(i); }
    real_type z0_at(size_t i) const { return m_p0->z_at(i); }
    real_type x1_at(size_t i) const { return m_p1->x_at(i); }
    real_type y1_at(size_t i) const { return m_p1->y_at(i); }
    real_type z1_at(size_t i) const { return m_p1->z_at(i); }
    real_type x2_at(size_t i) const { return m_p2->x_at(i); }
    real_type y2_at(size_t i) const { return m_p2->y_at(i); }
    real_type z2_at(size_t i) const { return m_p2->z_at(i); }
    real_type & x0_at(size_t i) { return m_p0->x_at(i); }
    real_type & y0_at(size_t i) { return m_p0->y_at(i); }
    real_type & z0_at(size_t i) { return m_p0->z_at(i); }
    real_type & x1_at(size_t i) { return m_p1->x_at(i); }
    real_type & y1_at(size_t i) { return m_p1->y_at(i); }
    real_type & z1_at(size_t i) { return m_p1->z_at(i); }
    real_type & x2_at(size_t i) { return m_p2->x_at(i); }
    real_type & y2_at(size_t i) { return m_p2->y_at(i); }
    real_type & z2_at(size_t i) { return m_p2->z_at(i); }

    real_type x0(size_t i) const { return m_p0->x(i); }
    real_type y0(size_t i) const { return m_p0->y(i); }
    real_type z0(size_t i) const { return m_p0->z(i); }
    real_type x1(size_t i) const { return m_p1->x(i); }
    real_type y1(size_t i) const { return m_p1->y(i); }
    real_type z1(size_t i) const { return m_p1->z(i); }
    real_type x2(size_t i) const { return m_p2->x(i); }
    real_type y2(size_t i) const { return m_p2->y(i); }
    real_type z2(size_t i) const { return m_p2->z(i); }
    real_type & x0(size_t i) { return m_p0->x(i); }
    real_type & y0(size_t i) { return m_p0->y(i); }
    real_type & z0(size_t i) { return m_p0->z(i); }
    real_type & x1(size_t i) { return m_p1->x(i); }
    real_type & y1(size_t i) { return m_p1->y(i); }
    real_type & z1(size_t i) { return m_p1->z(i); }
    real_type & x2(size_t i) { return m_p2->x(i); }
    real_type & y2(size_t i) { return m_p2->y(i); }
    real_type & z2(size_t i) { return m_p2->z(i); }

    point_type p0_at(size_t i) const { return m_p0->get_at(i); }
    point_type p1_at(size_t i) const { return m_p1->get_at(i); }
    point_type p2_at(size_t i) const { return m_p2->get_at(i); }
    void set_p0_at(size_t i, point_type const & p) { m_p0->set_at(i, p); }
    void set_p1_at(size_t i, point_type const & p) { m_p1->set_at(i, p); }
    void set_p2_at(size_t i, point_type const & p) { m_p2->set_at(i, p); }

    point_type p0(size_t i) const { return m_p0->get(i); }
    point_type p1(size_t i) const { return m_p1->get(i); }
    point_type p2(size_t i) const { return m_p2->get(i); }
    void set_p0(size_t i, point_type const & p) { m_p0->set(i, p); }
    void set_p1(size_t i, point_type const & p) { m_p1->set(i, p); }
    void set_p2(size_t i, point_type const & p) { m_p2->set(i, p); }

    SimpleArray<value_type> x0() { return m_p0->x(); }
    SimpleArray<value_type> y0() { return m_p0->y(); }
    SimpleArray<value_type> z0() { return m_p0->z(); }
    SimpleArray<value_type> x1() { return m_p1->x(); }
    SimpleArray<value_type> y1() { return m_p1->y(); }
    SimpleArray<value_type> z1() { return m_p1->z(); }
    SimpleArray<value_type> x2() { return m_p2->x(); }
    SimpleArray<value_type> y2() { return m_p2->y(); }
    SimpleArray<value_type> z2() { return m_p2->z(); }

    std::shared_ptr<point_pad_type> p0() { return m_p0; }
    std::shared_ptr<point_pad_type> p1() { return m_p1; }
    std::shared_ptr<point_pad_type> p2() { return m_p2; }

    triangle_type get_at(size_t i) const
    {
        if (ndim() == 3)
        {
            return triangle_type(point_type(x0_at(i), y0_at(i), z0_at(i)), point_type(x1_at(i), y1_at(i), z1_at(i)), point_type(x2_at(i), y2_at(i), z2_at(i)));
        }
        else
        {
            return triangle_type(point_type(x0_at(i), y0_at(i), 0.0), point_type(x1_at(i), y1_at(i), 0.0), point_type(x2_at(i), y2_at(i), 0.0));
        }
    }
    void set_at(size_t i, triangle_type const & t)
    {
        x0_at(i) = t.x0();
        y0_at(i) = t.y0();
        x1_at(i) = t.x1();
        y1_at(i) = t.y1();
        x2_at(i) = t.x2();
        y2_at(i) = t.y2();
        if (ndim() == 3)
        {
            z0_at(i) = t.z0();
            z1_at(i) = t.z1();
            z2_at(i) = t.z2();
        }
    }
    void set_at(size_t i, point_type const & p0, point_type const & p1, point_type const & p2)
    {
        x0_at(i) = p0.x();
        y0_at(i) = p0.y();
        x1_at(i) = p1.x();
        y1_at(i) = p1.y();
        x2_at(i) = p2.x();
        y2_at(i) = p2.y();
        if (ndim() == 3)
        {
            z0_at(i) = p0.z();
            z1_at(i) = p1.z();
            z2_at(i) = p2.z();
        }
    }
    void set_at(size_t i, value_type x0, value_type y0, value_type x1, value_type y1, value_type x2, value_type y2)
    {
        x0_at(i) = x0;
        y0_at(i) = y0;
        x1_at(i) = x1;
        y1_at(i) = y1;
        x2_at(i) = x2;
        y2_at(i) = y2;
    }
    void set_at(size_t i, value_type x0, value_type y0, value_type z0, value_type x1, value_type y1, value_type z1, value_type x2, value_type y2, value_type z2)
    {
        x0_at(i) = x0;
        y0_at(i) = y0;
        x1_at(i) = x1;
        y1_at(i) = y1;
        x2_at(i) = x2;
        y2_at(i) = y2;
        if (ndim() == 3)
        {
            z0_at(i) = z0;
            z1_at(i) = z1;
            z2_at(i) = z2;
        }
    }

    triangle_type get(size_t i) const
    {
        if (ndim() == 3)
        {
            return triangle_type(point_type(x0(i), y0(i), z0(i)), point_type(x1(i), y1(i), z1(i)), point_type(x2(i), y2(i), z2(i)));
        }
        else
        {
            return triangle_type(point_type(x0(i), y0(i), 0.0), point_type(x1(i), y1(i), 0.0), point_type(x2(i), y2(i), 0.0));
        }
    }
    void set(size_t i, triangle_type const & t)
    {
        x0(i) = t.x0();
        y0(i) = t.y0();
        x1(i) = t.x1();
        y1(i) = t.y1();
        x2(i) = t.x2();
        y2(i) = t.y2();
        if (ndim() == 3)
        {
            z0(i) = t.z0();
            z1(i) = t.z1();
            z2(i) = t.z2();
        }
    }
    void set(size_t i, point_type const & p0, point_type const & p1, point_type const & p2)
    {
        x0(i) = p0.x();
        y0(i) = p0.y();
        x1(i) = p1.x();
        y1(i) = p1.y();
        x2(i) = p2.x();
        y2(i) = p2.y();
        if (ndim() == 3)
        {
            z0(i) = p0.z();
            z1(i) = p1.z();
            z2(i) = p2.z();
        }
    }
    // clang-format off
    void set(size_t i, value_type x0_value, value_type y0_value,
             value_type x1_value, value_type y1_value,
             value_type x2_value, value_type y2_value)
    // clang-format on
    {
        x0(i) = x0_value;
        y0(i) = y0_value;
        x1(i) = x1_value;
        y1(i) = y1_value;
        x2(i) = x2_value;
        y2(i) = y2_value;
    }
    // clang-format off
    void set(size_t i, value_type x0_value, value_type y0_value, value_type z0_value,
             value_type x1_value, value_type y1_value, value_type z1_value,
             value_type x2_value, value_type y2_value, value_type z2_value)
    // clang-format on
    {
        x0(i) = x0_value;
        y0(i) = y0_value;
        x1(i) = x1_value;
        y1(i) = y1_value;
        x2(i) = x2_value;
        y2(i) = y2_value;
        if (ndim() == 3)
        {
            z0(i) = z0_value;
            z1(i) = z1_value;
            z2(i) = z2_value;
        }
    }

    /**
     * Mirror the triangle pad with respect to the X axis.
     * This negates Y and Z coordinates, keeping X unchanged.
     * Handles both 2D and 3D triangles.
     */
    void mirror_x()
    {
        size_t const ntri = size();
        for (size_t i = 0; i < ntri; ++i)
        {
            y0(i) = -y0(i);
            y1(i) = -y1(i);
            y2(i) = -y2(i);
            if (ndim() == 3)
            {
                z0(i) = -z0(i);
                z1(i) = -z1(i);
                z2(i) = -z2(i);
            }
        }
    }

    /**
     * Mirror the triangle pad with respect to the Y axis.
     * This negates X and Z coordinates, keeping Y unchanged.
     * Handles both 2D and 3D triangles.
     */
    void mirror_y()
    {
        size_t const ntri = size();
        for (size_t i = 0; i < ntri; ++i)
        {
            x0(i) = -x0(i);
            x1(i) = -x1(i);
            x2(i) = -x2(i);
            if (ndim() == 3)
            {
                z0(i) = -z0(i);
                z1(i) = -z1(i);
                z2(i) = -z2(i);
            }
        }
    }

    /**
     * Mirror the triangle pad with respect to the Z axis.
     * This negates X and Y coordinates, keeping Z unchanged.
     * Only works for 3D triangles.
     *
     * @throw std::out_of_range if ndim is not 3
     */
    void mirror_z()
    {
        if (ndim() != 3)
        {
            throw std::out_of_range(
                std::format("TrianglePad::mirror_z: cannot mirror Z axis for ndim {}", int(ndim())));
        }
        size_t const ntri = size();
        for (size_t i = 0; i < ntri; ++i)
        {
            x0(i) = -x0(i);
            x1(i) = -x1(i);
            x2(i) = -x2(i);
            y0(i) = -y0(i);
            y1(i) = -y1(i);
            y2(i) = -y2(i);
        }
    }

    void mirror(Axis axis)
    {
        switch (axis)
        {
        case Axis::X: mirror_x(); break;
        case Axis::Y: mirror_y(); break;
        case Axis::Z: mirror_z(); break;
        default: throw std::invalid_argument("TrianglePad::mirror: invalid axis"); break;
        }
    }

private:

    void check_constructor_point_size(point_pad_type const & p0, point_pad_type const & p1, point_pad_type const & p2)
    {
        if (m_p0->size() != m_p1->size() || m_p0->size() != m_p2->size())
        {
            throw std::invalid_argument(
                std::format("TrianglePad::TrianglePad: p0.size() {} p1.size() {} p2.size() {} are not the same",
                            p0.size(),
                            p1.size(),
                            p2.size()));
        }
    }

    std::shared_ptr<point_pad_type> m_p0;
    std::shared_ptr<point_pad_type> m_p1;
    std::shared_ptr<point_pad_type> m_p2;

}; /* end class TrianglePad */

using TrianglePadFp32 = TrianglePad<float>;
using TrianglePadFp64 = TrianglePad<double>;

/**
 * Polygon3d handle class - lightweight view into a polygon stored in PolygonPad.
 *
 * This is a lightweight handle that references a polygon stored in a PolygonPad
 * container. The handle keeps the underlying PolygonPad alive by holding a
 * shared pointer to it. Polygons are defined by an ordered list of nodes following the
 * right-hand rule: counter-clockwise for positive area, clockwise for negative.
 *
 * The handle uses polygon_id as public API, with internal offset/count for efficient access.
 *
 * @tparam T floating-point type
 * @ingroup group_geometry
 */
template <typename T>
class Polygon3d
{

private:

    struct ctor_passkey
    {
    };

public:

    using point_type = Point3d<T>;
    using value_type = T;
    using segment_type = Segment3d<T>;
    using polygon_pad_type = PolygonPad<T>;
    using pad_ptr_type = std::shared_ptr<polygon_pad_type const>;

    Polygon3d(pad_ptr_type pad, size_t polygon_id, ctor_passkey const &)
        : m_pad(std::move(pad))
        , m_id(polygon_id)
    {
        if (m_id >= m_pad->m_begins.size())
        {
            throw std::out_of_range(
                std::format("Polygon3d::Polygon3d: polygon_id {} >= num_polygons {}",
                            m_id,
                            m_pad->m_begins.size()));
        }
    }

    Polygon3d() = delete;
    Polygon3d(Polygon3d const &) = default;
    Polygon3d(Polygon3d &&) = default;
    Polygon3d & operator=(Polygon3d const &) = default;
    Polygon3d & operator=(Polygon3d &&) = default;
    ~Polygon3d() = default;

    size_t polygon_id() const { return m_id; }
    pad_ptr_type pad() const { return m_pad; }

    size_t nnode() const
    {
        if (m_id >= m_pad->m_begins.size())
        {
            throw std::out_of_range(std::format("Polygon3d::nnode: polygon_id {} >= num_polygons {}",
                                                m_id,
                                                m_pad->m_begins.size()));
        }
        typename polygon_pad_type::ssize_type const begin_index = m_pad->m_begins[m_id];
        typename polygon_pad_type::ssize_type const end_index = m_pad->m_ends[m_id];
        return static_cast<size_t>(end_index - begin_index);
    }

    uint8_t ndim() const { return m_pad->ndim(); }

    point_type node(size_t index) const { return m_pad->get_node(m_id, index); }

    segment_type edge(size_t index) const { return m_pad->get_edge(m_id, index); }

    value_type compute_signed_area() const { return m_pad->compute_signed_area(m_id); }

    bool is_counter_clockwise() const
    {
        if (!m_calculated_counter_clockwise)
        {
            calc_counter_clockwise();
            m_calculated_counter_clockwise = true;
        }
        return m_is_counter_clockwise;
    }

    BoundBox3d<T> calc_bound_box() const { return m_pad->calc_bound_box(m_id); }

    bool operator==(Polygon3d const & other) const
    {
        size_t const node_count = nnode();

        if (node_count != other.nnode())
        {
            return false;
        }
        if (ndim() != other.ndim())
        {
            return false;
        }
        for (size_t i = 0; i < node_count; ++i)
        {
            if (node(i) != other.node(i))
            {
                return false;
            }
        }
        return true;
    }

    bool operator!=(Polygon3d const & other) const { return !(*this == other); }

    bool is(Polygon3d const & other) const { return m_pad == other.m_pad && m_id == other.m_id; }

    bool is_not(Polygon3d const & other) const { return !is(other); }

private:
    bool calc_counter_clockwise() const
    {
        // Be careful for the zero area case
        m_is_counter_clockwise = compute_signed_area() >= 0;
        return m_is_counter_clockwise;
    }

private:
    pad_ptr_type m_pad = nullptr;
    size_t m_id = 0;
    mutable bool m_calculated_counter_clockwise = false; // lazy evaluation flag
    mutable bool m_is_counter_clockwise = false;

    friend class PolygonPad<T>;

}; /* end class Polygon3d */

/**
 * PolygonPad - container for multiple polygons stored as node lists.
 *
 * Polygons are stored efficiently as sequences of nodes in a shared PointPad,
 * with each polygon defined by a range [start, end) in the node list. This avoids
 * memory duplication compared to storing line segments.
 *
 * Note that PolygonPad assumes polygons are in a XY plane currently.
 * TODO: Extend to 3D polygons in arbitrary planes.
 *
 * All nodes follow the right-hand rule:
 *  - Counter-clockwise ordering = positive area polygon
 *  - Clockwise ordering = negative area polygon (holes)
 *
 * Future: Will integrate with trapezoidal map for polygon boolean operations.
 * Reference: http://www0.cs.ucl.ac.uk/staff/m.slater/Teaching/CG/1997-98/Solutions/Trap/
 *
 * @tparam T floating-point type
 * @ingroup group_geometry
 */
template <typename T>
class PolygonPad
    : public NumberBase<int32_t, T>
    , public std::enable_shared_from_this<PolygonPad<T>>
{

private:

    struct ctor_passkey
    {
    };

public:

    static_assert(std::is_arithmetic_v<T>, "T in PolygonPad<T> must be arithmetic type");

    using ssize_type = int32_t;
    using point_type = Point3d<T>;
    using value_type = T;
    using segment_type = Segment3d<T>;
    using point_pad_type = PointPad<T>;
    using polygon_type = Polygon3d<T>;
    using polygon_pad_type = PolygonPad<T>;
    using segment_pad_type = SegmentPad<T>;
    using curve_pad_type = CurvePad<T>;
    using rtree_type = RTree<segment_type, BoundBox3d<T>, RTreeValueOps<segment_type, BoundBox3d<T>>>;
    using trapezoid_decomposer_type = TrapezoidalDecomposer<T>;

    template <typename... Args>
    static std::shared_ptr<PolygonPad<T>> construct(Args &&... args)
    {
        return std::make_shared<PolygonPad<T>>(std::forward<Args>(args)..., ctor_passkey());
    }

    PolygonPad(uint8_t ndim, ctor_passkey const &)
        : m_points(point_pad_type::construct(ndim))
        , m_rtree(std::make_unique<rtree_type>())
        , m_decomposer(ndim)
    {
    }

    PolygonPad() = delete;
    PolygonPad(PolygonPad const &) = delete;
    PolygonPad(PolygonPad &&) = delete;
    PolygonPad & operator=(PolygonPad const &) = delete;
    PolygonPad & operator=(PolygonPad &&) = delete;
    ~PolygonPad() = default;

    uint8_t ndim() const { return m_points->ndim(); }

    size_t num_polygons() const { return m_begins.size(); }

    size_t num_points() const { return m_points->size(); }

    /**
     * Add a polygon from a list of nodes.
     * Nodes must follow right-hand rule: counter-clockwise for positive area.
     *
     * @param nodes Vector of points defining the polygon boundary
     * @return Polygon3d handle to the newly added polygon
     */
    polygon_type add_polygon(std::vector<point_type> const & nodes);

    /**
     * Add a polygon from a SegmentPad by extracting nodes.
     * Assumes segments form a connected chain.
     *
     * The input is interpreted as an ordered chain. Nodes are extracted from
     * `segments->p0(i)` for i in [0, size). The last closing segment is not
     * validated; it is the caller's responsibility to provide a consistent chain.
     *
     * @param segments SegmentPad containing connected line segments
     * @return Polygon3d handle to the newly added polygon
     */
    polygon_type add_polygon_from_segments(std::shared_ptr<segment_pad_type> const & segments);

    /**
     * Add a polygon from a CurvePad by sampling.
     *
     * @param curves CurvePad to sample
     * @param sample_length Sampling interval
     * @return Polygon3d handle to the newly added polygon
     */
    polygon_type add_polygon_from_curves(std::shared_ptr<curve_pad_type> const & curves, value_type sample_length);

    /**
     * Add a polygon from both segments and curves.
     *
     * @param segments SegmentPad containing line segments
     * @param curves CurvePad to sample
     * @param sample_length Sampling interval for curves
     * @return Polygon3d handle to the newly added polygon
     */
    polygon_type add_polygon_from_segments_and_curves(
        std::shared_ptr<segment_pad_type> const & segments,
        std::shared_ptr<curve_pad_type> const & curves,
        value_type sample_length);

    /**
     * Get a polygon handle by polygon_id.
     *
     * @param polygon_id ID of the polygon
     * @return Polygon3d handle
     */
    polygon_type get_polygon(size_t polygon_id) const
    {
        if (polygon_id >= m_begins.size())
        {
            throw std::out_of_range(
                std::format("PolygonPad::get_polygon: polygon_id {} >= num_polygons {}",
                            polygon_id,
                            m_begins.size()));
        }
        return polygon_type(this->shared_from_this(),
                            polygon_id,
                            typename polygon_type::ctor_passkey());
    }

    /**
     * Get number of nodes in a specific polygon.
     */
    size_t get_num_nodes(size_t polygon_id) const;

    /**
     * Get a node from a specific polygon.
     *
     * @param polygon_id Index of the polygon
     * @param node_index Index of the node within the polygon
     * @return Point at the specified position
     */
    point_type get_node(size_t polygon_id, size_t node_index) const;

    /**
     * Get an edge (segment) from a specific polygon.
     * Edge i connects node i to node (i+1) % nnode.
     *
     * @param polygon_id Index of the polygon
     * @param edge_index Index of the edge within the polygon
     * @return Segment representing the edge
     */
    segment_type get_edge(size_t polygon_id, size_t edge_index) const;

    /**
     * Compute signed area of a polygon using the shoelace formula.
     * Positive area indicates counter-clockwise node ordering (right-hand rule).
     * Negative area indicates clockwise ordering.
     *
     * This currently uses only x and y coordinates (i.e., the signed area of the
     * polygon projected onto the XY plane). For `ndim()==2`, this is the usual
     * 2D signed area. For `ndim()==3`, this is meaningful only when the polygon
     * is planar and aligned with the XY plane.
     *
     * @param polygon_id Index of the polygon
     * @return Signed area (positive = CCW, negative = CW)
     */
    value_type compute_signed_area(size_t polygon_id) const;

    /**
     * Check if polygon nodes are ordered counter-clockwise (right-hand rule).
     */
    bool is_counter_clockwise(size_t polygon_id) const { return get_polygon(polygon_id).is_counter_clockwise(); }

    /**
     * Calculate bounding box for a specific polygon.
     */
    BoundBox3d<T> calc_bound_box(size_t polygon_id) const;

    /**
     * Search for segments within a bounding box across all polygons.
     * Returns segments from all polygons that intersect the query box.
     *
     * @param box Query bounding box
     * @param output Vector to store found segments
     */
    void search_segments(BoundBox3d<T> const & box, std::vector<segment_type> & output) const
    {
        m_rtree->search(box, output);
    }

    /**
     * Rebuild the spatial index (RTree) for all polygons.
     *
     * The R-tree is updated incrementally when polygons are added through this
     * class. If in the future polygon nodes become mutable (e.g., via exposing
     * direct access to `m_points`), call `rebuild_rtree()` after any mutation so
     * `search_segments()` remains correct.
     */
    void rebuild_rtree();

    /**
     * Decompose a polygon into trapezoids using vertical sweep line algorithm.
     *
     * @param polygon_id ID of the polygon to decompose
     * @return Pair of begin and end indices into the decomposer's trapezoid pad
     * @throws std::out_of_range if polygon_id is invalid
     */
    std::pair<size_t, size_t> decompose_to_trapezoid(size_t polygon_id);

    /**
     * Access the trapezoid pad produced by the decomposer.
     * Valid after calling decompose_to_trapezoid().
     */
    std::shared_ptr<TrapezoidPad<T>> decomposed_trapezoids() { return m_decomposer.trapezoids(); }
    std::shared_ptr<TrapezoidPad<T> const> decomposed_trapezoids() const { return m_decomposer.trapezoids(); }

    /**
     * Compute union of two polygons using trapezoidal decomposition.
     *
     * @param p1 First polygon
     * @param p2 Second polygon
     * @return polygon pad forming the union
     */
    std::shared_ptr<polygon_pad_type> boolean_union(polygon_type const & p1, polygon_type const & p2)
    {
        return m_boolean_union.compute(this->shared_from_this(), p1.polygon_id(), p2.polygon_id());
    }

    /**
     * Compute intersection of two polygons using trapezoidal decomposition.
     *
     * @param p1 First polygon
     * @param p2 Second polygon
     * @return polygon pad forming the intersection
     */
    std::shared_ptr<polygon_pad_type> boolean_intersection(polygon_type const & p1, polygon_type const & p2)
    {
        return m_boolean_intersection.compute(this->shared_from_this(), p1.polygon_id(), p2.polygon_id());
    }

    /**
     * Compute difference of two polygons (p1 - p2).
     *
     * @param p1 First polygon
     * @param p2 Second polygon to subtract
     * @return polygon pad forming the difference
     */
    std::shared_ptr<polygon_pad_type> boolean_difference(polygon_type const & p1, polygon_type const & p2)
    {
        return m_boolean_difference.compute(this->shared_from_this(), p1.polygon_id(), p2.polygon_id());
    }

private:

    friend class Polygon3d<T>;

    void rebuild_polygon_rtree(polygon_type const & polygon);

    std::shared_ptr<point_pad_type> m_points;
    SimpleCollector<ssize_type> m_begins;
    SimpleCollector<ssize_type> m_ends;
    std::unique_ptr<rtree_type> m_rtree;

    trapezoid_decomposer_type m_decomposer;
    AreaBooleanUnion<T> m_boolean_union;
    AreaBooleanIntersection<T> m_boolean_intersection;
    AreaBooleanDifference<T> m_boolean_difference;
}; /* end class PolygonPad */

using PolygonPadFp32 = PolygonPad<float>;
using PolygonPadFp64 = PolygonPad<double>;

using Polygon3dFp32 = Polygon3d<float>;
using Polygon3dFp64 = Polygon3d<double>;

/**
 * RTreeValueOps specialization that bounds Segment3d items with BoundBox3d.
 *
 * Teaches the RTree how to compute the axis-aligned bounding box of a single
 * segment and of a group of segments, used when indexing polygon edges.
 *
 * @tparam T floating-point type
 * @ingroup group_geometry
 */
template <typename T>
struct RTreeValueOps<Segment3d<T>, BoundBox3d<T>>
{
    static BoundBox3d<T> calc_bound_box(Segment3d<T> const & item)
    {
        T min_x = std::min(item.x0(), item.x1());
        T min_y = std::min(item.y0(), item.y1());
        T min_z = std::min(item.z0(), item.z1());
        T max_x = std::max(item.x0(), item.x1());
        T max_y = std::max(item.y0(), item.y1());
        T max_z = std::max(item.z0(), item.z1());
        return BoundBox3d<T>(min_x, min_y, min_z, max_x, max_y, max_z);
    }

    static BoundBox3d<T> calc_group_bound_box(std::vector<Segment3d<T>> const & items)
    {
        if (items.empty())
        {
            return BoundBox3d<T>(0, 0, 0, 0, 0, 0);
        }

        BoundBox3d<T> result = calc_bound_box(items[0]);
        for (size_t i = 1; i < items.size(); ++i)
        {
            result.expand(calc_bound_box(items[i]));
        }
        return result;
    }
};

template <typename T>
Polygon3d<T> PolygonPad<T>::add_polygon(std::vector<point_type> const & nodes)
{
    if (nodes.empty())
    {
        throw std::invalid_argument("PolygonPad::add_polygon: cannot add empty polygon");
    }
    if (nodes.size() < 3)
    {
        throw std::invalid_argument("PolygonPad::add_polygon: polygon must have at least 3 nodes");
    }

    auto const begin_index = static_cast<ssize_type>(m_points->size());

    for (point_type const & node : nodes)
    {
        // check if the point is in XY plane
        if (node.z() != 0)
        {
            throw std::invalid_argument("PolygonPad::add_polygon: all nodes must lie in the XY plane (z=0)");
        }

        m_points->append(node);
    }

    auto const end_index = static_cast<ssize_type>(m_points->size());
    size_t const polygon_id = m_begins.size();
    m_begins.push_back(begin_index);
    m_ends.push_back(end_index);

    std::shared_ptr<PolygonPad<T> const> const & const_this = this->shared_from_this();
    polygon_type polygon(const_this, polygon_id, typename polygon_type::ctor_passkey());
    rebuild_polygon_rtree(polygon);

    return polygon;
}

template <typename T>
Polygon3d<T> PolygonPad<T>::add_polygon_from_segments(std::shared_ptr<segment_pad_type> const & segments)
{
    if (!segments)
    {
        throw std::invalid_argument("PolygonPad::add_polygon_from_segments: segments is null");
    }
    if (segments->size() == 0)
    {
        throw std::invalid_argument("PolygonPad::add_polygon_from_segments: empty segment pad");
    }
    if (segments->ndim() != ndim())
    {
        throw std::invalid_argument(
            std::format("PolygonPad::add_polygon_from_segments: segments.ndim() {} != pad.ndim() {}",
                        int(segments->ndim()),
                        int(ndim())));
    }

    std::vector<point_type> nodes;
    nodes.reserve(segments->size());

    for (size_t i = 0; i < segments->size(); ++i)
    {
        nodes.push_back(segments->p0(i));
    }

    return add_polygon(nodes);
}

template <typename T>
Polygon3d<T> PolygonPad<T>::add_polygon_from_curves(std::shared_ptr<curve_pad_type> const & curves, value_type sample_length)
{
    if (!curves)
    {
        throw std::invalid_argument("PolygonPad::add_polygon_from_curves: curves is null");
    }
    if (!(sample_length > 0))
    {
        throw std::invalid_argument(
            std::format("PolygonPad::add_polygon_from_curves: sample_length {} must be > 0",
                        sample_length));
    }
    if (curves->ndim() != ndim())
    {
        throw std::invalid_argument(
            std::format("PolygonPad::add_polygon_from_curves: curves.ndim() {} != pad.ndim() {}",
                        int(curves->ndim()),
                        int(ndim())));
    }
    std::shared_ptr<segment_pad_type> const & segments = curves->sample(sample_length);
    return add_polygon_from_segments(segments);
}

template <typename T>
Polygon3d<T> PolygonPad<T>::add_polygon_from_segments_and_curves(
    std::shared_ptr<segment_pad_type> const & segments,
    std::shared_ptr<curve_pad_type> const & curves,
    value_type sample_length)
{
    if (!segments)
    {
        throw std::invalid_argument("PolygonPad::add_polygon_from_segments_and_curves: segments is null");
    }
    if (!curves)
    {
        throw std::invalid_argument("PolygonPad::add_polygon_from_segments_and_curves: curves is null");
    }
    if (!(sample_length > 0))
    {
        throw std::invalid_argument(
            std::format("PolygonPad::add_polygon_from_segments_and_curves: sample_length {} must be > 0",
                        sample_length));
    }
    if (segments->ndim() != ndim() || curves->ndim() != ndim())
    {
        throw std::invalid_argument(
            std::format("PolygonPad::add_polygon_from_segments_and_curves: segments.ndim() {}, curves.ndim() {}, pad.ndim() {}",
                        int(segments->ndim()),
                        int(curves->ndim()),
                        int(ndim())));
    }

    std::vector<point_type> nodes;

    for (size_t i = 0; i < segments->size(); ++i)
    {
        nodes.push_back(segments->p0(i)); // FIXME: NOLINT(performance-inefficient-vector-operation)
    }

    std::shared_ptr<segment_pad_type> const & curve_segments = curves->sample(sample_length);
    for (size_t i = 0; i < curve_segments->size(); ++i)
    {
        nodes.push_back(curve_segments->p0(i));
    }

    return add_polygon(nodes);
}

template <typename T>
size_t PolygonPad<T>::get_num_nodes(size_t polygon_id) const
{
    if (polygon_id >= m_begins.size())
    {
        throw std::out_of_range(
            std::format("PolygonPad::get_num_nodes: polygon_id {} >= num_polygons {}",
                        polygon_id,
                        m_begins.size()));
    }
    ssize_type const begin_index = m_begins[polygon_id];
    ssize_type const end_index = m_ends[polygon_id];
    return static_cast<size_t>(end_index - begin_index);
}

template <typename T>
Point3d<T> PolygonPad<T>::get_node(size_t polygon_id, size_t node_index) const
{
    if (polygon_id >= m_begins.size())
    {
        throw std::out_of_range(
            std::format("PolygonPad::get_node: polygon_id {} >= num_polygons {}",
                        polygon_id,
                        m_begins.size()));
    }
    ssize_type const begin_index = m_begins[polygon_id];
    ssize_type const end_index = m_ends[polygon_id];
    auto const count = static_cast<size_t>(end_index - begin_index);
    if (node_index >= count)
    {
        throw std::out_of_range(
            std::format("PolygonPad::get_node: node_index {} >= count {}",
                        node_index,
                        count));
    }
    // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
    return m_points->get(static_cast<size_t>(begin_index + static_cast<ssize_type>(node_index)));
}

template <typename T>
Segment3d<T> PolygonPad<T>::get_edge(size_t polygon_id, size_t edge_index) const
{
    if (polygon_id >= m_begins.size())
    {
        throw std::out_of_range(
            std::format("PolygonPad::get_edge: polygon_id {} >= num_polygons {}",
                        polygon_id,
                        m_begins.size()));
    }
    ssize_type const begin_index = m_begins[polygon_id];
    ssize_type const end_index = m_ends[polygon_id];
    auto const count = static_cast<size_t>(end_index - begin_index);
    if (edge_index >= count)
    {
        throw std::out_of_range(
            std::format("PolygonPad::get_edge: edge_index {} >= count {}",
                        edge_index,
                        count));
    }
    // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
    point_type const p0 = m_points->get(static_cast<size_t>(begin_index + static_cast<ssize_type>(edge_index)));
    // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
    point_type const p1 = m_points->get(static_cast<size_t>(begin_index + static_cast<ssize_type>((edge_index + 1) % count)));
    return segment_type(p0, p1);
}

template <typename T>
T PolygonPad<T>::compute_signed_area(size_t polygon_id) const
{
    if (polygon_id >= m_begins.size())
    {
        throw std::out_of_range(
            std::format("PolygonPad::compute_signed_area: polygon_id {} >= num_polygons {}",
                        polygon_id,
                        m_begins.size()));
    }
    auto const begin_index = m_begins[polygon_id];
    auto const end_index = m_ends[polygon_id];
    auto const count = static_cast<size_t>(end_index - begin_index);
    if (count < 3)
    {
        return 0;
    }

    value_type area = 0;
    for (size_t i = 0; i < count; ++i)
    {
        // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
        point_type const p0 = m_points->get(static_cast<size_t>(begin_index + static_cast<ssize_type>(i)));
        // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
        point_type const p1 = m_points->get(static_cast<size_t>(begin_index + static_cast<ssize_type>((i + 1) % count)));
        area += p0.x() * p1.y() - p1.x() * p0.y();
    }

    return area / 2;
}

template <typename T>
BoundBox3d<T> PolygonPad<T>::calc_bound_box(size_t polygon_id) const
{
    if (polygon_id >= m_begins.size())
    {
        throw std::out_of_range(
            std::format("PolygonPad::calc_bound_box: polygon_id {} >= num_polygons {}",
                        polygon_id,
                        m_begins.size()));
    }
    auto const begin_index = m_begins[polygon_id];
    auto const end_index = m_ends[polygon_id];
    auto const count = static_cast<size_t>(end_index - begin_index);

    if (count == 0)
    {
        return BoundBox3d<T>(0, 0, 0, 0, 0, 0);
    }

    value_type min_x = std::numeric_limits<value_type>::max();
    value_type min_y = std::numeric_limits<value_type>::max();
    value_type min_z = std::numeric_limits<value_type>::max();
    value_type max_x = std::numeric_limits<value_type>::lowest();
    value_type max_y = std::numeric_limits<value_type>::lowest();
    value_type max_z = std::numeric_limits<value_type>::lowest();

    for (size_t i = 0; i < count; ++i)
    {
        // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
        point_type const node = m_points->get(static_cast<size_t>(begin_index + static_cast<ssize_type>(i)));
        min_x = std::min(min_x, node.x());
        min_y = std::min(min_y, node.y());
        min_z = std::min(min_z, node.z());
        max_x = std::max(max_x, node.x());
        max_y = std::max(max_y, node.y());
        max_z = std::max(max_z, node.z());
    }

    return BoundBox3d<T>(min_x, min_y, min_z, max_x, max_y, max_z);
}

template <typename T>
void PolygonPad<T>::rebuild_rtree()
{
    m_rtree = std::make_unique<rtree_type>();
    for (size_t i = 0; i < m_begins.size(); ++i)
    {
        polygon_type const polygon = get_polygon(i);
        rebuild_polygon_rtree(polygon);
    }
}

template <typename T>
void PolygonPad<T>::rebuild_polygon_rtree(polygon_type const & polygon)
{
    size_t const count = polygon.nnode();

    for (size_t i = 0; i < count; ++i)
    {
        segment_type const edge = polygon.edge(i);
        m_rtree->insert(edge);
    }
}

template <typename T>
std::pair<size_t, size_t> PolygonPad<T>::decompose_to_trapezoid(size_t polygon_id)
{
    if (polygon_id >= num_polygons())
    {
        throw std::out_of_range(
            std::format("PolygonPad::decompose_to_trapezoid: polygon_id {} >= num_polygons {}",
                        polygon_id,
                        num_polygons()));
    }

    polygon_type const polygon = get_polygon(polygon_id);
    std::vector<point_type> points;
    points.reserve(polygon.nnode());
    for (size_t i = 0; i < polygon.nnode(); ++i)
    {
        points.push_back(polygon.node(i));
    }

    return m_decomposer.decompose(polygon_id, points);
}

} /* end namespace solvcon */

namespace solvcon
{

namespace detail
{

/**
 * Boolean operations on two polygons using trapezoidal decomposition.
 * Delegates to BooleanDecompositionHelper.
 *
 * @see BooleanDecompositionHelper for algorithm details.
 */
template <typename T>
std::shared_ptr<PolygonPad<T>> compute_boolean_with_decomposition(
    const std::shared_ptr<PolygonPad<T>> & pad,
    size_t polygon_id1,
    size_t polygon_id2,
    BooleanOperation op)
{
    BooleanDecompositionHelper<T> helper(pad, polygon_id1, polygon_id2, op);
    return helper.compute();
} /* end function compute_boolean_with_decomposition */

} /* end namespace detail */

template <typename T>
std::shared_ptr<PolygonPad<T>> AreaBooleanUnion<T>::compute(const std::shared_ptr<PolygonPad<T>> & pad, size_t polygon_id1, size_t polygon_id2)
{
    return detail::compute_boolean_with_decomposition(pad, polygon_id1, polygon_id2, detail::BooleanOperation::Union);
}

template <typename T>
std::shared_ptr<PolygonPad<T>> AreaBooleanIntersection<T>::compute(const std::shared_ptr<PolygonPad<T>> & pad, size_t polygon_id1, size_t polygon_id2)
{
    return detail::compute_boolean_with_decomposition(pad, polygon_id1, polygon_id2, detail::BooleanOperation::Intersection);
}

template <typename T>
std::shared_ptr<PolygonPad<T>> AreaBooleanDifference<T>::compute(const std::shared_ptr<PolygonPad<T>> & pad, size_t polygon_id1, size_t polygon_id2)
{
    return detail::compute_boolean_with_decomposition(pad, polygon_id1, polygon_id2, detail::BooleanOperation::Difference);
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

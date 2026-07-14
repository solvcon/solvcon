#pragma once

/*
 * Copyright (c) 2018, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <functional>
#include <memory>
#include <vector>

#include <solvcon/solvcon.hpp>

namespace solvcon
{

namespace spacetime
{

class Grid;
class Field;

template <class ET>
class ElementBase
{

public:

    using value_type = real_type;
    using base_type = ElementBase;

    ElementBase(Field * field, value_type * xptr)
        : m_field(field)
        , m_xptr(xptr)
    {
    }

    ElementBase() = delete;

    ET duplicate() { return *static_cast<ET *>(this); }

    Grid const & grid() const;
    Field const & field() const { return *m_field; }
    Field & field() { return *m_field; }

    value_type time_increment() const { return field().time_increment(); }
    value_type dt() const { return field().dt(); }
    value_type hdt() const { return field().hdt(); }
    value_type qdt() const { return field().qdt(); }

    value_type x() const { return *m_xptr; }
    value_type dx() const { return xpos() - xneg(); }

    // By pass address sanitizer check, more information please reference to issue #257
    ASAN_NO_SANITIZE_ADDRESS value_type xneg() const { return *(m_xptr - 1); }
    ASAN_NO_SANITIZE_ADDRESS value_type xpos() const { return *(m_xptr + 1); }
    value_type xctr() const { return static_cast<ET const *>(this)->xctr(); }

    void move(ssize_t offset) { m_xptr += offset; }
    void move_at(ssize_t offset) { static_cast<ET *>(this)->move_at(static_cast<int_type>(offset)); }

    void move_left() { move(-2); }
    void move_right() { move(2); }
    void move_neg() { move(-1); }
    void move_pos() { move(1); }

    void move_left_at() { move_at(-2); }
    void move_right_at() { move_at(2); }
    void move_neg_at() { move_at(-1); }
    void move_pos_at() { move_at(1); }

    bool operator==(ET const & b) const { return (m_field == b.m_field) && (m_xptr == b.m_xptr); }
    bool operator!=(ET const & b) const { return (m_field != b.m_field) || (m_xptr != b.m_xptr); }
    bool operator<(ET const & b) const { return (m_field == b.m_field) && (m_xptr < b.m_xptr); }
    bool operator<=(ET const & b) const { return (m_field == b.m_field) && (m_xptr <= b.m_xptr); }
    bool operator>(ET const & b) const { return (m_field == b.m_field) && (m_xptr > b.m_xptr); }
    bool operator>=(ET const & b) const { return (m_field == b.m_field) && (m_xptr >= b.m_xptr); }

protected:

    size_t xindex() const;

private:

    Field * m_field;
    value_type * m_xptr;

    friend Grid;
    friend Field;

}; /* end class ElementBase */

class Celm;
class Selm;

class Grid
    : public std::enable_shared_from_this<Grid>
{

public:

    // Remove the two aliases duplicated in ElementBase.
    using value_type = real_type;
    using array_type = AscendantGrid1d::array_type;
    constexpr static size_t BOUND_COUNT = 2;
    static_assert(BOUND_COUNT >= 2, "BOUND_COUNT must be greater or equal to 2");

private:

    class ctor_passkey
    {
    };

public:

    template <class... Args>
    static std::shared_ptr<Grid> construct(Args &&... args)
    {
        return std::make_shared<Grid>(std::forward<Args>(args)..., ctor_passkey());
    }

    std::shared_ptr<Grid> clone() const
    {
        return std::make_shared<Grid>(*this);
    }

    Grid(real_type xmin, real_type xmax, size_t ncelm, ctor_passkey const &);

    // NOLINTNEXTLINE(hicpp-member-init,cppcoreguidelines-pro-type-member-init)
    Grid(array_type const & xloc, ctor_passkey const &) { init_from_array(xloc); }

    Grid() = delete;
    Grid(Grid const &) = default;
    Grid(Grid &&) = default;
    Grid & operator=(Grid const &) = default;
    Grid & operator=(Grid &&) = default;
    ~Grid() = default;

    real_type xmin() const { return m_xmin; }
    real_type xmax() const { return m_xmax; }
    size_t ncelm() const { return m_ncelm; }
    size_t nselm() const { return m_ncelm + 1; }

    size_t xsize() const { return m_agrid.size(); }

    array_type const & xcoord() const { return m_agrid.coord(); }
    array_type & xcoord() { return m_agrid.coord(); }

public:

    class CelmPK
    {
    private:
        CelmPK() = default;
        friend Celm;
    };

    /**
     * Get pointer to an coordinate value using conservation-element index.
     */
    real_type * xptr_celm(int_type ielm, bool odd_plane, CelmPK const &) { return xptr(xindex_celm(ielm, odd_plane)); }
    real_type const * xptr_celm(int_type ielm, bool odd_plane, CelmPK const &) const { return xptr(xindex_celm(ielm, odd_plane)); }

public:

    class SelmPK
    {
    private:
        SelmPK() = default;
        friend Selm;
    };

    /**
     * Get pointer to an coordinate value using conservation-element index.
     */
    real_type * xptr_selm(int_type ielm, bool odd_plane, SelmPK const &) { return xptr(xindex_selm(ielm, odd_plane)); }
    real_type const * xptr_selm(int_type ielm, bool odd_plane, SelmPK const &) const { return xptr(xindex_selm(ielm, odd_plane)); }

    /**
     * Convert celm index to coordinate index.
     */
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    size_t xindex_celm(int_type ielm) const { return 1 + BOUND_COUNT + static_cast<ssize_t>(ielm * 2); }
    size_t xindex_celm(int_type ielm, bool odd_plane) const { return xindex_celm(ielm) + (odd_plane ? 1 : 0); }

    /**
     * Convert selm index to coordinate index.
     */
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    size_t xindex_selm(int_type ielm) const { return BOUND_COUNT + static_cast<ssize_t>(ielm * 2); }
    size_t xindex_selm(int_type ielm, bool odd_plane) const { return xindex_selm(ielm) + (odd_plane ? 1 : 0); }

private:

    /**
     * Get pointer to an coordinate value using coordinate index.
     */
    real_type * xptr() { return m_agrid.data(); }
    real_type const * xptr() const { return m_agrid.data(); }
    real_type * xptr(size_t xindex) { return m_agrid.data() + xindex; }
    real_type const * xptr(size_t xindex) const { return m_agrid.data() + xindex; }

    void init_from_array(array_type const & xloc);

    real_type m_xmin;
    real_type m_xmax;
    size_t m_ncelm;

    AscendantGrid1d m_agrid;

    template <class ET>
    friend class ElementBase;

}; /* end class Grid */

template <class ET>
inline size_t ElementBase<ET>::xindex() const { return m_xptr - grid().xptr(); }

/**
 * Calculation kernel for the physical problem to be solved.  The kernel
 * defines how the solution elements calculate fluxes and other values.
 */
class Kernel
{

public:

    using value_type = Grid::value_type;

    using calc_type1 = std::function<value_type(Selm const &, size_t)>;
    using calc_type2 = std::function<void(Selm &)>;

    Kernel() { reset(); }
    void reset();

    Kernel(Kernel const &) = default;
    Kernel(Kernel &&) = default;
    Kernel & operator=(Kernel const &) = default;
    Kernel & operator=(Kernel &&) = default;
    ~Kernel() = default;

    // Accessors.
    calc_type1 const & xn_calc() const { return m_xn_calc; }
    calc_type1 & xn_calc() { return m_xn_calc; }
    calc_type1 const & xp_calc() const { return m_xp_calc; }
    calc_type1 & xp_calc() { return m_xp_calc; }
    calc_type1 const & tn_calc() const { return m_tn_calc; }
    calc_type1 & tn_calc() { return m_tn_calc; }
    calc_type1 const & tp_calc() const { return m_tp_calc; }
    calc_type1 & tp_calc() { return m_tp_calc; }
    calc_type1 const & so0p_calc() const { return m_so0p_calc; }
    calc_type1 & so0p_calc() { return m_so0p_calc; }
    calc_type2 const & cfl_updater() const { return m_cfl_updater; }
    calc_type2 & cfl_updater() { return m_cfl_updater; }

    // Calculating functions.
    value_type calc_xn(Selm const & se, size_t iv) const { return m_xn_calc(se, iv); }
    value_type calc_xp(Selm const & se, size_t iv) const { return m_xp_calc(se, iv); }
    value_type calc_tn(Selm const & se, size_t iv) const { return m_tn_calc(se, iv); }
    value_type calc_tp(Selm const & se, size_t iv) const { return m_tp_calc(se, iv); }
    value_type calc_so0p(Selm const & se, size_t iv) const { return m_so0p_calc(se, iv); }
    void update_cfl(Selm & se) { m_cfl_updater(se); }

private:

    calc_type1 m_xn_calc;
    calc_type1 m_xp_calc;
    calc_type1 m_tn_calc;
    calc_type1 m_tp_calc;
    calc_type1 m_so0p_calc;
    calc_type2 m_cfl_updater;

}; /* end class Kernel */

/**
 * Data class for solution.  It doesn't contain type information for the CE and
 * SE.  A Field declared as const is useless.
 */
class Field
{

public:

    using value_type = Grid::value_type;
    using array_type = Grid::array_type;

    Field(std::shared_ptr<Grid> const & grid, value_type time_increment, size_t nvar);

    Field() = delete;
    Field(Field const &) = default;
    Field(Field &&) = default;
    Field & operator=(Field const &) = default;
    Field & operator=(Field &&) = default;
    ~Field() = default;

    Field clone(bool grid = false) const
    {
        Field ret(*this);
        if (grid)
        {
            ret.m_grid = clone_grid();
        }
        return ret;
    }

    std::shared_ptr<Grid> clone_grid() const
    {
        return m_grid->clone();
    }

    void set_grid(std::shared_ptr<Grid> const & grid) { m_grid = grid; }

    Grid const & grid() const { return *m_grid; }
    Grid & grid() { return *m_grid; }

    array_type const & so0() const { return m_so0; }
    array_type & so0() { return m_so0; }
    array_type const & so1() const { return m_so1; }
    array_type & so1() { return m_so1; }
    array_type const & cfl() const { return m_cfl; }
    array_type & cfl() { return m_cfl; }

    value_type const & so0(size_t it, size_t iv) const { return m_so0(it, iv); }
    value_type & so0(size_t it, size_t iv) { return m_so0(it, iv); }
    value_type const & so1(size_t it, size_t iv) const { return m_so1(it, iv); }
    value_type & so1(size_t it, size_t iv) { return m_so1(it, iv); }
    value_type const & cfl(size_t it) const { return m_cfl(it); }
    value_type & cfl(size_t it) { return m_cfl(it); }

    size_t nvar() const { return m_so0.shape()[1]; }

    void set_time_increment(value_type time_increment);

    real_type time_increment() const { return m_time_increment; }
    real_type dt() const { return m_time_increment; }
    real_type hdt() const { return m_half_time_increment; }
    real_type qdt() const { return m_quarter_time_increment; }

    Kernel const & kernel() const { return m_kernel; }
    Kernel & kernel() { return m_kernel; }

    template <typename CE>
    CE const celm(int_type ielm, bool odd_plane) const { return CE(this, ielm, odd_plane, typename CE::const_ctor_passkey()); } // NOLINT(readability-const-return-type)
    template <typename CE>
    CE celm(int_type ielm, bool odd_plane) { return CE(this, ielm, odd_plane); }
    template <typename CE>
    CE const celm_at(int_type ielm, bool odd_plane) const; // NOLINT(readability-const-return-type)
    template <typename CE>
    CE celm_at(int_type ielm, bool odd_plane);

    template <typename SE>
    SE const selm(int_type ielm, bool odd_plane) const { return SE(this, ielm, odd_plane, typename SE::const_ctor_passkey()); } // NOLINT(readability-const-return-type)
    template <typename SE>
    SE selm(int_type ielm, bool odd_plane) { return SE(this, ielm, odd_plane); }
    template <typename SE>
    SE const selm_at(int_type ielm, bool odd_plane) const; // NOLINT(readability-const-return-type)
    template <typename SE>
    SE selm_at(int_type ielm, bool odd_plane);

private:

    std::shared_ptr<Grid> m_grid;

    array_type m_so0;
    array_type m_so1;
    array_type m_cfl;

    real_type m_time_increment = 0;
    // Cached value;
    real_type m_half_time_increment = 0;
    real_type m_quarter_time_increment = 0;

    Kernel m_kernel;

}; /* end class Field */

template <typename CE>
// NOLINTNEXTLINE(readability-const-return-type)
inline CE const Field::celm_at(int_type ielm, bool odd_plane) const
{
    const CE elm = celm<CE>(ielm, odd_plane);
    if (elm.xindex() < 2 || elm.xindex() >= grid().xsize() - 2)
    {
        throw std::out_of_range(
            std::format("Field::celm_at(ielm={}, odd_plane={}): xindex = {} outside the interval [2, {})",
                        ielm,
                        static_cast<int>(odd_plane),
                        elm.xindex(),
                        grid().xsize() - 2));
    }
    return elm;
}

template <typename CE>
inline CE Field::celm_at(int_type ielm, bool odd_plane)
{
    const CE elm = celm<CE>(ielm, odd_plane);
    if (elm.xindex() < 2 || elm.xindex() >= grid().xsize() - 2)
    {
        throw std::out_of_range(
            std::format("Field::celm_at(ielm={}, odd_plane={}): xindex = {} outside the interval [2, {})",
                        ielm,
                        static_cast<int>(odd_plane),
                        elm.xindex(),
                        grid().xsize() - 2));
    }
    return elm;
}

template <typename SE>
// NOLINTNEXTLINE(readability-const-return-type)
inline SE const Field::selm_at(int_type ielm, bool odd_plane) const
{
    const SE elm = selm<SE>(ielm, odd_plane);
    if (elm.xindex() < 1 || elm.xindex() >= grid().xsize() - 1)
    {
        throw std::out_of_range(
            std::format("Field::selm_at(ielm={}, odd_plane={}): xindex = {} outside the interval [1, {})",
                        ielm,
                        static_cast<int>(odd_plane),
                        elm.xindex(),
                        grid().xsize() - 1));
    }
    return elm;
}

template <typename SE>
inline SE Field::selm_at(int_type ielm, bool odd_plane)
{
    const SE elm = selm<SE>(ielm, odd_plane);
    if (elm.xindex() < 1 || elm.xindex() >= grid().xsize() - 1)
    {
        throw std::out_of_range(std::format("Field::selm_at(ielm={}, odd_plane={}): xindex = {} outside the interval [1, {})", ielm, static_cast<int>(odd_plane), elm.xindex(), grid().xsize() - 1));
    }
    return elm;
}

template <class ET>
inline Grid const & ElementBase<ET>::grid() const { return m_field->grid(); }

/**
 * A solution element.
 */
class Selm
    : public ElementBase<Selm>
{

public:

    Selm(Field * field, int_type index, bool odd_plane)
        : base_type(field, field->grid().xptr_selm(index, odd_plane, Grid::SelmPK()))
    {
    }

    class const_ctor_passkey
    {
        const_ctor_passkey() = default;
        friend Field;
    };

    Selm(Field const * field, int_type index, bool odd_plane, const_ctor_passkey)
        // The only intention of this workaround is to let const Field to
        // create const object derived from Selm. Do not abuse.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        : Selm(const_cast<Field *>(field), index, odd_plane)
    {
    }

    int_type index() const
    {
        static_assert(0 == (Grid::BOUND_COUNT % 2), "only work with even BOUND_COUNT");
        return (static_cast<int_type>(xindex()) >> 1) - 1;
    }

    /**
     * Return true for even plane, false for odd plane (temporal).
     */
    bool on_even_plane() const { return !on_odd_plane(); }
    bool on_odd_plane() const { return static_cast<bool>((xindex() - Grid::BOUND_COUNT) & 1); }

    value_type dxneg() const { return x() - xneg(); }
    value_type dxpos() const { return xpos() - x(); }
    value_type xctr() const { return (xneg() + xpos()) / 2; }

    void move_at(int_type offset);

    value_type const & so0(size_t iv) const { return field().so0(xindex(), iv); }
    value_type & so0(size_t iv) { return field().so0(xindex(), iv); }

    value_type const & so1(size_t iv) const { return field().so1(xindex(), iv); }
    value_type & so1(size_t iv) { return field().so1(xindex(), iv); }

    value_type const & cfl() const { return field().cfl(xindex()); }
    value_type & cfl() { return field().cfl(xindex()); }

    value_type xn(size_t iv) const { return field().kernel().calc_xn(*this, iv); }
    value_type xp(size_t iv) const { return field().kernel().calc_xp(*this, iv); }
    value_type tn(size_t iv) const { return field().kernel().calc_tn(*this, iv); }
    value_type tp(size_t iv) const { return field().kernel().calc_tp(*this, iv); }
    value_type so0p(size_t iv) const { return field().kernel().calc_so0p(*this, iv); }
    void update_cfl() { field().kernel().update_cfl(*this); }

}; /* end class Selm */

inline void Kernel::reset()
{
    m_xn_calc = [](Selm const &, size_t)
    { return 0.0; };
    m_xp_calc = [](Selm const &, size_t)
    { return 0.0; };
    m_tn_calc = [](Selm const &, size_t)
    { return 0.0; };
    m_tp_calc = [](Selm const &, size_t)
    { return 0.0; };
    m_so0p_calc = [](Selm const & se, size_t iv)
    { return se.so0(iv); };
    m_cfl_updater = [](Selm & se)
    { se.cfl() = 0.0; };
}

/* TODO: The naming of Celm and CelmBase is confusing: Celm is the base class
 * of CelmBase. */

/**
 * A compound conservation celm.
 */
class Celm
    : public ElementBase<Celm>
{

public:

    using selm_type = Selm;

    Celm(Field * field, int_type index, bool odd_plane)
        : base_type(field, field->grid().xptr_celm(index, odd_plane, Grid::CelmPK()))
    {
    }

    class const_ctor_passkey
    {
        const_ctor_passkey() = default;
        friend Field;
    };

    Celm(Field const * field, int_type index, bool odd_plane, const_ctor_passkey)
        // The only intention of this workaround is to let const Field to
        // create const object derived from CelmBase. Do not abuse.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        : Celm(const_cast<Field *>(field), index, odd_plane)
    {
    }

    int_type index() const
    {
        static_assert(0 == (Grid::BOUND_COUNT % 2), "only work with even BOUND_COUNT");
        return (static_cast<int_type>(xindex() - 1) >> 1) - 1;
    }

    /**
     * Return true for even plane, false for odd plane (temporal).
     */
    bool on_even_plane() const { return !on_odd_plane(); }
    bool on_odd_plane() const { return static_cast<bool>((xindex() - (1 + Grid::BOUND_COUNT)) & 1); }

    value_type xctr() const { return x(); }

    void move_at(int_type offset);

    value_type time_increment() const { return field().time_increment(); }
    value_type dt() const { return field().dt(); }
    value_type hdt() const { return field().hdt(); }
    value_type qdt() const { return field().qdt(); }

    template <typename SE>
    SE const selm_xn() const { return field().template selm<SE>(index(), on_odd_plane()); } // NOLINT(readability-const-return-type)
    template <typename SE>
    SE selm_xn() { return field().template selm<SE>(index(), on_odd_plane()); }
    template <typename SE>
    SE const selm_xp() const { return field().template selm<SE>(index() + 1, on_odd_plane()); } // NOLINT(readability-const-return-type)
    template <typename SE>
    SE selm_xp() { return field().template selm<SE>(index() + 1, on_odd_plane()); }
    template <typename SE>
    SE const selm_tn() const { return field().template selm<SE>(index() + on_odd_plane(), !on_odd_plane()); } // NOLINT(readability-const-return-type)
    template <typename SE>
    SE selm_tn() { return field().template selm<SE>(index() + on_odd_plane(), !on_odd_plane()); }
    template <typename SE>
    SE const selm_tp() const { return field().template selm<SE>(index() + on_odd_plane(), !on_odd_plane()); } // NOLINT(readability-const-return-type)
    template <typename SE>
    SE selm_tp() { return field().template selm<SE>(index() + on_odd_plane(), !on_odd_plane()); }

    template <typename SE>
    value_type calc_so0(size_t iv) const;
    template <typename SE, size_t ALPHA>
    value_type calc_so1_alpha(size_t iv) const;

    Selm const selm_xn() const { return selm_xn<Selm>(); } // NOLINT(readability-const-return-type)
    Selm selm_xn() { return selm_xn<Selm>(); }
    Selm const selm_xp() const { return selm_xp<Selm>(); } // NOLINT(readability-const-return-type)
    Selm selm_xp() { return selm_xp<Selm>(); }
    Selm const selm_tn() const { return selm_tn<Selm>(); } // NOLINT(readability-const-return-type)
    Selm selm_tn() { return selm_tn<Selm>(); }
    Selm const selm_tp() const { return selm_tn<Selm>(); } // NOLINT(readability-const-return-type)
    Selm selm_tp() { return selm_tn<Selm>(); }

    value_type calc_so0(size_t iv) const { return calc_so0<Selm>(iv); }
    template <size_t ALPHA>
    value_type calc_so1_alpha(size_t iv) const { return calc_so1_alpha<Selm, ALPHA>(iv); }

}; /* end class Celm */

template <typename SE>
class CelmBase
    : public Celm
{

public:

    using base_type = Celm;
    using base_type::base_type;
    using selm_type = SE;

    // NOLINTNEXTLINE(readability-const-return-type)
    SE const selm_xn() const { return this->Celm::selm_xn<SE>(); }
    SE selm_xn() { return this->Celm::selm_xn<SE>(); }
    // NOLINTNEXTLINE(readability-const-return-type)
    SE const selm_xp() const { return this->Celm::selm_xp<SE>(); }
    SE selm_xp() { return this->Celm::selm_xp<SE>(); }
    // NOLINTNEXTLINE(readability-const-return-type)
    SE const selm_tn() const { return this->Celm::selm_tn<SE>(); }
    SE selm_tn() { return this->Celm::selm_tn<SE>(); }
    // NOLINTNEXTLINE(readability-const-return-type)
    SE const selm_tp() const { return this->Celm::selm_tp<SE>(); }
    SE selm_tp() { return this->Celm::selm_tp<SE>(); }

    value_type calc_so0(size_t iv) const { return this->Celm::calc_so0<SE>(iv); }
    template <size_t ALPHA>
    value_type calc_so1_alpha(size_t iv) const
    {
        return this->Celm::calc_so1_alpha<SE, ALPHA>(iv);
    }

}; /* end class CelmBase */

template <typename SE>
inline typename Celm::value_type Celm::calc_so0(size_t iv) const
{
    const SE se_xn = selm_xn<SE>();
    const SE se_xp = selm_xp<SE>();
    const value_type flux_ll = se_xn.xp(iv) + se_xn.tp(iv);
    const value_type flux_ur = se_xp.xn(iv) - se_xp.tp(iv);
    const SE se_tp = selm_tp<SE>();
    return (flux_ll + flux_ur) / se_tp.dx();
}

template <typename SE, size_t ALPHA>
inline typename Celm::value_type Celm::calc_so1_alpha(size_t iv) const
{
    // Fetch value.
    const SE se_xn = selm_xn<SE>();
    const SE se_xp = selm_xp<SE>();
    const value_type upn = se_xn.so0p(iv); // u' at left SE
    const value_type upp = se_xp.so0p(iv); // u' at right SE
    const value_type utp = selm_tp().so0(iv); // u at top SE
    // alpha-scheme.
    const value_type duxn = (utp - upn) / se_xn.dxpos();
    const value_type duxp = (upp - utp) / se_xp.dxneg();
    const value_type fan = pow<ALPHA>(std::fabs(duxn));
    const value_type fap = pow<ALPHA>(std::fabs(duxp));
    constexpr value_type tiny = std::numeric_limits<value_type>::min();
    return (fap * duxn + fan * duxp) / (fap + fan + tiny);
}

class Selm;

/**
 * Algorithmic definition for solution.  It holds the type information for the
 * CE and SE.
 */
template <typename ST, typename CE, typename SE>
class SolverBase
    : public std::enable_shared_from_this<ST>
{

public:

    using value_type = Field::value_type;
    using array_type = Field::array_type;
    using celm_type = CE;
    using selm_type = SE;

protected:

    class ctor_passkey
    {
    };

    template <class... Args>
    static std::shared_ptr<ST> construct_impl(Args &&... args)
    {
        return std::make_shared<ST>(std::forward<Args>(args)..., ctor_passkey());
    }

public:

    std::shared_ptr<ST> clone(bool grid = false)
    {
        /* The only purpose of this reinterpret_cast is to workaround for
         * static polymorphism. */
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        auto ret = std::make_shared<ST>(*reinterpret_cast<ST *>(this));
        if (grid)
        {
            std::shared_ptr<Grid> const new_grid = m_field.clone_grid();
            ret->m_field.set_grid(new_grid);
        }
        return ret;
    }

    SolverBase(
        std::shared_ptr<Grid> const & grid, value_type time_increment, size_t nvar, ctor_passkey const &)
        : m_field(grid, time_increment, nvar)
    {
    }

    SolverBase() = delete;
    SolverBase(SolverBase const &) = default;
    SolverBase(SolverBase &&) = default;
    SolverBase & operator=(SolverBase const &) = default;
    SolverBase & operator=(SolverBase &&) = default;
    ~SolverBase() = default;

    Grid const & grid() const { return m_field.grid(); }
    Grid & grid() { return m_field.grid(); }

    array_type x(bool odd_plane) const;
    array_type xctr(bool odd_plane) const;

#define DECL_ST_ARRAY_ACCESS_0D(NAME)            \
    array_type const & NAME() const              \
    {                                            \
        return m_field.NAME();                   \
    }                                            \
    array_type & NAME()                          \
    {                                            \
        return m_field.NAME();                   \
    }                                            \
    array_type get_##NAME(bool odd_plane) const; \
    void set_##NAME(array_type const & arr, bool odd_plane);
#define DECL_ST_ARRAY_ACCESS_1D(NAME)                       \
    array_type const & NAME() const                         \
    {                                                       \
        return m_field.NAME();                              \
    }                                                       \
    array_type & NAME()                                     \
    {                                                       \
        return m_field.NAME();                              \
    }                                                       \
    array_type get_##NAME(size_t iv, bool odd_plane) const; \
    void set_##NAME(size_t iv, array_type const & arr, bool odd_plane);

    DECL_ST_ARRAY_ACCESS_0D(cfl)
    DECL_ST_ARRAY_ACCESS_1D(so0)
    DECL_ST_ARRAY_ACCESS_1D(so1)

#undef DECL_ST_ARRAY_ACCESS_1D
#undef DECL_ST_ARRAY_ACCESS_0D

    array_type get_so0p(size_t iv, bool odd_plane) const;

    size_t nvar() const { return m_field.nvar(); }

    void set_time_increment(value_type time_increment) { m_field.set_time_increment(time_increment); }

    real_type time_increment() const { return m_field.time_increment(); }
    real_type dt() const { return m_field.dt(); }
    real_type hdt() const { return m_field.hdt(); }
    real_type qdt() const { return m_field.qdt(); }

    Kernel const & kernel() const { return m_field.kernel(); }
    Kernel & kernel() { return m_field.kernel(); }

    // NOLINTNEXTLINE(readability-const-return-type)
    CE const celm(int_type ielm, bool odd_plane) const { return m_field.celm<CE>(ielm, odd_plane); }
    CE celm(int_type ielm, bool odd_plane) { return m_field.celm<CE>(ielm, odd_plane); }
    // NOLINTNEXTLINE(readability-const-return-type)
    CE const celm_at(int_type ielm, bool odd_plane) const { return m_field.celm_at<CE>(ielm, odd_plane); }
    CE celm_at(int_type ielm, bool odd_plane) { return m_field.celm_at<CE>(ielm, odd_plane); }

    // NOLINTNEXTLINE(readability-const-return-type)
    SE const selm(int_type ielm, bool odd_plane) const { return m_field.selm<SE>(ielm, odd_plane); }
    SE selm(int_type ielm, bool odd_plane) { return m_field.selm<SE>(ielm, odd_plane); }
    // NOLINTNEXTLINE(readability-const-return-type)
    SE const selm_at(int_type ielm, bool odd_plane) const { return m_field.selm_at<SE>(ielm, odd_plane); }
    SE selm_at(int_type ielm, bool odd_plane) { return m_field.selm_at<SE>(ielm, odd_plane); }

    void update_cfl(bool odd_plane);
    void march_half_so0(bool odd_plane);
    template <size_t ALPHA>
    void march_half_so1_alpha(bool odd_plane);
    void treat_boundary_so0();
    void treat_boundary_so1();

    void setup_march() { update_cfl(false); }
    template <size_t ALPHA>
    void march_half1_alpha();
    template <size_t ALPHA>
    void march_half2_alpha();
    template <size_t ALPHA>
    void march_alpha(size_t steps);

private:

    Field m_field;

}; /* end class SolverBase */

template <typename ST, typename CE, typename SE>
inline typename SolverBase<ST, CE, SE>::array_type
SolverBase<ST, CE, SE>::x(bool odd_plane) const
{
    const uint_type nselm = static_cast<uint_type>(grid().nselm()) - static_cast<uint_type>(odd_plane);
    array_type ret(small_vector<ssize_t>{static_cast<ssize_t>(nselm)});
    for (uint_type it = 0; it < nselm; ++it) { ret[it] = selm(it, odd_plane).x(); }
    return ret;
}

template <typename ST, typename CE, typename SE>
inline typename SolverBase<ST, CE, SE>::array_type
SolverBase<ST, CE, SE>::xctr(bool odd_plane) const
{
    const uint_type nselm = static_cast<uint_type>(grid().nselm()) - static_cast<uint_type>(odd_plane);
    array_type ret(small_vector<ssize_t>{static_cast<ssize_t>(nselm)});
    for (uint_type it = 0; it < nselm; ++it) { ret[it] = selm(it, odd_plane).xctr(); }
    return ret;
}

template <typename ST, typename CE, typename SE>
inline typename SolverBase<ST, CE, SE>::array_type
SolverBase<ST, CE, SE>::get_so0p(size_t iv, bool odd_plane) const
{
    if (iv >= m_field.nvar())
    {
        throw std::out_of_range("get_so0p(): out of nvar range");
    }
    const uint_type nselm = static_cast<uint_type>(grid().nselm()) - static_cast<uint_type>(odd_plane);
    array_type ret(small_vector<ssize_t>{static_cast<ssize_t>(nselm)});
    for (uint_type it = 0; it < nselm; ++it) { ret[it] = selm(it, odd_plane).so0p(iv); }
    return ret;
}

template <typename ST, typename CE, typename SE>
inline typename SolverBase<ST, CE, SE>::array_type
SolverBase<ST, CE, SE>::get_so0(size_t iv, bool odd_plane) const
{
    if (iv >= m_field.nvar())
    {
        throw std::out_of_range("get_so0(): out of nvar range");
    }
    const uint_type nselm = static_cast<uint_type>(grid().nselm()) - static_cast<uint_type>(odd_plane);
    array_type ret(small_vector<ssize_t>{static_cast<ssize_t>(nselm)});
    for (uint_type it = 0; it < nselm; ++it) { ret[it] = selm(it, odd_plane).so0(iv); }
    return ret;
}

template <typename ST, typename CE, typename SE>
inline typename SolverBase<ST, CE, SE>::array_type
SolverBase<ST, CE, SE>::get_so1(size_t iv, bool odd_plane) const
{
    if (iv >= m_field.nvar())
    {
        throw std::out_of_range("get_so1(): out of nvar range");
    }
    const uint_type nselm = static_cast<uint_type>(grid().nselm()) - static_cast<uint_type>(odd_plane);
    array_type ret(small_vector<ssize_t>{static_cast<ssize_t>(nselm)});
    for (uint_type it = 0; it < nselm; ++it) { ret[it] = selm(it, odd_plane).so1(iv); }
    return ret;
}

template <typename ST, typename CE, typename SE>
inline void
SolverBase<ST, CE, SE>::set_so0(size_t iv, typename SolverBase<ST, CE, SE>::array_type const & arr, bool odd_plane)
{
    if (iv >= m_field.nvar())
    {
        throw std::out_of_range(std::format("set_so0(): iv {} >= nvar {}", iv, m_field.nvar()));
    }
    if (1 != arr.shape().size())
    {
        throw std::out_of_range("set_so0(): input not 1D");
    }
    const uint_type nselm = static_cast<uint_type>(grid().nselm()) - static_cast<uint_type>(odd_plane);
    if (nselm != arr.size())
    {
        throw std::out_of_range(std::format("set_so0(): arr size {} != nselm {}", arr.size(), nselm));
    }
    for (uint_type it = 0; it < nselm; ++it) { selm(it, odd_plane).so0(iv) = arr[it]; }
}

template <typename ST, typename CE, typename SE>
inline void
SolverBase<ST, CE, SE>::set_so1(size_t iv, typename SolverBase<ST, CE, SE>::array_type const & arr, bool odd_plane)
{
    if (iv >= m_field.nvar())
    {
        throw std::out_of_range("set_so1(): out of nvar range");
    }
    if (1 != arr.shape().size())
    {
        throw std::out_of_range("set_so1(): input not 1D");
    }
    const uint_type nselm = static_cast<uint_type>(grid().nselm()) - static_cast<uint_type>(odd_plane);
    if (nselm != arr.size())
    {
        throw std::out_of_range("set_so1(): input wrong size");
    }
    for (uint_type it = 0; it < nselm; ++it) { selm(it, odd_plane).so1(iv) = arr[it]; }
}

template <typename ST, typename CE, typename SE>
inline typename SolverBase<ST, CE, SE>::array_type
SolverBase<ST, CE, SE>::get_cfl(bool odd_plane) const
{
    const uint_type nselm = static_cast<uint_type>(grid().nselm()) - static_cast<uint_type>(odd_plane);
    array_type ret(small_vector<ssize_t>{static_cast<ssize_t>(nselm)});
    for (uint_type it = 0; it < nselm; ++it) { ret[it] = selm(it, odd_plane).cfl(); }
    return ret;
}

template <typename ST, typename CE, typename SE>
inline void
SolverBase<ST, CE, SE>::set_cfl(typename SolverBase<ST, CE, SE>::array_type const & arr, bool odd_plane)
{
    if (1 != arr.shape().size())
    {
        throw std::out_of_range("set_so1(): input not 1D");
    }
    const uint_type nselm = static_cast<uint_type>(grid().nselm()) - static_cast<uint_type>(odd_plane);
    if (nselm != arr.size())
    {
        throw std::out_of_range("set_so1(): input wrong size");
    }
    for (uint_type it = 0; it < nselm; ++it) { selm(it, odd_plane).cfl() = arr[it]; }
}

template <typename ST, typename CE, typename SE>
inline void SolverBase<ST, CE, SE>::march_half_so0(bool odd_plane)
{
    const int_type start = odd_plane ? -1 : 0;
    const auto stop = static_cast<int_type>(grid().ncelm());
    for (int_type ic = start; ic < stop; ++ic)
    {
        auto ce = celm(ic, odd_plane);
        ce.selm_tp().so0(0) = ce.calc_so0(0);
    }
}

template <typename ST, typename CE, typename SE>
inline void SolverBase<ST, CE, SE>::update_cfl(bool odd_plane)
{
    const int_type start = odd_plane ? -1 : 0;
    const auto stop = static_cast<int_type>(grid().nselm());
    for (int_type ic = start; ic < stop; ++ic)
    {
        selm(ic, odd_plane).update_cfl();
    }
}

template <typename ST, typename CE, typename SE>
template <size_t ALPHA>
inline void SolverBase<ST, CE, SE>::march_half_so1_alpha(bool odd_plane)
{
    const int_type start = odd_plane ? -1 : 0;
    const auto stop = static_cast<int_type>(grid().ncelm());
    for (int_type ic = start; ic < stop; ++ic)
    {
        auto ce = celm(ic, odd_plane);
        ce.selm_tp().so1(0) = ce.template calc_so1_alpha<ALPHA>(0);
    }
}

template <typename ST, typename CE, typename SE>
inline void SolverBase<ST, CE, SE>::treat_boundary_so0()
{
    SE const selm_left_in = selm(0, true);
    SE selm_left_out = selm(-1, true);
    SE const selm_right_in = selm(static_cast<int_type>(grid().ncelm() - 1), true);
    SE selm_right_out = selm(static_cast<int_type>(grid().ncelm()), true);

    selm_left_out.so0(0) = selm_right_in.so0(0);
    selm_right_out.so0(0) = selm_left_in.so0(0);
}

template <typename ST, typename CE, typename SE>
inline void SolverBase<ST, CE, SE>::treat_boundary_so1()
{
    SE const selm_left_in = selm(0, true);
    SE selm_left_out = selm(-1, true);
    SE const selm_right_in = selm(static_cast<int_type>(grid().ncelm()) - 1, true);
    SE selm_right_out = selm(static_cast<int_type>(grid().ncelm()), true);

    selm_left_out.so1(0) = selm_right_in.so1(0);
    selm_right_out.so1(0) = selm_left_in.so1(0);
}

template <typename ST, typename CE, typename SE>
template <size_t ALPHA>
inline void SolverBase<ST, CE, SE>::march_half1_alpha()
{
    march_half_so0(false);
    treat_boundary_so0();
    update_cfl(true);
    march_half_so1_alpha<ALPHA>(false);
    treat_boundary_so1();
}

template <typename ST, typename CE, typename SE>
template <size_t ALPHA>
inline void SolverBase<ST, CE, SE>::march_half2_alpha()
{
    // In the second half step, no treating boundary conditions.
    march_half_so0(true);
    update_cfl(false);
    march_half_so1_alpha<ALPHA>(true);
}

template <typename ST, typename CE, typename SE>
template <size_t ALPHA>
inline void SolverBase<ST, CE, SE>::march_alpha(size_t steps)
{
    for (size_t it = 0; it < steps; ++it)
    {
        march_half1_alpha<ALPHA>();
        march_half2_alpha<ALPHA>();
    }
}

class Solver
    : public SolverBase<Solver, Celm, Selm>
{

public:

    using base_type = SolverBase<Solver, Celm, Selm>;
    using base_type::base_type;

    static std::shared_ptr<Solver>
    construct(std::shared_ptr<Grid> const & grid, value_type time_increment, size_t nvar)
    {
        return construct_impl(grid, time_increment, nvar);
    }

}; /* end class Solver */

} /* end namespace spacetime */

} /* end namespace solvcon */

#define SPACETIME_DERIVED_SELM_BODY_DEFAULT \
public:                                     \
    using base_type = Selm;                 \
    using base_type::base_type;             \
    value_type xn(size_t iv) const;         \
    value_type xp(size_t iv) const;         \
    value_type tn(size_t iv) const;         \
    value_type tp(size_t iv) const;         \
    value_type so0p(size_t iv) const;       \
    void update_cfl();

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

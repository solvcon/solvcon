#pragma once

/*
 * Copyright (c) 2017, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <memory>
#include <vector>

#include "march/core.hpp"
#include "march/mesh.hpp"

#include "march/gas/Solution.hpp"
#include "march/gas/Solver_decl.hpp"

namespace march {

namespace gas {

template< size_t NDIM > class Anchor;

/**
 * Anchor wrapper for treating both 2- and 3-dimensional anchor.
 */
class CommonAnchor
  : public std::enable_shared_from_this<CommonAnchor>
{

public:

    CommonAnchor() = delete;
    CommonAnchor(CommonAnchor const & ) = delete;
    CommonAnchor(CommonAnchor       &&) = delete;
    CommonAnchor & operator=(CommonAnchor const & ) = delete;
    CommonAnchor & operator=(CommonAnchor       &&) = delete;

    virtual ~CommonAnchor() {}

protected:

    class ctor_passkey {};

public:

    template <size_t NDIM> CommonAnchor(ctor_passkey const &, Solver<NDIM> & svr)
      : m_ndim(NDIM), m_solver(&svr) {}

    template <size_t NDIM>
    static std::shared_ptr<CommonAnchor> construct(Solver<NDIM> & svr) {
        return std::make_shared<CommonAnchor>(ctor_passkey(), svr);
    }

#define DECL_MARCH_GAS_COMMON_ANCHOR_METHOD(NAME) \
    virtual void NAME() {}

    DECL_MARCH_GAS_COMMON_ANCHOR_METHOD(provide)
    DECL_MARCH_GAS_COMMON_ANCHOR_METHOD(preloop)
    DECL_MARCH_GAS_COMMON_ANCHOR_METHOD(premarch)
    DECL_MARCH_GAS_COMMON_ANCHOR_METHOD(prefull)
    DECL_MARCH_GAS_COMMON_ANCHOR_METHOD(presub)
    DECL_MARCH_GAS_COMMON_ANCHOR_METHOD(postsub)
    DECL_MARCH_GAS_COMMON_ANCHOR_METHOD(postfull)
    DECL_MARCH_GAS_COMMON_ANCHOR_METHOD(postmarch)
    DECL_MARCH_GAS_COMMON_ANCHOR_METHOD(postloop)
    DECL_MARCH_GAS_COMMON_ANCHOR_METHOD(exhaust)

#undef DECL_MARCH_GAS_COMMON_ANCHOR_METHOD

    size_t ndim() const { return m_ndim; }
    size_t ndim()       { return m_ndim; }

    template <size_t NDIM> Solver<NDIM> & solver() const {
        assert(m_solver);
        if (NDIM != ndim()) { throw std::runtime_error("CommonAnchor::solver dimension mismatch"); }
        return *reinterpret_cast<Solver<NDIM>*>(m_solver);
    }

    template <size_t NDIM> std::shared_ptr<Anchor<NDIM>> make_owner();

private:

    size_t m_ndim = 0;
    void * m_solver = nullptr;

}; /* end class CommonAnchor */

template< size_t NDIM >
class Anchor
  : public std::enable_shared_from_this<Anchor<NDIM>>
{

public:

    using solver_type = Solver<NDIM>;

    Anchor() = delete;
    Anchor(Anchor const & ) = delete;
    Anchor(Anchor       &&) = delete;
    Anchor & operator=(Anchor const & ) = delete;
    Anchor & operator=(Anchor       &&) = delete;

protected:

    class ctor_passkey {};

public:

    Anchor(ctor_passkey const &, solver_type & svr, std::shared_ptr<CommonAnchor> const & common)
      : m_solver(svr), m_common(common) {}

    static std::shared_ptr<Anchor<NDIM>> construct(
        solver_type & svr
      , std::shared_ptr<CommonAnchor> const & common = std::shared_ptr<CommonAnchor>()
    ) {
        return std::make_shared<Anchor<NDIM>>(ctor_passkey(), svr, common);
    }

    virtual ~Anchor() = default;

    solver_type const & solver() const { return m_solver; }
    solver_type       & solver()       { return m_solver; }

#define DECL_MARCH_GAS_ANCHOR_METHOD(NAME) \
    virtual void NAME() { if (m_common) { m_common->NAME(); } }

    DECL_MARCH_GAS_ANCHOR_METHOD(provide)
    DECL_MARCH_GAS_ANCHOR_METHOD(preloop)
    DECL_MARCH_GAS_ANCHOR_METHOD(premarch)
    DECL_MARCH_GAS_ANCHOR_METHOD(prefull)
    DECL_MARCH_GAS_ANCHOR_METHOD(presub)
    DECL_MARCH_GAS_ANCHOR_METHOD(postsub)
    DECL_MARCH_GAS_ANCHOR_METHOD(postfull)
    DECL_MARCH_GAS_ANCHOR_METHOD(postmarch)
    DECL_MARCH_GAS_ANCHOR_METHOD(postloop)
    DECL_MARCH_GAS_ANCHOR_METHOD(exhaust)

#undef DECL_MARCH_GAS_ANCHOR_METHOD

private:

    solver_type & m_solver;
    std::shared_ptr<CommonAnchor> m_common;

}; /* end class Anchor */

template <size_t NDIM> std::shared_ptr<Anchor<NDIM>>
CommonAnchor::make_owner() {
    if (NDIM == ndim()) {
        return Anchor<NDIM>::construct(solver<NDIM>(), shared_from_this());
    } else {
        return std::shared_ptr<Anchor<NDIM>>();
    }
}

template< size_t NDIM >
class AnchorChain
{

public:

    using anchor_type = Anchor<NDIM>;
    using anchor_ptr = std::shared_ptr<anchor_type>;

    struct LifeManager { virtual ~LifeManager() {} };

    std::unique_ptr<LifeManager>       & life_manager()       { return m_life_manager; }
    std::unique_ptr<LifeManager> const & life_manager() const { return m_life_manager; }

    void push_back(anchor_ptr const & ptr) { m_anchors.push_back(ptr); }
    void append(anchor_ptr const & ptr, std::string const & name) {
        m_anchors.push_back(ptr);
        m_names.emplace(name, ptr);
    }

#define DECL_MARCH_GAS_ANCHOR_CALL_FORWARD(NAME) \
    void NAME() { \
        for (auto & anchor : m_anchors) { \
            anchor->NAME(); \
        } \
    }
    DECL_MARCH_GAS_ANCHOR_CALL_FORWARD(provide)
    DECL_MARCH_GAS_ANCHOR_CALL_FORWARD(preloop)
    DECL_MARCH_GAS_ANCHOR_CALL_FORWARD(premarch)
    DECL_MARCH_GAS_ANCHOR_CALL_FORWARD(prefull)
    DECL_MARCH_GAS_ANCHOR_CALL_FORWARD(presub)
    DECL_MARCH_GAS_ANCHOR_CALL_FORWARD(postsub)
    DECL_MARCH_GAS_ANCHOR_CALL_FORWARD(postfull)
    DECL_MARCH_GAS_ANCHOR_CALL_FORWARD(postmarch)
#undef DECL_MARCH_GAS_ANCHOR_CALL_FORWARD

// FIXME: reversed for each?
#define DECL_MARCH_GAS_ANCHOR_CALL_BACKWARD(NAME) \
    void NAME() { \
        for (auto it = m_anchors.rbegin(); it != m_anchors.rend(); ++it) { \
            (*it)->NAME(); \
        } \
    }
    DECL_MARCH_GAS_ANCHOR_CALL_BACKWARD(postloop)
    DECL_MARCH_GAS_ANCHOR_CALL_BACKWARD(exhaust)
#undef DECL_MARCH_GAS_ANCHOR_CALL_BACKWARD

private:

    std::vector<anchor_ptr> m_anchors;
    std::map<std::string, anchor_ptr> m_names;
    std::unique_ptr<LifeManager> m_life_manager;

}; /* end class AnchorChain */

} /* end namespace gas */

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:

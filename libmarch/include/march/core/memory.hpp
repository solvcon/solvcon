#pragma once

/*
 * Copyright (c) 2018, Yung-Yu Chen <yyc@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * \file
 * Memory facilities.
 */

namespace march {

/**
 * Mixin to help track the active instance count.
 */
template< class T >
class InstanceCounter {

public:

    InstanceCounter() { ++m_active_instance_count; }
    InstanceCounter(InstanceCounter const &) { ++m_active_instance_count; }
    ~InstanceCounter() { --m_active_instance_count; }

    static size_t active_instance_count() { return m_active_instance_count; }

private:

    static size_t m_active_instance_count;

}; /* end class InstanceCounter */

template< class T > size_t InstanceCounter<T>::m_active_instance_count = 0;

} /* end namespace march */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4:

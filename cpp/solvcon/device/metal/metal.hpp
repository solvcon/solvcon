#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Singleton manager for the Metal GPU device used by the linear algebra
 * backend.
 *
 * @ingroup group_core
 */

#include <memory>

namespace solvcon
{

namespace device
{

/**
 * Process-wide owner of the single Metal device handle.
 *
 * The instance is created on first access and acquires the Metal device in
 * startup(); shutdown() releases it. Copy and move are deleted so the device
 * has exactly one owner.
 *
 * @ingroup group_core
 */
class MetalManager
{

public:

    static MetalManager & instance();

    MetalManager(MetalManager const &) = delete;
    MetalManager(MetalManager &&) = delete;
    MetalManager & operator=(MetalManager const &) = delete;
    MetalManager & operator=(MetalManager &&) = delete;
    ~MetalManager();

    void startup();
    bool started() { return nullptr != m_impl; }
    void shutdown();

private:

    class Impl;

    MetalManager();

    std::unique_ptr<Impl> m_impl;

}; /* end class MetalManager */

} /* end namespace device */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

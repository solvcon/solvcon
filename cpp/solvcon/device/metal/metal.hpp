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

#include <solvcon/buffer/BufferStorage.hpp>

#include <cstddef>
#include <memory>

namespace solvcon
{

namespace device
{

/**
 * Process-wide owner of a unified-memory Metal device handle.
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

    /**
     * Allocate CPU-visible shared Metal storage.
     * @internal
     * @param nbytes Size of the storage in bytes.
     * @param alignment Alignment in bytes: 0, 16, 32, or 64; nbytes must be a multiple of a nonzero alignment.
     * @return Storage and the owner of its Metal resource.
     * @throw std::invalid_argument If the alignment or size is invalid.
     * @throw std::length_error If the requested storage exceeds the device limit.
     * @throw std::runtime_error If Metal storage is unavailable.
     * @throw std::bad_alloc If storage allocation fails.
     */
    detail::DeviceBufferStorage allocate_buffer(std::size_t nbytes, std::size_t alignment);

private:

    class Impl;

    MetalManager();

    std::unique_ptr<Impl> m_impl;

}; /* end class MetalManager */

} /* end namespace device */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

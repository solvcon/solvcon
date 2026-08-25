#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Ownership contracts for ConcreteBuffer storage.
 *
 * @ingroup group_core
 */

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace solvcon
{

namespace detail
{

/**
 * Base class for buffer-storage deallocation.
 *
 * ConcreteBuffer's data deleter calls this object to release the backing
 * storage. A derived remover may instead release its owner when the remover
 * itself is destroyed.
 */
struct ConcreteBufferRemover
{

    ConcreteBufferRemover() = default;
    ConcreteBufferRemover(ConcreteBufferRemover const &) = default;
    ConcreteBufferRemover(ConcreteBufferRemover &&) = default;
    ConcreteBufferRemover & operator=(ConcreteBufferRemover const &) = default;
    ConcreteBufferRemover & operator=(ConcreteBufferRemover &&) = default;
    virtual ~ConcreteBufferRemover() = default;

    /**
     * Release heap storage with its original alignment.
     * @param p Storage to release.
     * @param alignment Original alignment; 0 means no explicit alignment.
     */
    static void deallocate_memory(int8_t * p, size_t alignment);

    /**
     * Release storage or its backing resource.
     * @param p CPU-visible storage pointer.
     * @param alignment Storage alignment; 0 means no explicit alignment.
     */
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,readability-non-const-parameter)
    virtual void operator()(int8_t * p, size_t alignment) const { deallocate_memory(p, alignment); }

}; /* end struct ConcreteBufferRemover */

inline void ConcreteBufferRemover::deallocate_memory(int8_t * p, size_t alignment)
{
    if (alignment > 0) // NOLINT(bugprone-branch-clone)
    {
#ifdef _WIN32
        _aligned_free(p); // NOLINT(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
#else
        std::free(p); // NOLINT(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
#endif
    }
    else
    {
        std::free(p); // NOLINT(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
    }
}

/**
 * Transfer object for CPU-visible device-backed storage.
 *
 * The remover moves into ConcreteBuffer, where it keeps the backing resource
 * alive and handles release. A successful allocation has a non-null remover;
 * its data pointer is null only for zero-byte storage.
 */
struct DeviceBufferStorage
{

    int8_t * m_data = nullptr; ///< CPU-visible pointer, or nullptr for zero-byte storage.
    std::unique_ptr<ConcreteBufferRemover> m_remover; ///< Keeps the backing resource alive until destruction.

}; /* end struct DeviceBufferStorage */

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

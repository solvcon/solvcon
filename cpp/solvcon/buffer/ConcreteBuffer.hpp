#pragma once

/*
 * Copyright (c) 2019, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Reference-counted contiguous memory buffer.
 *
 * @ingroup group_core
 */

#include <solvcon/base.hpp>
#include <solvcon/buffer/BufferBase.hpp>
#include <solvcon/buffer/small_vector.hpp>

#include <algorithm>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace solvcon
{

namespace detail
{

// Take the remover and deleter classes outside ConcreteBuffer to work around
// https://bugzilla.redhat.com/show_bug.cgi?id=1569374

/**
 * The base class of memory deallocator for ConcreteBuffer.  When the object
 * exists in ConcreteBufferDataDeleter (the unique_ptr deleter), the deleter
 * calls it to release the memory of the ConcreteBuffer data buffer.
 */
struct ConcreteBufferRemover
{

    ConcreteBufferRemover() = default;
    ConcreteBufferRemover(ConcreteBufferRemover const &) = default;
    ConcreteBufferRemover(ConcreteBufferRemover &&) = default;
    ConcreteBufferRemover & operator=(ConcreteBufferRemover const &) = default;
    ConcreteBufferRemover & operator=(ConcreteBufferRemover &&) = default;
    virtual ~ConcreteBufferRemover() = default;

    static void deallocate_memory(int8_t * p, size_t alignment)
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

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,readability-non-const-parameter)
    virtual void operator()(int8_t * p, size_t alignment) const
    {
        deallocate_memory(p, alignment);
    }

}; /* end struct ConcreteBufferRemover */

struct ConcreteBufferNoRemove : public ConcreteBufferRemover
{

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,readability-non-const-parameter)
    void operator()(int8_t *, size_t) const override {}

}; /* end struct ConcreteBufferNoRemove */

struct ConcreteBufferDataDeleter
{

    using remover_type = ConcreteBufferRemover;

    ConcreteBufferDataDeleter(ConcreteBufferDataDeleter const &) = delete;
    ConcreteBufferDataDeleter & operator=(ConcreteBufferDataDeleter const &) = delete;

    ConcreteBufferDataDeleter() = default;
    ConcreteBufferDataDeleter(ConcreteBufferDataDeleter &&) = default;
    ConcreteBufferDataDeleter & operator=(ConcreteBufferDataDeleter &&) = default;
    ~ConcreteBufferDataDeleter() = default;
    explicit ConcreteBufferDataDeleter(std::unique_ptr<remover_type> && remover_in, size_t alignment_in = 0)
        : remover(std::move(remover_in))
        , alignment(alignment_in)
    {
    }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,readability-non-const-parameter)
    void operator()(int8_t * p) const
    {
        if (!remover)
        {
            remover_type::deallocate_memory(p, alignment);
        }
        else
        {
            (*remover)(p, alignment);
        }
    }

    std::unique_ptr<remover_type> remover{nullptr};
    size_t alignment = 0; // Alignment of the data buffer in bytes. 0 means no alignment.

}; /* end struct ConcreteBufferDataDeleter */

} /* end namespace detail */

/**
 * Untyped and unresizeable memory buffer for contiguous data storage.
 *
 * @ingroup group_core
 */
class ConcreteBuffer
    : public std::enable_shared_from_this<ConcreteBuffer>
    , public BufferBase<ConcreteBuffer>
{

private:

    struct ctor_passkey
    {
    }; /* end struct ctor_passkey */

    using data_deleter_type = detail::ConcreteBufferDataDeleter;

public:

    using remover_type = detail::ConcreteBufferRemover;
    using size_type = std::size_t;

    static std::shared_ptr<ConcreteBuffer> construct(size_t nbytes, size_t alignment = 0)
    {
        return std::make_shared<ConcreteBuffer>(nbytes, alignment, ctor_passkey());
    }

    /*
     * This factory method is dangerous since the data pointer passed in will
     * not be owned by the ConcreteBuffer created.  It is an error if the
     * number of bytes of the externally owned buffer doesn't match the value
     * passed in (but we cannot know here).
     */
    static std::shared_ptr<ConcreteBuffer> construct(size_t nbytes, int8_t * data, std::unique_ptr<remover_type> && remover, size_t alignment = 0)
    {
        return std::make_shared<ConcreteBuffer>(nbytes, data, std::move(remover), alignment, ctor_passkey());
    }

    static std::shared_ptr<ConcreteBuffer> construct(size_t nbytes, void * data, std::unique_ptr<remover_type> && remover, size_t alignment = 0)
    {
        return construct(nbytes, static_cast<int8_t *>(data), std::move(remover), alignment);
    }

    /// Construct an empty ConcreteBuffer with no data and no alignment.
    static std::shared_ptr<ConcreteBuffer> construct() { return construct(0, 0); }

    std::shared_ptr<ConcreteBuffer> clone() const
    {
        std::shared_ptr<ConcreteBuffer> ret = construct(nbytes(), m_alignment);
        std::copy_n(data(), size(), (*ret).data());
        return ret;
    }

    /**
     * @param[in] nbytes
     *      Size of the memory buffer in bytes.
     * @param[in] alignment
     *      Alignment for the memory buffer in bytes.
     *      0 means no alignment. Valid values are 0, 16, 32, or 64.
     */
    ConcreteBuffer(size_t nbytes, size_t alignment, const ctor_passkey &)
        : BufferBase<ConcreteBuffer>() // don't delegate m_begin and m_end, which will be overwritten later
        , m_nbytes(nbytes)
        , m_alignment(validate_alignment(alignment, "ConcreteBuffer::ConcreteBuffer"))
        , m_data(allocate(nbytes, m_alignment))
    {
        m_begin = m_data.get(); // overwrite m_begin and m_end once we have the data
        m_end = m_begin + m_nbytes;
    }

    /**
     * @param[in] nbytes
     *      Size of the memory buffer in bytes.
     * @param[in] data
     *      Pointer to the memory buffer that is not supposed to be owned by
     *      this ConcreteBuffer.
     * @param[in] remover
     *      The memory deallocator for the unowned data buffer passed in.
     * @param[in] alignment
     *      Alignment for the memory buffer in bytes.
     *      0 means no alignment. Valid values are 0, 16, 32, or 64.
     */
    // NOLINTNEXTLINE(readability-non-const-parameter)
    ConcreteBuffer(size_t nbytes, int8_t * data, std::unique_ptr<remover_type> && remover, size_t alignment, const ctor_passkey &)
        : BufferBase<ConcreteBuffer>() // don't delegate m_begin and m_end, which will be overwritten later
        , m_nbytes(nbytes)
        , m_alignment(validate_alignment(alignment, "ConcreteBuffer::ConcreteBuffer"))
        , m_data(data, data_deleter_type(std::move(remover), m_alignment))
    {
        m_begin = m_data.get(); // overwrite m_begin and m_end once we have the data
        m_end = m_begin + m_nbytes;
    }

    ~ConcreteBuffer() = default;

    ConcreteBuffer() = delete;
    ConcreteBuffer(ConcreteBuffer &&) = delete;
    ConcreteBuffer & operator=(ConcreteBuffer &&) = delete;
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#endif
    // Avoid enabled_shared_from_this copy constructor
    // NOLINTNEXTLINE(bugprone-copy-constructor-init)
    ConcreteBuffer(ConcreteBuffer const & other)
        : BufferBase<ConcreteBuffer>() // don't delegate m_begin and m_end, which will be overwritten later
        , m_nbytes(other.m_nbytes)
        , m_alignment(other.m_alignment)
        , m_data(allocate(other.m_nbytes, other.m_alignment))
    {
        m_begin = m_data.get(); // overwrite m_begin and m_end once we have the data
        m_end = m_begin + m_nbytes;
        if (size() != other.size())
        {
            throw std::out_of_range("Buffer size mismatch");
        }
        std::copy_n(other.data(), size(), data());
    }
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
    ConcreteBuffer & operator=(ConcreteBuffer const & other)
    {
        if (this != &other)
        {
            if (size() != other.size())
            {
                throw std::out_of_range("Buffer size mismatch");
            }
            std::copy_n(other.data(), size(), data());
        }
        return *this;
    }

    bool has_remover() const noexcept { return static_cast<bool>(m_data.get_deleter().remover); }
    remover_type const & get_remover() const { return *m_data.get_deleter().remover; }
    remover_type & get_remover() { return *m_data.get_deleter().remover; }

    size_type alignment() const noexcept { return m_alignment; }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
    using unique_ptr_type = std::unique_ptr<int8_t, data_deleter_type>;

    static constexpr const char * name() { return "ConcreteBuffer"; }

private:
    static unique_ptr_type allocate(size_t nbytes, size_t alignment)
    {
        unique_ptr_type ret(nullptr, data_deleter_type());
        if (0 != nbytes)
        {
            void * ptr = nullptr;
            if (alignment > 0)
            {
                validate_size_alignment(nbytes, alignment, "ConcreteBuffer::allocate");
#ifdef _WIN32
                ptr = _aligned_malloc(nbytes, alignment); // NOLINT(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
#else
                ptr = std::aligned_alloc(alignment, nbytes); // NOLINT(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
#endif
            }
            else
            {
                ptr = std::malloc(nbytes); // NOLINT(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
            }
            if (!ptr)
            {
                throw std::bad_alloc();
            }
            ret = unique_ptr_type(static_cast<int8_t *>(ptr), data_deleter_type(nullptr, alignment));
        }
        return ret;
    }

    size_t m_nbytes;
    size_t m_alignment = 0; // Alignment of the data buffer in bytes. 0 means no alignment.
    unique_ptr_type m_data;
}; /* end class ConcreteBuffer */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

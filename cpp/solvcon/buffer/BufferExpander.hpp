#pragma once

/*
 * Copyright (c) 2024, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/base.hpp>
#include <solvcon/buffer/BufferBase.hpp>
#include <solvcon/buffer/ConcreteBuffer.hpp>

#include <algorithm>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace solvcon
{

/**
 * Untyped and growing memory buffer for contiguous data storage.  The internal
 * expandable memory buffer cannot be used externally.
 */
class BufferExpander
    : public std::enable_shared_from_this<BufferExpander>
    , public BufferBase<BufferExpander>
{

private:

    struct ctor_passkey
    {
    }; /* end struct ctor_passkey */

public:

    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename... Args>
    static std::shared_ptr<BufferExpander> construct(Args &&... args)
    {
        return std::make_shared<BufferExpander>(std::forward<Args>(args)..., ctor_passkey());
    }

    std::shared_ptr<BufferExpander> clone()
    {
        return BufferExpander::construct(copy_concrete(), /*clone*/ false, m_alignment);
    }

    BufferExpander(std::shared_ptr<ConcreteBuffer> const & buf, bool clone, size_type alignment, ctor_passkey const &)
        : BufferBase<BufferExpander>()
        , m_concrete_buffer(clone ? buf->clone() : buf)
        , m_alignment(validate_alignment(alignment, "BufferExpander::BufferExpander"))
    {
        m_begin = m_concrete_buffer->data();
        m_end = m_begin + m_concrete_buffer->size();
        m_end_cap = m_begin + m_concrete_buffer->size(); // NOLINT(cppcoreguidelines-prefer-member-initializer)
    }

    BufferExpander(std::shared_ptr<ConcreteBuffer> const & buf, bool clone, ctor_passkey const &)
        : BufferExpander(buf, clone, 0, ctor_passkey())
    {
    }

    BufferExpander(size_type nbyte, size_type alignment, ctor_passkey const &)
        : BufferBase<BufferExpander>(nullptr, nullptr)
        , m_alignment(validate_alignment(alignment, "BufferExpander::BufferExpander"))
    {
        expand(nbyte);
    }

    BufferExpander(size_type nbyte, ctor_passkey const &)
        : BufferExpander(nbyte, 0, ctor_passkey())
    {
    }

    explicit BufferExpander(ctor_passkey const &)
        : BufferBase<BufferExpander>(nullptr, nullptr)
        , m_alignment(0) // NOLINT(cppcoreguidelines-use-default-member-init,modernize-use-default-member-init)
    {
    }

    BufferExpander() = delete;
    BufferExpander(BufferExpander const &) = delete;
    BufferExpander(BufferExpander &&) = delete;
    BufferExpander & operator=(BufferExpander const &) = delete;
    BufferExpander & operator=(BufferExpander &&) = delete;
    ~BufferExpander() = default;

    size_type capacity() const noexcept
    {
        if (!m_begin) // no data hence no capacity
        {
            return 0;
        }
        return static_cast<size_type>(this->m_end_cap - this->m_begin);
    }

    bool empty() const noexcept { return size() == 0; }

    void reserve(size_type cap);

    void expand(size_type length)
    {
        reserve(length);
        m_end = m_begin + length;
    }

    void clear() { m_end = m_begin; }

    /**
     * Push up the size by amount.
     * @param amount
     *  Number of bytes to push up.
     */
    void push_size(size_type amount)
    {
        if (size() + amount > capacity())
        {
            throw std::out_of_range(
                std::format("{}: size() {} + amount {} > capacity() {}",
                            name(),
                            size(),
                            amount,
                            capacity()));
        }
        m_end += amount;
    }

    /**
     * Pull the size down by amount, keeping the capacity intact.
     * @param amount Number of bytes to pull down.
     */
    void pop_size(size_type amount)
    {
        if (amount > size())
        {
            throw std::out_of_range(std::format("{}: amount {} > size() {}", name(), amount, size()));
        }
        m_end -= amount;
    }

    std::shared_ptr<ConcreteBuffer> copy_concrete(size_type cap = 0) const;
    std::shared_ptr<ConcreteBuffer> const & as_concrete(size_type cap = 0);
    bool is_concrete() const { return static_cast<bool>(m_concrete_buffer); }

    size_type alignment() const noexcept { return m_alignment; }

    static constexpr const char * name() { return "BufferExpander"; }

private:
    struct aligned_deleter
    {
        size_type alignment = 0;

        void operator()(int8_t * ptr) const
        {
            if (!ptr)
            {
                return;
            }
#ifdef _WIN32
            if (alignment > 0)
            {
                _aligned_free(ptr);
            }
            else
            {
                std::free(ptr);
            }
#else
            std::free(ptr); // NOLINT(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
#endif
        }
    }; /* end struct aligned_deleter */

    using unique_ptr_type = std::unique_ptr<int8_t, aligned_deleter>;

    static unique_ptr_type allocate(size_type nbytes, size_type alignment)
    {
        unique_ptr_type ret(nullptr, aligned_deleter{});
        if (0 != nbytes)
        {
            void * ptr = nullptr;
            if (alignment > 0)
            {
                validate_size_alignment(nbytes, alignment, "BufferExpander::allocate");
#ifdef _WIN32
                ptr = _aligned_malloc(nbytes, alignment);
#else
                ptr = std::aligned_alloc(alignment, nbytes); // NOLINT(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
#endif
            }
            else
            {
                // Use malloc instead of new[] so we can consistently use free in the deleter
                ptr = std::malloc(nbytes); // NOLINT(cppcoreguidelines-owning-memory,cppcoreguidelines-no-malloc)
            }
            if (!ptr)
            {
                throw std::bad_alloc();
            }
            ret = unique_ptr_type(static_cast<int8_t *>(ptr), aligned_deleter{alignment});
        }
        return ret;
    }

    unique_ptr_type m_data_holder{nullptr, aligned_deleter{}};
    std::shared_ptr<ConcreteBuffer> m_concrete_buffer = nullptr;
    size_type m_alignment = 0; // alignment for the buffer in bytes, 0 means no alignment

    int8_t * m_end_cap = nullptr;
}; /* end class BufferExpander */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
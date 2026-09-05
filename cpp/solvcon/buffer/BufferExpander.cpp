/*
 * Copyright (c) 2024, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/BufferExpander.hpp>

namespace solvcon
{

void BufferExpander::reserve(size_type cap)
{
    if (cap > capacity())
    {
        size_type const old_size = size();
        // Create new data holder and copy data.
        std::unique_ptr<int8_t, aligned_deleter> new_data_holder = allocate(cap, m_alignment);
        std::copy_n(m_begin, old_size, new_data_holder.get());
        // Process data holders.
        m_data_holder.swap(new_data_holder);
        if (m_concrete_buffer)
        {
            m_concrete_buffer.reset();
        }
        // Reset pointers.
        m_begin = m_data_holder.get();
        m_end = m_begin + old_size;
        m_end_cap = m_begin + cap;
    }
    else
    {
        return;
    }
}

std::shared_ptr<ConcreteBuffer> BufferExpander::copy_concrete(size_type cap) const
{
    size_type const old_size = size();
    size_type const csize = cap > old_size ? cap : old_size;
    auto buf = ConcreteBuffer::construct(csize, m_alignment);
    std::copy_n(m_begin, old_size, buf->data());
    return buf;
}

std::shared_ptr<ConcreteBuffer> const & BufferExpander::as_concrete(size_type cap)
{
    size_type const old_size = size();
    if (cap > 0 && m_alignment > 0)
    {
        validate_size_alignment(cap, m_alignment, "BufferExpander::as_concrete");
    }
    if (!m_concrete_buffer)
    {
        m_concrete_buffer = copy_concrete(cap);
        m_data_holder.reset();
    }
    m_concrete_buffer->export_host_access();
    m_begin = m_concrete_buffer->data();
    m_end = m_begin + old_size;
    m_end_cap = m_begin + m_concrete_buffer->size();
    return m_concrete_buffer;
}

} /* end namespace solvcon */

/* vim: set et ts=4 sw=4: */
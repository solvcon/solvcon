/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/ConcreteBuffer.hpp>

#ifdef SOLVCON_METAL
#include <solvcon/device/metal/metal.hpp>
#endif

namespace solvcon
{

std::shared_ptr<ConcreteBuffer> ConcreteBuffer::construct(size_t nbytes, size_t alignment, BufferDevice device)
{
    validate_alignment(alignment, "ConcreteBuffer::construct");
    validate_size_alignment(nbytes, alignment, "ConcreteBuffer::construct");

    switch (device)
    {
    case BufferDevice::Cpu:
        return construct(nbytes, alignment);
    case BufferDevice::Metal:
#ifdef SOLVCON_METAL
    {
        detail::DeviceBufferStorage device_storage = device::MetalManager::instance().allocate_buffer(nbytes, alignment);
        auto buffer = construct(nbytes, device_storage.m_data, std::move(device_storage.m_remover), alignment);
        buffer->m_access_state = std::make_unique<access_state_type>();
        return buffer;
    }
#else
        throw std::runtime_error("ConcreteBuffer::construct: Metal storage is unavailable");
#endif
    }
    throw std::invalid_argument("ConcreteBuffer::construct: unknown buffer device");
}

void ConcreteBuffer::wait() const
{
    if (m_access_state == nullptr)
    {
        return;
    }
    m_access_state->wait();
}

void ConcreteBuffer::export_host_access() const
{
    if (m_access_state == nullptr)
    {
        return;
    }
    m_access_state->export_host_access();
}

void ConcreteBuffer::copy_from(ConcreteBuffer const & other)
{
    auto const src_access = other.acquire_host_access();
    auto const dst_access = acquire_host_access();
    std::copy_n(other.data<int8_t>(), size(), data<int8_t>());
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

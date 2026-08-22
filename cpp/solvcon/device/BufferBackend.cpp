/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/ConcreteBuffer.hpp>
#include <solvcon/device/BufferBackend.hpp>

#include <format>
#include <stdexcept>

namespace solvcon
{

namespace device
{

namespace
{

class CpuBufferBackend final : public BufferBackend
{
public:
    BufferDevice device() const noexcept override { return BufferDevice::Cpu; }
    bool built() const noexcept override { return true; }
    bool available() const noexcept override { return true; }

    std::shared_ptr<ConcreteBuffer> allocate(size_t nbytes, size_t alignment) const override
    {
        return ConcreteBuffer::construct(nbytes, alignment);
    }
}; /* end class CpuBufferBackend */

class UnavailableBufferBackend final : public BufferBackend
{
public:
    explicit UnavailableBufferBackend(BufferDevice device)
        : m_device(device)
    {
    }

    BufferDevice device() const noexcept override { return m_device; }
    bool built() const noexcept override { return false; }
    bool available() const noexcept override { return false; }

    std::shared_ptr<ConcreteBuffer> allocate(size_t, size_t) const override;

private:
    BufferDevice m_device;
}; /* end class UnavailableBufferBackend */

std::shared_ptr<ConcreteBuffer> UnavailableBufferBackend::allocate(size_t, size_t) const
{
    throw std::runtime_error(std::format(
        "ConcreteBuffer: {} storage backend is unavailable",
        buffer_device_name(m_device)));
}

BufferBackend const & cpu_buffer_backend()
{
    static CpuBufferBackend backend;
    return backend;
}

BufferBackend const & unavailable_metal_backend()
{
    static UnavailableBufferBackend backend(BufferDevice::Metal);
    return backend;
}

} /* end namespace */

BufferBackend const & buffer_backend(BufferDevice device)
{
    switch (device)
    {
    case BufferDevice::Cpu:
        return cpu_buffer_backend();
    case BufferDevice::Metal:
        return unavailable_metal_backend();
    }
    throw std::invalid_argument("unknown buffer device");
}

std::shared_ptr<ConcreteBuffer> allocate_buffer(size_t nbytes, size_t alignment, BufferDevice device)
{
    return buffer_backend(device).allocate(nbytes, alignment);
}

} /* end namespace device */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#import <Metal/Metal.h>

#include <solvcon/buffer/BufferBase.hpp>
#include <solvcon/device/metal/metal.hpp>

#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

namespace solvcon
{

namespace device
{

static id<MTLDevice> select_unified_device()
{
    id<MTLDevice> const default_device = MTLCreateSystemDefaultDevice();
    if (default_device != nil && default_device.hasUnifiedMemory)
    {
        return default_device;
    }
    for (id<MTLDevice> device in MTLCopyAllDevices())
    {
        if (device.hasUnifiedMemory)
        {
            return device;
        }
    }
    return nil;
}

namespace
{

class MetalBufferRemover final : public detail::ConcreteBufferRemover
{

public:

    explicit MetalBufferRemover(id<MTLBuffer> buffer)
        : m_buffer(buffer)
    {
    }

    MetalBufferRemover(MetalBufferRemover const &) = delete;
    MetalBufferRemover(MetalBufferRemover &&) = delete;
    MetalBufferRemover & operator=(MetalBufferRemover const &) = delete;
    MetalBufferRemover & operator=(MetalBufferRemover &&) = delete;

    ~MetalBufferRemover() override
    {
        // Metal may autorelease related resources while releasing the buffer.
        @autoreleasepool
        {
            m_buffer = nil;
        }
    }

    void operator()(int8_t *, size_t) const override {}
    void reset(id<MTLBuffer> buffer) noexcept { m_buffer = buffer; }

private:

    id<MTLBuffer> m_buffer;

}; /* end class MetalBufferRemover */

} /* end namespace */

class MetalManager::Impl
{

public:

    Impl();
    bool started() const { return nil != m_device; }
    detail::DeviceBufferStorage allocate_buffer(size_t nbytes, size_t alignment) const;

private:

    id<MTLDevice> m_device = nil;

}; /* end class MetalManager::Impl */

MetalManager::Impl::Impl()
{
    @autoreleasepool
    {
        m_device = select_unified_device();
    }
}

detail::DeviceBufferStorage MetalManager::Impl::allocate_buffer(size_t nbytes, size_t alignment) const
{
    if (nbytes == 0)
    {
        return {.m_data = nullptr, .m_remover = std::make_unique<MetalBufferRemover>(nil)};
    }

    size_t const padding = alignment == 0 ? 0 : alignment - 1;
    size_t const capacity = nbytes + padding;

    // Metal calls may autorelease objects; leave each local pool before throwing.
    size_t max_buffer_length = 0;
    @autoreleasepool
    {
        max_buffer_length = m_device.maxBufferLength;
    }
    if (capacity > max_buffer_length)
    {
        throw std::length_error("MetalManager::allocate_buffer: Metal storage exceeds maxBufferLength");
    }

    auto remover = std::make_unique<MetalBufferRemover>(nil);
    id<MTLBuffer> buffer = nil;
    void * data = nullptr;
    @autoreleasepool
    {
        buffer = [m_device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
        remover->reset(buffer);
        data = buffer.contents;
    }
    if (buffer == nil || data == nullptr)
    {
        throw std::bad_alloc();
    }

    size_t space = capacity;
    if (alignment != 0 && std::align(alignment, nbytes, data, space) == nullptr)
    {
        throw std::bad_alloc();
    }
    return {
        .m_data = static_cast<int8_t *>(data),
        .m_remover = std::move(remover),
    };
}

MetalManager::MetalManager() { startup(); }

MetalManager::~MetalManager() = default;

MetalManager & MetalManager::instance()
{
    static MetalManager manager;
    return manager;
}

void MetalManager::startup()
{
    if (!m_impl)
    {
        m_impl = std::make_unique<Impl>();
        if (!m_impl->started())
        {
            m_impl.reset();
        }
    }
}

void MetalManager::shutdown() { m_impl.reset(); }

detail::DeviceBufferStorage MetalManager::allocate_buffer(std::size_t nbytes, std::size_t alignment)
{
    validate_alignment(alignment, "MetalManager::allocate_buffer");
    validate_size_alignment(nbytes, alignment, "MetalManager::allocate_buffer");

    if (!m_impl)
    {
        throw std::runtime_error("MetalManager::allocate_buffer: Metal storage is unavailable");
    }

    return m_impl->allocate_buffer(nbytes, alignment);
}

} /* end namespace device */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

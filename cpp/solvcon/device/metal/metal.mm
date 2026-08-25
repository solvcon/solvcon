/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#import <Metal/Metal.h>

#include <solvcon/device/metal/metal.hpp>

#include <memory>

namespace solvcon
{

namespace device
{

class MetalManager::Impl
{

public:

    Impl();
    bool started() const { return nil != m_device; }

private:

    id<MTLDevice> m_device = nil;

}; /* end class MetalManager::Impl */

MetalManager::Impl::Impl()
{
    @autoreleasepool
    {
        m_device = MTLCreateSystemDefaultDevice();
    }
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

} /* end namespace device */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

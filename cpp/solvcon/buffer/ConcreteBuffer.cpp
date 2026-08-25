/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/ConcreteBuffer.hpp>

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
        throw std::runtime_error("ConcreteBuffer::construct: Metal storage is unavailable");
    }
    throw std::invalid_argument("ConcreteBuffer::construct: unknown buffer device");
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

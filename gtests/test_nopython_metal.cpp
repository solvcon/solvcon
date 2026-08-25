#include <solvcon/buffer/ConcreteBuffer.hpp>
#include <solvcon/device/metal/metal.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

namespace solvcon
{

namespace device
{

TEST(MetalManager, repeated_lifecycle_calls_preserve_state)
{
    MetalManager & manager = MetalManager::instance();
    bool const available = manager.started();

    manager.startup();
    EXPECT_EQ(available, manager.started());

    manager.shutdown();
    EXPECT_FALSE(manager.started());
    manager.shutdown();
    EXPECT_FALSE(manager.started());
    EXPECT_THROW(ConcreteBuffer::construct(16, 0, BufferDevice::Metal), std::runtime_error);

    manager.startup();
    EXPECT_EQ(available, manager.started());
}

TEST(MetalManager, allocate_shared_buffer)
{
    MetalManager & manager = MetalManager::instance();
    if (!manager.started())
    {
        GTEST_SKIP() << "No unified-memory Metal device is available";
    }

    for (size_t const alignment : std::array<size_t, 4>{0, 16, 32, 64})
    {
        auto buffer = ConcreteBuffer::construct(64, alignment, BufferDevice::Metal);
        ASSERT_NE(nullptr, buffer->data());
        EXPECT_EQ(size_t{64}, buffer->size());
        EXPECT_EQ(alignment, buffer->alignment());
        EXPECT_TRUE(buffer->has_remover());
        if (alignment != 0)
        {
            auto const address = reinterpret_cast<std::uintptr_t>(buffer->data());
            EXPECT_EQ(std::uintptr_t{0}, address % alignment);
        }

        (*buffer)[0] = 12;
        (*buffer)[63] = 34;
        EXPECT_EQ(12, (*buffer)[0]);
        EXPECT_EQ(34, (*buffer)[63]);
    }
}

TEST(MetalManager, allocate_empty_shared_buffer)
{
    MetalManager & manager = MetalManager::instance();
    if (!manager.started())
    {
        GTEST_SKIP() << "No unified-memory Metal device is available";
    }

    auto buffer = ConcreteBuffer::construct(0, 64, BufferDevice::Metal);
    EXPECT_EQ(nullptr, buffer->data());
    EXPECT_EQ(size_t{0}, buffer->size());
    EXPECT_EQ(size_t{64}, buffer->alignment());
    EXPECT_TRUE(buffer->has_remover());
}

TEST(MetalManager, reject_oversized_shared_buffer)
{
    MetalManager & manager = MetalManager::instance();
    if (!manager.started())
    {
        GTEST_SKIP() << "No unified-memory Metal device is available";
    }

    EXPECT_THROW(
        ConcreteBuffer::construct(std::numeric_limits<size_t>::max(), 0, BufferDevice::Metal),
        std::length_error);
}

TEST(MetalManager, shared_buffer_owns_resource)
{
    MetalManager & manager = MetalManager::instance();
    if (!manager.started())
    {
        GTEST_SKIP() << "No unified-memory Metal device is available";
    }

    auto buffer = ConcreteBuffer::construct(16, 0, BufferDevice::Metal);
    (*buffer)[0] = 12;

    manager.shutdown();
    EXPECT_FALSE(manager.started());
    EXPECT_EQ(12, (*buffer)[0]);
    buffer.reset();

    manager.startup();
    EXPECT_TRUE(manager.started());
}

} /* end namespace device */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

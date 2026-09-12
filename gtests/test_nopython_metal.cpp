#include <solvcon/buffer/BufferExpander.hpp>
#include <solvcon/buffer/ConcreteBuffer.hpp>
#include <solvcon/buffer/SimpleArray.hpp>
#include <solvcon/device/metal/metal.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <semaphore>
#include <stdexcept>
#include <utility>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

namespace solvcon
{

namespace device
{

namespace
{

class PendingMatmulToken final : public detail::DeviceCompletionToken
{
public:
    bool ready() const override { return m_done.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }
    void wait() const override
    {
        m_started.release();
        m_done.wait();
    }
    bool wait_until_started() const { return m_started.try_acquire_for(std::chrono::seconds(5)); }
    void complete() { m_completion.set_value(); }

private:
    mutable std::counting_semaphore<> m_started{0};
    std::promise<void> m_completion;
    std::shared_future<void> m_done = m_completion.get_future().share();
}; /* end class PendingMatmulToken */

} /* end namespace */

TEST(MetalManager, cpu_matmul_waits_before_blas_and_naive)
{
    if (!MetalManager::instance().started())
    {
        GTEST_SKIP() << "No unified-memory Metal device is available";
    }
    for (auto kernel : {detail::MatmulKernel::Naive, detail::MatmulKernel::BlasGemm})
    {
        using array_type = SimpleArray<double>;
        using state_type = detail::BufferAccessState;
        auto lhs_buffer = ConcreteBuffer::construct(4 * sizeof(double), 0, BufferDevice::Metal);
        auto rhs_buffer = ConcreteBuffer::construct(4 * sizeof(double), 0, BufferDevice::Metal);
        auto output_buffer = ConcreteBuffer::construct(4 * sizeof(double), 0, BufferDevice::Metal);
        array_type lhs(array_type::shape_type{2, 2}, lhs_buffer);
        array_type rhs(array_type::shape_type{2, 2}, rhs_buffer);
        array_type output(array_type::shape_type{2, 2}, output_buffer);
        std::array<state_type *, 3> states{lhs_buffer->access_state(), rhs_buffer->access_state(), output_buffer->access_state()};
        // A distinct pending operand in each case catches a missing input or output lease.
        for (state_type * pending_state : states)
        {
            std::fill_n(lhs.logical_data(), 4, 0.0);
            std::fill_n(rhs.logical_data(), 4, 0.0);
            std::array<state_type *, 1> pending{pending_state};
            auto completion = std::make_shared<PendingMatmulToken>();
            state_type::Submission(std::span{pending}).publish(completion);
            auto compute = std::async(std::launch::async, [&]()
                                      {
                auto plan = detail::MatmulPlan::make(lhs, rhs);
                detail::MatmulExecutor<array_type> executor(std::move(plan), output, lhs, rhs);
                executor.execute(kernel); });
            if (!completion->wait_until_started())
            {
                completion->complete();
                compute.get();
                FAIL() << "CPU executor did not wait for pending device work";
            }
            EXPECT_THROW(state_type::Submission(std::span{pending}), std::runtime_error);
            // Simulate prior device writes while the CPU executor waits for their completion.
            std::fill_n(lhs.logical_data(), 4, 2.0);
            std::fill_n(rhs.logical_data(), 4, 3.0);
            completion->complete();
            compute.get();
            for (size_t index = 0; index < 4; ++index)
            {
                EXPECT_DOUBLE_EQ(12.0, output.logical_data()[index]);
            }
            EXPECT_NO_THROW({ state_type::Submission after_cpu(std::span{states}); });
        }
    }
}

TEST(MetalManager, unavailable_cpu_kernel_releases_host_access)
{
    if (!MetalManager::instance().started())
    {
        GTEST_SKIP() << "No unified-memory Metal device is available";
    }
    using array_type = SimpleArray<int32_t>;
    using state_type = detail::BufferAccessState;
    auto lhs_buffer = ConcreteBuffer::construct(4 * sizeof(int32_t), 0, BufferDevice::Metal);
    auto rhs_buffer = ConcreteBuffer::construct(4 * sizeof(int32_t), 0, BufferDevice::Metal);
    array_type lhs(array_type::shape_type{2, 2}, lhs_buffer);
    array_type rhs(array_type::shape_type{2, 2}, rhs_buffer);
    std::fill_n(lhs.logical_data(), 4, 2);
    std::fill_n(rhs.logical_data(), 4, 3);
    EXPECT_THROW(lhs.matmul(rhs, detail::MatmulKernel::BlasGemm), MatmulKernelUnavailable);
    auto output = lhs.matmul(rhs);
    for (size_t index = 0; index < 4; ++index)
    {
        EXPECT_EQ(12, output.logical_data()[index]);
    }
    std::array<state_type *, 2> states{lhs_buffer->access_state(), rhs_buffer->access_state()};
    EXPECT_NO_THROW({ state_type::Submission after_cpu(std::span{states}); });
}

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
        SimpleArray<int8_t> const array({8, 8}, {1, 8}, buffer);
        EXPECT_TRUE(array.to_row_major().is_c_contiguous());
        EXPECT_FALSE(buffer->access_state()->host_exported());
        ASSERT_NE(nullptr, buffer->data());
        EXPECT_FALSE(buffer->access_state()->host_exported());
        buffer->export_host_access();
        EXPECT_TRUE(buffer->access_state()->host_exported());
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

    auto buffer = ConcreteBuffer::construct(64, 0, BufferDevice::Metal);
    EXPECT_FALSE(buffer->access_state()->host_exported());
    auto expander = BufferExpander::construct(buffer, false);
    EXPECT_TRUE(buffer->access_state()->host_exported());
    EXPECT_EQ(buffer->data(), expander->data());
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

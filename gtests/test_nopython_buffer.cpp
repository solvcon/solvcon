#include <solvcon/buffer/buffer.hpp>

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <random>
#include <semaphore>
#include <type_traits>
#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif
#ifdef _MSC_VER
#pragma fenv_access(on)
#endif

namespace
{

using buffer_access_state_type = solvcon::detail::BufferAccessState;
using buffer_host_lease_type = buffer_access_state_type::HostLease;
using buffer_submission_type = buffer_access_state_type::Submission;

template <typename Evaluate, typename Expected>
void expect_rounding_result(int mode, Evaluate evaluate, Expected const & expected)
{
    std::fenv_t original_environment;
    ASSERT_EQ(0, std::fegetenv(&original_environment));

    int const mode_status = std::fesetround(mode);
    Expected const actual = evaluate();
    int const restore_status = std::fesetenv(&original_environment);

    EXPECT_EQ(0, mode_status);
    EXPECT_EQ(0, restore_status);
    EXPECT_EQ(expected, actual);
}

class BlockingDeviceCompletionToken final : public solvcon::detail::DeviceCompletionToken
{
public:
    explicit BlockingDeviceCompletionToken(buffer_access_state_type const & reentrant_state)
        : m_reentrant_state(reentrant_state)
    {
    }

    bool ready() const override { return m_done.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }
    void wait() const override
    {
        m_wait_started.release();
        static_cast<void>(m_reentrant_state.ready());
        m_done.wait();
    }
    bool wait_until_started() const { return m_wait_started.try_acquire_for(std::chrono::seconds(5)); }
    void complete() { m_done_promise.set_value(); }

private:
    buffer_access_state_type const & m_reentrant_state; ///< Borrowed state re-entered to catch waits under its mutex.
    mutable std::counting_semaphore<> m_wait_started{0};
    std::promise<void> m_done_promise; ///< Declared before m_done, which obtains its future.
    std::shared_future<void> m_done = m_done_promise.get_future().share();
}; /* end class BlockingDeviceCompletionToken */

class FailingDeviceCompletionToken final : public solvcon::detail::DeviceCompletionToken
{
public:
    bool ready() const override { return false; }
    void wait() const override { throw std::runtime_error("work failed"); }
}; /* end class FailingDeviceCompletionToken */

bool wait_started_or_complete(BlockingDeviceCompletionToken & completion)
{
    bool const started = completion.wait_until_started();
    if (!started)
    {
        completion.complete();
    }
    return started;
}

} /* end namespace */

TEST(ConcreteBuffer, construct_cpu_device)
{
    namespace sc = solvcon;

    auto buffer = sc::ConcreteBuffer::construct(32, 32, sc::BufferDevice::Cpu);
    EXPECT_EQ(size_t{32}, buffer->size());
    EXPECT_EQ(size_t{32}, buffer->alignment());
}

TEST(ConcreteBuffer, construct_empty)
{
    namespace sc = solvcon;

    auto buffer = sc::ConcreteBuffer::construct();
    EXPECT_EQ(nullptr, buffer->data());
    EXPECT_EQ(buffer->begin(), buffer->end());
    EXPECT_EQ(size_t{0}, buffer->size());
}

#ifndef SOLVCON_METAL
TEST(ConcreteBuffer, construct_unavailable_metal)
{
    namespace sc = solvcon;

    EXPECT_THROW(sc::ConcreteBuffer::construct(16, 0, sc::BufferDevice::Metal), std::runtime_error);
}
#endif

TEST(ConcreteBuffer, construct_invalid_metal_request)
{
    namespace sc = solvcon;

    EXPECT_THROW(sc::ConcreteBuffer::construct(15, 16, sc::BufferDevice::Metal), std::invalid_argument);
}

TEST(ConcreteBuffer, construct_unknown_device)
{
    namespace sc = solvcon;

    auto const unknown_device = static_cast<sc::BufferDevice>(255);
    EXPECT_THROW(sc::ConcreteBuffer::construct(16, 0, unknown_device), std::invalid_argument);
}

TEST(ConcreteBuffer, iterator)
{
    using namespace solvcon;

    auto buffer = ConcreteBuffer::construct(10);
    int8_t i = 0;
    for (auto & it : *buffer)
    {
        it = i++;
    }

    i = 0;
    for (const auto it : *buffer)
    {
        EXPECT_EQ(it, i++);
    }
}

TEST(BufferAccess, dependencies_and_wait_preserve_eligibility)
{
    buffer_access_state_type first, second;
    auto previous = std::make_shared<BlockingDeviceCompletionToken>(first);
    std::array<buffer_access_state_type *, 2> first_access{&first, &second};
    buffer_submission_type(std::span{first_access}).publish(previous);
    std::array<buffer_access_state_type *, 3> accesses{&second, &first, &first};
    auto first_wait = std::async(std::launch::async, &buffer_access_state_type::wait, &first);
    ASSERT_TRUE(wait_started_or_complete(*previous));
    buffer_submission_type submission(std::span{accesses});
    ASSERT_EQ(size_t{1}, submission.dependencies().size());
    EXPECT_EQ(previous, submission.dependencies().front());
    auto next = std::make_shared<BlockingDeviceCompletionToken>(first);
    submission.publish(next);
    previous->complete();
    first_wait.get();
    EXPECT_FALSE(first.ready());

    auto next_wait = std::async(std::launch::async, &buffer_access_state_type::wait, &first);
    ASSERT_TRUE(wait_started_or_complete(*next));
    next->complete();
    next_wait.get();
    second.wait();
    buffer_submission_type reused(std::span{accesses});
    EXPECT_THROW(reused.publish(previous), std::invalid_argument);
    auto final = std::make_shared<BlockingDeviceCompletionToken>(first);
    reused.publish(final);
    final->complete();
}

TEST(BufferAccess, scoped_host_access_blocks_submission)
{
    buffer_access_state_type access;
    std::array<buffer_access_state_type *, 1> accesses{&access};
    auto work = std::make_shared<BlockingDeviceCompletionToken>(access);
    buffer_submission_type(std::span{accesses}).publish(work);

    std::binary_semaphore release{0}, acquired{0};
    auto host = std::async(std::launch::async, [&]()
                           {
                               buffer_host_lease_type lease(&access);
                               acquired.release();
                               release.acquire(); });
    bool const started = wait_started_or_complete(*work);
    if (!started)
    {
        release.release();
    }
    ASSERT_TRUE(started);
    bool const acquired_early = acquired.try_acquire();
    EXPECT_FALSE(acquired_early);
    work->complete();
    EXPECT_TRUE(acquired_early || acquired.try_acquire_for(std::chrono::seconds(5)));

    EXPECT_THROW(buffer_submission_type(std::span{accesses}), std::runtime_error);
    release.release();
    host.get();
    EXPECT_NO_THROW({ buffer_submission_type canceled(std::span{accesses}); });
    EXPECT_NO_THROW({ buffer_submission_type after_cancel(std::span{accesses}); });
}

TEST(BufferAccess, raw_export_orders_with_submission)
{
    buffer_access_state_type access;
    std::array<buffer_access_state_type *, 1> accesses{&access};
    buffer_submission_type submission(std::span{accesses});
    auto exported = std::async(std::launch::async, &buffer_access_state_type::export_host_access, &access);
    auto work = std::make_shared<BlockingDeviceCompletionToken>(access);
    submission.publish(work);
    ASSERT_TRUE(wait_started_or_complete(*work));
    EXPECT_FALSE(access.host_exported());
    work->complete();
    exported.get();
    EXPECT_TRUE(access.host_exported());
    EXPECT_THROW(buffer_submission_type(std::span{accesses}), std::runtime_error);
}

TEST(BufferAccess, failed_wait_releases_host_access)
{
    buffer_access_state_type access;
    std::array<buffer_access_state_type *, 1> accesses{&access};
    buffer_submission_type(std::span{accesses}).publish(std::make_shared<FailingDeviceCompletionToken>());

    EXPECT_THROW(access.export_host_access(), std::runtime_error);
    EXPECT_FALSE(access.host_exported());
    EXPECT_NO_THROW({ buffer_submission_type next(std::span{accesses}); });
}

TEST(Float16, type_properties)
{
    namespace sc = solvcon;

    static_assert(sizeof(sc::Float16) == 2);
    static_assert(std::is_trivially_copyable_v<sc::Float16>);
    static_assert(std::is_convertible_v<float, sc::Float16>);
    static_assert(std::is_convertible_v<double, sc::Float16>);
    static_assert(std::is_convertible_v<int, sc::Float16>);
    static_assert(!std::is_convertible_v<sc::Float16, float>);
    static_assert(sc::Float16(1.0F).bits() == 0x3c00U);
}

TEST(Float16, arithmetic)
{
    namespace sc = solvcon;

    sc::Float16 const lhs(5.5F);
    sc::Float16 const rhs(2.0F);

    EXPECT_FLOAT_EQ(7.5F, static_cast<float>(lhs + rhs));
    EXPECT_FLOAT_EQ(3.5F, static_cast<float>(lhs - rhs));
    EXPECT_FLOAT_EQ(11.0F, static_cast<float>(lhs * rhs));
    EXPECT_FLOAT_EQ(2.75F, static_cast<float>(lhs / rhs));

    constexpr auto half_ulp = sc::Float16::from_bits(0x1000U);
    static_assert((sc::Float16(1.0F) + half_ulp).bits() == 0x3c00U);
    static_assert(sc::Float16(1.0F) < sc::Float16(2.0F));
}

TEST(Float16, arithmetic_rounding_modes)
{
    namespace sc = solvcon;

    using result_type = std::array<sc::Float16::storage_type, 3>;

    auto const evaluate = []()
    {
        volatile sc::Float16::storage_type one_bits = 0x3c00U;
        volatile sc::Float16::storage_type tiny_bits = 0x0001U;
        volatile sc::Float16::storage_type negative_one_bits = 0xbc00U;
        volatile sc::Float16::storage_type negative_tiny_bits = 0x8001U;
        auto const one = sc::Float16::from_bits(one_bits);
        auto const tiny = sc::Float16::from_bits(tiny_bits);
        auto const negative_one = sc::Float16::from_bits(negative_one_bits);
        auto const negative_tiny = sc::Float16::from_bits(negative_tiny_bits);
        return result_type{(one + tiny).bits(), (negative_one + negative_tiny).bits(), (one - one).bits()};
    };

    expect_rounding_result(FE_TONEAREST, evaluate, result_type{0x3c00U, 0xbc00U, 0x0000U});
    expect_rounding_result(FE_UPWARD, evaluate, result_type{0x3c01U, 0xbc00U, 0x0000U});
    expect_rounding_result(FE_DOWNWARD, evaluate, result_type{0x3c00U, 0xbc01U, 0x8000U});
    expect_rounding_result(FE_TOWARDZERO, evaluate, result_type{0x3c00U, 0xbc00U, 0x0000U});
}

TEST(Float16, compound_assignment)
{
    namespace sc = solvcon;

    sc::Float16 value(4.0F);

    EXPECT_EQ(&value, &(value += sc::Float16(2.0F)));
    EXPECT_FLOAT_EQ(6.0F, static_cast<float>(value));
    EXPECT_EQ(&value, &(value -= sc::Float16(1.0F)));
    EXPECT_FLOAT_EQ(5.0F, static_cast<float>(value));
    EXPECT_EQ(&value, &(value *= sc::Float16(3.0F)));
    EXPECT_FLOAT_EQ(15.0F, static_cast<float>(value));
    EXPECT_EQ(&value, &(value /= sc::Float16(2.0F)));
    EXPECT_FLOAT_EQ(7.5F, static_cast<float>(value));
}

TEST(Float16, comparison)
{
    namespace sc = solvcon;

    sc::Float16 const one(1.0F);
    sc::Float16 const two(2.0F);
    sc::Float16 const pos_zero(0.0F);
    sc::Float16 const neg_zero(-0.0F);
    sc::Float16 const pos_inf(std::numeric_limits<float>::infinity());
    sc::Float16 const neg_inf(-std::numeric_limits<float>::infinity());
    sc::Float16 const quiet_nan(std::numeric_limits<float>::quiet_NaN());

    EXPECT_LT(one, two);
    EXPECT_GT(two, one);
    EXPECT_EQ(pos_zero, neg_zero);
    EXPECT_EQ(std::partial_ordering::equivalent, pos_zero <=> neg_zero);
    EXPECT_LT(neg_inf, one);
    EXPECT_LT(one, pos_inf);
    EXPECT_NE(quiet_nan, quiet_nan);
    EXPECT_EQ(std::partial_ordering::unordered, quiet_nan <=> one);
}

TEST(Float16, binary_layout)
{
    namespace sc = solvcon;

    float const pos_input = 1.5F;
    auto const pos_bits = std::bit_cast<uint32_t>(pos_input);
    sc::Float16 const pos_output(pos_input);
    float const neg_input = -2.0F;
    auto const neg_bits = std::bit_cast<uint32_t>(neg_input);
    sc::Float16 const neg_output(neg_input);

    // Separators group the sign, exponent, and fraction fields.
    EXPECT_EQ(0b0'01111111'10000000000000000000000U, pos_bits);
    EXPECT_EQ(0b0'01111'1000000000U, pos_output.bits());
    EXPECT_FLOAT_EQ(pos_input, static_cast<float>(pos_output));
    EXPECT_EQ(0b1'10000000'00000000000000000000000U, neg_bits);
    EXPECT_EQ(0b1'10000'0000000000U, neg_output.bits());
    EXPECT_FLOAT_EQ(neg_input, static_cast<float>(neg_output));
}

TEST(Float16, explicit_conversion)
{
    namespace sc = solvcon;

    constexpr sc::Float16 value(1.5F);
    static_assert(static_cast<float>(value) == 1.5F);
    static_assert(static_cast<double>(value) == 1.5);
    static_assert(static_cast<int>(value) == 1);
    static_assert(static_cast<bool>(value));
    static_assert(!static_cast<bool>(sc::Float16(0.0F)));
    static_assert(!std::is_convertible_v<sc::Float16, float>);
    static_assert(!std::is_convertible_v<sc::Float16, int>);
}

TEST(Float16, boundaries)
{
    namespace sc = solvcon;

    float const max_finite = 65504.0F;
    float const min_subnormal = std::ldexp(1.0F, -24);
    float const underflow_tie = std::ldexp(1.0F, -25);

    EXPECT_EQ(0b0'11110'1111111111U, sc::Float16(max_finite).bits());
    EXPECT_EQ(0b0'00000'0000000001U, sc::Float16(min_subnormal).bits());
    EXPECT_EQ(0b0'00000'0000000000U, sc::Float16(underflow_tie).bits());

    EXPECT_FLOAT_EQ(max_finite, static_cast<float>(sc::Float16(max_finite)));
    EXPECT_FLOAT_EQ(min_subnormal, static_cast<float>(sc::Float16(min_subnormal)));
    EXPECT_FLOAT_EQ(0.0F, static_cast<float>(sc::Float16(underflow_tie)));
}

TEST(Float16, ties_to_even)
{
    namespace sc = solvcon;

    static_assert(sc::Float16(1.0F + 1.0F / 2048.0F).bits() == 0x3c00U);

    float const even_tie = 1.0F + std::ldexp(1.0F, -11);
    float const odd_tie = 1.0F + 3.0F * std::ldexp(1.0F, -11);

    // Midpoints select the result with an even least-significant fraction bit.
    EXPECT_EQ(0b0'01111'0000000000U, sc::Float16(even_tie).bits());
    EXPECT_EQ(0b0'01111'0000000010U, sc::Float16(odd_tie).bits());
}

TEST(Float16, rounding_modes)
{
    namespace sc = solvcon;

    using conversion_result_type = std::array<sc::Float16::storage_type, 4>;
    using mode_results_type = std::array<conversion_result_type, 3>;

    auto const evaluate_pair = [](float magnitude)
    {
        // Volatile prevents native casts from being folded before the runtime mode is set.
        volatile float positive_float = magnitude;
        volatile float negative_float = -magnitude;
        volatile double positive_double = magnitude;
        volatile double negative_double = -magnitude;
        return conversion_result_type{
            sc::Float16(positive_float).bits(),
            sc::Float16(negative_float).bits(),
            sc::Float16(positive_double).bits(),
            sc::Float16(negative_double).bits(),
        };
    };

    float const midpoint = 1.0F + std::ldexp(1.0F, -11);
    float const tiny = std::ldexp(1.0F, -26);
    float const overflow = 65536.0F;

    auto const evaluate = [&]()
    {
        return mode_results_type{
            evaluate_pair(midpoint),
            evaluate_pair(tiny),
            evaluate_pair(overflow),
        };
    };

    auto const expected_pair = [](sc::Float16::storage_type positive, sc::Float16::storage_type negative)
    {
        return conversion_result_type{positive, negative, positive, negative};
    };

    expect_rounding_result(FE_TONEAREST, evaluate, mode_results_type{
                                                       expected_pair(0x3c00U, 0xbc00U),
                                                       expected_pair(0x0000U, 0x8000U),
                                                       expected_pair(0x7c00U, 0xfc00U),
                                                   });
    expect_rounding_result(FE_UPWARD, evaluate, mode_results_type{
                                                    expected_pair(0x3c01U, 0xbc00U),
                                                    expected_pair(0x0001U, 0x8000U),
                                                    expected_pair(0x7c00U, 0xfbffU),
                                                });
    expect_rounding_result(FE_DOWNWARD, evaluate, mode_results_type{
                                                      expected_pair(0x3c00U, 0xbc01U),
                                                      expected_pair(0x0000U, 0x8001U),
                                                      expected_pair(0x7bffU, 0xfc00U),
                                                  });
    expect_rounding_result(FE_TOWARDZERO, evaluate, mode_results_type{
                                                        expected_pair(0x3c00U, 0xbc00U),
                                                        expected_pair(0x0000U, 0x8000U),
                                                        expected_pair(0x7bffU, 0xfbffU),
                                                    });
}

TEST(Float16, double_rounding)
{
    namespace sc = solvcon;

    double const input = 1.0 + std::ldexp(1.0, -11) + std::ldexp(1.0, -25);

    // Narrowing through float loses the term above the binary16 midpoint.
    EXPECT_EQ(0b0'01111'0000000001U, sc::Float16(input).bits());
    EXPECT_EQ(0b0'01111'0000000000U, sc::Float16(static_cast<float>(input)).bits());
}

TEST(Float16, value_vs_bits)
{
    namespace sc = solvcon;

    sc::Float16 const numeric(1);
    sc::Float16 const raw = sc::Float16::from_bits(1);

    EXPECT_EQ(0b0'01111'0000000000U, numeric.bits());
    EXPECT_EQ(0b0'00000'0000000001U, raw.bits());
    EXPECT_FLOAT_EQ(1.0F, static_cast<float>(numeric));
    EXPECT_FLOAT_EQ(std::ldexp(1.0F, -24), static_cast<float>(raw));
}

TEST(Float16, special_values)
{
    namespace sc = solvcon;

    sc::Float16 const pos_inf(std::numeric_limits<float>::infinity());
    sc::Float16 const neg_inf(-std::numeric_limits<float>::infinity());
    EXPECT_EQ(0x7c00U, pos_inf.bits());
    EXPECT_EQ(0xfc00U, neg_inf.bits());
    EXPECT_TRUE(std::isinf(static_cast<float>(pos_inf)));
    EXPECT_TRUE(std::signbit(static_cast<float>(neg_inf)));

    float const quiet_nan = std::bit_cast<float>(0x7fcaa000U);
    float const signaling_nan = std::bit_cast<float>(0x7f8aa000U);
    EXPECT_EQ(0x7e55U, sc::Float16(quiet_nan).bits());
    EXPECT_EQ(0x7e55U, sc::Float16(signaling_nan).bits());
    auto const decoded_quiet = static_cast<float>(sc::Float16::from_bits(0x7e55U));
    auto const decoded_signaling = static_cast<float>(sc::Float16::from_bits(0x7c55U));
    EXPECT_EQ(0x7fcaa000U, std::bit_cast<uint32_t>(decoded_quiet));
    EXPECT_EQ(0x7fcaa000U, std::bit_cast<uint32_t>(decoded_signaling));

    sc::Float16 const neg_zero(-0.0F);
    EXPECT_EQ(0x8000U, neg_zero.bits());
    EXPECT_TRUE(std::signbit(static_cast<float>(neg_zero)));
    EXPECT_EQ(0x7e55U, sc::Float16::from_bits(0x7e55U).bits());
}

TEST(SimpleArray, construction)
{
    namespace sc = solvcon;
    sc::SimpleArray<double> arr_double(10);
    EXPECT_EQ(arr_double.nbody(), 10);
    sc::SimpleArray<int> arr_int(17);
    EXPECT_EQ(arr_int.nbody(), 17);
}

TEST(SimpleArray, float16_storage)
{
    namespace sc = solvcon;

    sc::SimpleArrayFloat16 array(sc::small_vector<ssize_t>{2, 2});
    array(0, 0) = sc::Float16(1.5F);
    array(0, 1) = sc::Float16(-2.0F);
    array(1, 0) = sc::Float16(3.25F);
    array(1, 1) = sc::Float16(4.5F);

    sc::SimpleArrayFloat16 copy(array);

    EXPECT_NE(array.data(), copy.data());
    EXPECT_FLOAT_EQ(1.5F, static_cast<float>(copy(0, 0)));
    EXPECT_FLOAT_EQ(-2.0F, static_cast<float>(copy(0, 1)));
    EXPECT_FLOAT_EQ(3.25F, static_cast<float>(copy(1, 0)));
    EXPECT_FLOAT_EQ(4.5F, static_cast<float>(copy(1, 1)));
}

TEST(SimpleArray, minmaxsum)
{
    using namespace solvcon;

    SimpleArray<double> arr_double(small_vector<ssize_t>{10}, 0);
    EXPECT_EQ(arr_double.sum(), 0);
    EXPECT_EQ(arr_double.min(), 0);
    EXPECT_EQ(arr_double.max(), 0);
    arr_double.fill(3.14);
    EXPECT_EQ(arr_double.sum(), 3.14 * 10);
    EXPECT_EQ(arr_double.min(), 3.14);
    EXPECT_EQ(arr_double.max(), 3.14);
    arr_double(2) = -2.9;
    arr_double(4) = 12.7;
    EXPECT_EQ(arr_double.min(), -2.9);
    EXPECT_EQ(arr_double.max(), 12.7);

    SimpleArray<int> arr_int(small_vector<ssize_t>{3, 4}, -2);
    EXPECT_EQ(arr_int.sum(), -2 * 3 * 4);
    EXPECT_EQ(arr_int.min(), -2);
    EXPECT_EQ(arr_int.max(), -2);
    arr_int.fill(7);
    EXPECT_EQ(arr_int.sum(), 7 * 3 * 4);
    EXPECT_EQ(arr_int.min(), 7);
    EXPECT_EQ(arr_int.max(), 7);
    arr_int(1, 2) = -8;
    arr_int(2, 0) = 9;
    EXPECT_EQ(arr_int.min(), -8);
    EXPECT_EQ(arr_int.max(), 9);
}

TEST(SimpleArray, argminmax_axis_rejects_rank_zero_result)
{
    using namespace solvcon;

    SimpleArray<double> array(small_vector<ssize_t>{3}, 0.0);

    EXPECT_THROW(array.argmin(0), std::invalid_argument);
    EXPECT_THROW(array.argmin(-1), std::invalid_argument);
    EXPECT_THROW(array.argmax(0), std::invalid_argument);
    EXPECT_THROW(array.argmax(-1), std::invalid_argument);
}

TEST(SimpleArray, abs)
{
    using namespace solvcon;

    SimpleArray<double> arr(small_vector<ssize_t>{10}, -1.0);
    EXPECT_EQ(arr.sum(), -10.0);

    SimpleArray<double> brr = arr.abs();
    EXPECT_EQ(brr.sum(), 10.0);
}

TEST(SimpleArray, reshape_cross_type_preserves_logical_order)
{
    using namespace solvcon;

    SimpleArray<int64_t> array(small_vector<ssize_t>{2, 3}, 0);
    int64_t v = 0;
    for (auto & it : array)
    {
        it = v++;
    }

    array.transpose(false);
    auto result = array.reshape<uint64_t>(small_vector<ssize_t>{6});

    const uint64_t expected[6] = {0, 3, 1, 4, 2, 5};
    for (ssize_t i = 0; i < 6; ++i)
    {
        EXPECT_EQ(result(i), expected[i]);
    }
}

TEST(SimpleArray, reshape_rejects_mismatched_size)
{
    using namespace solvcon;

    SimpleArray<uint64_t> array(small_vector<ssize_t>{3}, 0);

    // A cross-type reshape to a different item size has no well-defined
    // element mapping and is rejected.
    EXPECT_THROW(array.reshape<uint32_t>(small_vector<ssize_t>{6}), std::runtime_error);

    // A same-type reshape must keep the element count.
    EXPECT_THROW(array.reshape(small_vector<ssize_t>{5}), std::runtime_error);
    EXPECT_THROW(array.reshape(small_vector<ssize_t>{}), std::runtime_error);
}

TEST(SimpleArray, iterator)
{
    using namespace solvcon;

    SimpleArray<double> arr(10);
    int8_t i = 0;
    for (auto & it : arr)
    {
        it = i++;
    }

    i = 0;
    for (const auto it : arr)
    {
        EXPECT_EQ(it, i++);
    }
}

TEST(SimpleArray, logical_data)
{
    using namespace solvcon;

    auto buffer = ConcreteBuffer::construct(6 * sizeof(double));
    double * const raw_data = buffer->data<double>();
    for (ssize_t it = 0; it < 6; ++it)
    {
        raw_data[it] = static_cast<double>(it);
    }

    SimpleArray<double> array(
        small_vector<ssize_t>{2, 3},
        small_vector<ssize_t>{-3, -1},
        buffer,
        5 * sizeof(double));
    EXPECT_EQ(raw_data, array.data());
    EXPECT_EQ(raw_data + 5, array.logical_data());
    EXPECT_EQ(5.0, array.at(small_vector<ssize_t>{0, 0}));
    EXPECT_EQ(0.0, array.at(small_vector<ssize_t>{1, 2}));

    SimpleArray<double> copy(array);
    EXPECT_NE(array.data(), copy.data());
    EXPECT_EQ(copy.data() + 5, copy.logical_data());
    EXPECT_EQ(5.0, copy.at(small_vector<ssize_t>{0, 0}));
    EXPECT_EQ(0.0, copy.at(small_vector<ssize_t>{1, 2}));

    auto shifted_buffer = ConcreteBuffer::construct(6 * sizeof(double));
    SimpleArray<double> shifted(
        small_vector<ssize_t>{4},
        shifted_buffer,
        2 * sizeof(double));
    SimpleArray<double> reshaped = shifted.reshape(
        small_vector<ssize_t>{2, 2});
    EXPECT_EQ(shifted.data(), reshaped.data());
    EXPECT_EQ(shifted.logical_data(), reshaped.logical_data());
}

TEST(SimpleArray_DataType, from_type)
{
    solvcon::DataType dt_half = solvcon::DataType::from<solvcon::Float16>();
    EXPECT_EQ(dt_half.type(), solvcon::DataType::Float16);

    solvcon::DataType dt_double = solvcon::DataType::from<double>();
    EXPECT_EQ(dt_double.type(), solvcon::DataType::Float64);

    solvcon::DataType dt_int = solvcon::DataType::from<int>();
    EXPECT_EQ(dt_int.type(), solvcon::DataType::Int32);
}

TEST(SimpleArray_DataType, from_string)
{
    solvcon::DataType dt_double = solvcon::DataType("float64");
    EXPECT_EQ(dt_double.type(), solvcon::DataType::Float64);

    solvcon::DataType dt_bool = solvcon::DataType("bool");
    EXPECT_EQ(dt_bool.type(), solvcon::DataType::Bool);

    solvcon::DataType dt_half = solvcon::DataType("float16");
    EXPECT_EQ(dt_half.type(), solvcon::DataType::Float16);

    EXPECT_THROW(solvcon::DataType("bool8"), std::invalid_argument); // bool8 does not exist
}

TEST(SimpleArrayPlex, float16_lifecycle)
{
    namespace sc = solvcon;

    sc::small_vector<ssize_t> const shape{16};
    sc::SimpleArrayPlex plex(shape, sc::DataType::Float16, 32);
    EXPECT_EQ(sc::DataType::Float16, plex.data_type());

    auto * array = static_cast<sc::SimpleArrayFloat16 *>(plex.mutable_instance_ptr());
    EXPECT_EQ(0U, reinterpret_cast<std::uintptr_t>(array->data()) % 32U);
    (*array)[0] = sc::Float16(2.0F);

    sc::SimpleArrayPlex copy(plex);
    auto const * copy_array = static_cast<sc::SimpleArrayFloat16 const *>(copy.instance_ptr());
    EXPECT_NE(array->data(), copy_array->data());
    EXPECT_FLOAT_EQ(2.0F, static_cast<float>((*copy_array)[0]));

    sc::SimpleArrayPlex assigned;
    assigned = plex;
    auto const * assigned_array = static_cast<sc::SimpleArrayFloat16 const *>(assigned.instance_ptr());
    EXPECT_NE(array->data(), assigned_array->data());
    EXPECT_FLOAT_EQ(2.0F, static_cast<float>((*assigned_array)[0]));

    auto buffer = sc::ConcreteBuffer::construct(shape[0] * sizeof(sc::Float16));
    sc::SimpleArrayPlex buffered(shape, buffer, sc::DataType::Float16);
    auto const * buffered_array = static_cast<sc::SimpleArrayFloat16 const *>(buffered.instance_ptr());
    EXPECT_EQ(buffer->data(), static_cast<void const *>(buffered_array->data()));
}

TEST(BufferExpander, iterator)
{
    using namespace solvcon;

    auto buffer = BufferExpander::construct(10);
    int8_t i = 0;
    for (auto & it : *buffer)
    {
        it = i++;
    }

    i = 0;
    for (const auto it : *buffer)
    {
        EXPECT_EQ(it, i++);
    }
}

TEST(BufferExpander, pop_size)
{
    using namespace solvcon;

    auto buffer = BufferExpander::construct(4);
    EXPECT_FALSE(buffer->empty());
    EXPECT_EQ(buffer->size(), 4);
    buffer->pop_size(3);
    EXPECT_EQ(buffer->size(), 1);
    // Capacity is left untouched when the size is pulled down.
    EXPECT_EQ(buffer->capacity(), 4);
    buffer->pop_size(1);
    EXPECT_TRUE(buffer->empty());
    EXPECT_THROW(buffer->pop_size(1), std::out_of_range);
}

TEST(SimpleCollector, pop_back)
{
    using namespace solvcon;

    SimpleCollector<int32_t> coll;
    EXPECT_TRUE(coll.empty());
    coll.push_back(10);
    coll.push_back(20);
    coll.push_back(30);
    EXPECT_EQ(coll.size(), 3);
    EXPECT_EQ(coll.front(), 10);
    EXPECT_EQ(coll.back(), 30);

    coll.pop_back();
    EXPECT_EQ(coll.size(), 2);
    EXPECT_EQ(coll.back(), 20);

    // pop_back keeps the surviving elements intact and reusable.
    coll.push_back(40);
    EXPECT_EQ(coll.front(), 10);
    EXPECT_EQ(coll.back(), 40);
    EXPECT_EQ(coll[0], 10);

    coll.pop_back();
    coll.pop_back();
    coll.pop_back();
    EXPECT_TRUE(coll.empty());
    EXPECT_THROW(coll.pop_back(), std::out_of_range);
}

TEST(small_vector, select_kth)
{
    const size_t n = 1024;
    std::vector<int> scrambled(n);
    // gcd(31, 1024) = 1,
    // so we can get all numbers from 0 to 1023.
    for (size_t i = 0; i < n; ++i)
    {
        scrambled[i] = static_cast<int>((i * 31) % n);
    }

    for (size_t k = 0; k < n; ++k)
    {
        solvcon::small_vector<int> sv(scrambled);
        int result = sv.select_kth(k);
        EXPECT_EQ(result, static_cast<int>(k));
    }
}

TEST(small_vector, select_kth_random)
{
    size_t n = 1024;
    std::vector<int> vec(n);
    std::iota(vec.begin(), vec.end(), 0);

    solvcon::small_vector<int> sv(vec);
    for (size_t i = 0; i < n; ++i)
    {
        auto rng = std::default_random_engine{};
        std::shuffle(sv.begin(), sv.end(), rng);
        int it = sv.select_kth(i);
        EXPECT_EQ(it, i);
    }
}

TEST(small_vector, equality_requires_same_size)
{
    solvcon::small_vector<int> prefix{1, 2};
    solvcon::small_vector<int> equal{1, 2};
    solvcon::small_vector<int> longer{1, 2, 3};

    EXPECT_TRUE(prefix == equal);
    EXPECT_FALSE(prefix == longer);
    EXPECT_FALSE(longer == prefix);
}

TEST(TakeAlongAxisSimd, basic_int32)
{
    using namespace solvcon;

    // Create a simple array with values [10, 20, 30, 40, 50]
    SimpleArray<int32_t> data(small_vector<ssize_t>{5});
    data[0] = 10;
    data[1] = 20;
    data[2] = 30;
    data[3] = 40;
    data[4] = 50;

    // Create indices [2, 0, 4, 1]
    SimpleArray<uint64_t> indices(small_vector<ssize_t>{4});
    indices[0] = 2;
    indices[1] = 0;
    indices[2] = 4;
    indices[3] = 1;

    // Call take_along_axis_simd
    SimpleArray<int32_t> result = data.take_along_axis_simd(indices);

    // Verify the result
    EXPECT_EQ(result.size(), 4);
    EXPECT_EQ(result[0], 30);
    EXPECT_EQ(result[1], 10);
    EXPECT_EQ(result[2], 50);
    EXPECT_EQ(result[3], 20);
}

TEST(TakeAlongAxisSimd, basic_float64)
{
    using namespace solvcon;

    // Create a simple array with float values
    SimpleArray<double> data(small_vector<ssize_t>{6});
    data[0] = 1.5;
    data[1] = 2.5;
    data[2] = 3.5;
    data[3] = 4.5;
    data[4] = 5.5;
    data[5] = 6.5;

    // Create indices [5, 2, 0, 3]
    SimpleArray<uint64_t> indices(small_vector<ssize_t>{4});
    indices[0] = 5;
    indices[1] = 2;
    indices[2] = 0;
    indices[3] = 3;

    // Call take_along_axis_simd
    SimpleArray<double> result = data.take_along_axis_simd(indices);

    // Verify the result
    EXPECT_EQ(result.size(), 4);
    EXPECT_DOUBLE_EQ(result[0], 6.5);
    EXPECT_DOUBLE_EQ(result[1], 3.5);
    EXPECT_DOUBLE_EQ(result[2], 1.5);
    EXPECT_DOUBLE_EQ(result[3], 4.5);
}

TEST(TakeAlongAxisSimd, large_array)
{
    using namespace solvcon;

    // Create a larger array
    const size_t data_size = 1000;
    SimpleArray<int64_t> data(small_vector<ssize_t>{static_cast<ssize_t>(data_size)});
    for (size_t i = 0; i < data_size; ++i)
    {
        data[i] = static_cast<int64_t>(i * 10);
    }

    // Create indices that sample from the array
    const size_t indices_size = 100;
    SimpleArray<uint64_t> indices(small_vector<ssize_t>{static_cast<ssize_t>(indices_size)});
    for (size_t i = 0; i < indices_size; ++i)
    {
        indices[i] = i * 10; // Sample every 10th element
    }

    // Call take_along_axis_simd
    SimpleArray<int64_t> result = data.take_along_axis_simd(indices);

    // Verify the result
    EXPECT_EQ(result.size(), indices_size);
    for (size_t i = 0; i < indices_size; ++i)
    {
        EXPECT_EQ(result[i], static_cast<int64_t>(i * 10 * 10));
    }
}

TEST(TakeAlongAxisSimd, out_of_range)
{
    using namespace solvcon;

    // Create a simple array
    SimpleArray<int32_t> data(small_vector<ssize_t>{5});
    data[0] = 10;
    data[1] = 20;
    data[2] = 30;
    data[3] = 40;
    data[4] = 50;

    // Create indices with out-of-range value
    SimpleArray<uint64_t> indices(small_vector<ssize_t>{3});
    indices[0] = 2;
    indices[1] = 10; // Out of range
    indices[2] = 1;

    // Should throw an exception
    EXPECT_THROW(data.take_along_axis_simd(indices), std::out_of_range);
}

TEST(TakeAlongAxisSimd, empty_indices)
{
    using namespace solvcon;

    // Create a simple array
    SimpleArray<int32_t> data(small_vector<ssize_t>{5});
    data[0] = 10;
    data[1] = 20;
    data[2] = 30;
    data[3] = 40;
    data[4] = 50;

    // Create empty indices
    SimpleArray<uint64_t> indices(small_vector<ssize_t>{0});

    // Call take_along_axis_simd
    SimpleArray<int32_t> result = data.take_along_axis_simd(indices);

    // Verify the result is empty
    EXPECT_EQ(result.size(), 0);
}

TEST(TakeAlongAxisSimd, sequential_indices)
{
    using namespace solvcon;

    // Create array [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    const size_t size = 10;
    SimpleArray<int32_t> data(small_vector<ssize_t>{static_cast<ssize_t>(size)});
    for (size_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<int32_t>(i);
    }

    // Create sequential indices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    SimpleArray<uint64_t> indices(small_vector<ssize_t>{static_cast<ssize_t>(size)});
    for (size_t i = 0; i < size; ++i)
    {
        indices[i] = i;
    }

    // Call take_along_axis_simd
    SimpleArray<int32_t> result = data.take_along_axis_simd(indices);

    // Result should be identical to input
    EXPECT_EQ(result.size(), size);
    for (size_t i = 0; i < size; ++i)
    {
        EXPECT_EQ(result[i], data[i]);
    }
}

TEST(TakeAlongAxisSimd, single_index_element)
{
    using namespace solvcon;

    // Create a data array with multiple elements
    SimpleArray<int32_t> data(small_vector<ssize_t>{10});
    for (size_t i = 0; i < 10; ++i)
    {
        data[i] = static_cast<int32_t>(i * 10);
    }

    // Create indices array with ONLY 1 ELEMENT (smaller than N_lane=2 on ARM NEON)
    // This should trigger the bug without the fix!
    SimpleArray<uint64_t> indices(small_vector<ssize_t>{1});
    indices[0] = 3;

    // Call take_along_axis_simd
    SimpleArray<int32_t> result = data.take_along_axis_simd(indices);

    // Verify the result
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], 30);
}
// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

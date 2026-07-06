#include <gtest/gtest.h>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

#include <solvcon/toggle/toggle.hpp>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

namespace solvcon
{

namespace detail
{

TEST(ToggleRefTest, declare_and_load)
{
    DynamicToggleTable table;
    ToggleRef<int32_t> const r = table.declare<int32_t>("a.n", 42);
    EXPECT_TRUE(r.valid());
    EXPECT_TRUE(static_cast<bool>(r));
    EXPECT_EQ(r.load(), 42);

    r.store(7);
    EXPECT_EQ(r.load(), 7);
    EXPECT_EQ(table.get<int32_t>("a.n", 0), 7);
}

TEST(ToggleRefTest, ref_resolves_existing_only)
{
    DynamicToggleTable table;
    table.declare<bool>("flag", true);

    ToggleRef<bool> const good = table.ref<bool>("flag");
    EXPECT_TRUE(good.valid());
    EXPECT_TRUE(good.load());

    // Missing key resolves to an invalid handle.
    ToggleRef<bool> const missing = table.ref<bool>("nope");
    EXPECT_FALSE(missing.valid());
    EXPECT_THROW(missing.load(), std::out_of_range);

    // Wrong type resolves to an invalid handle.
    ToggleRef<int32_t> const wrong = table.ref<int32_t>("flag");
    EXPECT_FALSE(wrong.valid());
}

TEST(ToggleRefTest, get_default_and_at_throw)
{
    DynamicToggleTable table;
    table.declare<double>("r", 2.5);

    EXPECT_DOUBLE_EQ(table.get<double>("r", 9.0), 2.5);
    EXPECT_DOUBLE_EQ(table.get<double>("missing", 9.0), 9.0);
    // Wrong-typed read falls back to the default.
    EXPECT_EQ(table.get<int32_t>("r", 9), 9);

    EXPECT_DOUBLE_EQ(table.at<double>("r"), 2.5);
    EXPECT_THROW(table.at<double>("missing"), std::out_of_range);
    EXPECT_THROW(table.at<int32_t>("r"), std::out_of_range);
}

TEST(ToggleRefTest, handle_survives_unrelated_growth)
{
    DynamicToggleTable table;
    ToggleRef<int32_t> const r = table.declare<int32_t>("first", 111);

    for (int i = 0; i < 10000; ++i)
    {
        table.declare<int32_t>("k" + std::to_string(i), i);
    }

    // The handle stays valid and reads its own value after the column has
    // grown well past any single chunk.
    EXPECT_TRUE(r.valid());
    EXPECT_EQ(r.load(), 111);
}

TEST(ToggleRefTest, handle_refused_after_clear)
{
    DynamicToggleTable table;
    ToggleRef<bool> const r = table.declare<bool>("flag", true);
    EXPECT_TRUE(r.valid());

    table.clear();

    EXPECT_FALSE(r.valid());
    EXPECT_THROW(r.load(), std::out_of_range);

    // A fresh handle after the clear works even at the same index.
    ToggleRef<bool> const r2 = table.declare<bool>("flag", false);
    EXPECT_TRUE(r2.valid());
    EXPECT_FALSE(r2.load());
}

TEST(ToggleRefTest, declare_idempotent_and_type_conflict)
{
    DynamicToggleTable table;
    ToggleRef<int32_t> const r1 = table.declare<int32_t>("n", 1);
    // Re-declaring the same type returns a handle to the same register and
    // does not overwrite the stored value.
    ToggleRef<int32_t> const r2 = table.declare<int32_t>("n", 99);
    EXPECT_EQ(r1.index(), r2.index());
    EXPECT_EQ(r2.load(), 1);

    // A conflicting type is an error.
    EXPECT_THROW(table.declare<bool>("n", true), std::invalid_argument);
}

TEST(ToggleRefTest, concurrent_readers_and_writers)
{
    DynamicToggleTable table;
    ToggleRef<int64_t> const r = table.declare<int64_t>("counter", 0);

    constexpr int readers = 4;
    constexpr int writers = 4;
    constexpr int iters = 20000;
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;

    // Readers load through the handle and the typed getter; an atomic
    // register is never torn, so every read is a value some writer stored.
    for (int i = 0; i < readers; ++i)
    {
        threads.emplace_back(
            [&]
            {
                while (!start.load())
                {
                }
                int64_t sink = 0;
                for (int k = 0; k < iters; ++k)
                {
                    sink += r.load();
                    sink += table.get<int64_t>("counter", -1);
                }
                (void)sink;
            });
    }
    // Writers store through the handle and through the table set path.
    for (int w = 0; w < writers; ++w)
    {
        threads.emplace_back(
            [&, w]
            {
                while (!start.load())
                {
                }
                for (int k = 0; k < iters; ++k)
                {
                    r.store(static_cast<int64_t>(k));
                    table.set_int64("counter", static_cast<int64_t>(w));
                }
            });
    }

    start.store(true);
    for (auto & t : threads)
    {
        t.join();
    }

    int64_t const final_value = r.load();
    EXPECT_GE(final_value, 0);
    EXPECT_LT(final_value, iters);
    EXPECT_TRUE(r.valid());
}

TEST(ToggleRefTest, concurrent_declare_and_read)
{
    DynamicToggleTable table;
    ToggleRef<int32_t> const base = table.declare<int32_t>("base", 7);

    std::atomic<bool> start{false};
    std::atomic<bool> reads_ok{true};
    std::vector<std::thread> threads;

    // Writers declare distinct new keys, growing the map and columns under
    // the table mutex.
    for (int w = 0; w < 4; ++w)
    {
        threads.emplace_back(
            [&, w]
            {
                while (!start.load())
                {
                }
                for (int k = 0; k < 2000; ++k)
                {
                    table.declare<int32_t>("w" + std::to_string(w) + "_" + std::to_string(k), k);
                }
            });
    }
    // A reader hammers the base handle while the table grows; its register is
    // heap-stable, so the lock-free load stays valid and correct.
    threads.emplace_back(
        [&]
        {
            while (!start.load())
            {
            }
            for (int k = 0; k < 40000; ++k)
            {
                if (base.load() != 7)
                {
                    reads_ok.store(false);
                }
            }
        });

    start.store(true);
    for (auto & t : threads)
    {
        t.join();
    }

    EXPECT_TRUE(reads_ok.load());
    EXPECT_EQ(base.load(), 7);
    EXPECT_TRUE(base.valid());
}

} /* end namespace detail */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

#include <gtest/gtest.h>
#include <thread>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

#define SOLVCON_PROFILE 1
#include <solvcon/profiling/RadixTree.hpp>
namespace solvcon
{

namespace detail
{
class CallProfilerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CallProfiler & profiler = CallProfiler::instance();
        pProfiler = &profiler;
    }

    RadixTree<CallerProfile> & radix_tree()
    {
        return pProfiler->m_radix_tree;
    }

    CallProfiler * pProfiler;
};

constexpr int uniqueTime1 = 19;
constexpr int uniqueTime2 = 35;
constexpr int uniqueTime3 = 7;

void foo3()
{
    SOLVCON_PROFILE_FUNCTION();
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() < uniqueTime1)
    {
        // use busy loop to get a precise duration
    }
}

void foo2()
{
    SOLVCON_PROFILE_FUNCTION();
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() < uniqueTime2)
    {
        // use busy loop to get a precise duration
    }
    foo3();
}

void foo1()
{
    SOLVCON_PROFILE_FUNCTION();
    foo2();
    auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() < uniqueTime3)
    {
        // use busy loop to get a precise duration
    }
}

TEST_F(CallProfilerTest, test_print_result)
{
    pProfiler->reset();

    foo1();

    std::stringstream ss;
    pProfiler->print_profiling_result(ss);
}

#ifdef _MSC_VER
auto foo1Name = "void __cdecl solvcon::detail::foo1(void)";
#else
auto foo1Name = "void solvcon::detail::foo1()";
#endif

#ifdef _MSC_VER
auto foo2Name = "void __cdecl solvcon::detail::foo2(void)";
#else
auto foo2Name = "void solvcon::detail::foo2()";
#endif

#ifdef _MSC_VER
auto foo3Name = "void __cdecl solvcon::detail::foo3(void)";
#else
auto foo3Name = "void solvcon::detail::foo3()";
#endif

TEST_F(CallProfilerTest, test_simple_case1)
{
    pProfiler->reset();

    foo1();

    // Example:
    // void solvcon::foo1() - Total Time: 61 ms, Call Count: 1
    //   void solvcon::foo2() - Total Time: 54 ms, Call Count: 1
    //      void solvcon::foo3() - Total Time: 19 ms, Call Count: 1

    int key = 0;

    auto * node1 = radix_tree().get_current_node()->get_child(key++);
    EXPECT_EQ(node1->data().caller_name, foo1Name);
    EXPECT_EQ(node1->data().call_count, 1);
    EXPECT_GE(node1->data().total_time.count() / 1e-6, uniqueTime1 + uniqueTime2 + uniqueTime3);

    auto * node2 = node1->get_child(key++);
    EXPECT_EQ(node2->data().caller_name, foo2Name);
    EXPECT_EQ(node2->data().call_count, 1);
    EXPECT_GE(node2->data().total_time.count() / 1e-6, uniqueTime1 + uniqueTime2);

    auto * node3 = node2->get_child(key++);
    EXPECT_EQ(node3->data().caller_name, foo3Name);
    EXPECT_EQ(node3->data().call_count, 1);
    EXPECT_GE(node3->data().total_time.count() / 1e-6, uniqueTime1);
}

TEST_F(CallProfilerTest, simple_case_2)
{
    pProfiler->reset();

    foo1();
    foo2();
    foo3();
    foo3();

    // Example:
    // void solvcon::foo1 - Total Time: 61 ms, Call Count: 1
    //   void solvcon::foo2 - Total Time: 54 ms, Call Count: 1
    //     void solvcon::foo3 - Total Time: 19 ms, Call Count: 1
    // void solvcon::foo2 - Total Time: 54 ms, Call Count: 1
    //   void solvcon::foo3 - Total Time: 19 ms, Call Count: 1
    // void solvcon::foo3 - Total Time: 38 ms, Call Count: 2

    // for first `foo1()` call
    {
        auto * node1 = radix_tree().get_current_node()->get_child(foo1Name);
        EXPECT_NE(node1, nullptr);
        EXPECT_EQ(node1->data().caller_name, foo1Name);
        EXPECT_EQ(node1->data().call_count, 1);
        EXPECT_GE(node1->data().total_time.count() / 1e-6, uniqueTime1 + uniqueTime2 + uniqueTime3);

        auto * node2 = node1->get_child(foo2Name);
        EXPECT_NE(node2, nullptr);
        EXPECT_EQ(node2->data().caller_name, foo2Name);
        EXPECT_EQ(node2->data().call_count, 1);
        EXPECT_GE(node2->data().total_time.count() / 1e-6, uniqueTime1 + uniqueTime2);

        auto * node3 = node2->get_child(foo3Name);
        EXPECT_NE(node3, nullptr);
        EXPECT_EQ(node3->data().caller_name, foo3Name);
        EXPECT_EQ(node3->data().call_count, 1);
        EXPECT_GE(node3->data().total_time.count() / 1e-6, uniqueTime1);
    }

    // for  `foo2()` call
    {
        auto * node1 = radix_tree().get_current_node()->get_child(foo2Name); // id = 1, because previously already assigned in the map, FIXME: probably find a better way than hard
        EXPECT_NE(node1, nullptr);
        EXPECT_EQ(node1->data().caller_name, foo2Name);
        EXPECT_EQ(node1->data().call_count, 1);
        EXPECT_GE(node1->data().total_time.count() / 1e-6, uniqueTime1 + uniqueTime2);

        auto * node2 = node1->get_child(foo3Name);
        EXPECT_NE(node2, nullptr);
        EXPECT_EQ(node2->data().caller_name, foo3Name);
        EXPECT_EQ(node2->data().call_count, 1);
        EXPECT_GE(node2->data().total_time.count() / 1e-6, uniqueTime1);
    }

    // for  two `foo3()` call
    {
        auto * node1 = radix_tree().get_current_node()->get_child(foo3Name);
        EXPECT_NE(node1, nullptr);
        EXPECT_EQ(node1->data().caller_name, foo3Name);
        EXPECT_EQ(node1->data().call_count, 2);
        EXPECT_GE(node1->data().total_time.count() / 1e-6, uniqueTime1 * 2);
    }
}

TEST_F(CallProfilerTest, cancel)
{
    pProfiler->reset();

    auto test1 = [&]()
    {
        SOLVCON_PROFILE_FUNCTION();

        auto test2 = [&]()
        {
            SOLVCON_PROFILE_FUNCTION();
            pProfiler->cancel();
        };

        test2();
    };
    test1();

    EXPECT_EQ(radix_tree().get_unique_node(), 0);
}

} // namespace detail
} // namespace solvcon

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

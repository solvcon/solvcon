#include <gtest/gtest.h>
#include <solvcon/profiling/RadixTree.hpp>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

/**
 * Simple Timed Entry for Testing
 */
class TestedTimedEntry
{
public:
    size_t count() const { return m_count; }
    double time() const { return m_time; }

    TestedTimedEntry & add_time(double time)
    {
        ++m_count;
        m_time += time;
        return *this;
    }

    friend std::ostream & operator<<(std::ostream & os, const TestedTimedEntry & entry)
    {
        os << "Count: " << entry.count() << " - Time: " << entry.time();
        return os;
    }

private:
    size_t m_count = 0;
    double m_time = 0.0;
}; /* end class TestedTimedEntry */

template <typename T>
std::string get_info(solvcon::RadixTreeNode<T> & r)
{
    std::ostringstream oss;
    if (nullptr != r.get_prev())
    {
        oss << r.name() << " - " << r.data();
    }
    return oss.str();
}

TEST(RadixTree, construction)
{
    namespace mm = solvcon;
    mm::RadixTree<TestedTimedEntry> radix_tree;
    EXPECT_EQ(radix_tree.get_unique_node(), 0);
}

TEST(RadixTree, single_insertion)
{
    namespace mm = solvcon;
    mm::RadixTree<TestedTimedEntry> radix_tree;
    TestedTimedEntry & entry1 = radix_tree.entry("a");
    entry1.add_time(5.2);

    EXPECT_EQ(entry1.count(), 1);
    EXPECT_DOUBLE_EQ(entry1.time(), 5.2);
    EXPECT_EQ(radix_tree.get_unique_node(), 1);

    mm::RadixTreeNode<TestedTimedEntry> * node = radix_tree.get_current_node();
    EXPECT_EQ(node->name(), "a");
    EXPECT_EQ(get_info(*node), "a - Count: 1 - Time: 5.2");
}

TEST(RadixTree, multiple_insertion)
{
    namespace mm = solvcon;
    mm::RadixTree<TestedTimedEntry> radix_tree;
    TestedTimedEntry & entry1 = radix_tree.entry("a");
    entry1.add_time(5.2);
    const mm::RadixTreeNode<TestedTimedEntry> * const node1 = radix_tree.get_current_node();

    TestedTimedEntry & entry2 = radix_tree.entry("b");
    entry2.add_time(10.1);

    EXPECT_EQ(radix_tree.get_unique_node(), 2);

    const mm::RadixTreeNode<TestedTimedEntry> * const node2 = radix_tree.get_current_node();
    EXPECT_EQ(node2->get_prev(), node1);
}

TEST(RadixTree, move_current_pointer)
{
    namespace mm = solvcon;
    mm::RadixTree<TestedTimedEntry> radix_tree;
    TestedTimedEntry & entry1 = radix_tree.entry("a");
    entry1.add_time(5.2);
    const mm::RadixTreeNode<TestedTimedEntry> * const node1 = radix_tree.get_current_node();

    TestedTimedEntry & entry2 = radix_tree.entry("b");
    entry2.add_time(10.1);

    radix_tree.move_current_to_parent();
    const mm::RadixTreeNode<TestedTimedEntry> * const node2 = radix_tree.get_current_node();

    EXPECT_EQ(radix_tree.get_unique_node(), 2);
    EXPECT_EQ(node2->name(), "a");
    EXPECT_EQ(node2, node1);
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

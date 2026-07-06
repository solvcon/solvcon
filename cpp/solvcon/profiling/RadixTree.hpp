#pragma once

/*
 * Copyright (c) 2023, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Radix tree container and the hierarchical call profiler built on it.
 *
 * @ingroup group_core
 */

#include <algorithm>
#include <chrono>
#include <vector>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <sstream>
#include <stack>
#include <unordered_map>
#include <queue>
#include <cstdint>
#include <set>

namespace solvcon
{
/**
 * A node in a RadixTree.
 *
 * Each node owns a signed integer key, a name, a value of type T,
 * a list of owned child nodes, and a back pointer to its parent.
 *
 * @ingroup group_core
 */
template <typename T>
class RadixTreeNode
{
public:

    using child_list_type = std::list<std::unique_ptr<RadixTreeNode<T>>>;
    using key_type = int32_t;

    static_assert(std::is_integral_v<key_type> && std::is_signed_v<key_type>, "signed integral required");

    RadixTreeNode(std::string name, key_type key)
        : m_key(key)
        , m_name(std::move(name))
        , m_prev(nullptr)
    {
    }

    RadixTreeNode() = default;
    RadixTreeNode(RadixTreeNode const &) = default;
    RadixTreeNode(RadixTreeNode &&) = default;
    RadixTreeNode & operator=(RadixTreeNode const &) = default;
    RadixTreeNode & operator=(RadixTreeNode &&) = default;
    ~RadixTreeNode() = default;

    key_type key() const { return m_key; }
    const std::string & name() const { return m_name; }
    T & data() { return m_data; }
    const T & data() const { return m_data; }
    const child_list_type & children() const { return m_children; }

    bool empty_children() const
    {
        return m_children.empty();
    }

    RadixTreeNode<T> * add_child(std::string const & name, key_type key)
    {
        auto new_child = std::make_unique<RadixTreeNode<T>>(name, key);
        new_child->m_prev = this;
        m_children.push_back(std::move(new_child));
        return m_children.back().get();
    }

    RadixTreeNode<T> * get_child(key_type key) const
    {
        auto it = std::find_if(m_children.begin(), m_children.end(), [&](const auto & child)
                               { return child->key() == key; });
        return (it != m_children.end()) ? it->get() : nullptr;
    }

    RadixTreeNode<T> * get_child(std::string name) const
    {
        auto it = std::find_if(m_children.begin(), m_children.end(), [&](const auto & child)
                               { return child->name() == name; });
        return (it != m_children.end()) ? it->get() : nullptr;
    }

    RadixTreeNode<T> * get_prev() const { return m_prev; }

private:
    key_type m_key = -1;
    std::string m_name;
    T m_data;
    child_list_type m_children;
    RadixTreeNode<T> * m_prev = nullptr;
}; /* end class RadixTreeNode */

/*
Ref:
https://kalkicode.com/radix-tree-implementation
https://www.algotree.org/algorithms/trie/
*/

namespace detail
{
class CallProfilerTest; // for gtest
} /* end namespace detail */

class SerializableRadixTree; // forward declaration
/**
 * A radix tree that maps names to stable integer ids.
 *
 * Each distinct name receives a unique signed integer id, and a current
 * node pointer walks down the tree as named entries are visited. A
 * stable snapshot of the id map supports re-entrant-safe reads.
 *
 * @ingroup group_core
 */
template <typename T>
class RadixTree
{
public:
    using key_type = typename RadixTreeNode<T>::key_type;
    RadixTree()
        : m_root(std::make_unique<RadixTreeNode<T>>())
        , m_current_node(m_root.get())
    {
    }

    T & entry(const std::string & name)
    {
        key_type const id = get_id(name);

        RadixTreeNode<T> * child = m_current_node->get_child(id);

        if (!child)
        {
            m_current_node = m_current_node->add_child(name, id);
        }
        else
        {
            m_current_node = child;
        }

        return m_current_node->data();
    }

    void move_current_to_parent()
    {
        if (m_current_node != m_root.get())
        {
            m_current_node = m_current_node->get_prev();
        }
    }

    void reset()
    {
        m_root = std::move(std::make_unique<RadixTreeNode<T>>());
        m_current_node = m_root.get();
        m_id_map.clear();
        m_stable_id_map.clear();
        m_unique_id = 0;
        m_stable_unique_id = 0;
    }

    bool is_root() const
    {
        return m_current_node == m_root.get();
    }

    void update_stable_items()
    {
        m_stable_id_map = m_id_map;
        m_stable_unique_id = m_unique_id;
    }

    RadixTreeNode<T> * get_root() const { return m_root.get(); }
    RadixTreeNode<T> * get_current_node() const { return m_current_node; }
    key_type get_unique_node() const { return m_unique_id; }
    key_type get_stable_unique_node() const { return m_stable_unique_id; }
    const std::unordered_map<std::string, key_type> & get_stable_id_map() const { return m_stable_id_map; }
    const std::unordered_map<std::string, key_type> & get_id_map() const { return m_id_map; }

private:
    key_type get_id(const std::string & name)
    {
        auto [it, inserted] = m_id_map.try_emplace(name, m_unique_id++);
        return it->second;
    }

    // FIXME: Use raw pointer for m_root.
    std::unique_ptr<RadixTreeNode<T>> m_root;
    RadixTreeNode<T> * m_current_node;
    std::unordered_map<std::string, key_type> m_id_map;
    std::unordered_map<std::string, key_type> m_stable_id_map; // Re-entrant safe
    key_type m_unique_id = 0;
    key_type m_stable_unique_id = 0; // Re-entrant safe
}; /* end class RadixTree */

/**
 * The profiling result of one caller.
 *
 * It records the stopwatch start time, the total elapsed time in
 * nanoseconds, the call count, and stable snapshots of the total time
 * and call count for re-entrant-safe reads.
 *
 * @ingroup group_core
 */
struct CallerProfile
{
    void start_stopwatch()
    {
        start_time = std::chrono::high_resolution_clock::now();
        is_running = true;
    }

    void stop_stopwatch()
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        total_time += elapsed_time;
        call_count++;
    }

    void update_stable_items()
    {
        stable_total_time = total_time;
        stable_call_count = call_count;
    }

    std::chrono::high_resolution_clock::time_point start_time;
    std::string caller_name;
    std::chrono::nanoseconds total_time = std::chrono::nanoseconds(0); /// use nanoseconds to have higher precision
    std::chrono::nanoseconds stable_total_time = std::chrono::nanoseconds(0); // Re-entrant safe
    int call_count = 0;
    int stable_call_count = 0; // Re-entrant safe
    bool is_running = false;
}; /* end struct CallerProfile */

/**
 * The profiler that profiles the hierarchical caller stack.
 *
 * @ingroup group_core
 */
class CallProfiler
{
private:
    CallProfiler() = default;

public:
    /// A singleton.
    static CallProfiler & instance()
    {
        static CallProfiler instance;
        return instance;
    }

    CallProfiler(CallProfiler const &) = delete;
    CallProfiler(CallProfiler &&) = delete;
    CallProfiler & operator=(CallProfiler const &) = delete;
    CallProfiler & operator=(CallProfiler &&) = delete;
    ~CallProfiler() = default;

    RadixTree<CallerProfile> & radix_tree()
    {
        return m_radix_tree;
    }

    /// Called when a function starts
    void start_caller(const std::string & caller_name, const std::function<void()> & cancel_callback);

    /// Called when a function ends
    void end_caller();

    /// Print the profiling information
    void print_profiling_result(std::ostream & outstream) const
    {
        print_profiling_result(*(m_radix_tree.get_current_node()), 0, outstream);
    }

    /// Reset the profiler
    void reset();

    /// Cancel the profiling from all probes
    void cancel()
    {
        for (auto & cancel_callback : m_cancel_callbacks)
        {
            cancel_callback();
        }
        reset();
    }

    const RadixTree<CallerProfile> & radix_tree() const { return m_radix_tree; }

private:
    void print_profiling_result(const RadixTreeNode<CallerProfile> & node, int depth, std::ostream & outstream) const;
    static void print_statistics(const RadixTreeNode<CallerProfile> & node, std::ostream & outstream);

    void update_pending_nodes()
    {
        for (auto const & node : m_pending_nodes)
        {
            node->data().update_stable_items();
        }
        m_pending_nodes.clear();
    }

private:
    RadixTree<CallerProfile> m_radix_tree; /// the data structure of the callers
    std::vector<std::function<void()>> m_cancel_callbacks; /// the callback to cancel the profiling from all probes
    std::set<RadixTreeNode<CallerProfile> *> m_pending_nodes; /// The nodes that are not yet updated
    friend detail::CallProfilerTest;
}; /* end class CallProfiler */

/**
 * Utility to profile a call.
 *
 * @ingroup group_core
 */
class CallProfilerProbe
{
public:
    CallProfilerProbe(CallProfiler & profiler, const char * caller_name)
        : m_caller_name(caller_name)
        , m_profiler(profiler)
    {
        auto cancel_callback = [&]()
        {
            cancel();
        };
        m_profiler.start_caller(m_caller_name, cancel_callback);
    }

    CallProfilerProbe(CallProfilerProbe const &) = delete;
    CallProfilerProbe(CallProfilerProbe &&) = delete;
    CallProfilerProbe & operator=(CallProfilerProbe const &) = delete;
    CallProfilerProbe & operator=(CallProfilerProbe &&) = delete;

    // FIXME: Remove the suppression NOLINTNEXTLINE(bugprone-exception-escape)
    ~CallProfilerProbe()
    {
        if (!m_cancel)
        {
            m_profiler.end_caller();
        }
    }

    void cancel()
    {
        m_cancel = true;
    }

private:
    const char * m_caller_name;
    bool m_cancel = false;
    CallProfiler & m_profiler;
}; /* end struct CallProfilerProbe */

// TODO: https://github.com/solvcon/solvcon/pull/559
#ifdef SOLVCON_PROFILE

#ifdef _MSC_VER
// ref: https://learn.microsoft.com/en-us/cpp/preprocessor/predefined-macros
#define __CROSS_PRETTY_FUNCTION__ __FUNCSIG__
#else
// ref: https://gcc.gnu.org/onlinedocs/gcc/Function-Names.html
#define __CROSS_PRETTY_FUNCTION__ __PRETTY_FUNCTION__
#endif
#define SOLVCON_PROFILE_FUNCTION() solvcon::CallProfilerProbe __profilerProbe##__COUNTER__(solvcon::CallProfiler::instance(), __CROSS_PRETTY_FUNCTION__)
#define SOLVCON_PROFILE_SCOPE(scopeName) solvcon::CallProfilerProbe __profilerProbe##__COUNTER__(solvcon::CallProfiler::instance(), scopeName)
#else
#define SOLVCON_PROFILE_FUNCTION() // do nothing
#define SOLVCON_PROFILE_SCOPE(scopeName) // do nothing
#endif

} /* end namespace solvcon */
// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

/*
 * Copyright (c) 2023, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/profiling/RadixTree.hpp>

namespace solvcon
{

void CallProfiler::reset()
{
    while (!m_radix_tree.is_root())
    {
        CallerProfile & profile = m_radix_tree.get_current_node()->data();
        if (profile.is_running)
        {
            profile.stop_stopwatch();
        }
        m_radix_tree.move_current_to_parent();
    }
    m_radix_tree.reset();
    m_cancel_callbacks.clear();
    m_pending_nodes.clear();
}

// Called when a function starts
void CallProfiler::start_caller(const std::string & caller_name, const std::function<void()> & cancel_callback)
{
    m_cancel_callbacks.push_back(cancel_callback);
    m_radix_tree.entry(caller_name);
    CallerProfile & callProfile = m_radix_tree.get_current_node()->data();
    callProfile.caller_name = caller_name;
    callProfile.start_stopwatch();
}

// Called when a function ends
void CallProfiler::end_caller()
{
    CallerProfile & call_profile = m_radix_tree.get_current_node()->data();
    call_profile.stop_stopwatch(); // Update profiling information to the pending time and count
    m_pending_nodes.insert(m_radix_tree.get_current_node());
    m_radix_tree.move_current_to_parent(); // Pop the caller from the call stack

    if (m_radix_tree.get_current_node() == m_radix_tree.get_root()) // If the root function ends, update all pending nodes and stable items
    {
        update_pending_nodes();
        m_radix_tree.update_stable_items();
    }
}
// NOLINTNEXTLINE(misc-no-recursion)
void CallProfiler::print_profiling_result(const RadixTreeNode<CallerProfile> & node, const int depth, std::ostream & outstream) const
{
    for (int i = 0; i < depth; ++i)
    {
        outstream << "  ";
    }

    auto const & profile = node.data();

    if (depth == 0)
    {
        outstream << "Profiling Result" << std::endl;
    }
    else
    {
        outstream << profile.caller_name << " - Total Time: " << profile.total_time.count() / 1e6 << " ms, Call Count: " << profile.call_count << std::endl;
    }

    for (const auto & child : node.children())
    {
        // NOLINTNEXTLINE(misc-no-recursion)
        print_profiling_result(*child, depth + 1, outstream);
    }
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

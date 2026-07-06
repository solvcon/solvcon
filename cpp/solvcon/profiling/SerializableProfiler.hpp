#pragma once

/*
 * Copyright (c) 2024, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Serializable mirror of the call profiler radix tree for JSON export.
 *
 * @ingroup group_core
 */

#include <solvcon/profiling/RadixTree.hpp>
#include <solvcon/serialization/SerializableItem.hpp>

namespace solvcon
{

/**
 * Serializable snapshot of one call profiler radix tree node.
 *
 * It captures the node key, name, accumulated total time (nanoseconds),
 * and call count, plus the child nodes whose stable call count is
 * positive. The running flag and start time are intentionally omitted
 * because they carry no meaning after deserialization.
 *
 * @ingroup group_core
 */
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class SerializableRadixTreeNode : public SerializableItem
{

public:

    using child_list_type = std::vector<SerializableRadixTreeNode>;
    using key_type = typename RadixTree<CallerProfile>::key_type;

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    SerializableRadixTreeNode() = default;
    SerializableRadixTreeNode(SerializableRadixTreeNode const &) = default;
    SerializableRadixTreeNode(SerializableRadixTreeNode &&) = default;
    SerializableRadixTreeNode & operator=(SerializableRadixTreeNode const &) = default;
    SerializableRadixTreeNode & operator=(SerializableRadixTreeNode &&) = default;
    ~SerializableRadixTreeNode() override = default;

    // FIXME: NOLINTNEXTLINE(misc-no-recursion)
    explicit SerializableRadixTreeNode(const RadixTreeNode<CallerProfile> * node)
        : m_key(node->key())
        , m_name(node->name())
        , m_total_time(node->data().stable_total_time)
        , m_call_count(node->data().stable_call_count)
    {
        for (const auto & child : node->children())
        {
            if (child->data().stable_call_count > 0)
            {
                m_children.push_back(SerializableRadixTreeNode(child.get())); // FIXME: NOLINT(modernize-use-emplace)
            }
        }
    }

    // NOLINTNEXTLINE(misc-no-recursion)
    MM_DECL_SERIALIZABLE(
        register_member("key", m_key);
        register_member("name", m_name);
        register_member("total_time", static_cast<double>(m_total_time.count()));
        register_member("call_count", m_call_count);
        register_member("children", m_children);)

private:

    /*
     * In T data, 'bool is_running' and
     * 'std::chrono::high_resolution_clock::time_point start_time' are not serialized
     * because they are useless when deserialize the object.
     */
    key_type m_key;
    std::string m_name;
    std::chrono::nanoseconds m_total_time;
    int m_call_count;
    child_list_type m_children;
}; /* end class SerializableRadixTreeNode */

/**
 * Serializable snapshot of a whole call profiler radix tree.
 *
 * It holds the root node, the name-to-key identifier map, and the next
 * unique node identifier, mirroring the stable state of a
 * RadixTree<CallerProfile>.
 *
 * @ingroup group_core
 */
// FIXME: NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class SerializableRadixTree : public SerializableItem
{

public:

    using key_type = typename RadixTree<CallerProfile>::key_type;
    SerializableRadixTree() = default;
    SerializableRadixTree(SerializableRadixTree const &) = default;
    SerializableRadixTree(SerializableRadixTree &&) = default;
    SerializableRadixTree & operator=(SerializableRadixTree const &) = default;
    SerializableRadixTree & operator=(SerializableRadixTree &&) = default;
    ~SerializableRadixTree() override = default;

    explicit SerializableRadixTree(const RadixTree<CallerProfile> & radix_tree)
        : m_root(radix_tree.get_root())
        , m_id_map(radix_tree.get_stable_id_map())
        , m_unique_id(radix_tree.get_stable_unique_node())
    {
    }

    MM_DECL_SERIALIZABLE(
        register_member("radix_tree", m_root);
        register_member("id_map", m_id_map);
        register_member("unique_id", m_unique_id);)

private:

    SerializableRadixTreeNode m_root;
    std::unordered_map<std::string, key_type> m_id_map;
    key_type m_unique_id;
}; /* end class SerializableRadixTree */

/**
 * Utility to serialize and deserialize CallProfiler.
 *
 * @ingroup group_core
 */
class CallProfilerSerializer
{

public:

    // It returns the json format of the CallProfiler.
    static std::string serialize(const CallProfiler & profiler)
    {
        auto * radix_tree_curent_node = profiler.radix_tree().get_current_node();
        auto * radix_tree_root = profiler.radix_tree().get_root();
        SerializableRadixTree const serializable_radix_tree(profiler.radix_tree());
        return serializable_radix_tree.to_json();
    }

}; /* end class CallProfilerSerializer */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

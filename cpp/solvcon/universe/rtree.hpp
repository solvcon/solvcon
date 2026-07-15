#pragma once

/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * R-tree spatial index for 2D and 3D bounding boxes.
 *
 * @ingroup group_geometry
 */

// This library implement R-tree based on "R Trees: A Dynamic Index Structure for Spatial Searchin" by Guttman in 1984
// https://doi.org/10.1145/971697.602266

#include <solvcon/base.hpp>

#include <algorithm>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace solvcon
{

/**
 * Bounding box for 2D and 3D objects, e.g., Point3d, Segment3d,
 * Triangle3d, etc. For 2D usage, callers pass min_z = max_z = 0
 * explicitly.
 *
 * @tparam T Floating-point coordinate type.
 *
 * @ingroup group_geometry
 */
template <typename T>
class BoundBox3d : public NumberBase<int32_t, T>
{
public:
    using value_type = T;

private:
    value_type m_min_x;
    value_type m_min_y;
    value_type m_min_z;
    value_type m_max_x;
    value_type m_max_y;
    value_type m_max_z;

public:

    /// construct a bounding box with given min and max coordinates
    BoundBox3d(value_type min_x_in, value_type min_y_in, value_type min_z_in, value_type max_x_in, value_type max_y_in, value_type max_z_in)
        : m_min_x(min_x_in)
        , m_min_y(min_y_in)
        , m_min_z(min_z_in)
        , m_max_x(max_x_in)
        , m_max_y(max_y_in)
        , m_max_z(max_z_in)
    {
    }

    /// construct a bounding box that encloses two bounding boxes
    BoundBox3d(const BoundBox3d & a, const BoundBox3d & b)
        : m_min_x(std::min(a.m_min_x, b.m_min_x))
        , m_min_y(std::min(a.m_min_y, b.m_min_y))
        , m_min_z(std::min(a.m_min_z, b.m_min_z))
        , m_max_x(std::max(a.m_max_x, b.m_max_x))
        , m_max_y(std::max(a.m_max_y, b.m_max_y))
        , m_max_z(std::max(a.m_max_z, b.m_max_z))
    {
    }

    value_type min_x() const { return m_min_x; }
    value_type min_y() const { return m_min_y; }
    value_type max_x() const { return m_max_x; }
    value_type max_y() const { return m_max_y; }
    value_type min_z() const { return m_min_z; }
    value_type max_z() const { return m_max_z; }

    bool overlap(BoundBox3d const & other) const
    {
        return !(other.m_min_x > m_max_x || other.m_max_x < m_min_x ||
                 other.m_min_y > m_max_y || other.m_max_y < m_min_y ||
                 other.m_min_z > m_max_z || other.m_max_z < m_min_z);
    }

    bool contain(BoundBox3d const & other) const
    {
        return (other.m_min_x >= m_min_x && other.m_max_x <= m_max_x &&
                other.m_min_y >= m_min_y && other.m_max_y <= m_max_y &&
                other.m_min_z >= m_min_z && other.m_max_z <= m_max_z);
    }

    /// 2D area (dx*dy) when z-extent = 0, else 3D volume (dx*dy*dz).
    /// Do not mix 2D and 3D boxes in one RTree: the split heuristic
    /// needs a monotonic measure within a single tree.
    value_type calc_area() const
    {
        const value_type dx = m_max_x - m_min_x;
        const value_type dy = m_max_y - m_min_y;
        const value_type dz = m_max_z - m_min_z;
        return dz == value_type(0) ? dx * dy : dx * dy * dz;
    }

    void expand(BoundBox3d const & other)
    {
        m_min_x = std::min(m_min_x, other.m_min_x);
        m_min_y = std::min(m_min_y, other.m_min_y);
        m_min_z = std::min(m_min_z, other.m_min_z);
        m_max_x = std::max(m_max_x, other.m_max_x);
        m_max_y = std::max(m_max_y, other.m_max_y);
        m_max_z = std::max(m_max_z, other.m_max_z);
    }
}; /* end class BoundBox3d */

/**
 * Value operations traits for the R-tree.
 *
 * @tparam E Item type to be stored in the R-tree.
 * @tparam B Bounding box type associated with the item E.
 *
 * @ingroup group_geometry
 */
template <typename E, typename B>
struct RTreeValueOps
{
    /// Calculate bounding box for the given item
    static B calc_bound_box(E const & item);

    /// Calculate bounding box for a group of items
    static B calc_group_bound_box(std::vector<E> const & items);
}; /* end struct RTreeValueOps */

/**
 * R-tree node structure.
 *
 * A node is a leaf when it holds items and an internal node when it
 * holds child nodes; the two vectors are not populated at the same
 * time.
 *
 * @tparam E Item type to be stored in the R-tree.
 * @tparam B Bounding box type associated with the item E.
 * @tparam ValueOpsType Value operations traits for E and B.
 *
 * @ingroup group_geometry
 */
template <typename E, typename B, typename ValueOpsType>
struct RTreeNode
{
    // TODO: optimize memory layout and  pointer operations (https://github.com/solvcon/solvcon/pull/637#discussion_r2552441239)
    B bbox;
    std::vector<E> items;

    using RTreeNodeType = RTreeNode<E, B, ValueOpsType>;
    std::vector<std::unique_ptr<RTreeNodeType>> nodes;

    explicit RTreeNode(B const & bbox_in)
        : bbox(bbox_in)
    {
    }

    RTreeNode() = delete;
    RTreeNode(RTreeNode const &) = delete;
    RTreeNode & operator=(RTreeNode const &) = delete;
    RTreeNode(RTreeNode &&) = delete;
    RTreeNode & operator=(RTreeNode &&) = delete;
    ~RTreeNode() = default;

    template <bool IS_LEAF>
    auto & get_children()
    {
        if constexpr (IS_LEAF)
        {
            return items;
        }
        else
        {
            return nodes;
        }
    }

    /// Recalculate bounding box based on contained items and nodes
    void recalculate_bound_box()
    {
        if (items.empty() && nodes.empty())
        {
            return; // empty node
        }

        bool first = true;
        B result_box = bbox;

        // if leaf node
        for (auto const & item : items)
        {
            B item_box = ValueOpsType::calc_bound_box(item);
            if (first)
            {
                result_box = item_box;
                first = false;
            }
            else
            {
                result_box.expand(item_box);
            }
        }

        // if internal node
        for (auto const & child : nodes)
        {
            if (first)
            {
                result_box = child->bbox;
                first = false;
            }
            else
            {
                result_box.expand(child->bbox);
            }
        }

        bbox = result_box;
    }
}; /* end struct RTreeNode */

/**
 * R-tree spatial index based on Guttman's 1984 R-tree paper.
 *
 * Supports insert, search by bounding box, and remove. A node holds at
 * most MAX_ITEMS_PER_NODE entries and splits when it overflows.
 *
 * @tparam E Item type to be stored in the R-tree.
 * @tparam B Bounding box type associated with E.
 * @tparam ValueOps Value operations traits for E and B.
 * @tparam MAX_ITEMS_PER_NODE Maximum number of entries per R-tree node.
 *
 * @ingroup group_geometry
 */
template <typename E, typename B, typename ValueOps = RTreeValueOps<E, B>, int MAX_ITEMS_PER_NODE = 64>
class RTree
{
private:
    static constexpr size_t MIN_ITEMS_PER_NODE = MAX_ITEMS_PER_NODE / 2;

    using value_type = typename B::value_type;
    using RTreeNodeType = RTreeNode<E, B, ValueOps>;
    using node_type = std::unique_ptr<RTreeNodeType>;
    node_type root;

public:
    RTree()
        : root(nullptr)
    {
    }
    RTree(RTree const &) = delete;
    RTree & operator=(RTree const &) = delete;
    RTree(RTree &&) = delete;
    RTree & operator=(RTree &&) = delete;
    ~RTree() = default;

    /// Insert item into R-tree
    void insert(E const & item)
    {
        const B box = ValueOps::calc_bound_box(item);

        // 1. find position for new record
        if (root == nullptr)
        {
            root = std::make_unique<RTreeNodeType>(box);
        }
        node_type & leaf = choose_leaf_for_new_entry(root, box);

        // 2. add record to leaf node
        leaf->items.push_back(item);
        leaf->recalculate_bound_box();

        node_type new_node = nullptr;
        if (leaf->items.size() > MAX_ITEMS_PER_NODE)
        {
            new_node = split_node(leaf);
        }

        // 3. propagate changes upward
        adjust_tree(leaf, std::move(new_node));
    }

    /// Search for items intersecting the given bounding box
    /// @param box the bounding box to search
    /// @param output the vector to store found items
    void search(const B & box, std::vector<E> & output) const
    {
        if (root == nullptr)
        {
            return; // empty tree
        }
        output.clear();

        search_internal(root, box, output);
    }

    /// Remove item from R-tree
    void remove(E const & item)
    {
        if (root == nullptr)
        {
            return; // empty tree
        }

        node_type * leaf = find_leaf_with_item(root, item);
        if (leaf == nullptr)
        {
            return; // item not found
        }

        // Remove item from leaf node
        auto it = std::find((*leaf)->items.begin(), (*leaf)->items.end(), item);
        if (it != (*leaf)->items.end())
        {
            (*leaf)->items.erase(it);
        }

        condense_tree(leaf->get());

        // shorten tree if necessary
        if (root->items.empty() && root->nodes.size() == 1)
        {
            root = std::move(root->nodes[0]);
        }
    }

private:

    // FIXME: NOLINTNEXTLINE(misc-no-recursion)
    void search_internal(const node_type & node, const B & box, std::vector<E> & output) const
    {
        assert(!(node->nodes.size() > 0 && node->items.size() > 0));

        // search subtree for item
        for (auto const & child : node->nodes)
        {
            if (child->bbox.overlap(box))
            {
                search_internal(child, box, output);
            }
        }

        // search leaf nodes for item
        for (auto const & it : node->items)
        {
            B item_box = ValueOps::calc_bound_box(it);
            if (item_box.overlap(box))
            {
                output.push_back(it);
            }
        }
    }

    // FIXME: NOLINTNEXTLINE(misc-no-recursion)
    node_type & choose_leaf_for_new_entry(node_type & node, const B & box)
    {
        if (node->nodes.empty())
        {
            return node;
        }

        // Initialize to first child to avoid undefined behavior when all
        // enlargements equal std::numeric_limits<value_type>::max().
        node_type * best_child = &node->nodes.front();
        value_type min_enlargement = std::numeric_limits<value_type>::max();

        for (auto & child : node->nodes)
        {
            value_type const enlargement = B(child->bbox, box).calc_area() - child->bbox.calc_area();

            if (enlargement < min_enlargement)
            {
                min_enlargement = enlargement;
                best_child = &child;
            }
        }

        return choose_leaf_for_new_entry(*best_child, box); // recurse down to leaf
    }

    // FIXME: NOLINTNEXTLINE(misc-no-recursion)
    node_type * get_parent(node_type & current, RTreeNodeType const * child)
    {
        for (auto & c : current->nodes)
        {
            if (c.get() == child)
            {
                return &current;
            }
            node_type * result = get_parent(c, child); // NOLINT(misc-const-correctness)
            if (result != nullptr)
            {
                return result;
            }
        }
        return nullptr;
    }

    // FIXME: NOLINTNEXTLINE(misc-no-recursion)
    node_type * find_leaf_with_item(node_type & current, E const & item)
    {
        for (auto const & it : current->items)
        {
            if (it == item)
            {
                return &current;
            }
        }

        for (auto & child : current->nodes)
        {
            if (child->bbox.contain(ValueOps::calc_bound_box(item)))
            {
                node_type * result = find_leaf_with_item(child, item); // NOLINT(misc-const-correctness)
                if (result != nullptr)
                {
                    return result;
                }
            }
        }

        return nullptr;
    }

    /// Adjust the tree after deletion
    void condense_tree(RTreeNodeType * node)
    {
        RTreeNodeType * current = node;
        std::vector<node_type> eliminated_nodes;

        while (current != root.get())
        {
            node_type * parent = get_parent(root, current);
            if (parent == nullptr)
            {
                break;
            }

            // Check if node is underfull
            size_t const entry_count = current->items.empty() ? current->nodes.size() : current->items.size();

            if (entry_count < MIN_ITEMS_PER_NODE && entry_count > 0)
            {
                // Remove node from parent
                auto it = std::find_if(
                    (*parent)->nodes.begin(), (*parent)->nodes.end(), [current](node_type const & child)
                    { return child.get() == current; });

                if (it != (*parent)->nodes.end())
                {
                    eliminated_nodes.push_back(std::move(*it));
                    (*parent)->nodes.erase(it);
                }
            }
            else
            {
                // Adjust bounding box
                current->recalculate_bound_box();
            }

            current = (*parent).get();
        }

        // Reinsert eliminated nodes' items
        for (auto & eliminated_node : eliminated_nodes)
        {
            for (auto & item : eliminated_node->items)
            {
                insert(item);
            }

            for (auto & child : eliminated_node->nodes)
            {
                // Reinsert all items from child nodes recursively
                reinsert_subtree(child);
            }
        }
    }

    // FIXME: NOLINTNEXTLINE(misc-no-recursion)
    void reinsert_subtree(node_type & node)
    {
        for (auto & item : node->items)
        {
            insert(item);
        }

        for (auto & child : node->nodes)
        {
            reinsert_subtree(child); // recursively reinsert child nodes
        }
    }

    /// split the provided node into two nodes
    /// @param node the node to split which will be modified
    /// @return the new node created from the split
    node_type split_node(node_type & node)
    {
        if (node->nodes.empty())
        {
            return split_node_impl<true>(node);
        }
        else
        {
            return split_node_impl<false>(node);
        }
    }

    /// split the provided node into two nodes
    /// @tparam IS_LEAF whether the node is a leaf node
    /// @tparam Container either the vector of items or vector of child nodes
    /// @param node1 the node to split which will be modified
    /// @return the new node created from the split
    template <bool IS_LEAF, typename Container = std::remove_reference_t<decltype(std::declval<node_type>()->template get_children<IS_LEAF>())>>
    node_type split_node_impl(node_type & node1)
    {
        Container entries;
        auto & original_entries = node1->template get_children<IS_LEAF>();
        entries.reserve(original_entries.size());
        for (auto & entry : original_entries)
        {
            entries.emplace_back(std::move(entry));
        }

        auto [seed1_index, seed2_index] = pick_seeds(entries);

        B box1 = calc_bound_box_from_entry(entries[seed1_index]);
        B box2 = calc_bound_box_from_entry(entries[seed2_index]);

        auto & node1_entries = node1->template get_children<IS_LEAF>();
        node1_entries.clear();
        node1_entries.emplace_back(std::move(entries[seed1_index]));

        node_type node2 = std::make_unique<RTreeNodeType>(box2);
        auto & node2_entries = node2->template get_children<IS_LEAF>();
        node2_entries.emplace_back(std::move(entries[seed2_index]));

        bool need_recalculate = false;
        for (size_t i = 0; i < entries.size(); ++i)
        {
            if (i == seed1_index || i == seed2_index)
            {
                continue;
            }

            // if one group has so few entries that all the rest must go there
            size_t const remaining = entries.size() - i;
            if (node1_entries.size() + remaining <= MIN_ITEMS_PER_NODE)
            {
                // assign all the remaining to group1
                node1_entries.emplace_back(std::move(entries[i]));
                need_recalculate = true;
                continue;
            }
            if (node2_entries.size() + remaining <= MIN_ITEMS_PER_NODE)
            {
                // assign all the remaining to group2
                node2_entries.emplace_back(std::move(entries[i]));
                need_recalculate = true;
                continue;
            }

            B box = calc_bound_box_from_entry(entries[i]);
            value_type const enlargement1 = B(box1, box).calc_area() - box1.calc_area();
            value_type const enlargement2 = B(box2, box).calc_area() - box2.calc_area();

            if (enlargement1 < enlargement2 ||
                (enlargement1 == enlargement2 && box1.calc_area() < box2.calc_area()))
            {
                node1_entries.emplace_back(std::move(entries[i]));
                node1->recalculate_bound_box();
            }
            else
            {
                node2_entries.emplace_back(std::move(entries[i]));
                node2->recalculate_bound_box();
            }
        }

        if (need_recalculate)
        {
            node1->recalculate_bound_box();
            node2->recalculate_bound_box();
        }

        return std::move(node2);
    }

    template <typename Entry>
    B calc_bound_box_from_entry(Entry const & entry)
    {
        if constexpr (std::is_same_v<Entry, E>)
        {
            return ValueOps::calc_bound_box(entry);
        }
        else
        {
            return entry->bbox;
        }
    }

    /// Select two seed entries to start the split process
    /// @tparam Container either the vector of items or vector of child nodes
    /// @param entries the entries to pick seeds from
    /// @return pair of indices of the selected seeds
    template <typename Container>
    std::pair<size_t, size_t> pick_seeds(Container const & entries)
    {
        value_type max_waste = std::numeric_limits<value_type>::lowest();
        size_t seed1 = 0, seed2 = 0;

        for (size_t i = 0; i < entries.size(); ++i)
        {
            for (size_t j = i + 1; j < entries.size(); ++j)
            {
                B box_i = calc_bound_box_from_entry(entries[i]);
                B box_j = calc_bound_box_from_entry(entries[j]);
                B combined_box = B(box_i, box_j);
                value_type const waste = combined_box.calc_area() - box_i.calc_area() - box_j.calc_area();
                if (waste > max_waste)
                {
                    max_waste = waste;
                    seed1 = i;
                    seed2 = j;
                }
            }
        }
        return {seed1, seed2};
    }

    void adjust_tree(node_type & leaf, node_type new_node)
    {
        RTreeNodeType const * current = leaf.get();
        node_type split_result = std::move(new_node);

        while (current != root.get())
        {
            node_type * parent = get_parent(root, current);
            if (parent == nullptr)
            {
                break;
            }

            if (split_result)
            {
                (*parent)->nodes.push_back(std::move(split_result));
                split_result = nullptr;

                if ((*parent)->nodes.size() > MAX_ITEMS_PER_NODE)
                {
                    split_result = split_node(*parent);
                }
            }

            (*parent)->recalculate_bound_box();

            current = (*parent).get(); // move up the tree
        }

        // Handle root split
        if (split_result && current == root.get())
        {
            B root_bbox(root->bbox, split_result->bbox);
            node_type new_root = std::make_unique<RTreeNodeType>(root_bbox);
            new_root->nodes.push_back(std::move(root));
            new_root->nodes.push_back(std::move(split_result));
            root = std::move(new_root);
        }
    }

}; /* end class RTree */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Toggle configuration system and process and command-line information.
 *
 * @ingroup group_core
 */

#include <solvcon/base.hpp>
#include <solvcon/buffer/buffer.hpp>
#include <solvcon/profiling/profile.hpp>
#include <solvcon/profiling/RadixTree.hpp>
#include <solvcon/toggle/build_config.hpp>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <deque>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace solvcon
{

int setenv(const char * name, const char * value, int overwrite);

/**
 * Lifecycle category of a toggle (Fowler's feature-flag categories).
 *
 * The category enforces nothing by itself; it records the expected lifetime
 * so a lint or test can flag, for example, a release toggle that outlived its
 * feature. Ops and UI settings are long-lived; a release toggle is removed
 * once its feature lands.
 *
 * @ingroup group_core
 */
enum class ToggleCategory : uint8_t
{
    Release,
    Ops,
    Experiment,
}; /* end enum class ToggleCategory */

template <typename T>
class ToggleRef;

/**
 * A single stored toggle value.
 *
 * Named after the atomic register of shared-memory theory: a single-value
 * shared location. A scalar register holds a std::atomic so its read and
 * write are lock-free and never torn; get and set use memory_order_relaxed
 * because a toggle publishes nothing but itself.
 *
 * @ingroup group_core
 */
template <typename T>
struct ToggleRegister
{
    std::atomic<T> value;
    T get() const { return value.load(std::memory_order_relaxed); }
    void set(T v) { value.store(v, std::memory_order_relaxed); }
}; /* end struct ToggleRegister */

/**
 * String register.
 *
 * A string is not a hot-path value: it is read by the UI and config code
 * only, never in a tight loop, so it is a plain std::string guarded by the
 * table mutex rather than an atomic. get returns a reference that is valid
 * only while no writer mutates the same key.
 *
 * @ingroup group_core
 */
template <>
struct ToggleRegister<std::string>
{
    std::string value;
    std::string const & get() const { return value; }
    void set(std::string v) { value = std::move(v); }
}; /* end struct ToggleRegister<std::string> */

/**
 * Column of ToggleRegister<T> with stable register addresses.
 *
 * Each register is heap-owned through a unique_ptr, so a register keeps its
 * address for its lifetime even as the column grows, and a resolved handle
 * stays valid. The heap box also lets a non-movable atomic register live in
 * a standard container. Indices are dense and start at zero, matching the
 * offsets held by DynamicToggleIndex.
 *
 * @ingroup group_core
 */
template <typename T>
class ToggleRegisterColumn
{

public:

    ToggleRegisterColumn() = default;

    ToggleRegisterColumn(ToggleRegisterColumn const & other)
    {
        for (auto const & reg : other.m_registers)
        {
            append(reg->get());
        }
    }
    ToggleRegisterColumn(ToggleRegisterColumn &&) = default;
    ToggleRegisterColumn & operator=(ToggleRegisterColumn const &) = delete;
    ToggleRegisterColumn & operator=(ToggleRegisterColumn &&) = default;
    ~ToggleRegisterColumn() = default;

    size_t size() const { return m_registers.size(); }

    ToggleRegister<T> & at(size_t index) { return *m_registers.at(index); }
    ToggleRegister<T> const & at(size_t index) const { return *m_registers.at(index); }

    size_t append(T const & value)
    {
        size_t const index = m_registers.size();
        m_registers.push_back(std::make_unique<ToggleRegister<T>>());
        m_registers.back()->set(value);
        return index;
    }

    void clear() { m_registers.clear(); }

private:

    std::deque<std::unique_ptr<ToggleRegister<T>>> m_registers;

}; /* end class ToggleRegisterColumn */

/**
 * Index and type tag that locates a dynamic toggle value in its typed
 * storage vector.
 *
 * The bool conversion is true when the type is not TYPE_NONE, and index
 * is a 32-bit offset into the storage vector for the given Type.
 *
 * @ingroup group_core
 */
struct DynamicToggleIndex
{

    /**
     * Data type tag for a dynamic toggle value.
     *
     * @ingroup group_core
     */
    enum Type : uint8_t
    {
        TYPE_NONE, // 0
        TYPE_BOOL,
        TYPE_INT8,
        TYPE_INT16,
        TYPE_INT32,
        TYPE_INT64,
        TYPE_REAL,
        TYPE_STRING,
        TYPE_SUBKEY
    };

    explicit operator bool() const { return type != TYPE_NONE; }
    bool is_bool() const { return type == TYPE_BOOL; }
    bool is_int8() const { return type == TYPE_INT8; }
    bool is_int16() const { return type == TYPE_INT16; }
    bool is_int32() const { return type == TYPE_INT32; }
    bool is_int64() const { return type == TYPE_INT64; }
    bool is_real() const { return type == TYPE_REAL; }
    bool is_string() const { return type == TYPE_STRING; }
    bool is_subkey() const { return type == TYPE_SUBKEY; }

    // Index upper bound 2**32 is more than sufficient.  2**16 (65536) may be
    // too little.
    uint32_t index = 0;
    Type type = TYPE_NONE;

}; /* end struct DynamicToggleIndex */

/**
 * Maps a C++ value type to its DynamicToggleIndex type tag.
 *
 * The primary template is left undefined so an unsupported type is a
 * compile error rather than a silent mismatch.
 *
 * @ingroup group_core
 */
template <typename T>
struct ToggleTypeTraits;

#define MM_TOGGLE_TYPE_TRAITS(CTYPE, TAG)               \
    template <>                                         \
    struct ToggleTypeTraits<CTYPE>                      \
    {                                                   \
        static constexpr DynamicToggleIndex::Type tag = \
            DynamicToggleIndex::TAG;                    \
    };
MM_TOGGLE_TYPE_TRAITS(bool, TYPE_BOOL)
MM_TOGGLE_TYPE_TRAITS(int8_t, TYPE_INT8)
MM_TOGGLE_TYPE_TRAITS(int16_t, TYPE_INT16)
MM_TOGGLE_TYPE_TRAITS(int32_t, TYPE_INT32)
MM_TOGGLE_TYPE_TRAITS(int64_t, TYPE_INT64)
MM_TOGGLE_TYPE_TRAITS(double, TYPE_REAL)
MM_TOGGLE_TYPE_TRAITS(std::string, TYPE_STRING)
#undef MM_TOGGLE_TYPE_TRAITS

namespace detail
{

/**
 * Shared registry of change callbacks, keyed by toggle key.
 *
 * It is owned by a shared_ptr so a ToggleSubscription can hold a weak
 * reference and unsubscribe safely even if the table is gone. Its own mutex
 * guards the map and is never held while a callback runs.
 *
 * @ingroup group_core
 */
struct ToggleChangeObservers
{
    using callback_type = std::shared_ptr<std::function<void()>>;
    std::mutex mutex;
    uint64_t next_id = 0;
    // The callback is held by shared_ptr so a firing thread can copy the
    // pointer (a cheap atomic) rather than the std::function itself. Copying
    // a std::function that wraps a Python callable would touch a Python
    // refcount without the GIL, from a solver thread that does not hold it.
    std::unordered_map<std::string, std::vector<std::pair<uint64_t, callback_type>>> map;
}; /* end struct ToggleChangeObservers */

} /* end namespace detail */

/**
 * RAII handle for one change subscription.
 *
 * Dropping it unsubscribes, so a callback never outlives the widget it
 * belongs to. It is move-only and tolerates the table being destroyed first
 * (the weak reference simply expires).
 *
 * @ingroup group_core
 */
class ToggleSubscription
{

public:

    ToggleSubscription() = default;

    ToggleSubscription(std::shared_ptr<detail::ToggleChangeObservers> const & observers, std::string key, uint64_t id)
        : m_observers(observers)
        , m_key(std::move(key))
        , m_id(id)
        , m_active(true)
    {
    }

    ToggleSubscription(ToggleSubscription const &) = delete;
    ToggleSubscription & operator=(ToggleSubscription const &) = delete;

    ToggleSubscription(ToggleSubscription && other) noexcept { swap(other); }
    ToggleSubscription & operator=(ToggleSubscription && other) noexcept
    {
        if (this != &other)
        {
            reset();
            swap(other);
        }
        return *this;
    }

    ~ToggleSubscription() { reset(); }

    bool active() const { return m_active; }

    void reset()
    {
        if (!m_active)
        {
            return;
        }
        m_active = false;
        if (std::shared_ptr<detail::ToggleChangeObservers> const obs = m_observers.lock())
        {
            std::scoped_lock const guard(obs->mutex);
            auto it = obs->map.find(m_key);
            if (it != obs->map.end())
            {
                auto & subs = it->second;
                subs.erase(
                    std::remove_if(subs.begin(), subs.end(), [this](auto const & pr)
                                   { return pr.first == m_id; }),
                    subs.end());
                if (subs.empty())
                {
                    obs->map.erase(it);
                }
            }
        }
    }

private:

    void swap(ToggleSubscription & other) noexcept
    {
        m_observers.swap(other.m_observers);
        m_key.swap(other.m_key);
        std::swap(m_id, other.m_id);
        std::swap(m_active, other.m_active);
    }

    std::weak_ptr<detail::ToggleChangeObservers> m_observers;
    std::string m_key;
    uint64_t m_id = 0;
    bool m_active = false;

}; /* end class ToggleSubscription */

class DynamicToggleTable;

/**
 * Hierarchical accessor that reads and writes a DynamicToggleTable
 * through dotted keys.
 *
 * It holds a base prefix and joins it to a key with a "." separator
 * (see rekey), so nested subkeys can be reached without repeating the
 * full path.
 *
 * @ingroup group_core
 */
class HierarchicalToggleAccess
{

public:

    explicit HierarchicalToggleAccess(DynamicToggleTable & table)
        : m_table(&table)
    {
    }

    HierarchicalToggleAccess(DynamicToggleTable & table, std::string base)
        : m_table(&table)
        , m_base(std::move(base))
    {
    }

    bool get_bool(std::string const & key) const;
    void set_bool(std::string const & key, bool value);
    int8_t get_int8(std::string const & key) const;
    void set_int8(std::string const & key, int8_t value);
    int16_t get_int16(std::string const & key) const;
    void set_int16(std::string const & key, int16_t value);
    int32_t get_int32(std::string const & key) const;
    void set_int32(std::string const & key, int32_t value);
    int64_t get_int64(std::string const & key) const;
    void set_int64(std::string const & key, int64_t value);
    double get_real(std::string const & key) const;
    void set_real(std::string const & key, double value);
    std::string const & get_string(std::string const & key) const;
    void set_string(std::string const & key, std::string const & value);

    HierarchicalToggleAccess get_subkey(std::string const & key);
    void add_subkey(std::string const & key);

    DynamicToggleIndex get_index(std::string const & key) const;

    std::string rekey(std::string const & key) const
    {
        return m_base.empty() ? key : std::format("{}.{}", m_base, key);
    }

private:

    DynamicToggleTable * m_table = nullptr;
    std::string m_base;

}; /* end class HierarchicalToggleAccess */

/**
 * Runtime table of dynamic toggles keyed by name.
 *
 * Each key maps to a DynamicToggleIndex that selects one of the per-type
 * storage vectors (bool, int8, int16, int32, int64, real, and string).
 * Values can be added, read, written, and cleared during runtime.
 *
 * @ingroup group_core
 */
// All data members are class types with their own default constructors.
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class DynamicToggleTable
{

public:

    using keymap_type = std::unordered_map<std::string, DynamicToggleIndex>;

    static std::string const sentinel_string; // FIXME: NOLINT(readability-redundant-string-init)

    DynamicToggleTable() = default;

    // The atomic generation and the mutex are not copyable. Delegate to a
    // private constructor that holds the source lock for the whole deep copy
    // (the scoped_lock temporary outlives the delegated-to body) and starts a
    // fresh mutex.
    DynamicToggleTable(DynamicToggleTable const & other)
        : DynamicToggleTable(other, std::scoped_lock<std::mutex>(other.m_mutex))
    {
    }
    DynamicToggleTable(DynamicToggleTable &&) = delete;
    DynamicToggleTable & operator=(DynamicToggleTable const &) = delete;
    DynamicToggleTable & operator=(DynamicToggleTable &&) = delete;
    ~DynamicToggleTable() = default;

    bool get_bool(std::string const & key) const;
    void set_bool(std::string const & key, bool value);
    int8_t get_int8(std::string const & key) const;
    void set_int8(std::string const & key, int8_t value);
    int16_t get_int16(std::string const & key) const;
    void set_int16(std::string const & key, int16_t value);
    int32_t get_int32(std::string const & key) const;
    void set_int32(std::string const & key, int32_t value);
    int64_t get_int64(std::string const & key) const;
    void set_int64(std::string const & key, int64_t value);
    double get_real(std::string const & key) const;
    void set_real(std::string const & key, double value);
    std::string const & get_string(std::string const & key) const;
    void set_string(std::string const & key, std::string const & value);

    HierarchicalToggleAccess get_subkey(std::string const & key)
    {
        return HierarchicalToggleAccess(*this, key);
    }
    void add_subkey(std::string const & key);

    // Reads the m_key2index member, so it cannot be made static.
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    DynamicToggleIndex get_index(std::string const & key) const
    {
        std::scoped_lock const guard(m_mutex);
        auto it = m_key2index.find(key);
        return (it != m_key2index.end()) ? it->second : DynamicToggleIndex{.index = 0, .type = DynamicToggleIndex::TYPE_NONE};
    }
    std::vector<std::string> keys() const;
    void clear();

    /**
     * Monotonic stamp bumped on every clear.
     *
     * A handle resolved before a clear carries the old generation, so it
     * can be detected as stale rather than aliasing a register reused
     * after the clear.
     */
    uint64_t generation() const { return m_generation.load(std::memory_order_relaxed); }

    /**
     * Create a toggle with a default and a category, and return a handle.
     *
     * Re-declaring an existing key of the same type is idempotent and
     * returns a handle to the existing register. A conflicting type is an
     * error.
     */
    template <typename T>
    ToggleRef<T> declare(std::string const & key, T const & default_value, ToggleCategory category = ToggleCategory::Ops);

    /// The category recorded for a key, or Ops if the key was never declared.
    ToggleCategory category(std::string const & key) const
    {
        std::scoped_lock const guard(m_mutex);
        auto it = m_categories.find(key);
        return (it != m_categories.end()) ? it->second : ToggleCategory::Ops;
    }

    /// Resolve an existing key to a handle; an invalid handle if it is
    /// missing or has a different type.
    template <typename T>
    ToggleRef<T> ref(std::string const & key);

    /// Typed read returning the caller default on a missing or wrong-typed
    /// key (the OpenFeature contract).
    template <typename T>
    T get(std::string const & key, T const & default_value) const;

    /// Strict typed read that throws on a missing or wrong-typed key.
    template <typename T>
    T at(std::string const & key) const;

    /**
     * Subscribe to changes of a key.
     *
     * The callback runs after the write lock is released, so it may call back
     * into the store (including set) without deadlock, and firing outside the
     * lock keeps a Python callback from inverting lock order against the GIL.
     * A no-op write (setting the value already stored) does not fire. Dropping
     * the returned token unsubscribes.
     */
    [[nodiscard]] ToggleSubscription on_change(std::string const & key, std::function<void()> callback)
    {
        std::scoped_lock const guard(m_observers->mutex);
        uint64_t const id = m_observers->next_id++;
        m_observers->map[key].emplace_back(id, std::make_shared<std::function<void()>>(std::move(callback)));
        return ToggleSubscription(m_observers, key, id);
    }

private:

    DynamicToggleTable(DynamicToggleTable const & other, std::scoped_lock<std::mutex> const &)
        : m_key2index(other.m_key2index)
        , m_categories(other.m_categories)
        , m_column_bool(other.m_column_bool)
        , m_column_int8(other.m_column_int8)
        , m_column_int16(other.m_column_int16)
        , m_column_int32(other.m_column_int32)
        , m_column_int64(other.m_column_int64)
        , m_column_real(other.m_column_real)
        , m_column_string(other.m_column_string)
        , m_generation(other.m_generation.load(std::memory_order_relaxed))
    {
    }

    template <typename T>
    ToggleRegisterColumn<T> & column();
    template <typename T>
    ToggleRegisterColumn<T> const & column() const;

    // Fire the change callbacks for a key, called after the write lock is
    // released. A throwing callback is caught so the store is never corrupted.
    void notify(std::string const & key) const;

    // Equality used by the no-op-write guard. Doubles compare by bit pattern
    // so NaN and -0.0 behave sanely under the load-compare-store.
    template <typename T>
    static bool value_equal(T const & lhs, T const & rhs)
    {
        if constexpr (std::is_same_v<T, double>)
        {
            return std::bit_cast<uint64_t>(lhs) == std::bit_cast<uint64_t>(rhs);
        }
        else
        {
            return lhs == rhs;
        }
    }

    mutable std::mutex m_mutex;
    keymap_type m_key2index;
    std::unordered_map<std::string, ToggleCategory> m_categories;
    ToggleRegisterColumn<bool> m_column_bool;
    ToggleRegisterColumn<int8_t> m_column_int8;
    ToggleRegisterColumn<int16_t> m_column_int16;
    ToggleRegisterColumn<int32_t> m_column_int32;
    ToggleRegisterColumn<int64_t> m_column_int64;
    ToggleRegisterColumn<double> m_column_real;
    ToggleRegisterColumn<std::string> m_column_string;
    std::atomic<uint64_t> m_generation{0};
    // A fresh registry per table; a clone does not inherit subscriptions.
    std::shared_ptr<detail::ToggleChangeObservers> m_observers = std::make_shared<detail::ToggleChangeObservers>();

}; /* end class DynamicToggleTable */

inline DynamicToggleIndex HierarchicalToggleAccess::get_index(std::string const & key) const
{
    return m_table->get_index(rekey(key));
}

/**
 * A resolved, typed handle to one toggle register.
 *
 * It binds a key to its register once (table, index, and the table
 * generation at resolution) so a hot path reads the value without touching
 * the string map again. Because the storage keeps register addresses
 * stable, the bound pointer survives unrelated toggles being added; because
 * the handle stamps the table generation, a handle taken before a
 * dynamic_clear is refused afterward instead of aliasing a reused register.
 *
 * @ingroup group_core
 */
template <typename T>
class ToggleRef
{

public:

    ToggleRef() = default;

    /// True when the handle points at a live register in the current
    /// generation (not default-constructed and not invalidated by a clear).
    bool valid() const
    {
        return nullptr != m_register && nullptr != m_table && m_table->generation() == m_generation;
    }
    explicit operator bool() const { return valid(); }

    T load() const
    {
        check();
        return m_register->get();
    }
    void store(T const & value) const
    {
        check();
        m_register->set(value);
    }

    uint32_t index() const { return m_index; }
    uint64_t generation() const { return m_generation; }

private:

    friend class DynamicToggleTable;

    ToggleRef(DynamicToggleTable * table, ToggleRegister<T> * reg, uint32_t index, uint64_t generation)
        : m_table(table)
        , m_register(reg)
        , m_index(index)
        , m_generation(generation)
    {
    }

    void check() const
    {
        if (!valid())
        {
            throw std::out_of_range("ToggleRef: stale or invalid handle");
        }
    }

    DynamicToggleTable * m_table = nullptr;
    ToggleRegister<T> * m_register = nullptr;
    uint32_t m_index = 0;
    uint64_t m_generation = 0;

}; /* end class ToggleRef */

template <typename T>
ToggleRegisterColumn<T> & DynamicToggleTable::column()
{
    if constexpr (std::is_same_v<T, bool>)
    {
        return m_column_bool;
    }
    else if constexpr (std::is_same_v<T, int8_t>)
    {
        return m_column_int8;
    }
    else if constexpr (std::is_same_v<T, int16_t>)
    {
        return m_column_int16;
    }
    else if constexpr (std::is_same_v<T, int32_t>)
    {
        return m_column_int32;
    }
    else if constexpr (std::is_same_v<T, int64_t>)
    {
        return m_column_int64;
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return m_column_real;
    }
    else
    {
        return m_column_string;
    }
}

template <typename T>
ToggleRegisterColumn<T> const & DynamicToggleTable::column() const
{
    if constexpr (std::is_same_v<T, bool>)
    {
        return m_column_bool;
    }
    else if constexpr (std::is_same_v<T, int8_t>)
    {
        return m_column_int8;
    }
    else if constexpr (std::is_same_v<T, int16_t>)
    {
        return m_column_int16;
    }
    else if constexpr (std::is_same_v<T, int32_t>)
    {
        return m_column_int32;
    }
    else if constexpr (std::is_same_v<T, int64_t>)
    {
        return m_column_int64;
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        return m_column_real;
    }
    else
    {
        return m_column_string;
    }
}

template <typename T>
ToggleRef<T> DynamicToggleTable::declare(std::string const & key, T const & default_value, ToggleCategory category)
{
    std::scoped_lock const guard(m_mutex);
    m_categories[key] = category;
    auto it = m_key2index.find(key);
    if (it != m_key2index.end())
    {
        if (it->second.type != ToggleTypeTraits<T>::tag)
        {
            throw std::invalid_argument(
                std::format("Toggle::declare: key \"{}\" already declared with a different type", key));
        }
        uint32_t const idx = it->second.index;
        return ToggleRef<T>(this, &column<T>().at(idx), idx, generation());
    }
    auto const idx = static_cast<uint32_t>(column<T>().append(default_value));
    m_key2index.insert({key, DynamicToggleIndex{idx, ToggleTypeTraits<T>::tag}});
    return ToggleRef<T>(this, &column<T>().at(idx), idx, generation());
}

template <typename T>
ToggleRef<T> DynamicToggleTable::ref(std::string const & key)
{
    std::scoped_lock const guard(m_mutex);
    auto it = m_key2index.find(key);
    if (it != m_key2index.end() && it->second.type == ToggleTypeTraits<T>::tag)
    {
        uint32_t const idx = it->second.index;
        return ToggleRef<T>(this, &column<T>().at(idx), idx, generation());
    }
    return ToggleRef<T>();
}

template <typename T>
T DynamicToggleTable::get(std::string const & key, T const & default_value) const
{
    std::scoped_lock const guard(m_mutex);
    auto it = m_key2index.find(key);
    if (it != m_key2index.end() && it->second.type == ToggleTypeTraits<T>::tag)
    {
        return column<T>().at(it->second.index).get();
    }
    return default_value;
}

template <typename T>
T DynamicToggleTable::at(std::string const & key) const
{
    std::scoped_lock const guard(m_mutex);
    auto it = m_key2index.find(key);
    if (it != m_key2index.end() && it->second.type == ToggleTypeTraits<T>::tag)
    {
        return column<T>().at(it->second.index).get();
    }
    throw std::out_of_range(std::format("Toggle::at: key \"{}\" is missing or has a different type", key));
}

/**
 * Compatibility facade for the former compile-time solid toggles.
 *
 * The one solid toggle, use_pyside, is now an inline constexpr build switch
 * (see build_config.hpp) that the optimizer folds away. This stateless
 * facade keeps the old accessor working until its callers move.
 *
 * @ingroup group_core
 */
class SolidToggle
{

public:

    // Kept as an instance method for API compatibility with the old facade.
    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    bool use_pyside() const { return build_config::use_pyside; }

}; /* end class SolidToggle */

/**
 * Compatibility facade for the former fixed toggles.
 *
 * python_redirect and show_axis are now ordinary declared toggles in the
 * store, so they gain change notification and serialization. This facade
 * reads and writes them through the store (with their startup defaults),
 * keeping the old get_/set_ accessors working until their callers move.
 *
 * @ingroup group_core
 */
class FixedToggle
{

public:

    explicit FixedToggle(DynamicToggleTable & table)
        : m_table(&table)
    {
    }

    bool get_python_redirect() const { return m_table->get<bool>("python_redirect", true); }
    void set_python_redirect(bool v) { m_table->set_bool("python_redirect", v); }
    bool get_show_axis() const { return m_table->get<bool>("show_axis", false); }
    void set_show_axis(bool v) { m_table->set_bool("show_axis", v); }

private:

    DynamicToggleTable * m_table = nullptr;

}; /* end class FixedToggle */

/**
 * The toggle system for solvcon. There are 3 types of toggles:
 *
 * 1. solid toggles: managed by SolidToggle class. It is the toggles whose value
 *    is determined during compile time. The value is read-only (const) through
 *    out the program lifecycle (the process).
 *
 *    The solid toggles have address and can be referenced. They cannot be
 *    optimized out (unlike macros and constexpr). It could add overhead when
 *    used in tight loops. The overhead may usually be too low to be noticed,
 *    but sometimes one needs to be careful about it.
 *
 * 2. fixed toggles: managed by FixedToggle class. It is the toggles whose name
 *    is determined during compile time. The value can be changed during
 *    runtime.
 *
 *    Because the names are determined during compile time, when accessing the
 *    toggles, no table lookup is needed. The address of the toggle variables
 *    has been determined by the compiler and linker.
 *
 *    The runtime cost of fixed toggles is the same as solid toggles. It may
 *    be used in tight loops. Just becareful about the potential runtime
 *    overhead.
 *
 * 3. dynamic toggles: managed by DynamicToggleTable. The toggles are
 *    hierarchical and the names and values can be added, removed, and modified
 *    during runtime. The value needs to use limited data types: bool, int8,
 *    int16, int32, int64, real, and string. It is intentional not to support
 *    unsigned integers.
 *
 *    Accessing dynamic toggles requires table lookup and string comparison. It
 *    is slow but flexible.
 *
 *    To access the dynamic toggles from C++, the data type of the toggle
 *    The hierarchical access (from C++) uses ".", like:
 *
 *      tg.get_int8("top_level.second_level.key_name")
 *
 *    In Python, the wrapper can determine the type dynamically, and the
 *    hierarchical access may use attribute syntax:
 *
 *      tg.top_level.second_level.key_name = value
 *
 * @ingroup group_core
 */
class Toggle
{

public:

    static Toggle & instance();

    Toggle * clone() const { return new Toggle(*this); /* NOLINT(cppcoreguidelines-owning-memory) */ }

    ~Toggle() = default;

    // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    SolidToggle solid() const { return SolidToggle{}; }

    FixedToggle fixed() { return FixedToggle(m_dynamic_table); }

    DynamicToggleTable const & dynamic() const { return m_dynamic_table; }
    DynamicToggleTable & dynamic() { return m_dynamic_table; }

    std::vector<std::string> dynamic_keys() const { return m_dynamic_table.keys(); }
    void dynamic_clear() { m_dynamic_table.clear(); }
    uint64_t dynamic_generation() const { return m_dynamic_table.generation(); }
    DynamicToggleIndex get_dynamic_index(std::string const & key) const { return m_dynamic_table.get_index(key); }

    template <typename T>
    ToggleRef<T> declare(std::string const & key, T const & default_value, ToggleCategory category = ToggleCategory::Ops)
    {
        return m_dynamic_table.declare<T>(key, default_value, category);
    }
    ToggleCategory category(std::string const & key) const { return m_dynamic_table.category(key); }
    template <typename T>
    ToggleRef<T> ref(std::string const & key) { return m_dynamic_table.ref<T>(key); }
    template <typename T>
    T get(std::string const & key, T const & default_value) const { return m_dynamic_table.get<T>(key, default_value); }
    template <typename T>
    T at(std::string const & key) const { return m_dynamic_table.at<T>(key); }

    [[nodiscard]] ToggleSubscription on_change(std::string const & key, std::function<void()> callback)
    {
        return m_dynamic_table.on_change(key, std::move(callback));
    }

    bool get_bool(std::string const & key) const { return m_dynamic_table.get_bool(key); }
    void set_bool(std::string const & key, bool value) { m_dynamic_table.set_bool(key, value); }
    int8_t get_int8(std::string const & key) const { return m_dynamic_table.get_int8(key); }
    void set_int8(std::string const & key, int8_t value) { m_dynamic_table.set_int8(key, value); }
    int16_t get_int16(std::string const & key) const { return m_dynamic_table.get_int16(key); }
    void set_int16(std::string const & key, int16_t value) { m_dynamic_table.set_int16(key, value); }
    int32_t get_int32(std::string const & key) const { return m_dynamic_table.get_int32(key); }
    void set_int32(std::string const & key, int32_t value) { m_dynamic_table.set_int32(key, value); }
    int64_t get_int64(std::string const & key) const { return m_dynamic_table.get_int64(key); }
    void set_int64(std::string const & key, int64_t value) { m_dynamic_table.set_int64(key, value); }
    double get_real(std::string const & key) const { return m_dynamic_table.get_real(key); }
    void set_real(std::string const & key, double value) { m_dynamic_table.set_real(key, value); }
    std::string const & get_string(std::string const & key) const { return m_dynamic_table.get_string(key); }
    void set_string(std::string const & key, std::string const & value) { m_dynamic_table.set_string(key, value); }
    HierarchicalToggleAccess get_subkey(std::string const & key) { return m_dynamic_table.get_subkey(key); }
    void add_subkey(std::string const & key) { m_dynamic_table.add_subkey(key); }

    // The store holds an atomic and a mutex, so it is not movable or
    // assignable, and neither is the Toggle that owns it. clone() still copies
    // through the private copy constructor.
    Toggle(Toggle &&) = delete;
    Toggle & operator=(Toggle const &) = delete;
    Toggle & operator=(Toggle &&) = delete;

private:

    // The formerly fixed toggles are now ordinary declared toggles with
    // startup defaults and a category, so they carry notification and
    // serialization like any other toggle.
    Toggle()
    {
        m_dynamic_table.declare<bool>("python_redirect", true, ToggleCategory::Ops);
        m_dynamic_table.declare<bool>("show_axis", false, ToggleCategory::Ops);
    }
    Toggle(Toggle const &) = default;

    DynamicToggleTable m_dynamic_table;

}; /* end class Toggle */

class ProcessInfo;

/**
 * Captured command-line state for the current process.
 *
 * It stores the executable basename and the original, populated, and
 * Python argument vectors. Once frozen, set_python_argv is ignored. It
 * also builds the C-style argv buffer used to launch a Python main.
 *
 * @ingroup group_core
 */
// The std::vector data members are default-constructed.
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class CommandLineInfo
{

public:

    CommandLineInfo() = default;
    CommandLineInfo(CommandLineInfo const &) = default;
    // NOLINTNEXTLINE(bugprone-exception-escape)
    CommandLineInfo(CommandLineInfo &&) = default;
    CommandLineInfo & operator=(CommandLineInfo const &) = default;
    CommandLineInfo & operator=(CommandLineInfo &&) = default;
    ~CommandLineInfo() = default;

    std::string const & executable_basename() const { return m_executable_basename; }
    std::vector<std::string> const & populated_argv() const { return m_populated_argv; }
    std::vector<std::string> const & python_argv() const { return m_python_argv; }
    // Assigns to the m_python_argv member, so it cannot be made const.
    // NOLINTNEXTLINE(readability-make-member-function-const)
    void set_python_argv(std::vector<std::string> const & argv)
    {
        if (!m_frozen)
        {
            m_python_argv = argv;
        }
    }

    class PopulatePasskey
    {
        friend ProcessInfo;
    };

    void populate(int argc, char ** argv, PopulatePasskey const &)
    {
        populate(argc, argv, /* repopulate */ false);
    }

    void freeze() { m_frozen = true; }

    bool frozen() const { return m_frozen; }
    bool populated() const { return m_populated; }

    bool python_main() const { return m_python_main; }
    int python_main_argc() const { return m_python_main_argc; }
    char ** python_main_argv_ptr() { return m_python_main_argv_ptr.data(); }

private:

    void unfreeze() { m_frozen = false; }

    void populate(int argc, char ** argv, bool repopulate);

    bool m_frozen = false;
    bool m_populated = false;
    std::string m_executable_basename;
    std::vector<std::string> m_populated_argv;
    std::vector<std::string> m_python_argv;

    bool m_python_main = false;
    int m_python_main_argc = 0;
    SimpleArray<char> m_python_main_argv_char;
    SimpleArray<char *> m_python_main_argv_ptr;

}; /* end class CommandLineInfo */

/**
 * Process-wide information accessed as a singleton.
 *
 * It owns the CommandLineInfo for the process and offers helpers to
 * populate the command line and set environment variables.
 *
 * @ingroup group_core
 */
// NOLINTNEXTLINE(bugprone-exception-escape)
class ProcessInfo
{

public:

    static ProcessInfo & instance();

    ProcessInfo & populate_command_line(int argc, char ** argv)
    {
        m_command_line.populate(argc, argv, CommandLineInfo::PopulatePasskey{});
        return *this;
    }

    ProcessInfo & set_environment_variables();

    CommandLineInfo const & command_line() const { return m_command_line; }
    CommandLineInfo & command_line() { return m_command_line; }

private:

    ProcessInfo();

    CommandLineInfo m_command_line;

}; /* end class ProcessInfo */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

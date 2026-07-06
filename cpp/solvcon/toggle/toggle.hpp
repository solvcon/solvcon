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

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace solvcon
{

int setenv(const char * name, const char * value, int overwrite);

/**
 * A single stored toggle value.
 *
 * Named after the atomic register of shared-memory theory: a single-value
 * shared location. For now it boxes a plain value; a later step makes the
 * scalar read and write atomic without changing this shape.
 *
 * @ingroup group_core
 */
template <typename T>
struct ToggleRegister
{
    T value;
}; /* end struct ToggleRegister */

/**
 * Segmented column of ToggleRegister<T> with stable element addresses.
 *
 * Registers append into a std::deque, so a register keeps its address as
 * the column grows, unlike a reallocating std::vector. A resolved handle
 * can therefore keep pointing at its register while unrelated toggles are
 * added. Indices are dense and start at zero, matching the offsets held by
 * DynamicToggleIndex.
 *
 * @ingroup group_core
 */
template <typename T>
class ToggleRegisterColumn
{

public:

    size_t size() const { return m_registers.size(); }

    ToggleRegister<T> & at(size_t index) { return m_registers.at(index); }
    ToggleRegister<T> const & at(size_t index) const { return m_registers.at(index); }

    size_t append(T const & value)
    {
        size_t const index = m_registers.size();
        m_registers.emplace_back();
        m_registers.back().value = value;
        return index;
    }

    void clear() { m_registers.clear(); }

private:

    std::deque<ToggleRegister<T>> m_registers;

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
    uint64_t generation() const { return m_generation; }

private:

    keymap_type m_key2index;
    ToggleRegisterColumn<bool> m_column_bool;
    ToggleRegisterColumn<int8_t> m_column_int8;
    ToggleRegisterColumn<int16_t> m_column_int16;
    ToggleRegisterColumn<int32_t> m_column_int32;
    ToggleRegisterColumn<int64_t> m_column_int64;
    ToggleRegisterColumn<double> m_column_real;
    ToggleRegisterColumn<std::string> m_column_string;
    uint64_t m_generation = 0;

}; /* end class DynamicToggleTable */

inline DynamicToggleIndex HierarchicalToggleAccess::get_index(std::string const & key) const
{
    return m_table->get_index(rekey(key));
}

#define MM_TOGGLE_SOLID_BOOL(NAME)         \
public:                                    \
    bool NAME() const { return m_##NAME; } \
                                           \
private:                                   \
    bool m_##NAME;

/**
 * Toggles whose values are determined at compile time and read-only
 * during the process lifetime.
 *
 * Each accessor (for example use_pyside) returns a bool member that has
 * an address and cannot be optimized out, unlike a macro or constexpr.
 *
 * @ingroup group_core
 */
class SolidToggle
{

public:

    SolidToggle();

    MM_TOGGLE_SOLID_BOOL(use_pyside)

}; /* end class SolidToggle */

#define MM_TOGGLE_FIXED_BOOL(NAME, DEFAULT)      \
public:                                          \
    bool get_##NAME() const { return m_##NAME; } \
    void set_##NAME(bool v) { m_##NAME = v; }    \
                                                 \
private:                                         \
    bool m_##NAME = DEFAULT;

/**
 * Toggles whose names are fixed at compile time but whose bool values
 * can change at runtime.
 *
 * Access needs no table lookup because the member addresses are known
 * to the compiler and linker (for example python_redirect and
 * show_axis).
 *
 * @ingroup group_core
 */
class FixedToggle
{

public:

    MM_TOGGLE_FIXED_BOOL(python_redirect, true)
    MM_TOGGLE_FIXED_BOOL(show_axis, false)

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

    SolidToggle const & solid() const { return m_solid; }

    FixedToggle const & fixed() const { return m_fixed; }
    FixedToggle & fixed() { return m_fixed; }

    DynamicToggleTable const & dynamic() const { return m_dynamic_table; }
    DynamicToggleTable & dynamic() { return m_dynamic_table; }

    std::vector<std::string> dynamic_keys() const { return m_dynamic_table.keys(); }
    void dynamic_clear() { m_dynamic_table.clear(); }
    uint64_t dynamic_generation() const { return m_dynamic_table.generation(); }
    DynamicToggleIndex get_dynamic_index(std::string const & key) const { return m_dynamic_table.get_index(key); }

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

private:

    Toggle() = default;
    Toggle(Toggle const &) = default;
    Toggle(Toggle &&) = default;
    Toggle & operator=(Toggle const &) = default;
    Toggle & operator=(Toggle &&) = default;

    SolidToggle m_solid;
    FixedToggle m_fixed;
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

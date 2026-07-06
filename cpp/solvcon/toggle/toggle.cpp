/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/toggle/toggle.hpp>
#ifdef SOLVCON_METAL
#include <solvcon/device/metal/metal.hpp>
#endif // SOLVCON_METAL

#include <cstdlib>
#include <cmath>

namespace solvcon
{

int setenv(const char * name, const char * value, int overwrite)
{
#ifdef _WIN32
    int errcode = 0;
    if (!overwrite)
    {
        size_t envsize = 0;
        errcode = getenv_s(&envsize, NULL, 0, name);
        if (errcode || envsize)
        {
            return errcode;
        }
    }
    return _putenv_s(name, value);
#else // _WIN32
    // NOLINTNEXTLINE(concurrency-mt-unsafe) TODO: make the wrapper thread safe.
    return ::setenv(name, value, overwrite);
#endif // _WIN32
}

Toggle & Toggle::instance()
{
    static Toggle o;
    return o;
}

// NOLINTNEXTLINE(bugprone-throwing-static-initialization,fuchsia-statically-constructed-objects,readability-redundant-string-init,cert-err58-cpp)
std::string const DynamicToggleTable::sentinel_string = "";

/* The macro gives debuggers a hard time. Manually expand it if you need to
 * step in a debugger. */
#define MM_DECL_DYNGET(CTYPE, MTYPE, SENTINEL)                                 \
    CTYPE DynamicToggleTable::get_##MTYPE(std::string const & key) const       \
    {                                                                          \
        std::scoped_lock const guard(m_mutex);                                 \
        auto it = m_key2index.find(key);                                       \
        if (it != m_key2index.end())                                           \
        {                                                                      \
            if (it->second.is_##MTYPE())                                       \
            {                                                                  \
                return m_column_##MTYPE.at(it->second.index).get();            \
            }                                                                  \
        }                                                                      \
        return SENTINEL;                                                       \
    }                                                                          \
    CTYPE HierarchicalToggleAccess::get_##MTYPE(std::string const & key) const \
    {                                                                          \
        return m_table->get_##MTYPE(rekey(key));                               \
    }
MM_DECL_DYNGET(bool, bool, false)
MM_DECL_DYNGET(int8_t, int8, 0)
MM_DECL_DYNGET(int16_t, int16, 0)
MM_DECL_DYNGET(int32_t, int32, 0)
MM_DECL_DYNGET(int64_t, int64, 0)
MM_DECL_DYNGET(double, real, std::nan("0"))
MM_DECL_DYNGET(std::string const &, string, sentinel_string)
#undef MM_DECL_DYNGET

/* The macro gives debuggers a hard time. Manually expand it if you need to
 * step in a debugger. */
#define MM_DECL_DYNSET(CTYPE, MTYPE, MTYPEC)                                                                                     \
    /* NOLINTNEXTLINE(bugprone-easily-swappable-parameters) */                                                                   \
    void DynamicToggleTable::set_##MTYPE(std::string const & key, CTYPE value)                                                   \
    {                                                                                                                            \
        bool changed = false;                                                                                                    \
        {                                                                                                                        \
            std::scoped_lock const guard(m_mutex);                                                                               \
            auto it = m_key2index.find(key);                                                                                     \
            if (it != m_key2index.end())                                                                                         \
            {                                                                                                                    \
                if (it->second.is_##MTYPE())                                                                                     \
                {                                                                                                                \
                    auto & reg = m_column_##MTYPE.at(it->second.index);                                                          \
                    if (!value_equal(reg.get(), value))                                                                          \
                    {                                                                                                            \
                        reg.set(value);                                                                                          \
                        changed = true;                                                                                          \
                    }                                                                                                            \
                }                                                                                                                \
            }                                                                                                                    \
            else                                                                                                                 \
            {                                                                                                                    \
                DynamicToggleIndex const idx{static_cast<uint32_t>(m_column_##MTYPE.size()), DynamicToggleIndex::TYPE_##MTYPEC}; \
                m_key2index.insert({key, idx});                                                                                  \
                m_column_##MTYPE.append(value);                                                                                  \
                changed = true;                                                                                                  \
            }                                                                                                                    \
        }                                                                                                                        \
        if (changed)                                                                                                             \
        {                                                                                                                        \
            notify(key);                                                                                                         \
        }                                                                                                                        \
    }                                                                                                                            \
    void HierarchicalToggleAccess::set_##MTYPE(std::string const & key, CTYPE value)                                             \
    {                                                                                                                            \
        m_table->set_##MTYPE(rekey(key), value);                                                                                 \
    }
MM_DECL_DYNSET(bool, bool, BOOL)
MM_DECL_DYNSET(int8_t, int8, INT8)
MM_DECL_DYNSET(int16_t, int16, INT16)
MM_DECL_DYNSET(int32_t, int32, INT32)
MM_DECL_DYNSET(int64_t, int64, INT64)
MM_DECL_DYNSET(double, real, REAL)
MM_DECL_DYNSET(std::string const &, string, STRING)
#undef MM_DECL_DYNSET

HierarchicalToggleAccess HierarchicalToggleAccess::get_subkey(const std::string & key)
{
    return m_table->get_subkey(rekey(key));
}

void HierarchicalToggleAccess::add_subkey(const std::string & key)
{
    m_table->add_subkey(rekey(key));
}

void DynamicToggleTable::add_subkey(std::string const & key)
{
    std::scoped_lock const guard(m_mutex);
    auto it = m_key2index.find(key);
    if (it == m_key2index.end())
    {
        DynamicToggleIndex const index{.index = 0, .type = DynamicToggleIndex::TYPE_SUBKEY};
        m_key2index.insert({key, index});
    }
}

std::vector<std::string> DynamicToggleTable::keys() const
{
    std::scoped_lock const guard(m_mutex);
    std::vector<std::string> ret;
    ret.reserve(m_key2index.size());
    for (auto const & it : m_key2index)
    {
        ret.push_back(it.first);
    }
    return ret;
}

void DynamicToggleTable::clear()
{
    std::scoped_lock const guard(m_mutex);
    m_key2index.clear();
    m_categories.clear();
    m_column_bool.clear();
    m_column_int8.clear();
    m_column_int16.clear();
    m_column_int32.clear();
    m_column_int64.clear();
    m_column_real.clear();
    m_column_string.clear();
    m_generation.fetch_add(1, std::memory_order_relaxed);
}

void DynamicToggleTable::notify(std::string const & key) const
{
    // Copy the callbacks out under the registry mutex, then run them with no
    // lock held so a callback may re-enter the store or the registry.
    std::vector<detail::ToggleChangeObservers::callback_type> callbacks;
    {
        std::scoped_lock const guard(m_observers->mutex);
        auto it = m_observers->map.find(key);
        if (it != m_observers->map.end())
        {
            for (auto const & pr : it->second)
            {
                callbacks.push_back(pr.second);
            }
        }
    }
    for (auto const & callback : callbacks)
    {
        try
        {
            (*callback)();
        }
        // A throwing observer must not corrupt the store or stop the other
        // observers, so the exception is intentionally swallowed here.
        // NOLINTNEXTLINE(bugprone-empty-catch)
        catch (...)
        {
        }
    }
}

// NOLINTNEXTLINE(modernize-use-equals-default) lack of SOLVCON_METAL
ProcessInfo::ProcessInfo()
{
#ifdef SOLVCON_METAL
    solvcon::device::MetalManager::instance();
#endif // SOLVCON_METAL
}

ProcessInfo & ProcessInfo::set_environment_variables()
{
    // No process environment variables need to be set at present.
    return *this;
}

ProcessInfo & ProcessInfo::instance()
{
    static ProcessInfo inst;
    return inst;
}

void CommandLineInfo::populate(int argc, char ** argv, bool repopulate)
{
    if (!m_frozen && (!m_populated || repopulate))
    {
        m_populated_argv.clear();
        m_populated_argv.reserve(argc);
        size_t nchar = 0;
        for (int i = 0; i < argc; ++i)
        {
            m_populated_argv.emplace_back(argv[i]);
            nchar += m_populated_argv.back().size() + 1;
        }
        m_populated = true;
        // Determine Python entry-point mode.
        {
            // Populate char buffer.
            m_python_main_argv_char = SimpleArray<char>(nchar);
            m_python_main_argv_ptr = SimpleArray<char *>(argc);
            m_python_main_argc = 0;
            size_t ichar = 0;
            size_t icnt = 0;
            for (int i = 0; i < argc; ++i)
            {
                std::string const & arg = m_populated_argv[i];
                if ("--mode=python" == arg)
                {
                    m_python_main = true;
                }
                else
                {
                    ++m_python_main_argc;
                    m_python_main_argv_ptr[icnt++] = &m_python_main_argv_char[ichar];
                    for (size_t j = 0; j < arg.size(); ++j)
                    {
                        m_python_main_argv_char[ichar++] = arg[j];
                    }
                    m_python_main_argv_char[ichar++] = 0;
                }
            }
        }
        // Copy to other containers.
        m_python_argv = m_populated_argv;
        // Get executable base name.
        if (!m_populated_argv.empty())
        {
            std::string const & path = m_populated_argv.front();
            if (!path.empty())
            {
                size_t idx = path.find_last_of("/\\");
                idx = (idx < path.size()) ? idx + 1 : 0;
                m_executable_basename = path.substr(idx);
            }
        }
    }
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

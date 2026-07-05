#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * A persistent, searchable ring of console command history.
 *
 * @ingroup group_domain
 */

#include <cstddef>
#include <deque>
#include <string>

namespace solvcon
{

/**
 * A capped ring of committed console commands, persisted to a file and
 * searchable backward for reverse incremental search.
 *
 * The store is intentionally free of Qt so the persistence and search
 * logic can be exercised by the C++ test suite. Commands may span several
 * lines; they are escaped when written so each entry stays on one line in
 * the file.
 *
 * @ingroup group_domain
 */
class RPythonConsoleHistory
{

public:

    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    explicit RPythonConsoleHistory(std::size_t limit = 1024)
        : m_limit(limit)
    {
    }

    void setFilePath(std::string path) { m_path = std::move(path); }
    std::string const & filePath() const { return m_path; }

    std::size_t size() const { return m_commands.size(); }
    bool empty() const { return m_commands.empty(); }
    std::string const & at(std::size_t index) const { return m_commands.at(index); }

    /// Append a command, skipping an empty one or a consecutive duplicate,
    /// cap the ring at the limit, and persist when a file path is set.
    void add(std::string const & command);

    /// Replace the in-memory ring with the content of the file path.
    void load();

    /// Persist the whole ring to the file path, creating parent directories.
    void save() const;

    /// The most recent entry at or before @p from whose text contains
    /// @p query. Returns npos when none matches; an empty query matches the
    /// entry at @p from.
    std::size_t searchBackward(std::string const & query, std::size_t from) const;

private:

    std::deque<std::string> m_commands;
    std::size_t m_limit;
    std::string m_path;

}; /* end class RPythonConsoleHistory */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

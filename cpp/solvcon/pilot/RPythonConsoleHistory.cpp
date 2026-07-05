/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RPythonConsoleHistory.hpp>

#include <filesystem>
#include <fstream>

namespace solvcon
{

namespace
{

// A command may span several lines, so escape backslash and newline to
// keep each entry on one line in the file.
std::string escape(std::string const & text)
{
    std::string out;
    out.reserve(text.size());
    for (char ch : text)
    {
        if (ch == '\\')
        {
            out += "\\\\";
        }
        else if (ch == '\n')
        {
            out += "\\n";
        }
        else
        {
            out += ch;
        }
    }
    return out;
}

std::string unescape(std::string const & text)
{
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i)
    {
        if (text[i] == '\\' && i + 1 < text.size())
        {
            char const next = text[i + 1];
            if (next == 'n')
            {
                out += '\n';
                ++i;
                continue;
            }
            if (next == '\\')
            {
                out += '\\';
                ++i;
                continue;
            }
        }
        out += text[i];
    }
    return out;
}

} /* end namespace */

void RPythonConsoleHistory::add(std::string const & command)
{
    if (command.empty())
    {
        return;
    }
    // Skip a consecutive duplicate so repeating a command does not flood
    // the ring, matching how a shell history behaves.
    if (!m_commands.empty() && m_commands.back() == command)
    {
        return;
    }
    m_commands.push_back(command);
    while (m_commands.size() > m_limit)
    {
        m_commands.pop_front();
    }
    save();
}

void RPythonConsoleHistory::load()
{
    m_commands.clear();
    if (m_path.empty())
    {
        return;
    }
    std::ifstream ifs(m_path);
    if (!ifs)
    {
        return;
    }
    std::string line;
    while (std::getline(ifs, line))
    {
        if (line.empty())
        {
            continue;
        }
        m_commands.push_back(unescape(line));
    }
    while (m_commands.size() > m_limit)
    {
        m_commands.pop_front();
    }
}

void RPythonConsoleHistory::save() const
{
    if (m_path.empty())
    {
        return;
    }
    std::filesystem::path const path(m_path);
    if (path.has_parent_path())
    {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
    }
    std::ofstream ofs(m_path, std::ios::trunc);
    if (!ofs)
    {
        return;
    }
    for (auto const & command : m_commands)
    {
        ofs << escape(command) << '\n';
    }
}

std::size_t RPythonConsoleHistory::searchBackward(std::string const & query, std::size_t from) const
{
    if (m_commands.empty())
    {
        return npos;
    }
    std::size_t index = from < m_commands.size() ? from : m_commands.size() - 1;
    while (true)
    {
        if (m_commands[index].find(query) != std::string::npos)
        {
            return index;
        }
        if (index == 0)
        {
            break;
        }
        --index;
    }
    return npos;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RPythonSyntaxRules.hpp>

#include <cctype>
#include <unordered_set>

namespace solvcon
{

namespace
{

bool isIdentStart(char ch)
{
    return (std::isalpha(static_cast<unsigned char>(ch)) != 0) || ch == '_';
}

bool isIdentChar(char ch)
{
    return (std::isalnum(static_cast<unsigned char>(ch)) != 0) || ch == '_';
}

bool isDigit(char ch)
{
    return std::isdigit(static_cast<unsigned char>(ch)) != 0;
}

std::unordered_set<std::string> const & keywords()
{
    static std::unordered_set<std::string> const set = {
        "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield"};
    return set;
}

std::unordered_set<std::string> const & builtins()
{
    static std::unordered_set<std::string> const set = {
        "abs", "dict", "dir", "enumerate", "float", "getattr", "hasattr", "int", "isinstance", "len", "list", "map", "max", "min", "open", "print", "range", "repr", "set", "sorted", "str", "sum", "tuple", "type", "zip"};
    return set;
}

} /* end namespace */

std::vector<SyntaxSpan> tokenizePython(std::string const & line)
{
    std::vector<SyntaxSpan> spans;
    std::size_t const size = line.size();
    std::size_t i = 0;
    while (i < size)
    {
        char const ch = line[i];
        if (ch == '#')
        {
            spans.push_back({i, size - i, SyntaxTokenKind::Comment});
            break;
        }
        if (ch == '\'' || ch == '"')
        {
            std::size_t const start = i;
            ++i;
            while (i < size)
            {
                if (line[i] == '\\' && i + 1 < size)
                {
                    i += 2;
                    continue;
                }
                if (line[i] == ch)
                {
                    ++i;
                    break;
                }
                ++i;
            }
            spans.push_back({start, i - start, SyntaxTokenKind::String});
            continue;
        }
        if (isIdentStart(ch))
        {
            std::size_t const start = i;
            while (i < size && isIdentChar(line[i]))
            {
                ++i;
            }
            std::string const word = line.substr(start, i - start);
            if (keywords().count(word) != 0)
            {
                spans.push_back({start, i - start, SyntaxTokenKind::Keyword});
            }
            else if (builtins().count(word) != 0)
            {
                spans.push_back({start, i - start, SyntaxTokenKind::Builtin});
            }
            continue;
        }
        if (isDigit(ch))
        {
            std::size_t const start = i;
            while (i < size && (isIdentChar(line[i]) || line[i] == '.'))
            {
                ++i;
            }
            spans.push_back({start, i - start, SyntaxTokenKind::Number});
            continue;
        }
        ++i;
    }
    return spans;
}

std::string nextLineIndent(std::string const & line)
{
    std::size_t lead = 0;
    while (lead < line.size() && (line[lead] == ' ' || line[lead] == '\t'))
    {
        ++lead;
    }
    std::string indent = line.substr(0, lead);

    std::size_t end = line.size();
    while (end > 0 && (line[end - 1] == ' ' || line[end - 1] == '\t'))
    {
        --end;
    }
    if (end > 0 && line[end - 1] == ':')
    {
        indent += "    ";
    }
    return indent;
}

std::size_t matchBracket(std::string const & text, std::size_t pos)
{
    if (pos >= text.size())
    {
        return syntax_npos;
    }

    std::string const opens = "([{";
    std::string const closes = ")]}";
    char const ch = text[pos];

    std::size_t const open_at = opens.find(ch);
    if (open_at != std::string::npos)
    {
        char const close = closes[open_at];
        int depth = 0;
        for (std::size_t i = pos; i < text.size(); ++i)
        {
            if (text[i] == ch)
            {
                ++depth;
            }
            else if (text[i] == close)
            {
                --depth;
                if (depth == 0)
                {
                    return i;
                }
            }
        }
        return syntax_npos;
    }

    std::size_t const close_at = closes.find(ch);
    if (close_at != std::string::npos)
    {
        char const open = opens[close_at];
        int depth = 0;
        for (std::size_t i = pos + 1; i-- > 0;)
        {
            if (text[i] == ch)
            {
                ++depth;
            }
            else if (text[i] == open)
            {
                --depth;
                if (depth == 0)
                {
                    return i;
                }
            }
        }
        return syntax_npos;
    }

    return syntax_npos;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

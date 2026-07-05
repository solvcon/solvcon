#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Qt-free lexical rules for the console editor: single-line Python
 * tokenizing for syntax highlighting, next-line indentation, and bracket
 * matching. Kept free of Qt so the logic can be tested without a display.
 *
 * @ingroup group_domain
 */

#include <cstddef>
#include <string>
#include <vector>

namespace solvcon
{

/// The lexical category a highlighter paints.
enum class SyntaxTokenKind
{
    Keyword,
    Builtin,
    String,
    Comment,
    Number
};

/// A half-open range [start, start + length) of one category on a line.
struct SyntaxSpan
{
    std::size_t start;
    std::size_t length;
    SyntaxTokenKind kind;
};

constexpr std::size_t syntax_npos = static_cast<std::size_t>(-1);

/// Tokenize one line of Python for highlighting. Strings and comments are
/// single-line, which is enough for a command editor.
std::vector<SyntaxSpan> tokenizePython(std::string const & line);

/// The indentation that opens the next line: the current line's leading
/// whitespace, plus one four-space level when the line ends with a colon.
std::string nextLineIndent(std::string const & line);

/// The index of the bracket matching the one at @p pos, or syntax_npos when
/// @p pos is not a bracket or has no match. Counts nesting per bracket type
/// and is not string-aware.
std::size_t matchBracket(std::string const & text, std::size_t pos);

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

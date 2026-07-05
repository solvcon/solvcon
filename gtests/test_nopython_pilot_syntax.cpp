/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RPythonSyntaxRules.hpp>

#include <string>

#include <gtest/gtest.h>

using solvcon::matchBracket;
using solvcon::nextLineIndent;
using solvcon::syntax_npos;
using solvcon::SyntaxSpan;
using solvcon::SyntaxTokenKind;
using solvcon::tokenizePython;

namespace
{

bool hasSpan(std::vector<SyntaxSpan> const & spans, std::size_t start, std::size_t length, SyntaxTokenKind kind)
{
    for (auto const & span : spans)
    {
        if (span.start == start && span.length == length && span.kind == kind)
        {
            return true;
        }
    }
    return false;
}

} /* end namespace */

TEST(PilotSyntaxTokenize, MarksKeywordStringComment)
{
    std::string const line = "for x in y:  # loop";
    auto const spans = tokenizePython(line);
    EXPECT_TRUE(hasSpan(spans, 0, 3, SyntaxTokenKind::Keyword)); // for
    EXPECT_TRUE(hasSpan(spans, 6, 2, SyntaxTokenKind::Keyword)); // in
    EXPECT_TRUE(hasSpan(spans, 13, 6, SyntaxTokenKind::Comment)); // # loop
}

TEST(PilotSyntaxTokenize, MarksStringAndBuiltinAndNumber)
{
    std::string const line = "print('a', 42)";
    auto const spans = tokenizePython(line);
    EXPECT_TRUE(hasSpan(spans, 0, 5, SyntaxTokenKind::Builtin)); // print
    EXPECT_TRUE(hasSpan(spans, 6, 3, SyntaxTokenKind::String)); // 'a'
    EXPECT_TRUE(hasSpan(spans, 11, 2, SyntaxTokenKind::Number)); // 42
}

TEST(PilotSyntaxTokenize, HandlesEscapedQuoteInString)
{
    std::string const line = R"(s = 'a\'b')";
    auto const spans = tokenizePython(line);
    // The string spans from the opening quote to the final closing quote,
    // skipping the escaped quote in the middle.
    EXPECT_TRUE(hasSpan(spans, 4, 6, SyntaxTokenKind::String));
}

TEST(PilotSyntaxTokenize, DoesNotMarkAKeywordSubstring)
{
    // "format" contains "for" but is not the keyword.
    auto const spans = tokenizePython("format");
    EXPECT_TRUE(spans.empty());
}

TEST(PilotSyntaxIndent, KeepsIndentAndOpensAfterColon)
{
    EXPECT_EQ(nextLineIndent("    x = 1"), "    ");
    EXPECT_EQ(nextLineIndent("    if x:"), "        ");
    EXPECT_EQ(nextLineIndent("for i in range(3):  "), "    ");
    EXPECT_EQ(nextLineIndent("plain"), "");
}

TEST(PilotSyntaxBracket, MatchesForwardBackwardAndNested)
{
    std::string const text = "f(a, (b, c))";
    EXPECT_EQ(matchBracket(text, 1), 11u); // outer ( -> last )
    EXPECT_EQ(matchBracket(text, 11), 1u); // last ) -> outer (
    EXPECT_EQ(matchBracket(text, 5), 10u); // inner ( -> inner )
}

TEST(PilotSyntaxBracket, ReturnsNposForUnmatchedOrNonBracket)
{
    EXPECT_EQ(matchBracket("(a", 0), syntax_npos);
    EXPECT_EQ(matchBracket("ab", 0), syntax_npos);
    EXPECT_EQ(matchBracket("()", 5), syntax_npos);
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

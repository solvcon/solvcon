/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/console/RPythonConsoleHistory.hpp>

#include <filesystem>
#include <string>

#include <gtest/gtest.h>

using solvcon::RPythonConsoleHistory;

namespace
{

std::string temp_path(char const * name)
{
    return (std::filesystem::temp_directory_path() / name).string();
}

} /* end namespace */

TEST(PilotConsoleHistory, AddSkipsEmptyAndConsecutiveDuplicate)
{
    RPythonConsoleHistory hist;
    hist.add("");
    hist.add("a");
    hist.add("a");
    hist.add("b");
    ASSERT_EQ(hist.size(), 2u);
    EXPECT_EQ(hist.at(0), "a");
    EXPECT_EQ(hist.at(1), "b");
}

TEST(PilotConsoleHistory, CapsAtLimit)
{
    RPythonConsoleHistory hist(3);
    hist.add("1");
    hist.add("2");
    hist.add("3");
    hist.add("4");
    ASSERT_EQ(hist.size(), 3u);
    EXPECT_EQ(hist.at(0), "2");
    EXPECT_EQ(hist.at(2), "4");
}

TEST(PilotConsoleHistory, RoundTripsThroughFileWithMultilineAndBackslash)
{
    std::string const path = temp_path("solvcon_hist_roundtrip.txt");
    std::filesystem::remove(path);

    std::string const multiline = "for i in range(2):\n    print(i)";
    std::string const backslash = R"(path = 'a\b')";
    {
        RPythonConsoleHistory hist;
        hist.setFilePath(path);
        hist.add("x = 1");
        hist.add(multiline);
        hist.add(backslash);
    }

    RPythonConsoleHistory loaded;
    loaded.setFilePath(path);
    loaded.load();
    ASSERT_EQ(loaded.size(), 3u);
    EXPECT_EQ(loaded.at(0), "x = 1");
    EXPECT_EQ(loaded.at(1), multiline);
    EXPECT_EQ(loaded.at(2), backslash);

    std::filesystem::remove(path);
}

TEST(PilotConsoleHistory, SearchBackwardFindsMostRecentMatch)
{
    RPythonConsoleHistory hist;
    hist.add("import os");
    hist.add("mesh = load()");
    hist.add("import sys");
    hist.add("print(mesh)");

    std::size_t index = hist.searchBackward("import", hist.size() - 1);
    EXPECT_EQ(index, 2u);
    index = hist.searchBackward("import", index - 1);
    EXPECT_EQ(index, 0u);
    EXPECT_EQ(hist.searchBackward("mesh", hist.size() - 1), 3u);
    EXPECT_EQ(
        hist.searchBackward("zzz", hist.size() - 1),
        RPythonConsoleHistory::npos);
}

TEST(PilotConsoleHistory, EmptyQueryMatchesAtClampedFrom)
{
    RPythonConsoleHistory hist;
    hist.add("a");
    hist.add("b");
    hist.add("c");
    EXPECT_EQ(hist.searchBackward("", 1), 1u);
    EXPECT_EQ(hist.searchBackward("", 100), 2u);
}

TEST(PilotConsoleHistory, SearchBackwardOnEmptyReturnsNpos)
{
    RPythonConsoleHistory hist;
    EXPECT_EQ(hist.searchBackward("x", 0), RPythonConsoleHistory::npos);
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

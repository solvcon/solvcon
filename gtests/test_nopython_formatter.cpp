/*
 * Copyright (c) 2025, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

// Intended to use deprecated Formatter as the demo of comparison with std::format
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <solvcon/base.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <format>
#include <string>

#ifdef Py_PYTHON_H
#error "Python.h should not be included."
#endif

TEST(Formatter, basic_usage)
{
    solvcon::Formatter formatter;
    formatter << "Hello, "
              << "World!"
              << " " << 42;
    std::string formatter_result = formatter.str();

    std::string format_result = std::format("Hello, {}! {}", "World", 42);

    EXPECT_EQ(formatter_result, "Hello, World! 42");
    EXPECT_EQ(format_result, "Hello, World! 42");
    EXPECT_EQ(formatter_result, format_result);
}

TEST(Formatter, numeric_types)
{
    solvcon::Formatter formatter;
    formatter << "int: " << 123 << ", double: " << 3.14159 << ", bool: " << true;
    std::string formatter_result = formatter.str();

    std::string format_result = std::format("int: {}, double: {}, bool: {}", 123, 3.14159, true);

    EXPECT_EQ(formatter_result, "int: 123, double: 3.14159, bool: 1");
    EXPECT_EQ(format_result, "int: 123, double: 3.14159, bool: true");
}

TEST(Formatter, chaining)
{
    std::string formatter_result = (solvcon::Formatter() << "x = " << 10 << ", y = " << 20).str();

    std::string format_result = std::format("x = {}, y = {}", 10, 20);

    EXPECT_EQ(formatter_result, "x = 10, y = 20");
    EXPECT_EQ(format_result, "x = 10, y = 20");
    EXPECT_EQ(formatter_result, format_result);
}

TEST(Formatter, conversion_operator)
{
    std::string formatter_result = solvcon::Formatter() << "Test " << 123;

    std::string format_result = std::format("Test {}", 123);

    EXPECT_EQ(formatter_result, "Test 123");
    EXPECT_EQ(format_result, "Test 123");
    EXPECT_EQ(formatter_result, format_result);
}

TEST(StdFormat, basic_usage)
{
    std::string format_result = std::format("Hello, {}! {}", "World", 42);

    std::string formatter_result = (solvcon::Formatter() << "Hello, "
                                                         << "World!"
                                                         << " " << 42)
                                       .str();

    EXPECT_EQ(format_result, "Hello, World! 42");
    EXPECT_EQ(formatter_result, "Hello, World! 42");
    EXPECT_EQ(format_result, formatter_result);
}

TEST(StdFormat, numeric_types)
{
    std::string format_result = std::format("int: {}, double: {}, bool: {}", 123, 3.14159, true);

    solvcon::Formatter formatter;
    formatter << "int: " << 123 << ", double: " << 3.14159 << ", bool: " << true;
    std::string formatter_result = formatter.str();

    EXPECT_EQ(format_result, "int: 123, double: 3.14159, bool: true");
    EXPECT_EQ(formatter_result, "int: 123, double: 3.14159, bool: 1");
}

TEST(StdFormat, formatting_options)
{
    std::string format_result = std::format("hex: {:#x}, precision: {:.2f}", 255, 3.14159);

    std::string formatter_result = (solvcon::Formatter() << "hex: " << 255 << ", precision: " << 3.14159).str();

    EXPECT_EQ(format_result, "hex: 0xff, precision: 3.14");
    EXPECT_EQ(formatter_result, "hex: 255, precision: 3.14159");
}

TEST(FormatterVsStdFormat, simple_string_comparison)
{
    std::string formatter_result = (solvcon::Formatter() << "Value: " << 42).str();
    std::string format_result = std::format("Value: {}", 42);

    EXPECT_EQ(formatter_result, format_result);
}

TEST(FormatterVsStdFormat, multiple_values_comparison)
{
    std::string formatter_result = (solvcon::Formatter() << "x = " << 10 << ", y = " << 20 << ", z = " << 30).str();
    std::string format_result = std::format("x = {}, y = {}, z = {}", 10, 20, 30);

    EXPECT_EQ(formatter_result, format_result);
}

TEST(FormatterVsStdFormat, performance_formatter)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::string result;
    for (int32_t i = 0; i < 10000; ++i)
    {
        result = (solvcon::Formatter() << "Iteration: " << i << ", Value: " << i * 2).str();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Formatter time: " << duration.count() << " microseconds" << std::endl;

    EXPECT_FALSE(result.empty());
}

TEST(FormatterVsStdFormat, performance_std_format)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::string result;
    for (int32_t i = 0; i < 10000; ++i)
    {
        result = std::format("Iteration: {}, Value: {}", i, i * 2);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "std::format time: " << duration.count() << " microseconds" << std::endl;

    EXPECT_FALSE(result.empty());
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

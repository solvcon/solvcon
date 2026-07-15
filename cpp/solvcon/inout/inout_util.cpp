/*
 * Copyright (c) 2024, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/inout/inout_util.hpp>

namespace solvcon
{

namespace inout
{

std::vector<std::string> tokenize(const std::string & str, const std::regex & regex_delim)
{
    std::vector<std::string> output;
    std::sregex_iterator iter(str.begin(), str.end(), regex_delim);
    const std::sregex_iterator end;
    while (iter != end)
    {
        output.push_back(iter->str());
        ++iter;
    }
    return output;
}

std::vector<std::string> tokenize(const std::string & str, const char delim)
{
    std::vector<std::string> output;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delim))
    {
        output.push_back(token);
    }
    return output;
}

} /* end namespace inout */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

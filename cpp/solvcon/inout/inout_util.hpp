#pragma once

/*
 * Copyright (c) 2024, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <regex>
#include <string>
#include <sstream>
#include <vector>

#include <solvcon/buffer/buffer.hpp>

namespace solvcon
{

namespace inout
{

std::vector<std::string> tokenize(const std::string & str, const std::regex & regex_delim);
std::vector<std::string> tokenize(const std::string & str, char delim);

} /* end namespace inout */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

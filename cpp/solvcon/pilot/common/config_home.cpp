/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/common/config_home.hpp> // Must be the first include.

#include <string>

#include <QDir>

namespace solvcon
{

QString configPath(QString const & name)
{
    pybind11::gil_scoped_acquire const gil;
    std::string const home = pybind11::module_::import("solvcon.config").attr("Config").attr("config_home")().cast<std::string>();
    return QDir(QString::fromStdString(home)).filePath(name);
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

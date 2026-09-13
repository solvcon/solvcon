/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/app/RThreadManager.hpp>

namespace solvcon
{

struct RThreadManager::Scheduler
{
    ThreadStateFactory factory;
}; /* end struct RThreadManager::Scheduler */

// TODO(#1527): the prototype implements no scheduling.
struct RThreadManager::Impl
{
    std::unordered_map<std::string, Scheduler> schedulers;
}; /* end struct RThreadManager::Impl */

RThreadManager::RThreadManager(QObject * parent)
    : QObject(parent)
    , m_impl(std::make_unique<Impl>())
{
}

RThreadManager::~RThreadManager() = default;

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

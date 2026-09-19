#pragma once

/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Shared PySide conversion declarations and the pybind11 QPointer holder.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <QObject>
#include <QPointer>

#ifdef SOLVCON_PYSIDE6_FULL
#include <pyside.h>
#else
namespace PySide
{

PyTypeObject * getTypeForQObject(const QObject * cpp_self);
PyObject * getWrapperForQObject(QObject * cpp_self, PyTypeObject * type);
QObject * convertToQObject(PyObject * object, bool raise_error);

} /* end namespace PySide */
#endif

PYBIND11_DECLARE_HOLDER_TYPE(T, QPointer<T>);

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

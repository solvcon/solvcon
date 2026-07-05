/*
 * Copyright (c) 2019, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/python/common.hpp> // Must be the first include.
#include <pybind11/stl.h> // Required for automatic conversion.

#include <solvcon/toggle/toggle.hpp>

#include <utility>

#include <pybind11/numpy.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace solvcon
{

namespace python
{

void import_numpy()
{
    auto local_import_numpy = []()
    {
        import_array2("cannot import numpy", false); // or numpy c api segfault.
        return true;
    };
    if (!local_import_numpy())
    {
        throw pybind11::error_already_set();
    }
}

Interpreter & Interpreter::instance()
{
    static Interpreter o;
    return o;
}

Interpreter & Interpreter::initialize()
{
    if (nullptr == m_interpreter)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
        m_interpreter = new pybind11::scoped_interpreter(true, 0, nullptr, true);
    }
    return *this;
}

Interpreter & Interpreter::finalize()
{
    if (nullptr != m_interpreter)
    {
        // Py_Main and Py_BytesMain may finalize the interpreter before this is reached.
        if (0 != Py_IsInitialized())
        {
            delete m_interpreter;
        }
        m_interpreter = nullptr;
    }
    return *this;
}

Interpreter & Interpreter::setup_modmesh_path()
{
    // The hard-coded Python in C++ is difficult to debug. Any better way?
    std::string const cmd = R""""(def _set_modmesh_path():
    import os
    import sys
    filename = os.path.join('solvcon', '__init__.py')
    path = os.getcwd()
    while True:
        if os.path.exists(os.path.join(path, filename)):
            break
        if path == os.path.dirname(path):
            path = None
            break
        else:
            path = os.path.dirname(path)
    if path:
        sys.path.insert(0, path)
_set_modmesh_path())"""";
    try
    {
        pybind11::exec(cmd);
    }
    catch (const pybind11::error_already_set & e)
    {
        std::cerr << e.what() << std::endl;
    }
    return *this;
}

Interpreter & Interpreter::setup_process()
{
    CommandLineInfo const & cmdinfo = ProcessInfo::instance().command_line();
    try
    {
        // NOLINTNEXTLINE(misc-const-correctness)
        pybind11::object mod_sys = pybind11::module_::import("solvcon.system");
        mod_sys.attr("setup_process")(cmdinfo.python_argv());
    }
    catch (const pybind11::error_already_set & e)
    {
        std::cerr << e.what() << std::endl;
    }
    return *this;
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
int Interpreter::enter_main()
{
    int ret = -1;
    CommandLineInfo const & cmdinfo = ProcessInfo::instance().command_line();
    try
    {
        // NOLINTNEXTLINE(misc-const-correctness)
        pybind11::object mod_sys = pybind11::module_::import("solvcon.system");
        ret = pybind11::cast<int>(mod_sys.attr("enter_main")(cmdinfo.python_argv()));
    }
    catch (const pybind11::error_already_set & e)
    {
        std::cerr << e.what() << std::endl;
    }
    return ret;
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static): implicitly requires m_interpreter
void Interpreter::preload_module(std::string const & name)
{
    namespace py = pybind11;

    std::cerr << "Loading " << name << " ... ";
    bool load_failure = false;
    try
    {
        py::module_::import(name.c_str());
    }
    catch (const py::error_already_set & e)
    {
        if (e.matches(PyExc_ModuleNotFoundError))
        {
            throw;
        }
        else // NOLINT(llvm-else-after-return,readability-else-after-return)
        {
            std::cerr << "fails" << std::endl;
            load_failure = true;
        }
    }
    if (!load_failure)
    {
        std::cerr << "succeeds" << std::endl;
        // Load into the namespace.
        std::ostringstream ms;
        ms << "import " << name;
        py::exec(ms.str());
    }
}

void Interpreter::preload_modules(std::vector<std::string> const & names)
{
    for (auto const & name : names)
    {
        preload_module(name);
    }
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void Interpreter::exec_code(std::string const & code)
{
    try
    {
        // NOLINTNEXTLINE(misc-const-correctness)
        pybind11::object mod_sys = pybind11::module_::import("solvcon.system");
        mod_sys.attr("exec_code")(code);
    }
    catch (const pybind11::error_already_set & e)
    {
        std::cerr << e.what() << std::endl;
    }
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
std::vector<std::string> Interpreter::get_completions(std::string const & text)
{
    std::vector<std::string> result;
    try
    {
        pybind11::gil_scoped_acquire const gil;
        // NOLINTNEXTLINE(misc-const-correctness)
        pybind11::object mod_sys = pybind11::module_::import("solvcon.system");
        pybind11::object const py_result = mod_sys.attr("get_completions")(text);
        result = py_result.cast<std::vector<std::string>>();
    }
    catch (const pybind11::error_already_set & e)
    {
        std::cerr << e.what() << std::endl;
    }
    catch (const std::exception & e)
    {
        std::cerr << "get_completions error: " << e.what() << std::endl;
    }
    return result;
}

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
std::string Interpreter::get_call_tip(std::string const & text)
{
    std::string result;
    try
    {
        pybind11::gil_scoped_acquire const gil;
        // NOLINTNEXTLINE(misc-const-correctness)
        pybind11::object mod_sys = pybind11::module_::import("solvcon.system");
        pybind11::object const py_result = mod_sys.attr("get_call_tip")(text);
        result = py_result.cast<std::string>();
    }
    catch (const pybind11::error_already_set & e)
    {
        std::cerr << e.what() << std::endl;
    }
    catch (const std::exception & e)
    {
        std::cerr << "get_call_tip error: " << e.what() << std::endl;
    }
    return result;
}

PythonStreamRedirect & PythonStreamRedirect::activate()
{
    if (is_enabled())
    {
        auto sys_module = pybind11::module::import("sys");
        // Back up old streams.
        if (!static_cast<bool>(m_stdout_backup))
        {
            m_stdout_backup = sys_module.attr("stdout");
        }
        if (!static_cast<bool>(m_stderr_backup))
        {
            m_stderr_backup = sys_module.attr("stderr");
        }

        // Create string-IO objects. Other file-like object can be used here as
        // well, such as objects created by pybind11.
        {
            auto string_io = pybind11::module::import("io").attr("StringIO");
            m_stdout_buffer = string_io();
            sys_module.attr("stdout") = m_stdout_buffer;
            m_stderr_buffer = string_io();
            sys_module.attr("stderr") = m_stderr_buffer;
        }
    }

    return *this;
}

PythonStreamRedirect & PythonStreamRedirect::deactivate()
{
    auto sys_module = pybind11::module::import("sys");
    if (static_cast<bool>(m_stdout_backup))
    {
        sys_module.attr("stdout") = m_stdout_backup;
        m_stdout_backup.release().dec_ref();
    }
    if (static_cast<bool>(m_stderr_backup))
    {
        sys_module.attr("stderr") = m_stderr_backup;
        m_stderr_backup.release().dec_ref();
    }
    return *this;
}

std::string PythonStreamRedirect::stdout_string() const
{
    m_stdout_buffer.attr("seek")(0);
    return pybind11::str(m_stdout_buffer.attr("read")());
}

std::string PythonStreamRedirect::stderr_string() const
{
    m_stderr_buffer.attr("seek")(0);
    return pybind11::str(m_stderr_buffer.attr("read")());
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

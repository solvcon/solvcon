/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/stl.h> // Must be the first include.

#include <solvcon/pilot/app/RThreadManager.hpp>
#include <solvcon/pilot/common/pyside.hpp>
#include <solvcon/pilot/wrap_pilot.hpp>

#include <memory>
#include <string>
#include <utility>

namespace solvcon::python
{

namespace
{

namespace py = pybind11;

QObject * to_qobject(py::object const & object)
{
    if (object.is_none())
    {
        throw py::value_error("owner must not be None");
    }
    QObject * owner = PySide::convertToQObject(object.ptr(), true);
    if (owner == nullptr)
    {
        if (PyErr_Occurred())
        {
            throw py::error_already_set();
        }
        throw py::type_error("owner must be a PySide QObject");
    }
    return owner;
}

/// Take the GIL around the call; PythonResult takes it in the destructor.
template <typename Arg>
std::function<void(Arg)> gil_callback(py::function callback)
{
    auto holder = std::make_shared<PythonResult>(std::move(callback));
    return [holder](Arg arg)
    {
        if (Py_IsInitialized() == 0)
        {
            return;
        }
        py::gil_scoped_acquire const gil;
        try
        {
            holder->object()(std::move(arg));
        }
        catch (py::error_already_set & error)
        {
            error.discard_as_unraisable("Pilot workflow callback");
        }
    };
}

py::object succeeded_result(Succeeded const & self)
{
    auto const * held = std::any_cast<std::shared_ptr<PythonResult>>(&self.result);
    return held != nullptr ? (*held)->object() : py::none();
}

} /* end namespace */

void wrap_thread_manager(pybind11::module & mod)
{
    py::enum_<WorkflowState>(mod, "WorkflowState")
        .value("QUEUED", WorkflowState::Queued)
        .value("RUNNING", WorkflowState::Running)
        .value("FINISHED", WorkflowState::Finished);

    py::class_<Error>(mod, "Error")
        .def(py::init<std::string, std::string>(), py::arg("kind"), py::arg("message"))
        .def_readonly("kind", &Error::kind)
        .def_readonly("message", &Error::message);

    py::class_<Succeeded>(mod, "Succeeded")
        .def(
            py::init([](py::object result)
                     { return Succeeded{.result = std::make_shared<PythonResult>(std::move(result))}; }),
            py::arg("result") = py::none())
        .def_property_readonly("result", &succeeded_result);

    py::class_<Failed>(mod, "Failed")
        .def(py::init<Error>(), py::arg("error"))
        .def_readonly("error", &Failed::error);

    py::class_<Cancelled>(mod, "Cancelled").def(py::init<>());

    py::class_<WorkflowContext>(mod, "WorkflowContext")
        .def_property_readonly("workflow_id", &WorkflowContext::workflow_id)
        .def("finish", &WorkflowContext::finish, py::arg("result"));

    py::class_<RWorkflowHandle, QPointer<RWorkflowHandle>>(mod, "WorkflowHandle")
        .def_property_readonly("workflow_id", &RWorkflowHandle::workflowId)
        .def_property_readonly("state", &RWorkflowHandle::state)
        .def(
            "on_state_changed",
            [](RWorkflowHandle & self, py::function callback)
            { self.onStateChanged(gil_callback<WorkflowState>(std::move(callback))); },
            py::arg("callback"))
        .def(
            "on_finished",
            [](RWorkflowHandle & self, py::function callback)
            { self.onFinished(gil_callback<Result>(std::move(callback))); },
            py::arg("callback"));

    py::class_<RThreadManager, QPointer<RThreadManager>>(mod, "RThreadManager")
        .def(
            "submit",
            [](RThreadManager & self, py::object workflow, py::object const & owner)
            { return self.submit(std::make_unique<PythonWorkflow>(std::move(workflow)), to_qobject(owner)); },
            py::arg("workflow"),
            py::kw_only(),
            py::arg("owner"));
}

} /* end namespace solvcon::python */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

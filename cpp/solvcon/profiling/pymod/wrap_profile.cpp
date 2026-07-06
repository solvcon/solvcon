/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/profiling/pymod/profiling_pymod.hpp> // Must be the first include.
#include <solvcon/profiling/SerializableProfiler.hpp>
#include <solvcon/solvcon.hpp>
#include <queue>
#include <unordered_map>

namespace solvcon
{

namespace python
{

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapWrapperProfilerStatus
    : public WrapBase<WrapWrapperProfilerStatus, WrapperProfilerStatus>
{

public:

    friend root_base_type;

protected:

    WrapWrapperProfilerStatus(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            // clang-format off
            .def_property_readonly_static("me", [](py::object const &) -> wrapped_type& { return wrapped_type::me(); })
            // clang-format on
            .def_property_readonly("enabled", &wrapped_type::enabled)
            .def("enable", &wrapped_type::enable)
            .def("disable", &wrapped_type::disable)
            //
            ;

        mod.attr("wrapper_profiler_status") = mod.attr("WrapperProfilerStatus").attr("me");
    }

}; /* end class WrapWrapperTimerStatus */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapStopWatch
    : public WrapBase<WrapStopWatch, StopWatch>
{

public:

    friend root_base_type;

protected:

    WrapStopWatch(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .def_property_readonly_static(
                "me",
                [](py::object const &) -> wrapped_type &
                { return wrapped_type::me(); })
            .def("lap", &wrapped_type::lap)
            .def_property_readonly("duration", &wrapped_type::duration)
            .def_property_readonly_static(
                "resolution",
                [](py::object const &)
                { return wrapped_type::resolution(); })
            //
            ;

        mod.attr("stop_watch") = mod.attr("StopWatch").attr("me");
    }

}; /* end class WrapStopWatch */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapCallProfiler : public WrapBase<WrapCallProfiler, CallProfiler>
{
public:
    friend root_base_type;
    static pybind11::dict result(CallProfiler & profiler);

protected:
    WrapCallProfiler(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .def_property_readonly_static(
                "instance",
                [](py::object const &) -> wrapped_type &
                { return wrapped_type::instance(); })
            .def("result", &result)
            .def("reset", &wrapped_type::reset)
            .def("serialize", &CallProfilerSerializer::serialize);

        mod.attr("call_profiler") = mod.attr("CallProfiler").attr("instance");
    }
}; /* end class WrapCallProfiler */

pybind11::dict WrapCallProfiler::result(CallProfiler & profiler)
{
    namespace py = pybind11;

    const RadixTreeNode<CallerProfile> * root = profiler.radix_tree().get_root();
    if (root->empty_children())
    {
        return {};
    }
    py::dict result;
    std::queue<const RadixTreeNode<CallerProfile> *> node_queue;
    std::unordered_map<const RadixTreeNode<CallerProfile> *, py::dict> dict_storage;

    node_queue.push(root);
    dict_storage[root] = result;

    while (!node_queue.empty())
    {
        const RadixTreeNode<CallerProfile> * cur_node = node_queue.front();
        const py::dict & current_dict = dict_storage[cur_node];
        node_queue.pop();

        current_dict["name"] = cur_node->name();
        current_dict["total_time"] = cur_node->data().total_time.count() / 1e6;
        current_dict["count"] = cur_node->data().call_count;
        if (cur_node == profiler.radix_tree().get_current_node())
        {
            current_dict["current_node"] = true;
        }

        py::list children_list;
        for (const auto & child : cur_node->children())
        {
            dict_storage[child.get()] = py::dict();
            py::dict & child_dict = dict_storage[child.get()];
            children_list.append(child_dict);
            node_queue.push(child.get());
        }
        current_dict["children"] = children_list;
    }
    return result;
}

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapCallProfilerProbe : public WrapBase<WrapCallProfilerProbe, CallProfilerProbe>
{
public:
    friend root_base_type;

protected:
    WrapCallProfilerProbe(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .def(
                py::init(
                    [](const char * caller_name)
                    {
                        return std::make_unique<CallProfilerProbe>(CallProfiler::instance(), caller_name);
                    }),
                py::arg("caller_name"))
            .def("cancel", &wrapped_type::cancel);
    }
}; /* end class WrapCallProfilerProbe */

void wrap_profile(pybind11::module & mod)
{
    WrapWrapperProfilerStatus::commit(mod, "WrapperProfilerStatus", "WrapperProfilerStatus");
    WrapStopWatch::commit(mod, "StopWatch", "StopWatch");
    WrapCallProfiler::commit(mod, "CallProfiler", "CallProfiler");
    WrapCallProfilerProbe::commit(mod, "CallProfilerProbe", "CallProfilerProbe");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/python/python.hpp> // Must be the first include.

#include <solvcon/buffer/pymod/buffer_pymod.hpp>
#include <solvcon/inout/pymod/inout_pymod.hpp>
#include <solvcon/mesh/pymod/mesh_pymod.hpp>
#include <solvcon/onedim/pymod/onedim_pymod.hpp>
#include <solvcon/multidim/pymod/multidim_pymod.hpp>
#include <solvcon/python/module.hpp>
#include <solvcon/spacetime/pymod/spacetime_pymod.hpp>
#include <solvcon/profiling/pymod/profiling_pymod.hpp>
#include <solvcon/toggle/pymod/toggle_pymod.hpp>
#include <solvcon/universe/pymod/universe_pymod.hpp>
#include <solvcon/math/pymod/math_pymod.hpp>
#include <solvcon/transform/pymod/transform_pymod.hpp>
#include <solvcon/linalg/pymod/linalg_pymod.hpp>
#include <solvcon/oasis/pymod/oasis_pymod.hpp>

#ifdef QT_CORE_LIB
#include <solvcon/pilot/wrap_pilot.hpp>
#endif // QT_CORE_LIB

namespace solvcon
{

namespace python
{

void initialize(pybind11::module_ mod)
{
    initialize_toggle(mod);
    initialize_profiling(mod);
    initialize_buffer(mod);
    initialize_universe(mod);
    initialize_mesh(mod);
    initialize_multidim(mod);
    initialize_inout(mod);
    initialize_math(mod);
    initialize_linalg(mod);
    pybind11::module_ spacetime_mod = mod.def_submodule("spacetime", "spacetime");
    initialize_spacetime(spacetime_mod);
    pybind11::module_ onedim_mod = mod.def_submodule("onedim", "onedim");
    initialize_onedim(onedim_mod);
    initialize_transform(mod);
    initialize_oasis(mod);

#ifdef QT_CORE_LIB
    mod.attr("HAS_PILOT") = true;
    pybind11::module_ view_mod = mod.def_submodule("pilot", "pilot");
    initialize_pilot(view_mod);
#else // QT_CORE_LIB
    mod.attr("HAS_PILOT") = false;
#endif // QT_CORE_LIB
}

int program_entrance(int argc, char ** argv)
{
    ProcessInfo::instance().populate_command_line(argc, argv);
    ProcessInfo::instance().set_environment_variables();
    auto & clinfo = ProcessInfo::instance().command_line();

    // Initialize the Python interpreter.
    Interpreter::instance()
        .initialize()
        .setup_modmesh_path()
        .setup_process();

    int ret = 0;

    if (clinfo.python_main())
    {
        ret = Py_BytesMain(clinfo.python_main_argc(), clinfo.python_main_argv_ptr());
    }
    else
    {
        ret = Interpreter::instance().enter_main();
    }

    Interpreter::instance().finalize();
    return ret;
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:

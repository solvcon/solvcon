/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/toggle/pymod/toggle_pymod.hpp> // Must be the first include.
#include <solvcon/solvcon.hpp>

#ifdef SOLVCON_METAL
#include <solvcon/device/metal/metal.hpp>
#endif // SOLVCON_METAL

namespace solvcon
{

namespace python
{

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapSolidToggle
    : public WrapBase<WrapSolidToggle, SolidToggle>
{

public:

    using base_type = WrapBase<WrapSolidToggle, SolidToggle>;
    using wrapped_type = typename base_type::wrapped_type;

    friend root_base_type;

protected:

    WrapSolidToggle(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapSolidToggle */

WrapSolidToggle::WrapSolidToggle(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(
            "get_names",
            [](wrapped_type const &)
            {
                // Hardcoding the property names in a lambda does not scale,
                // but I have only 1 property at the moment.
                py::list r;
                r.append("use_pyside");
                return r;
            })
        //
        ;

    // Instance properties.
    (*this)
        .def_property_readonly("use_pyside", &wrapped_type::use_pyside)
        //
        ;
}

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapFixedToggle
    : public WrapBase<WrapFixedToggle, FixedToggle>
{

public:

    using base_type = WrapBase<WrapFixedToggle, FixedToggle>;
    using wrapped_type = typename base_type::wrapped_type;

    friend root_base_type;

protected:

    WrapFixedToggle(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapFixedToggle */

WrapFixedToggle::WrapFixedToggle(pybind11::module & mod, const char * pyname, const char * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def(
            "get_names",
            [](py::object const & self)
            {
                return self.attr("NAMES");
            });

    std::list<std::string> names;

#define MM_PYTHON_TOGGLE_FIXED(NAME)                                                   \
    (*this).def_property(#NAME, &wrapped_type::get_##NAME, &wrapped_type::set_##NAME); \
    names.emplace_back(#NAME);

    // Instance properties.
    // clang-format off
    MM_PYTHON_TOGGLE_FIXED(python_redirect)
    MM_PYTHON_TOGGLE_FIXED(show_axis)
    // clang-format on

    names.sort();
    cls().attr("NAMES") = names;

#undef MM_PYTHON_TOGGLE_FIXED
}

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapHierarchicalToggleAccess
    : public WrapBase<WrapHierarchicalToggleAccess, HierarchicalToggleAccess>
{

public:

    using base_type = WrapBase<WrapHierarchicalToggleAccess, HierarchicalToggleAccess>;
    using wrapped_type = typename base_type::wrapped_type;

    friend root_base_type;

    static pybind11::object getattr(wrapped_type & self, std::string const & key);
    static void setattr(wrapped_type & self, std::string const & key, pybind11::object & value);

protected:

    WrapHierarchicalToggleAccess(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapHierarchicalToggleAccess */

WrapHierarchicalToggleAccess::WrapHierarchicalToggleAccess(pybind11::module & mod, char const * pyname, char const * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    // Dynamic properties.  Number of the properties can be freely changed
    // during runtime.
    (*this)
        .def("__getattr__", getattr)
        .def("__setattr__", setattr)
        .def("get_bool", &wrapped_type::get_bool, py::arg("key"))
        .def("set_bool", &wrapped_type::set_bool, py::arg("key"), py::arg("value"))
        .def("get_int8", &wrapped_type::get_int8, py::arg("key"))
        .def("set_int8", &wrapped_type::set_int8, py::arg("key"), py::arg("value"))
        .def("get_int16", &wrapped_type::get_int16, py::arg("key"))
        .def("set_int16", &wrapped_type::set_int16, py::arg("key"), py::arg("value"))
        .def("get_int32", &wrapped_type::get_int32, py::arg("key"))
        .def("set_int32", &wrapped_type::set_int32, py::arg("key"), py::arg("value"))
        .def("get_int64", &wrapped_type::get_int64, py::arg("key"))
        .def("set_int64", &wrapped_type::set_int64, py::arg("key"), py::arg("value"))
        .def("get_real", &wrapped_type::get_real, py::arg("key"))
        .def("set_real", &wrapped_type::set_real, py::arg("key"), py::arg("value"))
        .def("get_string", &wrapped_type::get_string, py::arg("key"))
        .def("set_string", &wrapped_type::set_string, py::arg("key"), py::arg("value"))
        .def("get_subkey", &wrapped_type::get_subkey, py::arg("key"))
        .def("add_subkey", &wrapped_type::add_subkey, py::arg("key"))
        //
        ;
}

pybind11::object WrapHierarchicalToggleAccess::getattr(wrapped_type & self, std::string const & key)
{
    namespace py = pybind11;

    DynamicToggleIndex const index = self.get_index(key);
    switch (index.type)
    {
    case DynamicToggleIndex::TYPE_NONE:
        throw py::attribute_error(
            std::format("Cannot get non-existing key \"{}\"", self.rekey(key)));
        break;
    case DynamicToggleIndex::TYPE_BOOL:
        return py::cast(self.get_bool(key));
        break;
    case DynamicToggleIndex::TYPE_INT8:
        return py::cast(self.get_int8(key));
        break;
    case DynamicToggleIndex::TYPE_INT16:
        return py::cast(self.get_int16(key));
        break;
    case DynamicToggleIndex::TYPE_INT32:
        return py::cast(self.get_int32(key));
        break;
    case DynamicToggleIndex::TYPE_INT64:
        return py::cast(self.get_int64(key));
        break;
    case DynamicToggleIndex::TYPE_REAL:
        return py::cast(self.get_real(key));
        break;
    case DynamicToggleIndex::TYPE_STRING:
        return py::cast(self.get_string(key));
        break;
    case DynamicToggleIndex::TYPE_SUBKEY:
        return py::cast(self.get_subkey(key));
        break;
    default:
        return py::none();
        break;
    }
}

void WrapHierarchicalToggleAccess::setattr(wrapped_type & self, std::string const & key, pybind11::object & value)
{
    namespace py = pybind11;

    DynamicToggleIndex const index = self.get_index(key);
    switch (index.type)
    {
    case DynamicToggleIndex::TYPE_NONE:
        /* It is intentional to throw an exception when the key does not
         * exist.  Key-value pairs in the toggle object are supposed to be
         * added using the set_TYPE() functions, not the Pythonic
         * __setattr__().
         *
         * Do not try to "fix" the exception using RTTI. */
        throw pybind11::attribute_error(
            std::format("Cannot set non-existing key \"{}\"; use set_TYPE() instead", self.rekey(key)));
        break;
    case DynamicToggleIndex::TYPE_BOOL:
        self.set_bool(key, py::cast<bool>(value));
        break;
    case DynamicToggleIndex::TYPE_INT8:
        self.set_int8(key, py::cast<int8_t>(value));
        break;
    case DynamicToggleIndex::TYPE_INT16:
        self.set_int16(key, py::cast<int16_t>(value));
        break;
    case DynamicToggleIndex::TYPE_INT32:
        self.set_int32(key, py::cast<int32_t>(value));
        break;
    case DynamicToggleIndex::TYPE_INT64:
        self.set_int64(key, py::cast<int64_t>(value));
        break;
    case DynamicToggleIndex::TYPE_REAL:
        self.set_real(key, py::cast<double>(value));
        break;
    case DynamicToggleIndex::TYPE_STRING:
        self.set_string(key, py::cast<std::string>(value));
        break;
    default:
        break;
    }
}

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapToggle
    : public WrapBase<WrapToggle, Toggle>
{

public:

    using base_type = WrapBase<WrapToggle, Toggle>;
    using wrapped_type = typename base_type::wrapped_type;

    friend root_base_type;

protected:

    WrapToggle(pybind11::module & mod, char const * pyname, char const * pydoc);

    static std::string report(wrapped_type const & self);

}; /* end class WrapToggle */

namespace detail
{

struct Toggle2Python
{

    explicit Toggle2Python(Toggle & toggle)
        : m_toggle(toggle)
    {
    }

    static pybind11::object from_toggle(Toggle & toggle, std::string const & type)
    {
        pybind11::object r;
        if (type == "solid")
        {
            Toggle2Python const o(toggle);
            r = o.load_solid();
        }
        else if (type == "fixed")
        {
            Toggle2Python const o(toggle);
            r = o.load_fixed();
        }
        else if (type == "dynamic")
        {
            Toggle2Python o(toggle);
            pybind11::dict d = o.load_dynamic();
            return d;
        }
        else
        {
            Toggle2Python o(toggle);
            r = o.load();
        }
        return r;
    }

    pybind11::list load()
    {
        namespace py = pybind11;
        py::list ret;
        {
            py::dict d;
            d["fixed"] = load_fixed();
            ret.append(d);
        }
        {
            py::dict d;
            d["dynamic"] = load_dynamic();
            ret.append(d);
        }
        return ret;
    }

    pybind11::dict load_solid() const
    {
        namespace py = pybind11;
        SolidToggle const & solid = m_toggle.solid();
        py::dict ret;
        ret["use_pyside"] = solid.use_pyside();
        return ret;
    }

    pybind11::dict load_fixed() const
    {
        namespace py = pybind11;
        FixedToggle const & fixed = m_toggle.fixed();
        py::object const pfixed = py::cast(fixed);
        py::dict ret;
        for (py::handle const name : pfixed.attr("NAMES"))
        {
            ret[name] = pfixed.attr(name);
        }
        return ret;
    }

    pybind11::dict load_dynamic()
    {
        namespace py = pybind11;
        DynamicToggleTable & table = m_toggle.dynamic();
        std::vector<std::string> const keys = table.keys();
        HierarchicalToggleAccess access(table);
        py::dict ret;

        for (auto const & fk : keys)
        {
            std::vector<std::string> const hk = split(fk);
            DynamicToggleIndex const index = table.get_index(fk);
            py::dict d = ret;
            for (size_t i = 0; i < hk.size(); ++i)
            {
                auto const & k = hk[i];
                if (!d.contains(k))
                {
                    if ((hk.size() > 1 && i != (hk.size() - 1)) ||
                        (index.type == DynamicToggleIndex::TYPE_SUBKEY))
                    {
                        d[k.c_str()] = py::dict();
                    }
                    else
                    {
                        d[k.c_str()] = WrapHierarchicalToggleAccess::getattr(access, fk);
                    }
                }
                else
                {
                    // do nothing if the key is already there
                }
                if (i != (hk.size() - 1))
                {
                    d = d[k.c_str()];
                }
            }
        }

        return ret;
    }

    Toggle & m_toggle;

private:

    static std::vector<std::string> split(std::string const & input, char c = '.')
    {
        char const * str = input.c_str();
        std::vector<std::string> result;
        do
        {
            const char * begin = str;
            while (*str != c && *str)
            {
                str++;
            }
            result.emplace_back(begin, str);
        } while (0 != *str++);
        return result;
    }

}; /* end struct Toggle2Python */

} /* end namespace detail */

namespace detail
{

template <typename T>
void declare_toggle(Toggle & self, std::string const & key, T const & value)
{
    self.declare<T>(key, value);
}

void warn_deprecated_getter(char const * name)
{
    PyErr_WarnEx(
        PyExc_DeprecationWarning,
        std::format("Toggle.{} returns a silent sentinel on a missing or wrong-typed "
                    "key and is deprecated; use get(key, default) or at(key)",
                    name)
            .c_str(),
        1);
}

} /* end namespace detail */

WrapToggle::WrapToggle(pybind11::module & mod, char const * pyname, char const * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def("clone", &wrapped_type::clone, py::return_value_policy::take_ownership)
        .def("report", &report)
        .def("to_python", detail::Toggle2Python::from_toggle, py::arg("type") = "")
        //
        ;

    // Accssors to dynamic properties.  Number of the properties can be freely
    // changed during runtime.
    (*this)
        .def(
            "get_value",
            [](wrapped_type & self, std::string const & key)
            {
                HierarchicalToggleAccess access(self.dynamic());
                return WrapHierarchicalToggleAccess::getattr(access, key);
            },
            py::arg("key"))
        .def(
            "get_value",
            [](wrapped_type & self, std::string const & key, py::object & obj)
            {
                HierarchicalToggleAccess access(self.dynamic());
                py::object ret;
                if (access.get_index(key).type != DynamicToggleIndex::TYPE_NONE)
                {
                    ret = WrapHierarchicalToggleAccess::getattr(access, key);
                }
                else
                {
                    ret = obj;
                }
                return ret;
            },
            py::arg("key"),
            py::arg("default"))
        .def(
            "get",
            [](wrapped_type & self, std::string const & key, py::object & obj)
            {
                HierarchicalToggleAccess access(self.dynamic());
                if (access.get_index(key).type != DynamicToggleIndex::TYPE_NONE)
                {
                    return WrapHierarchicalToggleAccess::getattr(access, key);
                }
                return obj;
            },
            py::arg("key"),
            py::arg("default"))
        .def(
            "at",
            [](wrapped_type & self, std::string const & key)
            {
                HierarchicalToggleAccess access(self.dynamic());
                if (access.get_index(key).type == DynamicToggleIndex::TYPE_NONE)
                {
                    throw py::key_error(std::format("Toggle.at: key \"{}\" is missing", key));
                }
                return WrapHierarchicalToggleAccess::getattr(access, key);
            },
            py::arg("key"))
        .def("declare_bool", &detail::declare_toggle<bool>, py::arg("key"), py::arg("default"))
        .def("declare_int8", &detail::declare_toggle<int8_t>, py::arg("key"), py::arg("default"))
        .def("declare_int16", &detail::declare_toggle<int16_t>, py::arg("key"), py::arg("default"))
        .def("declare_int32", &detail::declare_toggle<int32_t>, py::arg("key"), py::arg("default"))
        .def("declare_int64", &detail::declare_toggle<int64_t>, py::arg("key"), py::arg("default"))
        .def("declare_real", &detail::declare_toggle<double>, py::arg("key"), py::arg("default"))
        .def("declare_string", &detail::declare_toggle<std::string>, py::arg("key"), py::arg("default"))
        .def(
            "on_change",
            [](wrapped_type & self, std::string const & key, py::function const & callback)
            {
                // Run the Python callback under the GIL (it may fire from a
                // solver thread that does not hold it) and never let it
                // propagate a C++ exception out of the store.
                // The whole body is wrapped so nothing escapes into the
                // store's notify loop (which also guards callbacks); the
                // analysis cannot always see that through the GIL guard.
                // NOLINTNEXTLINE(bugprone-exception-escape)
                auto wrapped = [callback]()
                {
                    try
                    {
                        py::gil_scoped_acquire const acquire;
                        try
                        {
                            callback();
                        }
                        catch (py::error_already_set & e)
                        {
                            e.discard_as_unraisable("solvcon.Toggle on_change callback");
                        }
                    }
                    // NOLINTNEXTLINE(bugprone-empty-catch)
                    catch (...)
                    {
                    }
                };
                return self.on_change(key, std::move(wrapped));
            },
            py::arg("key"),
            py::arg("callback"))
        .def(
            "__getattr__",
            [](wrapped_type & self, std::string const & key)
            {
                HierarchicalToggleAccess access(self.dynamic());
                py::object ret;
                if (access.get_index(key).type != DynamicToggleIndex::TYPE_NONE) // dynamic
                {
                    ret = WrapHierarchicalToggleAccess::getattr(access, key);
                }
                else if (key == "show_axis") // fixed
                {
                    ret = py::cast(self.fixed().get_show_axis());
                }
                else if (key == "use_pyside") // solid
                {
                    ret = py::cast(self.solid().use_pyside());
                }
                else
                {
                    throw py::attribute_error(std::format("Cannot get by key \"{}\"", key));
                }
                return ret;
            })
        .def(
            "__setattr__",
            [](wrapped_type & self, std::string const & key, pybind11::object & value)
            {
                HierarchicalToggleAccess access(self.dynamic());
                if (access.get_index(key).type != DynamicToggleIndex::TYPE_NONE) // dynamic
                {
                    // This call only sets an existing key.
                    WrapHierarchicalToggleAccess::setattr(access, key, value);
                }
                else if (key == "show_axis") // fixed
                {
                    self.fixed().set_show_axis(py::cast<bool>(value));
                }
                /* solid toggle is not settable */
                else
                {
                    throw py::attribute_error(std::format("Cannot set by key \"{}\"", key));
                }
            })
        .def("dynamic_keys", &wrapped_type::dynamic_keys)
        .def("dynamic_clear", &wrapped_type::dynamic_clear)
        .def("set_bool", &wrapped_type::set_bool, py::arg("key"), py::arg("value"))
        .def("set_int8", &wrapped_type::set_int8, py::arg("key"), py::arg("value"))
        .def("set_int16", &wrapped_type::set_int16, py::arg("key"), py::arg("value"))
        .def("set_int32", &wrapped_type::set_int32, py::arg("key"), py::arg("value"))
        .def("set_int64", &wrapped_type::set_int64, py::arg("key"), py::arg("value"))
        .def("set_real", &wrapped_type::set_real, py::arg("key"), py::arg("value"))
        .def("set_string", &wrapped_type::set_string, py::arg("key"), py::arg("value"))
        .def("get_subkey", &wrapped_type::get_subkey, py::arg("key"))
        .def("add_subkey", &wrapped_type::add_subkey, py::arg("key"))
        //
        ;

    // The type-specific getters return a silent sentinel on a missing or
    // wrong-typed key. They are kept for compatibility but deprecated in
    // favor of get(key, default) and at(key); each emits a
    // DeprecationWarning.
#define MM_PYTHON_TOGGLE_DEPRECATED_GET(MTYPE)             \
    (*this).def(                                           \
        "get_" #MTYPE,                                     \
        [](wrapped_type & self, std::string const & key)   \
        {                                                  \
            detail::warn_deprecated_getter("get_" #MTYPE); \
            return self.get_##MTYPE(key);                  \
        },                                                 \
        py::arg("key"));
    MM_PYTHON_TOGGLE_DEPRECATED_GET(bool)
    MM_PYTHON_TOGGLE_DEPRECATED_GET(int8)
    MM_PYTHON_TOGGLE_DEPRECATED_GET(int16)
    MM_PYTHON_TOGGLE_DEPRECATED_GET(int32)
    MM_PYTHON_TOGGLE_DEPRECATED_GET(int64)
    MM_PYTHON_TOGGLE_DEPRECATED_GET(real)
    MM_PYTHON_TOGGLE_DEPRECATED_GET(string)
#undef MM_PYTHON_TOGGLE_DEPRECATED_GET

    // Static properties.
    (*this)
        .def_property_readonly_static(
            "instance",
            [](py::object const &) -> auto &
            { return wrapped_type::instance(); })
        //
        ;

    // Instance properties.
    (*this)
        .def_property_readonly(
            "solid",
            [](wrapped_type & self) -> auto &
            {
                return self.solid();
            })
        .def_property_readonly(
            "fixed",
            [](wrapped_type & self) -> auto &
            {
                return self.fixed();
            })
        .def_property_readonly(
            "dynamic",
            [](wrapped_type & self)
            {
                return HierarchicalToggleAccess(self.dynamic());
            })
        //
        ;
}

std::string WrapToggle::report(WrapToggle::wrapped_type const & self)
{
    return std::format("Toggle: USE_PYSIDE={}", self.solid().use_pyside());
}

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapCommandLineInfo
    : public WrapBase<WrapCommandLineInfo, CommandLineInfo>
{

public:

    using base_type = WrapBase<WrapCommandLineInfo, CommandLineInfo>;
    using wrapped_type = typename base_type::wrapped_type;

    friend root_base_type;

protected:

    WrapCommandLineInfo(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapCommandLineInfo */

WrapCommandLineInfo::WrapCommandLineInfo(pybind11::module & mod, char const * pyname, char const * pydoc)
    : base_type(mod, pyname, pydoc)
{
    (*this)
        .def("freeze", &wrapped_type::freeze)
        .def_property_readonly("frozen", &wrapped_type::frozen)
        .def_property_readonly("populated", &wrapped_type::populated)
        .def_property_readonly("populated_argv", &wrapped_type::populated_argv)
        .def_property_readonly("executable_basename", &wrapped_type::executable_basename)
        .def_property("python_argv", &wrapped_type::python_argv, &wrapped_type::set_python_argv);
}

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapProcessInfo
    : public WrapBase<WrapProcessInfo, ProcessInfo>
{

public:

    using base_type = WrapBase<WrapProcessInfo, ProcessInfo>;
    using wrapped_type = typename base_type::wrapped_type;

    friend root_base_type;

protected:

    WrapProcessInfo(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapProcessInfo */

WrapProcessInfo::WrapProcessInfo(pybind11::module & mod, char const * pyname, char const * pydoc)
    : base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;
    (*this)
        .def_property_readonly_static(
            "instance",
            [](py::object const &) -> auto &
            {
                return wrapped_type::instance();
            })
        .def(
            "set_environment_variables",
            [](wrapped_type & self) -> auto &
            {
                return self.set_environment_variables();
            },
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "command_line",
            [](wrapped_type & self) -> auto &
            {
                return self.command_line();
            },
            py::return_value_policy::reference_internal);
}

void wrap_Toggle(pybind11::module & mod)
{
    namespace py = pybind11;

    // The RAII subscription token returned by Toggle.on_change. Dropping the
    // Python handle (or calling unsubscribe) removes the callback.
    py::class_<ToggleSubscription>(mod, "ToggleSubscription")
        .def("unsubscribe", &ToggleSubscription::reset)
        .def_property_readonly("active", &ToggleSubscription::active);

    WrapSolidToggle::commit(mod, "SolidToggle", "SolidToggle");
    WrapFixedToggle::commit(mod, "FixedToggle", "FixedToggle");
    WrapHierarchicalToggleAccess::commit(mod, "HierarchicalToggleAccess", "HierarchicalToggleAccess");
    WrapToggle::commit(mod, "Toggle", "Toggle");
    WrapCommandLineInfo::commit(mod, "CommandLineInfo", "CommandLineInfo");
    WrapProcessInfo::commit(mod, "ProcessInfo", "ProcessInfo");

#ifdef SOLVCON_METAL
    mod.attr("METAL_BUILT") = true;
#else // SOLVCON_METAL
    mod.attr("METAL_BUILT") = false;
#endif // SOLVCON_METAL
    mod.def(
        "metal_running",
        []()
        {
#ifdef SOLVCON_METAL
            return ::solvcon::device::MetalManager::instance().started();
#else // SOLVCON_METAL
            return false;
#endif // SOLVCON_METAL
        });
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

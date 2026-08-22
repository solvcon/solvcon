/*
 * Copyright (c) 2026, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/mcap/pymod/mcap_pymod.hpp>
#include <solvcon/solvcon.hpp>

namespace solvcon
{

namespace python
{

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapMcapSchema
    : public WrapBase<WrapMcapSchema, mcap::SchemaRecord>
{

public:

    using base_type = WrapBase<WrapMcapSchema, mcap::SchemaRecord>;
    using wrapped_type = typename base_type::wrapped_type;

    friend root_base_type;

protected:

    WrapMcapSchema(pybind11::module & mod, char const * pyname, char const * pydoc)
        : base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

        (*this)
            .def_readonly("id", &wrapped_type::id)
            .def_readonly("name", &wrapped_type::name)
            .def_readonly("encoding", &wrapped_type::encoding)
            // The definition is text for a ROS 2 encoding but bytes for a
            // binary one, so it stays bytes for every encoding.
            .def_property_readonly(
                "data",
                [](wrapped_type const & self)
                { return py::bytes(self.data); })
            //
            ;
    }

}; /* end class WrapMcapSchema */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapMcapReader
    : public WrapBase<WrapMcapReader, mcap::Reader, std::shared_ptr<mcap::Reader>>
{

public:

    using base_type = WrapBase<WrapMcapReader, mcap::Reader, std::shared_ptr<mcap::Reader>>;
    using wrapped_type = typename base_type::wrapped_type;

    friend root_base_type;

protected:

    WrapMcapReader(pybind11::module & mod, char const * pyname, char const * pydoc)
        : base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

        (*this)
            .def(
                py::init(
                    [](std::string const & path)
                    { return std::make_shared<mcap::Reader>(path); }),
                py::arg("path"))
            //
            ;

        (*this)
            .def_property_readonly("path", &wrapped_type::path)
            .def("topics", &wrapped_type::topics)
            .def("chunk_count", &wrapped_type::chunk_count)
            .def("schema", &wrapped_type::schema, py::arg("topic"))
            .def("time_range", &wrapped_type::time_range)
            .def("has_time_range", &wrapped_type::has_time_range)
            //
            ;
    }

}; /* end class WrapMcapReader */

void wrap_McapReader(pybind11::module & mod)
{
    WrapMcapSchema::commit(mod, "McapSchema", "McapSchema");
    WrapMcapReader::commit(mod, "McapReader", "McapReader");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

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

/// The log time column and the list of field columns a plan extracts.
static pybind11::tuple extract_columns(mcap::Reader & reader, std::string const & topic, mcap::DecodePlan const & plan)
{
    namespace py = pybind11;

    mcap::ColumnSet columns = mcap::extract(reader, topic, plan);

    // A SimpleArray copy clones its buffer, so the arrays move into the
    // objects Python holds.
    py::list arrays;
    for (mcap::Column & column : columns.columns)
    {
        arrays.append(std::visit([](auto & array)
                                 { return py::cast(std::move(array)); },
                                 column));
    }

    return py::make_tuple(py::cast(std::move(columns.time)), arrays);
}

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

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapMcapMessageIterator
    : public WrapBase<WrapMcapMessageIterator, mcap::MessageIterator, std::shared_ptr<mcap::MessageIterator>>
{

public:

    using base_type = WrapBase<WrapMcapMessageIterator, mcap::MessageIterator, std::shared_ptr<mcap::MessageIterator>>;
    using wrapped_type = typename base_type::wrapped_type;

    friend root_base_type;

protected:

    WrapMcapMessageIterator(pybind11::module & mod, char const * pyname, char const * pydoc)
        : base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

        (*this)
            .def(
                "__iter__",
                [](py::object self)
                { return self; })
            .def(
                "__next__",
                [](wrapped_type & self)
                {
                    uint64_t log_time = 0;
                    std::string_view payload;
                    if (!self.next(log_time, payload))
                    {
                        throw py::stop_iteration();
                    }
                    return py::make_tuple(log_time, py::bytes(payload.data(), payload.size()));
                })
            .def("selected_chunk_count", &wrapped_type::selected_chunk_count)
            //
            ;
    }

}; /* end class WrapMcapMessageIterator */

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
            // The iterator reads from the file the reader holds open.
            .def("messages", &wrapped_type::messages, py::arg("topic"), py::keep_alive<0, 1>())
            .def("extract", &extract_columns, py::arg("topic"), py::arg("plan"))
            //
            ;
    }

}; /* end class WrapMcapReader */

void wrap_McapReader(pybind11::module & mod)
{
    WrapMcapSchema::commit(mod, "McapSchema", "McapSchema");
    WrapMcapMessageIterator::commit(mod, "McapMessageIterator", "McapMessageIterator");
    WrapMcapReader::commit(mod, "McapReader", "McapReader");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

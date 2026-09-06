/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/pymod/buffer_pymod.hpp> // Must be the first include.
#include <solvcon/buffer/buffer.hpp>

namespace solvcon
{

namespace python
{

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapConcreteBuffer
    : public WrapBase<WrapConcreteBuffer, ConcreteBuffer, std::shared_ptr<ConcreteBuffer>>
{

    friend root_base_type;

    WrapConcreteBuffer(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapConcreteBuffer */

WrapConcreteBuffer::WrapConcreteBuffer(pybind11::module & mod, char const * pyname, char const * pydoc)
    : root_base_type(mod, pyname, pydoc, pybind11::buffer_protocol())
{
    namespace py = pybind11;

    (*this)
        .def_timed(
            py::init(
                [](size_t nbytes, size_t alignment)
                { return wrapped_type::construct(nbytes, alignment); }),
            py::arg("nbytes"),
            py::arg("alignment") = 0)
        .def(
            py::init(
                [](py::array & arr_in, size_t alignment)
                {
                    if ((arr_in.flags() & (py::array::c_style | py::array::f_style)) == 0)
                    {
                        throw std::invalid_argument("ConcreteBuffer: input array must be C- or F-contiguous");
                    }
                    return wrapped_type::construct(
                        arr_in.nbytes(), arr_in.mutable_data(), std::make_unique<ConcreteBufferNdarrayRemover>(arr_in), alignment);
                }),
            py::arg("array"),
            py::arg("alignment") = 0)
        .def_timed("clone", &wrapped_type::clone)
        .def_property_readonly("nbytes", &wrapped_type::nbytes)
        .def_property_readonly("alignment", &wrapped_type::alignment)
        .def("__len__", &wrapped_type::size)
        .def(
            "__getitem__",
            [](wrapped_type const & self, size_t it)
            { return self.at(it); })
        .def(
            "__setitem__",
            [](wrapped_type & self, size_t it, int8_t val)
            { self.at(it) = val; })
        .def_buffer(
            [](wrapped_type & self)
            {
                return py::buffer_info(
                    self.data(), /* Pointer to buffer */
                    sizeof(int8_t), /* Size of one scalar */
                    py::format_descriptor<int8_t>::format(), /* Python struct-style format descriptor */
                    1, /* Number of dimensions */
                    {self.size()}, /* Buffer dimensions */
                    {1} /* Strides (in bytes) for each index */
                );
            })
        .def_property_readonly(
            "ndarray",
            [](wrapped_type & self)
            {
                namespace py = pybind11;
                return py::array(
                    py::detail::npy_format_descriptor<int8_t>::dtype(), /* Numpy dtype */
                    {self.size()}, /* Buffer dimensions */
                    {1}, /* Strides (in bytes) for each index */
                    self.data(), /* Pointer to buffer */
                    py::cast(self.shared_from_this()) /* Owning Python object */
                );
            })
        .def_property_readonly(
            "is_from_python",
            [](wrapped_type const & self)
            {
                return self.has_remover() && ConcreteBufferNdarrayRemover::is_same_type(self.get_remover());
            })
        //
        ;
}

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapBufferExpander
    : public WrapBase<WrapBufferExpander, BufferExpander, std::shared_ptr<BufferExpander>>
{

    friend root_base_type;

    WrapBufferExpander(pybind11::module & mod, char const * pyname, char const * pydoc);

}; /* end class WrapBufferExpander */

WrapBufferExpander::WrapBufferExpander(pybind11::module & mod, char const * pyname, char const * pydoc)
    : root_base_type(mod, pyname, pydoc)
{
    namespace py = pybind11;

    (*this)
        .def_timed(
            py::init(
                [](size_t length, size_t alignment)
                { return wrapped_type::construct(length, alignment); }),
            py::arg("length"),
            py::arg("alignment") = 0)
        .def_timed(py::init([]()
                            { return wrapped_type::construct(); }))
        .def_timed(py::init([](std::shared_ptr<ConcreteBuffer> const & buf, size_t alignment)
                            { return wrapped_type::construct(buf, /*clone*/ true, alignment); }),
                   py::arg("buffer"),
                   py::arg("alignment") = 0)
        .def_timed("reserve", &wrapped_type::reserve, py::arg("cap"))
        .def_timed("expand", &wrapped_type::expand, py::arg("length"))
        .def_property_readonly("capacity", &wrapped_type::capacity)
        .def_property_readonly("alignment", &wrapped_type::alignment)
        .def("__len__", &wrapped_type::size)
        .def(
            "__getitem__",
            [](wrapped_type const & self, size_t it)
            { return self.at(it); })
        .def(
            "__setitem__",
            [](wrapped_type & self, size_t it, int8_t val)
            { self.at(it) = val; })
        .def_timed("clone", &wrapped_type::clone)
        .def_timed("copy_concrete", &wrapped_type::copy_concrete, py::arg("cap") = 0)
        .def_timed("as_concrete", &wrapped_type::as_concrete, py::arg("cap") = 0)
        .def_property_readonly("is_concrete", &wrapped_type::is_concrete)
        //
        ;
}

void wrap_ConcreteBuffer(pybind11::module & mod)
{
    WrapConcreteBuffer::commit(mod, "ConcreteBuffer", "ConcreteBuffer");
    WrapBufferExpander::commit(mod, "BufferExpander", "BufferExpander");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

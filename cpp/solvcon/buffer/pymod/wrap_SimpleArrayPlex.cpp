/*
 * Copyright (c) 2024, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/pymod/buffer_pymod.hpp> // Must be the first include.

#include <solvcon/buffer/pymod/array_common.hpp>
#include <type_traits>

namespace solvcon
{

namespace python
{

/// Execute the callback function with the typed array
/// @tparam A either `SimpleArrayPlex` or `const SimpleArrayPlex`
/// @tparam C the type of the callback function
/// @param arrayplex the plex array, which is the wrapper of the typed array
/// @param callback the callback function, which has the mutable typed array as the argument
/// @return the return type of the callback function
template <typename A, typename C>
requires solvcon::IsSameRemoveConstType<A, SimpleArrayPlex>
// NOLINTNEXTLINE(misc-use-anonymous-namespace,cppcoreguidelines-missing-std-forward)
static auto execute_callback_with_typed_array(A & arrayplex, C && callback)
{
// We get the typed array from the arrayplex and call the callback function with the typed array.
#define DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType, ArrayType)                                            \
    case DataType:                                                                                            \
    {                                                                                                         \
        using ArrayTypePtr = typename std::conditional_t<std::is_const_v<A>, const ArrayType *, ArrayType *>; \
        /* NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) */                                     \
        auto array = reinterpret_cast<ArrayTypePtr>(arrayplex.mutable_instance_ptr());                        \
        return callback(*array);                                                                              \
    }

    switch (arrayplex.data_type())
    {
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Bool, SimpleArrayBool)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Int8, SimpleArrayInt8)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Int16, SimpleArrayInt16)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Int32, SimpleArrayInt32)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Int64, SimpleArrayInt64)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Uint8, SimpleArrayUint8)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Uint16, SimpleArrayUint16)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Uint32, SimpleArrayUint32)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Uint64, SimpleArrayUint64)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Float32, SimpleArrayFloat32)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Float64, SimpleArrayFloat64)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Complex64, SimpleArrayComplex64)
        DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY(DataType::Complex128, SimpleArrayComplex128)
    default:
    {
        throw std::invalid_argument("Unsupported datatype");
    }
    }

#undef DECL_MM_RUN_CALLBACK_WITH_TYPED_ARRAY
}

/// Check the data type of the python value match the given data type. If not, throw a type error.
// NOLINTNEXTLINE(misc-use-anonymous-namespace)
static void verify_python_value_datatype(pybind11::object const & value, DataType datatype)
{
    switch (datatype)
    {
    case DataType::Bool:
    {
        if (!pybind11::isinstance<pybind11::bool_>(value))
        {
            throw pybind11::type_error("Data type mismatch, expected Python bool");
        }
        break;
    }
    case DataType::Int8:
    case DataType::Int16:
    case DataType::Int32:
    case DataType::Int64:
    case DataType::Uint8:
    case DataType::Uint16:
    case DataType::Uint32:
    case DataType::Uint64:
    {
        if (!pybind11::isinstance<pybind11::int_>(value))
        {
            throw pybind11::type_error("Data type mismatch, expected Python int");
        }
        break;
    }
    case DataType::Float32:
    case DataType::Float64:
    {
        if (!pybind11::isinstance<pybind11::float_>(value))
        {
            throw pybind11::type_error("Data type mismatch, expected Python float");
        }
        break;
    }
    case DataType::Complex64:
    {
        if (!pybind11::isinstance<Complex<float>>(value))
        {
            throw pybind11::type_error("Data type mismatch, expected Complex64");
        }
        break;
    }
    case DataType::Complex128:
    {
        if (!pybind11::isinstance<Complex<double>>(value))
        {
            throw pybind11::type_error("Data type mismatch, expected Complex128");
        }
        break;
    }

    default:
        throw std::runtime_error("Unsupported datatype");
    }
}

/// Get the typed array value by the key
/// @tparam T the type of the key
template <typename T>
// NOLINTNEXTLINE(misc-use-anonymous-namespace)
static pybind11::object get_typed_array_value(const SimpleArrayPlex & array_plex, T key)
{
#define DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType, ArrayType)                     \
    case DataType:                                                                      \
    {                                                                                   \
        const auto * array = static_cast<const ArrayType *>(array_plex.instance_ptr()); \
        return pybind11::cast(array->at(key));                                          \
    }

    switch (array_plex.data_type())
    {
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Bool, SimpleArrayBool)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Int8, SimpleArrayInt8)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Int16, SimpleArrayInt16)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Int32, SimpleArrayInt32)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Int64, SimpleArrayInt64)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Uint8, SimpleArrayUint8)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Uint16, SimpleArrayUint16)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Uint32, SimpleArrayUint32)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Uint64, SimpleArrayUint64)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Float32, SimpleArrayFloat32)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Float64, SimpleArrayFloat64)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Complex64, SimpleArrayComplex64)
        DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX(DataType::Complex128, SimpleArrayComplex128)
    default:
    {
        throw std::runtime_error("Unsupported datatype");
    }
    }
#undef DECL_MM_GET_TYPED_ARRAY_VALUE_BY_INDEX
}

/// Get the typed array from the arrayplex
// NOLINTNEXTLINE(misc-use-anonymous-namespace)
static pybind11::object get_typed_array(const SimpleArrayPlex & array_plex)
{
#define DECL_MM_GET_TYPED_ARRAY(DataType, ArrayType)                                    \
    case DataType:                                                                      \
    {                                                                                   \
        const auto * array = static_cast<const ArrayType *>(array_plex.instance_ptr()); \
        return pybind11::cast(std::move(ArrayType(*array)));                            \
    }

    switch (array_plex.data_type())
    {
        DECL_MM_GET_TYPED_ARRAY(DataType::Bool, SimpleArrayBool)
        DECL_MM_GET_TYPED_ARRAY(DataType::Int8, SimpleArrayInt8)
        DECL_MM_GET_TYPED_ARRAY(DataType::Int16, SimpleArrayInt16)
        DECL_MM_GET_TYPED_ARRAY(DataType::Int32, SimpleArrayInt32)
        DECL_MM_GET_TYPED_ARRAY(DataType::Int64, SimpleArrayInt64)
        DECL_MM_GET_TYPED_ARRAY(DataType::Uint8, SimpleArrayUint8)
        DECL_MM_GET_TYPED_ARRAY(DataType::Uint16, SimpleArrayUint16)
        DECL_MM_GET_TYPED_ARRAY(DataType::Uint32, SimpleArrayUint32)
        DECL_MM_GET_TYPED_ARRAY(DataType::Uint64, SimpleArrayUint64)
        DECL_MM_GET_TYPED_ARRAY(DataType::Float32, SimpleArrayFloat32)
        DECL_MM_GET_TYPED_ARRAY(DataType::Float64, SimpleArrayFloat64)
        DECL_MM_GET_TYPED_ARRAY(DataType::Complex64, SimpleArrayComplex64)
        DECL_MM_GET_TYPED_ARRAY(DataType::Complex128, SimpleArrayComplex128)
    default:
    {
        throw std::runtime_error("Unsupported datatype");
    }
    }
#undef DECL_MM_GET_TYPED_ARRAY
}

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapSimpleArrayPlex : public WrapBase<WrapSimpleArrayPlex, SimpleArrayPlex>
{
    using root_base_type = WrapBase<WrapSimpleArrayPlex, SimpleArrayPlex>;
    using wrapped_type = typename root_base_type::wrapped_type;
    using wrapper_type = typename root_base_type::wrapper_type;

    friend root_base_type;

#define DECL_MM_EXECUTE_TYPED_ARRAY_METHOD(typed_array_method)          \
    [](wrapped_type & self) { return execute_callback_with_typed_array( \
                                  self, [](auto & array) { return array.typed_array_method(); }); }

    WrapSimpleArrayPlex(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc, pybind11::buffer_protocol())
    {
        (*this)
            .def_timed(
                pybind11::init(
                    [](pybind11::object const & shape, std::string const & datatype)
                    { return wrapped_type(make_shape(shape), datatype); }),
                pybind11::arg("shape"),
                pybind11::arg("dtype"))
            .def_timed(
                pybind11::init(
                    [](pybind11::object const & shape, std::string const & datatype, size_t alignment)
                    { return wrapped_type(make_shape(shape), DataType(datatype), alignment); }),
                pybind11::arg("shape"),
                pybind11::arg("dtype"),
                pybind11::arg("alignment"))
            .def_timed(
                pybind11::init(&init_array_plex_with_value),
                pybind11::arg("shape"),
                pybind11::arg("value"),
                pybind11::arg("dtype"))
            .def_timed(
                pybind11::init(&init_array_plex_with_value_and_alignment),
                pybind11::arg("shape"),
                pybind11::arg("value"),
                pybind11::arg("dtype"),
                pybind11::arg("alignment"))
            .def(
                pybind11::init(
                    [](pybind11::array & arr_in)
                    {
                        solvcon::detail::shape_type shape;
                        for (ssize_t i = 0; i < arr_in.ndim(); ++i)
                        {
                            shape.push_back(arr_in.shape(i));
                        }
                        std::shared_ptr<ConcreteBuffer> const buffer = ConcreteBuffer::construct(
                            arr_in.nbytes(),
                            arr_in.mutable_data(),
                            std::make_unique<ConcreteBufferNdarrayRemover>(arr_in));
                        return wrapped_type(shape, buffer, pybind11::str(arr_in.dtype()));
                    }),
                pybind11::arg("array"))
            .def("clone",
                 [](wrapped_type const & self)
                 { return wrapped_type(self); }) // cloning the object using the copy constructor. never add the clone method to the C++ class.
            .def_property_readonly("typed", &get_typed_array)
            .def_buffer(
                [](wrapped_type & self)
                {
                    return execute_callback_with_typed_array(
                        self,
                        [](auto & array)
                        {
                            using data_type = typename std::remove_reference_t<decltype(array[0])>;
                            return ArrayPropertyHelper<data_type>::get_buffer_info(array); });
                })
            .def_property_readonly("nbytes", DECL_MM_EXECUTE_TYPED_ARRAY_METHOD(nbytes))
            .def_property_readonly("size", DECL_MM_EXECUTE_TYPED_ARRAY_METHOD(size))
            .def_property_readonly("itemsize", DECL_MM_EXECUTE_TYPED_ARRAY_METHOD(itemsize))
            .def_property_readonly("alignment", &wrapped_type::alignment)
            .def_property_readonly(
                "shape",
                [](wrapped_type const & self)
                {
                    return execute_callback_with_typed_array(
                        self, [](const auto & array)
                        {
                            pybind11::tuple ret(array.shape().size());
                            for (size_t i = 0; i < array.shape().size(); ++i)
                            {
                                ret[i] = array.shape()[i];
                            }
                            return ret; });
                })
            .def_property_readonly(
                "stride",
                [](wrapped_type const & self)
                {
                    return execute_callback_with_typed_array(
                        self,
                        [](const auto & array)
                        {
                            pybind11::tuple ret(array.stride().size());
                            for (size_t i = 0; i < array.stride().size(); ++i)
                            {
                                ret[i] = array.stride()[i];
                            }
                            return ret;
                        });
                })
            .def("__len__", DECL_MM_EXECUTE_TYPED_ARRAY_METHOD(size))
            .def("__getitem__", &get_typed_array_value<ssize_t>)
            .def("__getitem__", &get_typed_array_value<const std::vector<ssize_t> &>)
            .def("__setitem__",
                 [](wrapped_type & self, pybind11::args const & args)
                 {
                     execute_callback_with_typed_array(
                         self,
                         [&args](auto & array)
                         {
                             using data_type = typename std::remove_reference_t<decltype(array[0])>;
                             ArrayPropertyHelper<data_type>::setitem_parser(array, args);
                         });
                 })
            .def("reshape", [](wrapped_type const & self, pybind11::object const & py_shape)
                 { return execute_callback_with_typed_array(
                       self,
                       // NOLINTNEXTLINE(fuchsia-trailing-return)
                       [&](const auto & array) -> wrapped_type // need the return type to get correct deduced type
                       {
                        const auto shape = make_shape(py_shape);
                        return array.reshape(shape); }); })
            .def_property_readonly("has_ghost", DECL_MM_EXECUTE_TYPED_ARRAY_METHOD(has_ghost))
            .def_property(
                "nghost",
                // getter
                DECL_MM_EXECUTE_TYPED_ARRAY_METHOD(nghost),
                // setter
                [](wrapped_type & self, ssize_t nghost)
                {
                    execute_callback_with_typed_array(
                        self,
                        [nghost](auto & array)
                        {
                            return array.set_nghost(nghost);
                        });
                })
            .def_property_readonly("nbody", DECL_MM_EXECUTE_TYPED_ARRAY_METHOD(nbody))
            .wrap_modifiers()
            .wrap_calculators()
            // ATTENTION: always keep the same interface between WrapSimpleArrayPlex and WrapSimpleArray
            ;
    }

    wrapper_type & wrap_modifiers()
    {
        namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

        (*this)
            .def(
                "fill",
                [](wrapped_type & self, pybind11::object const & py_value)
                {
                    auto datatype = self.data_type();
                    execute_callback_with_typed_array(
                        self, [&py_value, datatype](auto & array)
                        {
                            using value_type = typename std::remove_reference_t<decltype(array[0])>;
                            verify_python_value_datatype(py_value, datatype);
                            const auto value = py_value.cast<value_type>();
                            array.fill(value);
                        }); },
                py::arg("value"))
            //
            ;

        return *this;
    }

    wrapper_type & wrap_calculators()
    {
#define DECL_MM_EXECUTE_TYPED_ARRAY_METHOD_RETUN_TYPED_VALUE(typed_array_method) \
    [](wrapped_type & self) { return execute_callback_with_typed_array(          \
                                  self, [](auto & array) { return pybind11::cast(array.typed_array_method()); }); }

        (*this)
            .def("min", DECL_MM_EXECUTE_TYPED_ARRAY_METHOD_RETUN_TYPED_VALUE(min))
            .def("max", DECL_MM_EXECUTE_TYPED_ARRAY_METHOD_RETUN_TYPED_VALUE(max))
            .def("sum", DECL_MM_EXECUTE_TYPED_ARRAY_METHOD_RETUN_TYPED_VALUE(sum))
            .def("abs", DECL_MM_EXECUTE_TYPED_ARRAY_METHOD_RETUN_TYPED_VALUE(abs))
            //
            ;

#undef DECL_MM_EXECUTE_TYPED_ARRAY_METHOD_RETUN_TYPED_VALUE
        return *this;
    }

    /// Initialize the arrayplex with the given value
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    static wrapped_type init_array_plex_with_value(
        pybind11::object const & shape_in,
        pybind11::object const & py_value,
        std::string const & datatype_str)
    {
        const auto shape = make_shape(shape_in);
        wrapped_type array_plex(shape, datatype_str);
        auto datatype = array_plex.data_type();
        execute_callback_with_typed_array(
            array_plex, [&py_value, datatype](auto & array)
            {
                using value_type = typename std::remove_reference_t<decltype(array[0])>;
                verify_python_value_datatype(py_value, datatype);
                const auto value = py_value.cast<value_type>();
                array.fill(value); });
        return array_plex;
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    /// Initialize the arrayplex with the given value and alignment
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    static wrapped_type init_array_plex_with_value_and_alignment(
        pybind11::object const & shape_in,
        pybind11::object const & py_value,
        std::string const & datatype_str,
        size_t alignment)
    {
        const auto shape = make_shape(shape_in);
        wrapped_type array_plex(shape, DataType(datatype_str), alignment);
        auto datatype = array_plex.data_type();
        execute_callback_with_typed_array(
            array_plex, [&py_value, datatype](auto & array)
            {
                using value_type = typename std::remove_reference_t<decltype(array[0])>;
                verify_python_value_datatype(py_value, datatype);
                const auto value = py_value.cast<value_type>();
                array.fill(value); });
        return array_plex;
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

#undef DECL_MM_EXECUTE_TYPED_ARRAY_METHOD

}; /* end class WrapSimpleArrayPlex */

void wrap_SimpleArrayPlex(pybind11::module & mod)
{
    WrapSimpleArrayPlex::commit(mod, "SimpleArray", "SimpleArray");
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

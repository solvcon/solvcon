#pragma once

/*
 * Copyright (c) 2019, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/pybind11.h> // Must be the first include.
#include <pybind11/attr.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

#include <atomic>

#include <solvcon/toggle/toggle.hpp>

#ifdef __GNUG__
#define SOLVCON_PYTHON_WRAPPER_VISIBILITY __attribute__((visibility("hidden")))
#else
#define SOLVCON_PYTHON_WRAPPER_VISIBILITY PYBIND11_EXPORT
#endif

namespace solvcon
{

namespace python
{

namespace detail
{

template <class T>
std::string to_str(T const & self)
{
    std::ostringstream oss;
    oss << self;
    return oss.str();
}

} /* end namespace detail */

template <typename T>
bool dtype_is_type(pybind11::array const & arr)
{
    return pybind11::detail::npy_format_descriptor<T>::dtype().equal(arr.dtype());
}

class WrapperProfilerStatus
{

public:

    static WrapperProfilerStatus & me()
    {
        static WrapperProfilerStatus instance;
        return instance;
    }

    WrapperProfilerStatus(WrapperProfilerStatus const &) = delete;
    WrapperProfilerStatus(WrapperProfilerStatus &&) = delete;
    WrapperProfilerStatus & operator=(WrapperProfilerStatus const &) = delete;
    WrapperProfilerStatus & operator=(WrapperProfilerStatus &&) = delete;
    ~WrapperProfilerStatus() = default;

    bool enabled() const { return m_enabled; }
    WrapperProfilerStatus & enable()
    {
        m_enabled = true;
        return *this;
    }
    WrapperProfilerStatus & disable()
    {
        m_enabled = false;
        return *this;
    }

private:

    WrapperProfilerStatus()
        : m_enabled(true)
    {
    }

    std::atomic<bool> m_enabled;

}; /* end class WrapperProfilerStatus */

struct mmtag
{
};

} /* end namespace python */
} /* end namespace solvcon */

namespace pybind11
{
namespace detail
{

template <>
struct process_attribute<solvcon::python::mmtag>
    : process_attribute_default<solvcon::python::mmtag>
{

    static void precall(function_call & call)
    {
        if (solvcon::python::WrapperProfilerStatus::me().enabled())
        {
            solvcon::CallProfiler::instance().start_caller(get_name(call), nullptr);
        }
    }

    static void postcall(function_call &, handle &)
    {
        if (solvcon::python::WrapperProfilerStatus::me().enabled())
        {
            solvcon::CallProfiler::instance().end_caller();
        }
    }

private:

    static std::string get_name(function_call const & call)
    {
        function_record const & r = call.func;
        return std::string(str(r.scope.attr("__name__"))) + std::string(".") + r.name;
    }
};

} /* end namespace detail */
} /* end namespace pybind11 */

namespace solvcon
{

namespace python
{

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4251) // needs to have dll-interface to be used by clients of class
#pragma warning(disable : 4275) // non dll-interface struct used as base for dll-interface struct
#endif
// clang-format off
struct
SOLVCON_PYTHON_WRAPPER_VISIBILITY
ConcreteBufferNdarrayRemover : ConcreteBuffer::remover_type
// clang-format on
{

    static bool is_same_type(ConcreteBuffer::remover_type const & other)
    {
        return typeid(other) == typeid(ConcreteBufferNdarrayRemover);
    }

    ConcreteBufferNdarrayRemover() = delete;

    explicit ConcreteBufferNdarrayRemover(pybind11::array arr_in)
        : ndarray(std::move(arr_in))
    {
    }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,readability-non-const-parameter)
    void operator()(int8_t *, size_t) const override {}

    pybind11::array ndarray;

}; /* end struct ConcreteBufferNdarrayRemover */
#ifdef _MSC_VER
#pragma warning(pop)
#endif

template <typename S>
// FIXME: NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
std::enable_if_t<is_simple_array_v<S>, pybind11::array> to_ndarray(S && sarr)
{
    namespace py = pybind11;
    using T = typename std::remove_reference_t<S>::value_type;
    std::vector<size_t> const shape(sarr.shape().begin(), sarr.shape().end());
    std::vector<py::ssize_t> stride;
    stride.reserve(sarr.stride().size());
    auto const itemsize = static_cast<py::ssize_t>(sarr.itemsize());
    for (ssize_t const v : sarr.stride())
    {
        stride.push_back(static_cast<py::ssize_t>(v) * itemsize);
    }
    return py::array(
        py::detail::npy_format_descriptor<T>::dtype(), // Numpy dtype
        shape, // Buffer dimensions
        stride, // Strides (in bytes) for each index
        sarr.data(), // Pointer to buffer
        py::cast(sarr.buffer().shared_from_this()) // Create the Python object owning the buffer
    );
}

template <typename T>
static SimpleArray<T> makeSimpleArray(pybind11::array_t<T> & ndarr)
{
    solvcon::detail::shape_type shape;
    for (ssize_t i = 0; i < ndarr.ndim(); ++i)
    {
        shape.push_back(ndarr.shape(i));
    }
    std::shared_ptr<ConcreteBuffer> const buffer = ConcreteBuffer::construct(
        ndarr.nbytes(), // Number of bytes
        ndarr.mutable_data(), // Pointer to buffer
        std::make_unique<ConcreteBufferNdarrayRemover>(ndarr) // Use ndarray to own the buffer
    );
    return SimpleArray<T>(shape, buffer);
}

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4251) // needs to have dll-interface to be used by clients of class
#endif
/**
 * Helper template for pybind11 class wrappers.
 */
// clang-format off
template
<
    class Wrapper
  , class Wrapped
  , class Holder = std::unique_ptr<Wrapped>
  , class WrappedBase = Wrapped
>
/*
 * Use CRTP to detect type error during compile time.
 */
class
SOLVCON_PYTHON_WRAPPER_VISIBILITY
WrapBase
// clang-format on
{

public:

    using wrapper_type = Wrapper;
    using wrapped_type = Wrapped;
    using wrapped_base_type = WrappedBase;
    using holder_type = Holder;
    // clang-format off
    using root_base_type = WrapBase
    <
        wrapper_type
      , wrapped_type
      , holder_type
      , wrapped_base_type
    >;
    using class_ = typename std::conditional_t
    <
        std::is_same_v< Wrapped, WrappedBase >
      , pybind11::class_< wrapped_type, holder_type >
      , pybind11::class_< wrapped_type, wrapped_base_type, holder_type >
    >;
    // clang-format on

    template <class... Extra>
    static wrapper_type & commit(pybind11::module & mod, char const * pyname, char const * pydoc, Extra &&... extra)
    {
        // The static local is constructed exactly once, on the first call, so
        // only the first call's pyname/pydoc/extra take effect; later calls
        // return the already-registered singleton. Existing three-argument call
        // sites deduce an empty Extra pack and keep working unchanged.
        static wrapper_type derived(mod, pyname, pydoc, std::forward<Extra>(extra)...);
        return derived;
    }

    WrapBase() = delete;
    WrapBase(WrapBase const &) = delete;
    WrapBase(WrapBase &&) = delete;
    WrapBase & operator=(WrapBase const &) = delete;
    WrapBase & operator=(WrapBase &&) = delete;
    ~WrapBase() = default;

#define DECL_MM_PYBIND_CLASS_METHOD_UNTIMED(METHOD)                           \
    template <class... Args> /* NOLINTNEXTLINE(bugprone-macro-parentheses) */ \
    wrapper_type & METHOD(Args &&... args)                                    \
    {                                                                         \
        m_cls.METHOD(std::forward<Args>(args)...);                            \
        return *static_cast<std::add_pointer_t<wrapper_type>>(this);          \
    }

#define DECL_MM_PYBIND_CLASS_METHOD_TIMED(METHOD)                             \
    template <class... Args> /* NOLINTNEXTLINE(bugprone-macro-parentheses) */ \
    wrapper_type & METHOD##_timed(Args &&... args)                            \
    {                                                                         \
        m_cls.METHOD(std::forward<Args>(args)..., mmtag());                   \
        return *static_cast<std::add_pointer_t<wrapper_type>>(this);          \
    }

#define DECL_MM_PYBIND_CLASS_METHOD(METHOD)     \
    DECL_MM_PYBIND_CLASS_METHOD_UNTIMED(METHOD) \
    DECL_MM_PYBIND_CLASS_METHOD_TIMED(METHOD)

    // The args pack is forwarded inside the macro-generated bodies above;
    // clang-tidy misattributes the forwarding through the macro expansion.
    // NOLINTBEGIN(cppcoreguidelines-missing-std-forward)
    DECL_MM_PYBIND_CLASS_METHOD(def)
    DECL_MM_PYBIND_CLASS_METHOD(def_static)

    DECL_MM_PYBIND_CLASS_METHOD(def_readwrite)
    DECL_MM_PYBIND_CLASS_METHOD(def_readonly)
    DECL_MM_PYBIND_CLASS_METHOD(def_readwrite_static)
    DECL_MM_PYBIND_CLASS_METHOD(def_readonly_static)

    DECL_MM_PYBIND_CLASS_METHOD(def_property)
    DECL_MM_PYBIND_CLASS_METHOD(def_property_static)
    DECL_MM_PYBIND_CLASS_METHOD(def_property_readonly)
    DECL_MM_PYBIND_CLASS_METHOD(def_property_readonly_static)

    DECL_MM_PYBIND_CLASS_METHOD_UNTIMED(def_buffer)
    // NOLINTEND(cppcoreguidelines-missing-std-forward)

#undef DECL_MM_PYBIND_CLASS_METHOD_UNTIMED
#undef DECL_MM_PYBIND_CLASS_METHOD_TIMED
#undef DECL_MM_PYBIND_CLASS_METHOD

    wrapper_type & def_alias(char const * from_name, char const * to_name)
    {
        cls().attr(to_name) = cls().attr(from_name);
        return *static_cast<std::add_pointer_t<wrapper_type>>(this);
    }

    template <typename Func>
    wrapper_type & expose_SimpleArray(char const * name, Func && f)
    {
        namespace py = pybind11;

        using array_reference = typename std::invoke_result_t<Func &, wrapped_type &>;
        static_assert(std::is_reference_v<array_reference>, "this_array_reference is not a reference");
        static_assert(!std::is_const_v<array_reference>, "this_array_reference cannot be const");
        using array_type = typename std::remove_reference_t<array_reference>;

        // Capture the accessor by value so the stored property callbacks do not
        // dangle once this function returns. Build the getter first (a copy),
        // then let the setter take ownership of f, so neither reads a moved-from
        // value regardless of argument evaluation order.
        auto getter = [f](wrapped_type & self) -> array_reference
        { return f(self); };
        (*this)
            .def_property(
                name,
                std::move(getter),
                [f = std::forward<Func>(f)](wrapped_type & self, py::array_t<typename array_type::value_type> & ndarr)
                {
                    array_reference this_array = f(self);
                    if (this_array.nbytes() != static_cast<size_t>(ndarr.nbytes()))
                    {
                        std::ostringstream msg; // NOLINT(misc-const-correctness) tidy bug
                        msg << ndarr.nbytes() << " bytes of input array differ from "
                            << this_array.nbytes() << " bytes of internal array";
                        throw std::length_error(msg.str());
                    }
                    makeSimpleArray(ndarr).swap(this_array);
                })
            //
            ;

        return *static_cast<std::add_pointer_t<wrapper_type>>(this);
    }

    template <typename Func>
    wrapper_type & expose_SimpleArrayAsNdarray(char const * name, Func && f)
    {
        namespace py = pybind11;

        using array_reference = typename std::invoke_result_t<Func &, wrapped_type &>;
        static_assert(std::is_reference_v<array_reference>, "this_array_reference is not a reference");
        static_assert(!std::is_const_v<array_reference>, "this_array_reference cannot be const");
        using array_type = typename std::remove_reference_t<array_reference>;

        // Capture the accessor by value so the stored property callbacks do not
        // dangle once this function returns. Build the getter first (a copy),
        // then let the setter take ownership of f, so neither reads a moved-from
        // value regardless of argument evaluation order.
        auto getter = [f](wrapped_type & self)
        { return to_ndarray(f(self)); };
        (*this)
            .def_property(
                name,
                std::move(getter),
                [f = std::forward<Func>(f)](wrapped_type & self, py::array_t<typename array_type::value_type> & ndarr)
                {
                    array_reference this_array = f(self);
                    if (this_array.nbytes() != static_cast<size_t>(ndarr.nbytes()))
                    {
                        std::ostringstream msg; // NOLINT(misc-const-correctness) tidy bug
                        msg << ndarr.nbytes() << " bytes of input array differ from "
                            << this_array.nbytes() << " bytes of internal array";
                        throw std::length_error(msg.str());
                    }
                    this_array.swap(makeSimpleArray(ndarr));
                })
            //
            ;

        return *static_cast<std::add_pointer_t<wrapper_type>>(this);
    }

    class_ & cls() { return m_cls; }

protected:

    template <typename... Extra>
    // m_cls is initialized in the member initializer list below.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    WrapBase(pybind11::module & mod, char const * pyname, char const * pydoc, const Extra &... extra)
        : m_cls(mod, pyname, pydoc, extra...)
    {
    }

private:

    class_ m_cls;

}; /* end class WrapBase */
#ifdef _MSC_VER
#pragma warning(pop)
#endif

void import_numpy();
#ifdef __GNUC__
#pragma GCC diagnostic push
// Suppress the warning "greater visibility than the type of its field"
#pragma GCC diagnostic ignored "-Wattributes"
#endif
/**
 * Take a pybind11 module and an initializing function and only run the
 * initializing function once.
 */
template <typename T>
class OneTimeInitializer
{

public:

    OneTimeInitializer(OneTimeInitializer const &) = delete;
    OneTimeInitializer(OneTimeInitializer &&) = delete;
    OneTimeInitializer & operator=(OneTimeInitializer const &) = delete;
    OneTimeInitializer & operator=(OneTimeInitializer &&) = delete;
    ~OneTimeInitializer() = default;

    // Do not implement this function as a template. It should use a specialization in only one compilation unit.
    static OneTimeInitializer<T> & me();

    OneTimeInitializer<T> & operator()(
        pybind11::module & mod, std::function<void(pybind11::module &)> const & initializer)
    {
        if (!initialized())
        {
            m_mod = &mod;
            m_initializer = initializer;
            m_initializer(*m_mod);
        }
        m_initialized = true;
        return *this;
    }

    pybind11::module const & mod() const { return *m_mod; }
    pybind11::module & mod() { return *m_mod; }

    bool initialized() const { return m_initialized && nullptr != m_mod; }

private:

    OneTimeInitializer() = default;

    bool m_initialized = false;
    pybind11::module * m_mod = nullptr;
    std::function<void(pybind11::module &)> m_initializer;

}; /* end class OneTimeInitializer */
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
class SOLVCON_PYTHON_WRAPPER_VISIBILITY Interpreter
{

public:

    static Interpreter & instance();

    Interpreter(Interpreter const &) = delete;
    Interpreter(Interpreter &&) = delete;
    Interpreter & operator=(Interpreter const &) = delete;
    Interpreter & operator=(Interpreter &&) = delete;
    ~Interpreter() { finalize(); };

    Interpreter & initialize();
    Interpreter & finalize();
    bool initialized() const { return nullptr != m_interpreter; }

    void preload_module(std::string const & name);
    void preload_modules(std::vector<std::string> const & names);

    Interpreter & setup_modmesh_path();
    Interpreter & setup_process();

    int enter_main();
    void exec_code(std::string const & code);
    std::vector<std::string> get_completions(std::string const & text);
    std::string get_call_tip(std::string const & text);

private:

    Interpreter() = default;

    pybind11::scoped_interpreter * m_interpreter = nullptr;

}; /* end class Interpreter */

#ifdef __GNUC__
#pragma GCC diagnostic push
// Suppress the warning "greater visibility than the type of its field"
#pragma GCC diagnostic ignored "-Wattributes"
#endif
class PythonStreamRedirect
{

public:

    explicit PythonStreamRedirect(bool enabled)
    {
        set_enabled(enabled);
    }

    std::string stdout_string() const;
    std::string stderr_string() const;

    PythonStreamRedirect & set_enabled(bool enabled)
    {
        Toggle::instance().fixed().set_python_redirect(enabled);
        return *this;
    }

    // FIXME: NOLINTNEXTLINE(readability-convert-member-functions-to-static)
    bool is_enabled() const { return Toggle::instance().fixed().get_python_redirect(); }
    bool is_disabled() const { return !is_enabled(); }

    PythonStreamRedirect & activate();
    PythonStreamRedirect & deactivate();

    bool is_activated() const { return static_cast<bool>(m_stdout_backup) || static_cast<bool>(m_stderr_backup); }
    bool is_deactivated() const { return !is_activated(); }

private:

    pybind11::object m_stdout_backup;
    pybind11::object m_stderr_backup;
    pybind11::object m_stdout_buffer;
    pybind11::object m_stderr_buffer;

}; /* end class PythonStreamRedirect */
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

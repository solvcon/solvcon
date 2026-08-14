# Binding C++ to Python

Numerical software must run fast and change fast. The code not only gives
results, but also explains the physics and mathematics behind them. solvcon
runs fast by using C++ in the computing engine, and changes fast by providing a
Python scripting layer.

A system does not become fast by being written entirely in a fast language. It
becomes fast through an optimization loop: (1) profile, (2) find the spot that
dominates the runtime, (3) rewrite to improve that spot, and (4) measure again.
Each pass changes the structure of the program, so a previous hotspot gives way
to a new one.

Cheap restructuring makes the loop easy, and Python enables the cheap
restructuring. This is why the slow Python makes a fast computing system. We
need a binding layer between C++ and Python code. To make the two programming
languages work in harmony and help us find the best place to implement code,
the binding layer needs to provide assistive facilities, in addition to plain
wrapping.

## Rich Binding

solvcon uses a layer on top of pybind11 for wrapping between Python and C++ to
provide features that pybind11 alone does not offer: runtime profiling,
aliasing, wrapper code organization, and `SimpleArray` helpers. The additional
abstraction layer does not cost anything at runtime, because binding code runs
once per process while the module is imported.

The rich binding centers on `WrapBase`, the base class of every wrapper. It is
a class template holding one private data member, `m_cls`, and giving the
derived wrapper a set of aliases to write against.

```cpp
template <class Wrapper, class Wrapped,
          class Holder = std::unique_ptr<Wrapped>,
          class WrappedBase = Wrapped>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapBase;
```

`SOLVCON_PYTHON_WRAPPER_VISIBILITY` is used to keep the symbols from being
exported. pybind11 forces its whole namespace to hidden visibility, so `m_cls`,
a `pybind11::class_`, is a hidden type, and GCC warns when a default-visibility
class holds a hidden field. The macro hides the wrapper to match. Hiding the
symbols keeps the template-heavy mangled names out of the dynamic symbol table,
and prevents symbol clashes with another pybind11.

We use the class attribute rather than a compiler flag so that we can control
which symbols are exported. Carry the macro on every wrapper.

| Parameter     | What it names                        | Default            |
|:--------------|:-------------------------------------|:-------------------|
| `Wrapper`     | the derived wrapper class itself     | none               |
| `Wrapped`     | the C++ class being exposed          | none               |
| `Holder`      | how pybind11 owns the instance       | `std::unique_ptr`  |
| `WrappedBase` | the registered C++ base, if any      | same as `Wrapped`  |

The class template `WrapBase` has four template parameters, as shown in the
table above.  `Wrapper` and `Wrapped` are required.  `Wrapper` is the CRTP
(curiously recurring template pattern) parameter for the derived wrapper
class itself.  `Wrapped` is the C++ class being wrapped.

The other two parameters are optional.  `Holder` can be a `std::unique_ptr`
(default) or `std::shared_ptr`.  When wrapping a class managed exclusively
with `std::shared_ptr`, `Holder` must be set accordingly.  A holder can also
be a foreign smart pointer once pybind11 is told about it, e.g.,
`QPointer<T>`.

`WrappedBase` names the C++ base class of `Wrapped` when that base is also
registered with pybind11, so a Python object of the derived class is
accepted wherever the base is expected.  It defaults to `Wrapped` itself,
which means no base: the `class_` alias passes the extra argument to
`pybind11::class_` only when `WrappedBase` differs from `Wrapped`.  No
wrapper in the tree overrides the default today.

## Singleton for Binding

Registering the same Python class twice in one interpreter is not meaningful.
As such, a `Wrapper` class should be a singleton.  `Wrapper::commit()`
constructs the `Wrapper`, once per process:

```cpp
static wrapper_type & commit(
    pybind11::module & mod, char const * pyname, char const * pydoc)
{
    // Meyers singleton
    static wrapper_type derived(mod, pyname, pydoc);
    return derived;
}
```

The static local is the Meyers singleton: the first call constructs the
wrapper, and every later call returns the same instance, with its `pyname` and
`pydoc` ignored. C++ guarantees the initialization is not raced, which does no
real work here.  In addition, the registration runs under the Python GIL
(global interpreter lock).

We put all wrapping code in the constructor. Because the class is made a
singleton, it is guaranteed to run the wrapping code only once per process.

To make the class a singleton, the constructor is non-public. Every wrapper
carries a `friend root_base_type;` declaration, which is what allows `commit()`
to construct the derived class. Omit it and the build fails at the `commit()`
call site with an access error.

## Augment pybind11 Class Verbs

solvcon uses pybind11 for two-way wrapping. The `WrapBase` binding layer
supports all of pybind11's {doc}`class def verbs <pybind11:classes>` and
augments them. Nearly every pybind11 verb has a `*_timed` counterpart that adds
a node to the solvcon profiling system.

The augmentation is implemented by using function-like C preprocessor macros.
The body is written once and expanded for every verb:

```cpp
#define SC_DECL_PYBIND_CLASS_METHOD_UNTIMED(METHOD)                  \
    template <class... Args>                                         \
    wrapper_type & METHOD(Args &&... args)                           \
    {                                                                \
        m_cls.METHOD(std::forward<Args>(args)...);                   \
        return *static_cast<std::add_pointer_t<wrapper_type>>(this); \
    }
```

The macro returns `*this` (with proper casting) so that calls to `METHOD` can
be chained (i.e., the [fluent
API](https://martinfowler.com/bliki/FluentInterface.html) style). We do not
return `m_cls.METHOD(...)`, which is a `pybind11::class_` and drops the rest of
the chain into pybind11's own type.

The timed macro defines `METHOD##_timed`. The second name is built with the
operator [`##`](https://en.cppreference.com/w/cpp/preprocessor/replace), which
concatenates a macro parameter with adjacent tokens into one identifier (see
also the GCC [concatenation
notes](https://gcc.gnu.org/onlinedocs/cpp/Concatenation.html)). A `*_timed`
verb costs one guard object per call plus the profiler's own bookkeeping. A
binding opts in by writing `def_timed` instead of `def`. We do not time
everything. A rule of thumb is to only time costly functions and leave fast
functions (e.g., array element access) untimed.

The macro uses `mmtag` and `guard_type` (explained later in the section) to
install the solvcon profiler:

```cpp
#define SC_DECL_PYBIND_CLASS_METHOD_TIMED(METHOD)                    \
    template <class... Args>                                         \
    wrapper_type & METHOD##_timed(Args &&... args)                   \
    {                                                                \
        using guard_type =                                           \
            pybind11::call_guard<WrapperProfilerGuard>;              \
        m_cls.METHOD(std::forward<Args>(args)...,                    \
                     mmtag(), guard_type());                         \
        return *static_cast<std::add_pointer_t<wrapper_type>>(this); \
    }
```

`SC_DECL_PYBIND_CLASS_METHOD(METHOD)` expands to both forms at once, so one
line per verb declares the plain and the timed member together:

```cpp
#define SC_DECL_PYBIND_CLASS_METHOD(METHOD)     \
    SC_DECL_PYBIND_CLASS_METHOD_UNTIMED(METHOD) \
    SC_DECL_PYBIND_CLASS_METHOD_TIMED(METHOD)
```

A macro respects no scope, so it is `#undef`ed as soon as its expansions are
done. Leaving it defined would leak it into every file that includes the
header, and `common.hpp` reaches every pymod source in the tree:

```cpp
#define SC_DECL_PYBIND_CLASS_METHOD_UNTIMED(METHOD) ...
#define SC_DECL_PYBIND_CLASS_METHOD_TIMED(METHOD) ...
#define SC_DECL_PYBIND_CLASS_METHOD(METHOD) ...
// ...
SC_DECL_PYBIND_CLASS_METHOD(def)
SC_DECL_PYBIND_CLASS_METHOD(def_static)
// ...
#undef SC_DECL_PYBIND_CLASS_METHOD_UNTIMED
#undef SC_DECL_PYBIND_CLASS_METHOD_TIMED
#undef SC_DECL_PYBIND_CLASS_METHOD
```

The following verbs carry both the plain and the timed form:

- `def` and `def_static`, for methods and static methods
  ({doc}`Object-oriented code <pybind11:classes>`)
- `def_readwrite`, `def_readonly`, `def_readwrite_static`, and
  `def_readonly_static`, for fields ({ref}`pybind11:properties`)
- `def_property`, `def_property_static`, `def_property_readonly`, and
  `def_property_readonly_static`, for properties
  ({ref}`pybind11:properties`, {ref}`pybind11:static_properties`)

`def_buffer`, which implements the
{doc}`buffer protocol <pybind11:advanced/pycpp/numpy>`, gets the untimed form
alone, and not by choice: all three of pybind11's `class_::def_buffer`
overloads take the function and nothing else, so a `def_buffer_timed`
appending two call attributes could not compile.

### Tag for Profiling

A binding selects, member by member, what the profiler sees.  `def` and
`def_timed` differ in exactly one respect: whether the bound member reports
itself to `CallProfiler`, solvcon's runtime call-tree profiler. `mmtag` is an
empty struct that means nothing on its own, serving only as a key for a
pybind11 attribute specialization. pybind11 walks a registration's extra
arguments on every call and invokes `process_attribute<T>::precall` for each,
which inserts a per-call hook without touching the bound function:

```cpp
template <>
struct process_attribute<solvcon::python::mmtag>
    : process_attribute_default<solvcon::python::mmtag>
{
    static void precall(function_call & call)
    {
        if (solvcon::python::WrapperProfilerStatus::me().enabled())
        {
            solvcon::CallProfiler::instance().start_caller(
                get_name(call), nullptr);
        }
    } // get_name() joins the scope __name__ and the function name
};
```

`WrapperProfilerGuard` closes the interval: its constructor records whether
profiling was enabled at entry, and its destructor calls `end_caller()` only
if it was.  The flag is therefore read twice, once in `precall` to decide the
start and once in the guard constructor to decide the end.  The pair stays
balanced because nothing between the two reads releases the GIL, so a toggle
from Python cannot land between them.

## Verbs Added by solvcon

`WrapBase` adds two members that do not have a pybind11 equivalent.

`def_alias(from_name, to_name)` copies an already-registered attribute onto a
second name, which is how a renamed C++ function could keep its old Python
spelling.

`expose_SimpleArray(name, accessor)` registers a read-write property
over a `SimpleArray` member:

```cpp
.expose_SimpleArray("coord", [](wrapped_type & self) -> decltype(auto)
                    { return self.coord(); })
```

It is convoluted to expose a `SimpleArray` to Python without
`expose_SimpleArray()`.

## Wrapper for Class Templates

A wrapper is itself a template when the wrapped class is. `WrapSimpleArray<T>`
in `cpp/solvcon/buffer/pymod/wrap_SimpleArray.hpp` wraps `SimpleArray<T>` for
every element type:

```cpp
template <typename T>
class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapSimpleArray
    : public WrapBase<WrapSimpleArray<T>, SimpleArray<T>>
{
    using root_base_type = WrapBase<WrapSimpleArray<T>, SimpleArray<T>>;
    using wrapped_type = typename root_base_type::wrapped_type;
    friend root_base_type;

    WrapSimpleArray(pybind11::module & mod, char const * pyname,
                    char const * pydoc)
        : root_base_type(mod, pyname, pydoc, pybind11::buffer_protocol())
    { /* ... one long registration chain ... */ }
};
```

The wrapper redeclares `root_base_type` locally, because unqualified lookup
does not reach into a dependent base. The `friend root_base_type;` then works
as in any wrapper. Everything is private to make `commit()` the only legitimate
entry point. Constructors are bound inside that chain with `py::init`, into the
holder chosen above. The class lives in a header so that the per-type
registrations can split across translation units.

Calling `commit()` instantiates the wrapper for one element type, and the
`pyname` argument becomes the Python class name:

```cpp
void wrap_SimpleArray_int(pybind11::module & mod)
{
    WrapSimpleArray<int8_t>::commit(mod, "SimpleArrayInt8", "SimpleArrayInt8");
    // ... int16, int32, int64, and the matching collectors ...
}
```

## Wrapper Grouping Helper

A wrapper with dozens of registrations becomes one unreadable chain.  The
remedy is a private member per group returning `wrapper_type &`, which is also
where a conditional registration belongs (e.g., using `if constexpr`).

```cpp
wrapper_type & wrap_matrix()
{
    (*this)
        .def_static("eye", &wrapped_type::eye, py::arg("n"), "...")
        .def("trace", &wrapped_type::trace, "...");
    return *this;
}
```

## Organize Files for Wrapper and Binding

Each subsystem under `cpp/solvcon/` keeps its bindings in a `pymod/`
subdirectory, conventionally holding three kinds of file.

| File               | What it holds                                    |
|:-------------------|:-------------------------------------------------|
| `<name>_pymod.hpp` | the declarations the subsystem exports           |
| `<name>_pymod.cpp` | the tag, the one-time gate, `initialize_<name>`  |
| `wrap_<Class>.cpp` | one wrapper class or family and its commit       |

It is a convention, not a rule. When seeing an exception, look around before
forming an opinion.

Open every pymod source and header with a header that brings in pybind11,
marked `// Must be the first include.`, before anything else the unit includes.
The rule comes from CPython: `Python.h` defines feature-test macros, such as
`_POSIX_C_SOURCE`, that change what the standard headers declare, so it must be
included before any standard header or the unit sees inconsistent declarations
(see {ref}`Include Files <python:api-includes>` in the Python C API manual).
pybind11 includes `Python.h`, so putting the pybind11-bearing header first also
satisfies the rule.

## Initialization Order

Wrapper initialization has two requirements: each subsystem registers exactly
once per process, and the subsystems register in a clear order. We use a
per-subsystem gate for the first, and a single driver function for the second.

### Registration Gating

The `OneTimeInitializer` class template implements the gate with a tag class.
`<name>_pymod.cpp` includes the code:

```cpp
template <>
OneTimeInitializer<toggle_pymod_tag> &
OneTimeInitializer<toggle_pymod_tag>::me()
{
    static OneTimeInitializer<toggle_pymod_tag> instance;
    return instance;
}

void initialize_toggle(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    { wrap_Toggle(mod); };
    OneTimeInitializer<toggle_pymod_tag>::me()(mod, initialize_impl);
}
```

The tag gives each subsystem (`toggle` in this example) its own gate.
`OneTimeInitializer` is a class template, so each distinct template argument
mints a distinct type with its own `me()` singleton and initialized flag.

`operator()` is the gate itself:

```cpp
OneTimeInitializer<T> & operator()(
    pybind11::module & mod,
    std::function<void(pybind11::module &)> const & initializer)
{
    // Run the initializer only when the initialized bit is not set.
    if (!initialized())
    {
        m_mod = &mod;
        m_initializer = initializer;
        m_initializer(*m_mod);
    }
    m_initialized = true;
    return *this;
}
```

### Registration Order

The driver `solvcon::python::initialize()` is in
`cpp/solvcon/python/module.cpp` and calls every `initialize_*` in a fixed
order. `initialize_buffer` runs `import_numpy()` first, and nothing may touch
the numpy C API before that.

The driver is run in both module and embedded mode. `PYBIND11_MODULE(_solvcon,
mod)` in `cpp/binary/pymod_solvcon/module.cpp` is one call to `initialize()`,
and `cpp/binary/pilot/pilot.cpp` declares the same module with
`PYBIND11_EMBEDDED_MODULE` calling the same function, so the pilot's embedded
interpreter sees exactly the `_solvcon` that an ordinary `import` sees.

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->

# Coding Style Guide

A code style guideline is to help developers align how they write and change
code. The consistency reduces the cost to maintain and develop the code, and
the former matters more than the latter, because the former costs more than the
latter.

solvcon uses [clang-format](https://clang.llvm.org/docs/ClangFormat.html) to
lint C++ code and [flake8](https://flake8.pycqa.org/) to lint Python code
according to [PEP-8](https://peps.python.org/pep-0008/). We mind the code
style when adding new code and changing existing code. The rules of thumb are:

1. The linters must be clean. Before creating and updating a
   [pull request](https://docs.github.com/en/pull-requests/), run:

   ```bash
   make lint
   ```

2. Read the code nearby and follow the style. Start from the functions and
   classes that the code resides in. Then get familiar with the style in the
   file and follow it. Familiar with the code in the module(s) if time permits.
3. Use the style guide.

## Line Economy and Breathing Room

Two forces balance here, and they do not conflict. Line economy keeps a single
thought on as few lines as it honestly needs. Breathing room separates distinct
thoughts with a blank line so a reader can see the structure. Economy fights
spreading one idea across many lines; breathing room fights packing many ideas
into one unbroken block. Both serve the same goal: code a human can scan.

### Keep one thought compact

Prefer fewer lines for a single thought. Dense code that fits within the
line-width limits is easier to scan than the same logic spread across many
lines. This applies to both C++ and Python.

- Group related short declarations on one line when natural, e.g.
  `double x, y;`.
- Prefer compact forms over spread-out forms when both are equally clear.
- Do not pad a short block: a 3-line function body needs no internal blank
  lines.
- Respect the linting line-width limits (below); never sacrifice them to
  shorten the line count.
- Never put two consecutive executable statements (separated by `;`) on one
  line, because debuggers and stack traces need line granularity. A
  single-statement inline body like `void set_flag(bool v) { m_flag = v; }` is
  one statement, not two, and remains the preferred accessor form.

### Let the code breathe

A longer body packed edge to edge with no blank line is hard to read even when
every line is necessary. The most common failure is the opposite of padding: a
wall of statements with no seams. Keep consecutive statements of the same
construct together, then insert one blank line at the seam between those groups
so each reads as a unit. Blank lines pay off once a body is long enough to have
seams; a short body has none to mark. Let the blank line mark the seam; do not
reach for a narrating comment to do it (see "Comments").

```cpp
void Solver::march_one_step()
{
    SimpleArray<double> left = gather_left();
    SimpleArray<double> right = gather_right();
    SimpleArray<double> flux = solve_riemann(left, right);

    // A couple of more lines ...

    apply_flux(flux);
    advance_time();
}
```

Where a blank line earns its place:

- Between functions, methods, and class definitions (required).
- Between the logical sections of a longer body, one blank line per seam.
- Around access specifiers in a C++ class, and between accessor pairs.
- Before a multi-line expression or a concluding `return` that wants setting
  off from the work above it.

Where it does not:

- Inside a short block that is already a single thought.
- More than one consecutive blank line.
- Right after an opening brace or right before a closing brace.

This is a readability guideline in both directions. Do not compress until the
code is an unbroken wall, and do not pad until the logic drowns in whitespace.
When either rule hurts clarity, choose clarity.

## Indentation and file format

Use 4 white spaces for indentation. Do not use a tab.

C++ files do not have a text width limit, but it is good for a line to be less
than 120 characters. Python files should use a text width of 79 characters.

Use UTF-8 as file encoding and [UNIX text file
format](http://en.wikipedia.org/wiki/Newline). Do not use DOS file format.

### Vim modelines

Even if you do not use vim, add the modeline at the end of files to document
the required file format:

* `ff=unix`: Use the [UNIX text file
  format](http://en.wikipedia.org/wiki/Newline) (`\n` line ending).
* `fenc=utf8`: Use UTF-8 for encoding.
* `et`: Expand tabs. Do not use tabs for solvcon.
* `sw=4 ts=4 sts=4`: Use 4 white spaces for tabs.

The modeline for C++ is:

```cpp
// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
```

The modeline for Python is:

```python
# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
```

In Python, set the text width to 79 (`tw=79`).

### Space and Blank Line

Leave a space behind `,` (commas):

```cpp
void help_something(int32_t serial, double value);
```

Use a blank line between the definitions of classes and functions.

## Naming

Do not use a name (especially for a variable) with only 1 character.

Prefer to use `UPPER_CASE` for constants. In C++ sometimes `snake_case`
is preferred when it involves a foreign code base.

Functions and variables use `snake_case` and classes use `CamelCase` in
both C++ and Python.

Member data and functions in a C++ class use the same naming convention
regardless of access (`public`, `protected`, and `private`). Member data
should be prefixed with `m_` like `m_snake_case`, unless it is for a POD
(plain-old-data) struct.

C++ types (classes) for type aliasing and template meta-programming follow STL
and use `snake_case_t` or `snake_case_type`, e.g., `size_type`

```cpp
class MyPowerHouse
{

public:

    void do_something();

private:

    void help_something();

    int32_t m_serial_number;

}; /* end class MyPowerHouse */

struct PureData
{

    // Member data names in POD are usually short for easy access.
    int32_t serial;
    double x, y;

}; /* end struct PureData */
```

In a Python class, public attributes and methods (member functions) use normal
`snake_case`. Non-public (nothing is really private in Python) attributes and
methods use `_leading_underscore_snake_case` (unmangled) and
`__double_leading_underscore_snake_case` (mangled).

Python exceptions are Python classes and use `CamelCase`.

Do the best to name a function like `verb_objective()` (in both C++ and
Python).

```python
# function.
take_some_action(from_this, with_that)
# method.
some_object.do_something(with_some_information)
```

### Acronym

Treat acronyms like a word. Do not make them all-upper-cases in names.
The rule holds in both C++ and Python, and applies no matter how short
the acronym is or how habitually it is written in capitals elsewhere.

```cpp
// "Http" is treated like a word in CamelCase.
class HttpRequest
{
    // "http" is treated like a word in snake_case.
    void update_http_header();
} /* end class HttpRequest */
```

```python
# "Ui" is treated like a word in CamelCase, so it is not "UIState".
class UiState(object):
    # "ui" is treated like a word in snake_case.
    def apply_ui_section(self, section):
        pass
```

### Qt

For Qt sub-classes, follow the Qt naming style, but prefix with `R` instead
of `Q` and put them in the `solvcon` namespace. (Why "`R`"? It is the
next character than "`Q`" and we want to distinguish the classes derived in
solvcon.) Use `camelCase` (note the leading lower-case character) for
functions. Member data should use `m_snake_case` as other solvcon C++ class.

### Iterating Counter

Iterating counters start with `i`, `j`, `k`.

- Trivial indexing variables can be named as `it`, `jt`, or `kt`.
- Standalone `i`, `j`, and `k` should never be used to name a variable
  because they are too short.

### Shorthands for Unstructured Meshes

Code for the unstructured meshes carries geometrical terms and needs shorthands
to keep the line width reasonable.

- Two-character names for nodes, faces, and cells:

  - `nd`: node/vertex.
  - `fc`: face.
  - `cl`: cell.
- For example, `icl` is a counter of cell.
- The following prefices often (but not always) means serial numbers:

  - `nxx`: number of `xx`, e.g., `ncl` is number of cells.
  - `mxx`: maximum number of `xx`, e.g., `mfc` is the maximum number of
    faces.

More examples:

- `clnnd` means number of nodes belonging to a cell.
- `FCMND` means maximum number of nodes for a face.
- `icl` means the first-level (iterating) index of cell.
- `jfc` means the second-level (iterating) index of face.
- Some special iterators used in code, such as:

  - `clfcs(icl, ifl)`: get the `ifl`-th face in `icl`-th cell.
  - `fcnds(ifc, inf)`: get the `inf`-th fact in `ifc`-th face.

## Python import

Never import everything ("`import *`" or "`from mod import *`").

Only import modules, not module content (classes, functions, or constants).
Use `from` to specify the module path and `import` to import the module:

```python
# Mind the order of the lines importing the modules.
# Modules in standard library.
import os
import sys
import typing
import dataclasses
import pathlib

# Modules from third-party.
import numpy as np
from matplotlib.backends import backend_qtagg
from matplotlib import figure
from PySide6 import QtCore, QtWidgets, QtGui

# Modules in the current project.
import solvcon as mm
from solvcon import onedim
from solvcon.plot import svg

# Explicit relative import is OK.
from . import core
from . import _base_app
```

> **Note:**
>
> `solvcon` can be shorthanded as `mm`.

Do not import module content (classes, functions, or constants) directly.
Always use the `foo.bar` pattern to access classes, functions, or constants:

```python
# BAD: imports a class, not a module.
from typing import Any, Callable
from dataclasses import dataclass
from pathlib import Path

def foo(x: Any) -> Callable:
    ...
p = Path("/tmp")

# GOOD: import the module and access content through it.
import typing
import dataclasses
import pathlib

def foo(x: typing.Any) -> typing.Callable:
    ...
p = pathlib.Path("/tmp")
```

```python
# BAD: imports content from a sub-module.
from matplotlib.figure import Figure
from ._base_app import QuantityLine

fig = Figure()
line = QuantityLine(...)

# GOOD: import the module.
from matplotlib import figure
from . import _base_app

fig = figure.Figure()
line = _base_app.QuantityLine(...)
```

> **Note:**
>
> **Exception for Qt (PySide6):** Qt classes may be imported directly because
> almost all Qt classes have a `Q` or `Qt` prefix that makes them
> unmistakable, the module-qualified form (e.g., `QtWidgets.QDockWidget`)
> reads mouthy with repeated `Q`, and the C++ counterpart does not use the
> module name.
>
> ```python
> # OK: import Qt classes directly.
> from PySide6.QtCore import QTimer, Slot, Qt
> from PySide6.QtWidgets import QDockWidget, QWidget
> ```

Use relative import for peer modules in the same package:

```python
# For a module file in solvcon/pilot/
# BAD: use absolute import for peer modules.
from solvcon.pilot import _gui
from solvcon.pilot.airfoil import _naca

# GOOD: use relative import for peer modules.
from . import _gui
from .airfoil import _naca
```

Relative import may not be required for modules outside the current package:

```python
# For a module file in solvcon/pilot/
# GOOD: use absolute import for non-peer modules.
from solvcon.plot import svg

# OK but may be clumsy: use relative import.
from ..plot import svg
```

Do not use dotted import path with `import` for project modules:

```python
# BAD: uses dotted path instead of from...import.
import solvcon.plot.svg as svg

# GOOD: use from to specify the path.
from solvcon.plot import svg
```

Do not import multiple modules in one line:

```python
# BAD BAD BAD
import os, sys
```

Never do implicit relative import:

```python
# BAD for modules in the current project.
import onedim
```

## NumPy Array Creation

Always specify the `dtype` when creating a NumPy `ndarray`, and spell the
dtype as a string. The default floating-point and integer types depend on the
platform and the NumPy version, so an explicit dtype keeps the array layout
deterministic and mirrors the fixed-width integer rule below.

```python
# GOOD: explicit dtype given as a string.
a = np.empty(10, dtype='float64')
b = np.zeros((3, 4), dtype='int32')
c = np.array([1, 2, 3], dtype='uint8')

# BAD: no dtype; the type is implicit and platform-dependent.
a = np.empty(10)
b = np.zeros((3, 4))

# BAD: dtype given as a type object instead of a string.
a = np.empty(10, dtype=np.float64)
```

This applies to every array-creating call (`np.empty`, `np.zeros`, `np.ones`,
`np.full`, `np.array`, `np.arange`, etc.).

## Testing

Python tests are the default. Write tests in Python (`tests/`, pytest, files
named `test_*.py`) whenever the code can be exercised through the Python
bindings. Use C++ gtest (`gtests/`, files named `test_nopython_*.cpp`) only
when the code cannot or should not be tested from Python -- for example,
internals with no Python binding, or behavior that must be verified at the C++
level. Do not duplicate the same test in both layers without a reason.

A test should encode why a behavior matters, not merely what it does. A test
that cannot fail when the logic it covers changes is not pulling its weight.

## Comments

Code says what happens. A comment says why, and what a reader must know but
cannot see. The default is no comment: add one only when it carries
information the code itself cannot, and prefer fewer, load-bearing comments
over many thin ones. When in doubt, leave it out. Write comments clearly and
concisely, and keep two levels separate:

- Interface comments sit before a function, class, or member and describe how
  to use it: what it does, the meaning and units of each argument and return
  value, who owns any allocated memory, which values are sentinels (a null
  pointer, `-1`, an empty array), and the invariants the caller must uphold.
- Implementation comments sit inside the body and explain what reading it does
  not reveal: a non-trivial algorithm, why one approach was chosen over a
  viable alternative, or a single tricky line. Do not repeat the interface
  comment.

**Never write a comment that only restates the code; it is noise the next reader
must wade through and then delete.** In particular, do not write any of these:

- A paraphrase of the line below it: `// increment the counter` over
  `++count;`, or `# return the result` over `return result`.
- A label for self-evident structure: `// constructor`, `// the main loop`,
  `// imports`, `# getter for width`, or a banner over code short enough to
  read at a glance.
- A step narration that names the call: `// run the solver` over
  `solver.run()`.
- Prose that repeats an already-clear name: `# the cell count` over
  `cell_count`.
- Explain an obvious value or type: `// An absent optional is "null"` over 
  a `std::optional`.

If the only honest comment you can write restates the code, write none. A clear
name beats a comment; rename rather than annotate.

Because this codebase solves numerical problems, state the physical and
structural facts the types do not encode:

- Units, coordinate conventions, and index bases.
- The expected shape, layout, or contiguity of an array or buffer.
- Invariants tying several variables together (e.g. a count that must equal a
  buffer length).
- For any formula, cite the literature, equation, or document it comes from.

Further conventions:

- Prefer a clear name over a comment; rename rather than explain an obscure
  name. A comment cannot rescue an unclear design.
- Keep the rationale next to the code, not only in a commit message or pull
  request. When you change the code, change the comment in the same edit.
- Write comments as full sentences with capitalization and a closing period.
  The verb mood differs by language -- the C++ and Python sections below fix
  it. Use ASCII only.
- A comment describes the code, never the task or conversation that produced
  it. Do not write "as requested" or reference a review thread.
- If possible, provide references (with URL) to literature or documents in
  comments.

### C++ Comment

C++ interface comments are [Doxygen](https://www.doxygen.nl/manual/docblocks.html)
blocks. The general comment rules above still apply (full sentences,
capitalized, closing period, ASCII only; state units, coordinate conventions,
index base, array shape and contiguity, and a literature or equation citation
with URL for any formula). Doxygen output is XML only, consumed by breathe and
rendered by Sphinx (see `doc/Doxyfile` and `doc/source/api/cpp.md`).

- **Markers.** Use `/** ... */` for any block of two or more lines, `///` for a
  one-line brief on a declaration, and `///<` for a trailing comment on a
  member or enumerator. Do not use `/*!`, `//!`, or a plain `//` or `/* */`
  comment for interface documentation that must appear in the reference.
- **Commands.** Always use the at-sign form (`@brief`, `@param`, `@file`).
  Backslash commands are not allowed.
- **Brief.** `JAVADOC_AUTOBRIEF` is on, so the first sentence is the brief.
  Open each block with one capitalized sentence ending in a period. Add an
  explicit `@brief` only when the brief spans multiple lines or its first
  sentence contains an abbreviation or version number that would truncate it.
- **Required tags.** Every header carries an `@file` block (bare `@file`,
  description on the next line) with one `@ingroup` for its architectural
  group. Every public class, struct, enum, and function has a brief; template
  parameters use `@tparam`; value-returning functions use `@return` (not
  `@returns`). Document each parameter with `@param`; use directional
  `@param[in]`, `@param[out]`, or `@param[in,out]` only when an output
  parameter is present, and then on every parameter of that function. Members
  and enumerators carry a `///<` brief when their meaning, units, or sentinel
  value is not obvious.
- **Placement.** The doc block lives on the declaration in the header, never
  duplicated on the `.cpp` definition. Implementation notes inside a `.cpp`
  body stay plain `//`, or use `@internal` to keep them out of the public
  reference.
- **Grouping.** Coarse topic groups are defined once in `doc/groups.dox` with
  `@defgroup`; tag types with one `@ingroup`. Inside a large class, group
  related members with `@name` and `///@{ ... ///@}`. Keep grouping shallow.
- **Markdown.** Doxygen Markdown is on, but the comment is parsed by Doxygen
  and serialized to XML; Sphinx and MyST never see the raw text. Use `-` (not
  `*`) for bullet lists, because leading-asterisk stripping breaks `*` bullets.
  Use double backticks around inline code that contains a single quote. Prefer
  `@ref Name` over a bare Markdown link for an internal cross-reference so the
  breathe resolver can link it.

### Python Comment

The general comment rules above apply to Python. The interface comment is the
docstring, not a `#` comment: give every public module, class, function, and
method a `"""triple-quoted"""` docstring, per [PEP
257](https://peps.python.org/pep-0257/). Reserve `#` comments for
implementation notes inside a body.

- If a function or class is obvious, do not add a docstring.
- Begin a docstring with a one-line summary that ends with a period. It
  prescribes what a call does as a command ("Run the solver.", "Return the cell
  count.").
- For a multi-line docstring, follow the summary with a blank line, then the
  detail, and keep the closing `"""` on its own line.
- When documenting arguments and return values, use the
  Sphinx/reStructuredText field syntax already used in the codebase (`:param
  name:`, `:return:`, `:rtype:`) rather than introducing a second docstring
  dialect.

## Integer Type

Use fixed-width integers (`int32_t`, `uint8_t`, etc.) Do not use the basic
integer types (`int`, `long`, etc.) unless there is not another choice.

## C++ Include File

The inclusion guard uses `#pragma once` in the first line before everything.

Always use path-first inclusion (angle bracket). Do not use current-first
(double quote).

```cpp
// Use this: search for include file start with the paths to the compiler.
#include <solvcon/base.hpp>
// Do not use this. This starts to search from the directory of the file.
#include "solvcon/toggle.hpp"
```

## C++ Namespace

Put everything in the `solvcon` namespace.

Never `using namespace` outside a local scope (like a function). Another
namespace is not a local scope and should not `using namespace`. When
accessing something in a namespace (e.g., `solvcon`) from outside, spell
out the full name:

```cpp
// An anonymous namespace
namespace
{

solvcon::real_type local_function(solvcon::int_type value);

} /* end namespace */
```

The namespace `solvcon` may be aliased to `mm` in a local scope. No alias
should be use outside a local scope.

```cpp
solvcon::real_type local_function(solvcon::int_type value)
{
    // Alias the solvcon namespace to mm.
    namespace mm = solvcon;
    return mm::real_type(value); // Same as solvcon::real_type(value);
}
```

Needless to say that `using namespace std;` is absolutely forbidden.

### Implementation Detail

Name the namespace for implementation details to `detail`.

```cpp
namespace solvcon
{

namespace detail
{
    // Implementation detail
} /* end namespace detail */

} /* end namespace solvcon */
```

## C Pre-Processor Macro

Prefix macros with `MM_DECL_`. If they are not supposed to be used as a
global helper, delete them after consumption.

## C++ Standard

Use C++-17 and beyond.

Follow the [rule of
five](https://en.cppreference.com/w/cpp/language/rule_of_three). Most of the
time just spell out all default implementation of constructors and assignment
operators and group them together:

```cpp
class MyClass
{
public:
    // Listing all default implementation will make the intention clear and
    // it is easier to change from default to delete.

    // Default constructor.
    MyClass() = default;
    // Copy constructor.
    MyClass(MyClass const &) = default;
    // Move constructor.
    MyClass(MyClass &&) = default;
    // Copy assignment operator.
    MyClass & operator=(MyClass const &) = default;
    // Move assignment operator.
    MyClass & operator=(MyClass &&) = default;
    // Destructor.
    ~MyClass() = default;
}; /* end class MyClass */
```

## C++ Encapsulation

Prefer encapsulated classes over POD struct so that we always provide
accessors. We provide accessors for even scalars of fundamental types.

```cpp
class MyPowerHouse
{

public:

    void calculate_internal_data();

    // Use the same-name style for accessors.
    double internal_value() const { return m_internal_value; }
    double & internal_value() { return m_internal_value; }

    // It may be good to have a blank line between accessor pairs.
    SimpleArray<double> const & internal_data() const { return m_internal_data; }
    SimpleArray<double> & internal_data() { return m_internal_data; }

private:

    double m_internal_value = 0.0;
    SimpleArray<double> m_internal_data;

}; /* end class MyPowerHouse */
```

### Prefer Same-Name Accessors

(Python does not need accessors. Do not add accessors in Python code.)

Prefer same-name accessors because we expose a lot of internal containers:

```cpp
// Getter is const and return a copy of a fundamental type.
double internal_value() const { return m_internal_value; }
// Setter is non-const and return a reference.
double & internal_value() { return m_internal_value; }

// Getter is const and return a const reference of a non-fundamental type.
SimpleArray<double> const & internal_data() const { return m_internal_data; }
// Setter is non-const and return a reference.
SimpleArray<double> & internal_data() { return m_internal_data; }
```

Sometimes we may use the getter-and-setter style to supplement the same-name
accessors:

```cpp
// Getter is const and return a copy of a fundamental type.
double get_internal_value() const { return m_internal_value; }
// Setter takes
void set_internal_value(double v) { m_internal_value = v; }

// Getter is const and return a const reference of a non-fundamental type.
SimpleArray<double> const & internal_data() const { return m_internal_data; }
// Setter is non-const and return a reference.
SimpleArray<double> & internal_data() { return m_internal_data; }
```

It is OK for accessors of the same-name and getter-and-setter styles to be
available for the same member datum, but we should only do it when necessary.

## C++ Ending Mark

Add ending marks to classes and namespaces.  They are usually too long (across
hundreds of lines) to keep track of.

```cpp
namespace solvcon
{

class MyClass
{
    // Code.
}; /* end class MyClass */

} /* end namespace solvcon */
```

## C++ STL Containers

Replace `std::vector` with `SimpleCollector` when `value_type` is a
fundamental type. Use `small_vector` for a small amount of data.

Do not use STL containers for member data unless it is just in a prototype
phase. In that case, add a `TODO` comment and create a follow-up PR or issue
to replace them:

```cpp
class MyClass
{

private:

    // GOOD: use SimpleCollector for fundamental types.
    SimpleCollector<double> m_values;

    // BAD: do not use std::vector for member data.
    std::vector<double> m_values;

    // OK only in prototype phase with a TODO comment.
    // TODO: Replace with SimpleCollector (see issue #NNN).
    std::vector<double> m_values;

}; /* end class MyClass */
```

For local variables, STL is sometimes acceptable but discouraged:

```cpp
void do_something()
{
    // Discouraged but sometimes OK for local variables.
    std::vector<int32_t> temp_indices;
}
```

## C++ Function Body Placement

Move non-accessor function bodies to be outside the class declaration when the
code is not 2-3 times longer than an accessor. Keep short accessors inline in
the class declaration as described in the encapsulation section. Other function
bodies should be defined outside.

If a function body is very simple (e.g., a single return or assignment
statement), write it as a one-liner to keep the code compact:

```cpp
// GOOD: very simple function as a one-liner.
double internal_value() const { return m_internal_value; }
void set_flag(bool v) { m_flag = v; }

// BAD: unnecessary multi-line form for a trivial body.
double internal_value() const
{
    return m_internal_value;
}
```

Other function bodies should be defined outside:

```cpp
class MyPowerHouse
{

public:

    // Short accessors stay inline in the class.
    double internal_value() const { return m_internal_value; }
    double & internal_value() { return m_internal_value; }

    // Declare non-trivial functions in the class, define outside.
    void calculate_internal_data();

private:

    double m_internal_value = 0.0;

}; /* end class MyPowerHouse */

// Define non-accessor function bodies outside the class.
void MyPowerHouse::calculate_internal_data()
{
    m_internal_value = 42.0;
    // ... more logic ...
}
```

## C++ pybind11 Binding Style

When writing pybind11 bindings, separate constructors and other bindings
(methods, properties, etc.) into two `(*this)` sections for readability. This
can also be addressed in a future PR if not done immediately:

```cpp
// Inside a wrapper class constructor:
(*this)
    .def(pybind11::init<>())
    .def(pybind11::init<int32_t>())
    //
    ;

(*this)
    .def("do_something", &wrapped_type::do_something)
    .def_property_readonly("value", &wrapped_type::value)
    //
    ;
```

## C++ Curly Braces

Always add curly braces and always add them in standalone lines:

```cpp
if (condition)
{
    return;
}
```

That is, never drop curly braces even when you can:

```cpp
// NEVER DROP CURLY BRACES
if (condition)
    return;
```

## Commit Log

A commit message records why a change happened, which the diff cannot show.
Write it for the person who runs `git log` or `git blame` a year from now.  We
do not use semantic (conventional) commit prefixes such as `feat:` or `fix:`.

We write comments clearly and concisely, like carefully-thinking professionals
do.

Structure a message as a subject line, a blank line, and an optional body:

- Write the subject in the imperative mood, completing the sentence "If
  applied, this commit will ..." -- "Add the oblique-shock Euler driver", not
  "Added ..." or "Adds ...". This matches git's own messages ("Merge ...",
  "Revert ...").
- Capitalize the subject and do not end it with a period.
- Aim for a subject of 50 characters; treat 72 as the hard limit.
- Separate the subject from the body with one blank line; many git tools rely
  on this split.
- Wrap the body at 72 characters. Git does not wrap it for you.

Use the body to explain what changed and why, not how -- the diff already
shows how. Make it self-contained: a reader should judge the change without
opening the patch. Describe the behavior before the change in the present
tense ("The solver drains to vacuum at a slip wall"), then say why the new
behavior is better. If you considered and rejected an alternative, name it. Do
not point at a chat log or mailing-list thread for the reasoning; summarize it
in the message.

A one-line subject is enough when the change is trivial and needs no context
(e.g. "Fix typo in the buffer header"). Add a body whenever the reason is not
obvious from the subject.

Do not reference issues or PRs directly in the commit log, unless you have to.
In the rare occasions to reference issues, follow the rule in `CLAUDE.md`: end
the body with "Related to #xxx" or "For issue #xxx". Never use closing keywords
(`close`, `fixes`, `resolves`, ...); commit text must not drive issue
management.

## Copyright Notice

solvcon uses the [BSD license](http://opensource.org/licenses/BSD-3-Clause).
When creating a new file, put the following text at the top of the file
(replace `<Year>` with the year you create the file and `<Your Name>` with
your name and maybe email).  The license text formatted for C++ files:

```cpp
/*
 * Copyright (c) <Year>, <Your Name>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the copyright holder nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
```

The license text formatted for Python files:

```python
# -*- coding: UTF-8 -*-
#
# Copyright (c) <Year>, <Your Name>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
```

<!-- vim: set ft=markdown ff=unix fenc=utf8 et sw=2 ts=2 sts=2 tw=79: -->

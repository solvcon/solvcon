:orphan:

==================
Python Style Guide
==================

It's important to have consistent coding style.  Quoted from
http://google-styleguide.googlecode.com/svn/trunk/pyguide.html:

  BE CONSISTENT.

  If you're editing code, take a few minutes to look at the code around you and
  determine its style. If they use spaces around all their arithmetic
  operators, you should too. If their comments have little boxes of hash marks
  around them, make your comments have little boxes of hash marks around them
  too.

  The point of having style guidelines is to have a common vocabulary of coding
  so people can concentrate on what you're saying rather than on how you're
  saying it. We present global style rules here so people know the vocabulary,
  but local style is also important. If code you add to a file looks
  drastically different from the existing code around it, it throws readers out
  of their rhythm when they go to read it. Avoid this.

Import
======

Only import modules, like:

.. code-block:: python

  # modules in standard library.
  import os
  import sys

  # modules from third-party.
  import numpy as np

  # modules in the current project.
  import solvcon as sc
  from solvcon import boundcond
  from solvcon.io import vtkxml

  # explicit relative import is OK.
  from . import solver
  from . import case

Never import multiple modules in one line:

.. code-block:: python

  # BAD BAD BAD
  import os, sys

Never do implicit relative import:

.. code-block:: python

  # BAD for modules in the current project.
  import block

Py3k Compatibility
==================

Enable three Py3k features by adding the following line at top of modules:

.. code-block:: python

  from __future__ import division, absolute_import, print_function

Indentation
===========

Always use **four** white spaces for indentation.  Never use a tab.  Below is
an example vim mode line for Python:

.. code-block:: python

  # vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:

It's good to limit a line to 79 characters.  Width of everyone's monitor is
different.

File Format
===========

- Always use UTF-8 as file encoding.
- Always use `UNIX text file format <http://en.wikipedia.org/wiki/Newline>`__;
  NEVER use DOS text file format.

Blank Lines
===========

Major sections are seperated by two blank lines, while lower-level entities use
one blank line.

.. code-block:: python

  import os
  import sys


  class Class(object):
      def __init__(self):
          pass

      def method(self):
          pass


  class Another(object):
      def __init__(self):
          pass

Naming
======

Here show some naming rules that help readability.  These conventions should be
followed as much as possible, so that the code can be self-explanary.

- Names of frequently used instances should use 3 letters:

  - ``blk``: :py:class:`Block <solvcon.block.Block>`.
  - ``svr``: :py:class:`MeshSolver <solvcon.solver.MeshSolver>`.
  - ``ank``: :py:class:`MeshAnchor <solvcon.anchor.MeshAnchor>`.
  - ``cse``: :py:class:`MeshCase <solvcon.case.MeshCase>`.
  - ``hok``: :py:class:`MeshHook <solvcon.hook.MeshHook>`.
- The following two-character names have specific meaning:

  - ``nd``: node/vertex.
  - ``fc``: face.
  - ``cl``: cell.
- The following prefices often (but not always) have specific meanings:

  - ``nxx``: number of ``xx``.
  - ``mxx``: maximum number of ``xx``.
- Names of iterating counters start with ``i``, ``j``, ``k``, e.g., ``icl``
  denoting a counter of cell.

  - However standalone ``i``, ``j``, and ``k`` should NEVER be used to name a
    variable.  Variables must not use only one character.
  - Trivial indexing variables can be named as ``it``, ``jt``, or ``kt``.

For example,

- ``clnnd`` means number of nodes belonging to a cell.
- ``FCMND`` means maximum number of nodes for a face.
- ``icl`` means the first-level (iterating) index of cell.
- ``jfc`` means the second-level (iterating) index of face.
- Some special iterators used in code, such as:

  - ``clfcs[icl,ifl]``: get the ``ifl``-th face in ``icl``-th cell.
  - ``fcnds[ifc,inf]``: get the ``inf``-th fact in ``ifc``-th face.

Other than the above specific rules, here is a table for other stuff:

.. list-table:: General Naming Convention
  :widths: 15 10 25
  :header-rows: 1

  - - Type
    - Public
    - Internal
  - - Packages
    - ``lower_with_under``
    -
  - - Modules
    - ``lower_with_under``
    - ``_lower_with_under``
  - - Classes
    - ``CapWords``
    - ``_CapWords``
  - - Exceptions
    - ``CapWords``
    -
  - - Functions
    - ``lower_with_under()``
    - ``_lower_with_under()``
  - - Global/Class Constants
    - ``CAPS_WITH_UNDER``
    - ``_CAPS_WITH_UNDER``
  - - Global/Class Variablesi
    - ``lower_with_under``
    - ``_lower_with_under``
  - - Instance Variables
    - ``lower_with_under``
    - ``_lower_with_under`` (protected) or ``__lower_with_under`` (private)
  - - Method Names
    - ``lower_with_under()``
    - ``_lower_with_under()`` (protected) or ``__lower_with_under()`` (private)
  - - Function/Method Parameters
    - ``lower_with_under``
    -
  - - Local Variables
    - ``lower_with_under``
    -

It's good to name functions or methods as ``verb_objective()``, such that code
can look like:
  
.. code-block:: python
  
  # function.
  make_some_action(from_this, with_that)
  # method.
  some_object.do_something(with_some_information)

Copyright Notice
================

SOLVCON uses the `BSD license <http://opensource.org/licenses/BSD-3-Clause>`__.
When creating a new file, put the following text at the top of the file
(replace ``<Year>`` with the year you create the file and ``<Your Name>`` with
your name and maybe email)::

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

The first line tells Python interpreter to use UTF-8, as required in `File
Format`_.  It is not part of the copyright notice.

.. vim: set ft=rst ff=unix fenc=utf8 ai:

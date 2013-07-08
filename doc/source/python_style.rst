==================
Python Style Guide
==================

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

And for C:

.. code-block:: c

  // vim: set ff=unix fenc=utf8 ft=c ai et sw=4 ts=4 tw=79:

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

.. vim: set ft=rst ff=unix fenc=utf8 ai:

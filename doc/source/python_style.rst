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

Indentation
===========

Always use **four** white spaces for indentation.  Never use a tab.

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

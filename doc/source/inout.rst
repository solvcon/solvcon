===========================
Input and Output Facilities
===========================

.. py:module:: solvcon.io

.. py:module:: solvcon.io.gmsh

:py:mod:`solvcon.io.gmsh`
=========================

.. autoclass:: Gmsh

  .. automethod:: __init__

  .. autoattribute:: nnode

  .. autoattribute:: ncell

  .. automethod:: load

  .. rubric:: Private Helpers for Loading and Parsing the Mesh File

  These private methods are documented for demonstrating the data structure of
  the loaded meshes.  Do not rely on their implementation.

  .. automethod:: _check_meta

  .. automethod:: _load_nodes

  .. automethod:: _load_elements

  .. automethod:: _load_physics

  .. automethod:: _load_periodic

  .. automethod:: _parse_physics

  .. rubric:: Mesh Definition and Data Attributes

  .. autoattribute:: ELMAP
    :annotation:

  .. autoinstanceattribute:: stream
    :annotation:

  .. autoinstanceattribute:: ndim
    :annotation:

  .. autoinstanceattribute:: nodes
    :annotation:

  .. autoinstanceattribute:: usnds
    :annotation:

  .. autoinstanceattribute:: ndmap
    :annotation:

  .. autoinstanceattribute:: cltpn
    :annotation:

  .. autoinstanceattribute:: elgrp
    :annotation:

  .. autoinstanceattribute:: elgeo
    :annotation:

  .. autoinstanceattribute:: eldim
    :annotation:

  .. autoinstanceattribute:: elems
    :annotation:

  .. autoinstanceattribute:: intels
    :annotation:

  .. autoinstanceattribute:: physics
    :annotation:

  .. autoinstanceattribute:: periodics
    :annotation:

.. autoclass:: GmshIO

  .. inheritance-diagram:: GmshIO

  .. automethod:: load

.. vim: set spell ff=unix fenc=utf8 ft=rst:

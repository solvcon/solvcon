============
Verification
============

Driving Scripts (to Be Corrected)
=================================

The simplest example: located at ``$SCSRC/examples/euler/hbnt/``.

Objective:

#. Understand the concept of "driving script" (programmable input file).
#. Perform simulation with SOLVCON.

Course:

#. Run the code::

     $ PYTHONPATH=../../.. ./go run

#. Simulation results are stored in the sub-directory ``result/``.  Use
   ParaView to load the VTK files.
#. Code organization:

   - First line: indicating it is a Python script.
   - Second line: indicating encoding of the file.
   - Line 4-18: comments for copyright information.
   - Line 20-27: docstring in the form of `string literal
     <http://docs.python.org/reference/lexical_analysis.html#string-literals>`_.
   - Line 29: module-level import for arrangement decorator.
   - Line 31-99: main body for instantiating the ``Case`` object in the form
     of a Python function/callable; "creation function".
   - Line 101-112: `decorated
     <http://en.wikipedia.org/wiki/Decorator_pattern>`_ arrangement
     (function).
   - Line 114-116: invoking command-line interface of SOLVCON.
#. Customization goes into the creation function:

   - Specify BC: line 54-59.
   - Feed parameter to Case: line 60-64.
#. SOLVCON modules to hack:

   - ``solvcon.boundcond``
   - ``solvcon.case``
   - ``solvcon.solver``

The Hook System (to Be Corrected)
=================================

Located at ``$SCSRC/examples/euler/obrefl/``.

Objective:

- Use the programmability of input file for properties specification.
- Understand the Hook system for custom post-processing.

Question:

- Where is the creation function?

Course:

#. Run and inspect the simulation.
#. Change the flow properties in line 263-275 and see the difference.

   - Utility code is organized as a class in line 52-164, for calculating shock
     properties.
#. How to extend SOLVCON by using Hook, i.e., line 166-244, 318-320.
#. SOLVCON modules to hack:

   - ``solvcon.hook``
   - ``solvcon.kerpak.euler``

Change Physical Model (to Be Corrected)
=======================================

Located at ``$SCSRC/examples/elastic/grpv/``.

Objective:

- Change the physical model.
- Understand the Anchor system for parallel processing.

Questions:

#. What is the path of the mesh file used in this simulation?
#. What is the equivalent code of line 123-125 in the previous two examples?

Course:

#. Run and inspect the simulation.
#. Note the difference of line 144.  It uses a different calculator to the
   Euler solver.
#. Line 76-89, 135-142 uses the Anchor system to insert source term.
#. Line 35-74 calculate the source value.
#. SOLVCON modules to hack:

   - ``solvcon.anchor``
   - ``solvcon.kerpak.elastic``

Output Control (to Be Corrected)
================================

Located at ``$SCSRC/examples/visout/pvtk/``.

Objective:

- Parallel run.
- Specify the variables to output.

Questions:

#. Guess what problem is it?
#. Where is the code for sequential VTK output in legacy format?

Course:

#. Run the simulation in parallel by following the docstring.
#. Inspect the solution.
#. Line 90-102 specifies three kinds of quantities:

   - Negative integer for array.
   - Zero for scalar.
   - Positive value for vector.
#. Try to turn off some of the variables by commenting out.

   - Before rerun the simulation, clean the ``result/`` directory.
#. SOLVCON sub-package to hack:

   - ``solvcon.io``

Set up Jenkins Slave
====================

If you have extra machines, we welcome you to set them up as Jenkins slaves for
SOLVCON.  The first step is to contact the site admin to set up a node for you.
Then, you can configure your machines as headless slaves.  If you are running
Debian or Ubuntu, we've prepared the configuration scripts for you: (i) the
init script file `/etc/init.d/jenkins-slave`_ and (ii) the default file
`/etc/default/jenkins-slave`_.  To run the slave, you need at least two
prerequisite packages and they can be installed by::

  apt-get install daemon sun-java6-jre

Before starting the slave, don't forget to supply the settings
``JENKINS_SLAVE_USER`` and ``JENKINS_SLAVE_HOME`` in the default file
`/etc/default/jenkins-slave`_.

If everything runs correctly, then you can install the init script to rc.d::

  update-rc.d jenkins-slave defaults

so that the slave can automatically run on machine start-up.

In the directory ``contrib/`` of the source package of SOLVCON,
`/etc/init.d/jenkins-slave`_, `/etc/default/jenkins-slave`_, and a install
script that does the above actions are supplied for your convenience.

File Listings
=============

/etc/init.d/jenkins-slave
+++++++++++++++++++++++++

.. literalinclude:: ../../contrib/jenkins-slave
  :language: bash

/etc/default/jenkins-slave
++++++++++++++++++++++++++

.. literalinclude:: ../../contrib/jenkins-slave-default
  :language: bash

.. vim: set ft=rst ff=unix fenc=utf8 ai:

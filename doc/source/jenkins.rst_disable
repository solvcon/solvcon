:orphan:

===================
Jenkins Slave Setup
===================

If you have extra machines, we welcome you to set them up as Jenkins slaves for
SOLVCON.  The first step is to contact the site admin to set up a node for you.
Then, you can configure your machines as headless slaves.  If you are running
Debian or Ubuntu, we've prepared the configuration scripts for you: (i) the
init script file `/etc/init.d/jenkins-slave`_ and (ii) the default file
`/etc/default/jenkins-slave`_.  To run the slave, you need at least two
prerequisite packages and they can be installed by::

  $ apt-get install daemon sun-java6-jre

Before starting the slave, don't forget to supply the settings
``JENKINS_SLAVE_USER`` and ``JENKINS_SLAVE_HOME`` in the default file
`/etc/default/jenkins-slave`_.

If everything runs correctly, then you can install the init script to rc.d::

  $ update-rc.d jenkins-slave defaults

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

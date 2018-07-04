SOLVCON Singularity-based image build file.

In short, run this command to get everything ready.

  $ sudo singularity build ./ubuntu-xenial-singularity.img ./Singularity && singularity exec ubuntu-xenial-singularity.img /bin/bash -c 'source /opt/solvcon/dist/SOLVCON-0.1.4+/contrib/singularity/activate.sh; cd /opt/solvcon/dist/SOLVCON-0.1.4+/; nosetests --with-doctest

Pre-requisite
=============

Install `Singularity <http://singularity.lbl.gov/>`_.

Create a file `host-username.txt` by

  $ whoami > /tmp/host-username.txt

This is a trick for you to touch the SOLVCON folder in the container later after building the singularity image.

This instruction is verified by running on Ubuntu Xenial.

Build
=====

Build the image by

  $ sudo singularity build ./<name-of-the-image>.img ./Singularity

In this case, `sudo` is necessary to execute post actions of the build process.

Use the Container
=================

Use the container by

  $ singularity shell ./<name-of-the-image>.img

Activate the runtime by

  [container] :SOLVCON-SRC-ROOT> source contrib/singularity/activate.sh

You could find the built SOLVCON is under `/opt/solvcon/dist/SOLVCON-<Version Number>`. Go there and run some jobs

  [container] :/opt/solvcon/dist/SOLVCON-0.1.4+> nosetests --with-doctest

So far the feature of parallel mode does not work in the singularity container.

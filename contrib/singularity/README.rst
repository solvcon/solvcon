SOLVCON Singularity-based image build file.

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

You could find the built SOLVCON is under `/opt/solvcon/dist/SOLVCON-<Version Number>`.

Activate the runtime by

  [container] $ source contrib/singularity/activate.sh

So far the feature of single process run on the local container works only.

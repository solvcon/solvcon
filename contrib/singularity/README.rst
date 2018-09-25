SOLVCON Singularity-based image build file.

In short, run this command to get everything ready. You will get a SOLVCON singularity image `solvcon.img`.

  $ ./build-solvcon.sh

Pre-requisite
=============

Install `Singularity <http://singularity.lbl.gov/>`_.

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

Now you can develop your driving script and use SOLVCON.

So far the feature of parallel mode does not work in the singularity container.

If you use `build-solvcon.sh` to build the image, you could follow the instruction of the build message at the end of the build script to run an example.


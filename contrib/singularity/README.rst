SOLVCON Singularity-based image build file.

In short, run this command to get everything ready. You will get a SOLVCON singularity image `solvcon.img`.

  $ ./build-solvcon.sh

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

  $ singularity shell --home <your-directory-used-as-container-home> ./<name-of-the-image>.img

Activate the runtime by

  [container] :SOLVCON-SRC-ROOT> source /opt/solvcon/contrib/singularity/activate.sh

Now you can develop your driving script and use SOLVCON.

So far the feature of parallel mode does not work in the singularity container. Please note the option `--home` is recommended. To use a clean directory as your home in the container will help you to avoid many python virtual environment issues, e.g. conda is confused by your configuration files in your host home directory.

You could refer to `contrib/singularity/sod-tube` to see how a SOLVCON driving script works with the container, and how `build-solvcon.sh` leverages it.


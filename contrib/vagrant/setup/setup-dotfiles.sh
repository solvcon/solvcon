#!/bin/bash
SRCDIR=/vagrant/setup/dotfiles
DSTDIR=${HOME}

mkdir -p ${DSTDIR}/.config/upstart
cp ${SRCDIR}/dot_bashrc ${DSTDIR}/.bashrc
cp ${SRCDIR}/dot_screenrc ${DSTDIR}/.screenrc
cp ${SRCDIR}/dot_git-completion.bash ${DSTDIR}/.git-completion.bash
cp ${SRCDIR}/dot_git-prompt.bash ${DSTDIR}/.git-prompt.bash
cp ${SRCDIR}/dot_vimrc ${DSTDIR}/.vimrc
cp ${SRCDIR}/dot_gvimrc ${DSTDIR}/.gvimrc
cp ${SRCDIR}/dot_setup-desktop.sh ${DSTDIR}/.setup-desktop.sh
cp ${SRCDIR}/dot_setup-desktop.conf ${DSTDIR}/.config/upstart/setup-desktop.conf
cp ${SRCDIR}/dot_install-anaconda.sh ${DSTDIR}/.install-anaconda.sh

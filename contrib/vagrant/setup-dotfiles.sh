#!/bin/bash
SRCDIR=/vagrant/dotfiles
DSTDIR=${HOME}

cp ${SRCDIR}/dot_bashrc ${DSTDIR}/.bashrc
cp ${SRCDIR}/dot_git-completion.bash ${DSTDIR}/.git-completion.bash
cp ${SRCDIR}/dot_git-prompt.bash ${DSTDIR}/.git-prompt.bash
cp ${SRCDIR}/dot_screenrc ${DSTDIR}/.screenrc
cp ${SRCDIR}/dot_gitconfig ${DSTDIR}/.gitconfig
cp ${SRCDIR}/dot_dir_colors ${DSTDIR}/.dir_colors

#!/bin/bash

# Bring repository up to date
apt-get update
apt-get dist-upgrade -y

# NOTE: linux-generic contains drm kernel module, which is required for
# vboxvideo kernel module.  Without vboxvideo Ubuntu unity graphics will be
# extremely slow.  See
# http://askubuntu.com/questions/287532/how-do-i-resolve-slow-and-choppy-performance-in-virtualbox.
# "/usr/lib/nux/unity_support_test -p" is a convenient tool to check Ubuntu's
# video acceleration status.
pkgs="linux-generic"
pkgs="$pkgs virtualbox-guest-dkms virtualbox-guest-utils virtualbox-guest-x11"

# For desktop environment.
# http://askubuntu.com/questions/42964/installing-ubuntu-desktop-without-all-the-bloat
## Ubuntu desktop
pkgs="$pkgs ubuntu-desktop ubuntu-software"
pkgs="$pkgs indicator-session network-manager-gnome unity-lens-applications"
## Desktop must-haves
pkgs="$pkgs fonts-inconsolata fonts-dejavu-core fonts-freefont-ttf"
pkgs="$pkgs fonts-noto-cjk"
pkgs="$pkgs gksu gnome-terminal firefox vim-gnome"

# For development environment.
pkgs="$pkgs build-essential gfortran m4"
pkgs="$pkgs zlib1g zlib1g-dev libreadline libreadline-dev"
pkgs="$pkgs git exuberant-ctags cscope"
pkgs="$pkgs freeglut3-dev"

# Make apt-get download faster: https://github.com/ilikenwf/apt-fast
/bin/bash -c "$(curl -sL https://git.io/vokNn)"
echo "apt-fast install --no-install-recommends -y $pkgs"
apt-fast install --no-install-recommends -y $pkgs

# vim: set et nobomb fenc=utf8 ft=sh ff=unix sw=2 ts=2:

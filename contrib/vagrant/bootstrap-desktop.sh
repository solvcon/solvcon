#!/bin/bash

# NOTE: linux-generic contains drm kernel module, which is required for
# vboxvideo kernel module.  Without vboxvideo Ubuntu unity graphics will be
# extremely slow.  See
# http://askubuntu.com/questions/287532/how-do-i-resolve-slow-and-choppy-performance-in-virtualbox.
# "/usr/lib/nux/unity_support_test -p" is a convenient tool to check Ubuntu's
# video acceleration status.
apt-get install -y \
  ubuntu-desktop linux-generic \
  virtualbox-guest-dkms virtualbox-guest-utils virtualbox-guest-x11 \

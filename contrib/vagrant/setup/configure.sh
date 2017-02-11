#!/bin/bash

# ubuntu/xenial64 box has non-standard username/password, so work it around by
# resetting the password.  For more information, see
# https://bugs.launchpad.net/cloud-images/+bug/1569237
chpasswd << END
ubuntu:ubuntu
END

timedatectl set-timezone Asia/Taipei

# vim: set et nobomb fenc=utf8 ft=sh ff=unix sw=2 ts=2:

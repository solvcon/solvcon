#!/bin/bash

sudo timedatectl set-timezone Asia/Taipei

# Update repository.
apt-get update

# Make the system up-to-date.
apt-get dist-upgrade -y

# For development environment.
apt-get install -y \
  build-essential liblapack-pic liblapack-dev \
  git mercurial vim exuberant-ctags cscope

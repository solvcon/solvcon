#!/bin/sh
#PBS -l nodes=1:ppn=8:newdual:pvfs,walltime=2:00:00
#PBS -N buildground
#PBS -j oe
#PBS -S /bin/sh
#
# Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

echo "Customized paths for job:"
if [ -r $HOME/.bashrc ]
then
  . $HOME/.bashrc
fi
echo "Run @`date`:"
if [ -n "`echo $SCSRC`" ]
then
  cd $SCSRC/ground
fi
NP=8 make all
echo "Finish @`date`."

# vim: set ai et nu:

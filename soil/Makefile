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

export SCROOT?=$(HOME)/opt/scruntime
export SCBIN=$(SCROOT)/bin
export SCETC=$(SCROOT)/etc
export TMPDL=downloaded/
export TMPBLD=build/
#export NP?=$(`cat /proc/cpuinfo | grep processor | wc -l`)
export NP?=1
## Additional options to pass to gcc compiler, e.g. --program-suffix=4.7 ... 
export GCC_CONFIG_OPTS?=

default: gcc

$(SCETC)/scgccvars.sh:
	mkdir -p $(TMPBLD) $(SCBIN) $(SCETC)
	./create-scgccvars.sh

################################################################################
# Begin gcc
################################################################################

gcc: $(SCETC)/scgccvars.sh \
	$(SCETC)/gmp-5.0.2 $(SCETC)/mpfr-3.0.1 $(SCETC)/mpc-0.9 \
	$(SCETC)/gcc-4.7.2

$(SCETC)/gmp-5.0.2:
	./build-gmp.sh gmp-5.0.2
	touch $(SCETC)/gmp-5.0.2

$(SCETC)/mpfr-3.0.1: $(SCETC)/gmp-5.0.2
	./build-mpfr.sh mpfr-3.0.1
	touch $(SCETC)/mpfr-3.0.1

$(SCETC)/mpc-0.9: $(SCETC)/gmp-5.0.2 $(SCETC)/mpfr-3.0.1
	./build-mpc.sh mpc-0.9
	touch $(SCETC)/mpc-0.9

$(SCETC)/gcc-4.7.2: $(SCETC)/gmp-5.0.2 $(SCETC)/mpfr-3.0.1 $(SCETC)/mpc-0.9
	./build-gcc.sh gcc-4.7.2
	touch $(SCETC)/gcc-4.7.2

################################################################################
# End gcc
################################################################################

.PHONY: default gcc

# vim: set ai noet nu:

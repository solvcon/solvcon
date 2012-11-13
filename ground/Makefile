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
export SCLIB=$(SCROOT)/lib
export SCETC=$(SCROOT)/etc
export TMPDL=downloaded/
export TMPBLD=build/
#export NP?=$(`cat /proc/cpuinfo | grep processor | wc -l`)
export NP?=1

default: basictool basiclib pybin numlib pylib pytool mesher

all: default vislib

$(SCETC)/scvars.sh:
	mkdir -p $(TMPBLD) $(SCBIN) $(SCLIB) $(SCETC)
	./create-scvars.sh

################################################################################
# Start basictool.
################################################################################

basictool: $(SCETC)/scvars.sh \
	$(SCETC)/cmake-2.8.5

$(SCETC)/cmake-2.8.5:
	./build-cmake.sh cmake-2.8.5
	touch $(SCETC)/cmake-2.8.5

################################################################################
# End basictool.
################################################################################

################################################################################
# Begin basiclib.
################################################################################

basiclib: $(SCETC)/scvars.sh \
	$(SCETC)/blas $(SCETC)/lapack-3.3.1

$(SCETC)/blas:
	./build-blas.sh blas
	touch $(SCETC)/blas

$(SCETC)/lapack-3.3.1: $(SCETC)/blas
	./build-lapack.sh lapack-3.3.1
	touch $(SCETC)/lapack-3.3.1

$(SCETC)/atlas3.8.4:
	./build-atlas.sh atlas3.8.4
	touch $(SCETC)/atlas3.8.4

################################################################################
# End basiclib.
################################################################################

################################################################################
# Begin pybin.
################################################################################

pybin: $(SCETC)/scvars.sh \
	$(SCETC)/Python-2.7.2

$(SCETC)/Python-2.7.2:
	./build-python.sh Python-2.7.2
	touch $(SCETC)/Python-2.7.2

################################################################################
# End pybin.
################################################################################

################################################################################
# Start pylib.
################################################################################

pylib: $(SCETC)/scvars.sh \
	$(SCETC)/numpy-1.6.1 $(SCETC)/distribute-0.6.24

$(SCETC)/numpy-1.6.1:
	cd $(TMPBLD) && tar xfz ../$(TMPDL)/numpy-1.6.1.tar.gz
	cd $(TMPBLD)/numpy-1.6.1 && ../../generate-numpy.py
	cd $(TMPBLD)/numpy-1.6.1 && \
		$(SCBIN)/python2.7 setup.py install > install2.7.log 2>&1
	touch $(SCETC)/numpy-1.6.1

$(SCETC)/distribute-0.6.24:
	cd $(TMPBLD) && tar xfz ../$(TMPDL)/distribute-0.6.24.tar.gz
	cd $(TMPBLD)/distribute-0.6.24 && \
		$(SCBIN)/python2.7 setup.py install > install2.7.log 2>&1
	touch $(SCETC)/distribute-0.6.24

################################################################################
# End pylib.
################################################################################

################################################################################
# Begin numlib.
################################################################################

numlib: $(SCETC)/scvars.sh \
	$(SCETC)/netcdf-4.1.2 $(SCETC)/scotch_5.1.11

$(SCETC)/netcdf-4.1.2:
	./build-netcdf.sh netcdf-4.1.2
	touch $(SCETC)/netcdf-4.1.2

$(SCETC)/scotch_5.1.11:
	./build-scotch.sh scotch_5.1.11
	touch $(SCETC)/scotch_5.1.11

################################################################################
# End numlib.
################################################################################

################################################################################
# Start pytool.
################################################################################

pytool: $(SCETC)/scvars.sh \
	$(SCETC)/mercurial-2.0.2 $(SCETC)/scons-2.1.0 $(SCETC)/Cython-0.17.1 \
	$(SCETC)/nose-1.0.0 $(SCETC)/epydoc-3.0.1 $(SCETC)/docutils-0.8.1 \
	$(SCETC)/Jinja2-2.6 $(SCETC)/Pygments-1.4 $(SCETC)/Sphinx-1.1.2

$(SCETC)/mercurial-2.0.2:
	cd $(TMPBLD) && tar xfz ../$(TMPDL)/mercurial-2.0.2.tar.gz
	cd $(TMPBLD)/mercurial-2.0.2 && \
		$(SCBIN)/python2.7 setup.py install > install2.7.log 2>&1
	touch $(SCETC)/mercurial-2.0.2

$(SCETC)/scons-2.1.0:
	cd $(TMPBLD) && tar xfz ../$(TMPDL)/scons-2.1.0.tar.gz
	cd $(TMPBLD)/scons-2.1.0 && \
		$(SCBIN)/python2.7 setup.py install > install2.7.log 2>&1
	$(SCBIN)/python2.7 modify-pyshebang.py $(SCBIN)/scons*
	touch $(SCETC)/scons-2.1.0

$(SCETC)/Cython-0.17.1:
	cd $(TMPBLD) && tar xfz ../$(TMPDL)/Cython-0.17.1.tar.gz
	cd $(TMPBLD)/Cython-0.17.1 && \
		$(SCBIN)/python2.7 setup.py install > install2.7.log 2>&1
	$(SCBIN)/python2.7 modify-pyshebang.py $(SCBIN)/scons*
	touch $(SCETC)/Cython-0.17.1

$(SCETC)/nose-1.0.0:
	cd $(TMPBLD) && tar xfz ../$(TMPDL)/nose-1.0.0.tar.gz
	cd $(TMPBLD)/nose-1.0.0 && \
		$(SCBIN)/python2.7 setup.py install > install2.7.log 2>&1
	$(SCBIN)/python2.7 modify-pyshebang.py $(SCBIN)/nosetests*
	touch $(SCETC)/nose-1.0.0

$(SCETC)/epydoc-3.0.1:
	cd $(TMPBLD) && tar xfz ../$(TMPDL)/epydoc-3.0.1.tar.gz
	cd $(TMPBLD)/epydoc-3.0.1 && \
		$(SCBIN)/python2.7 setup.py install > install2.7.log 2>&1
	$(SCBIN)/python2.7 modify-pyshebang.py $(SCBIN)/epydoc*
	touch $(SCETC)/epydoc-3.0.1

$(SCETC)/docutils-0.8.1:
	cd $(TMPBLD) && tar xfz ../$(TMPDL)/docutils-0.8.1.tar.gz
	cd $(TMPBLD)/docutils-0.8.1 && \
		$(SCBIN)/python2.7 setup.py install > install2.7.log 2>&1
	touch $(SCETC)/docutils-0.8.1

$(SCETC)/Jinja2-2.6: $(SCETC)/distribute-0.6.24
	cd $(TMPBLD) && tar xfz ../$(TMPDL)/Jinja2-2.6.tar.gz
	cd $(TMPBLD)/Jinja2-2.6 && \
		$(SCBIN)/python2.7 setup.py install > install2.7.log 2>&1
	touch $(SCETC)/Jinja2-2.6

$(SCETC)/Pygments-1.4:
	cd $(TMPBLD) && tar xfz ../$(TMPDL)/Pygments-1.4.tar.gz
	cd $(TMPBLD)/Pygments-1.4 && \
		$(SCBIN)/python2.7 setup.py install > install2.7.log 2>&1
	touch $(SCETC)/Pygments-1.4

$(SCETC)/Sphinx-1.1.2: $(SCETC)/docutils-0.8.1 $(SCETC)/Jinja2-2.6 \
		$(SCETC)/Pygments-1.4
	cd $(TMPBLD) && tar xfz ../$(TMPDL)/Sphinx-1.1.2.tar.gz
	cd $(TMPBLD)/Sphinx-1.1.2 && \
		$(SCBIN)/python2.7 setup.py install > install2.7.log 2>&1
	touch $(SCETC)/Sphinx-1.1.2

################################################################################
# End pytool.
################################################################################

################################################################################
# Start mesher.
################################################################################

mesher: $(SCETC)/scvars.sh \
	$(SCETC)/gmsh-2.5.0-source

$(SCETC)/gmsh-2.5.0-source:
	./build-gmsh.sh gmsh-2.5.0-source
	touch $(SCETC)/gmsh-2.5.0-source

################################################################################
# End mesher.
################################################################################

################################################################################
# Start vislib.
################################################################################

vislib: $(SCETC)/scvars.sh \
	$(SCETC)/vtk-5.6.1

$(SCETC)/vtk-5.6.1:
	./build-vtk.sh vtk-5.6.1 vtkdata-5.6.1
	touch $(SCETC)/vtk-5.6.1

################################################################################
# End vislib.
################################################################################

.PHONY: default basictool basiclib pybin numlib pylib pytool

# vim: set ai noet nu:

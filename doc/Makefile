# Minimal documentation build for solvcon.
#
# Quick start:
#   pip install -r requirements.txt   # plus a system Doxygen
#   make html                         # site -> build/html/index.html

SPHINXBUILD ?= sphinx-build
PYTHON ?= python3
SOURCEDIR = source
BUILDDIR = build
PORT ?= 8000
BIND ?= 127.0.0.1
FORCE_BUILD ?=

.PHONY: help html doxygen serve clean

help:
	@echo "make doxygen   generate C++ API XML (run before html for C++)"
	@echo "make html      build the HTML site into $(BUILDDIR)/html"
	@echo "make serve     build, then serve at http://localhost:$(PORT)/"
	@echo "make clean     remove built docs and Doxygen output"
	@echo ""
	@echo "Set FORCE_BUILD=1 to force a full rebuild:"
	@echo "  make FORCE_BUILD=1 html"

doxygen:
	mkdir -p "$(BUILDDIR)/doxygen"
	doxygen Doxyfile

html: doxygen
ifdef FORCE_BUILD
	rm -rf "$(BUILDDIR)/html"
endif
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

serve: html
	@echo "Serving docs at http://$(BIND):$(PORT)/ -- Ctrl-C to stop"
	cd "$(BUILDDIR)/html" && "$(PYTHON)" -m http.server "$(PORT)" \
		--bind "$(BIND)"

clean:
	rm -rf "$(BUILDDIR)"

# vim: set ft=make ff=unix fenc=utf8 noet sw=8 ts=8 sts=8 tw=79:

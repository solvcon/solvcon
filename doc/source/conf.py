# Configuration file for the Sphinx documentation builder.
#
# Minimal prototype of a hybrid C++/Python documentation toolchain for
# solvcon.  The pieces, and why each is here:
#
#   myst_parser          author pages in Markdown (matches the repo
#                        culture: README.md, STYLE.md, CLAUDE.md)
#   autodoc + napoleon   Python API straight from docstrings
#   breathe              C++ API bridged from Doxygen XML
#   mathjax              CESE / conservation-law equations
#   sphinxcontrib.bibtex academic citations (the CESE literature)
#
# Build with ``make html`` from the doc/ directory.  Run ``make doxygen``
# first if you want the C++ API pages populated.

import os
import re
import sys

from docutils import nodes
from sphinx import addnodes
from sphinx.transforms.post_transforms import SphinxPostTransform

# Make the in-tree ``solvcon`` package importable for autodoc.  The repo
# root is two levels up from this file (doc/source/conf.py).
sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../ext"))

# -- Project information ----------------------------------------------------

project = "solvcon"
copyright = "2019-2026, Yung-Yu Chen and solvcon contributors"
author = "Yung-Yu Chen and solvcon contributors"

# -- General configuration --------------------------------------------------

extensions = [
    "myst_parser",            # Markdown authoring
    "sphinx.ext.autodoc",     # pull Python docstrings
    "sphinx.ext.autosummary",  # API summary tables
    "sphinx.ext.napoleon",    # NumPy / Google docstring styles
    "sphinx.ext.viewcode",    # link to highlighted source
    "sphinx.ext.intersphinx",  # cross-link python / numpy docs
    "sphinx.ext.mathjax",     # render LaTeX math
    "breathe",                # C++ via Doxygen XML
    "sphinxcontrib.bibtex",   # citations
    "sphinxcontrib.mermaid",  # mermaid diagrams
    "pstake",                 # PSTricks .tex -> PNG at build time
]

# MyST Markdown extensions: $...$ and $$...$$ math, amsmath
# environments, ::: fences, and definition lists.
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
]

# Treat a ```mermaid fence as the mermaid directive, so the same fenced
# block renders natively on GitHub and through sphinxcontrib.mermaid here.
myst_fence_as_directive = ["mermaid"]

autosummary_generate = True

# The compiled _solvcon extension is a build artifact, absent on a clean
# checkout (and on Read the Docs).  Mock it so autodoc can still import
# the pure-Python layers that sit on top of it.
autodoc_mock_imports = ["_solvcon"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -- Breathe (C++ bridge) ---------------------------------------------------

# Points at the Doxygen XML produced by ``make doxygen`` (see Doxyfile).
breathe_projects = {"solvcon": "../build/doxygen/xml"}
breathe_default_project = "solvcon"

# Render members by default so directives need not repeat ``:members:``.
breathe_default_members = ("members",)

# -- sphinxcontrib.bibtex ---------------------------------------------------

bibtex_bibfiles = ["reference.bib"]

# -- HTML output ------------------------------------------------------------

# furo is a clean, responsive theme with first-class light and dark modes
# and good defaults for long-form technical documentation.
html_theme = "furo"
html_title = "solvcon"
html_static_path = ["_static"]

# -- MathJax configuration --------------------------------------------------

mathjax3_config = {
    "tex": {
        "macros": {
            "defeq": r"\overset{\text{def}}{=}",
            "dif": r"\mathrm{d}",
        },
        "packages": {"[+]": ["cancel"]},
    }
}

numfig = True

# -- Link Qt types in the C++ API to the Qt documentation -------------------

# Doxygen and breathe render Qt types (QRhiWidget, QImage, QMatrix4x4, ...) as
# bare C++ identifiers, and there is no Sphinx inventory to resolve them
# against, so the C++ domain leaves them as plain text in the generated
# reference.  Catch those unresolved C++ references and point any Qt class (a
# "Q" followed by CamelCase) at its Qt 6 manual page, so the Doxygen-bridged
# pages cross-link to the official Qt docs.

_qt_type_re = re.compile(r"^Q[A-Z][A-Za-z0-9]+$")


def _resolve_qt_reference(app, env, node, contnode):
    if node.get("refdomain") != "cpp":
        return None
    target = node.get("reftarget", "")
    if not _qt_type_re.match(target):
        return None
    uri = "https://doc.qt.io/qt-6/%s.html" % target.lower()
    refnode = nodes.reference("", "", internal=False, refuri=uri)
    refnode += contnode
    return refnode


# -- Resolve cross-references written inside the Doxygen comments -----------

# Breathe renders a class or function named in a Doxygen "///" or "/**" comment
# as a std ":ref:" keyed by the raw Doxygen refid (e.g.
# "classsolvcon_1_1_r_scene"), which the std domain cannot resolve, so the
# reference falls back to plain text.  The post-transform below decodes the
# refid back to a C++ name and resolves it through the cpp domain, so the
# comment prose cross-links like the rest of the generated reference.


_doxy_compound_re = re.compile(r"^(?:class|struct|union|namespace|group)(.+)$")
_doxy_member_re = re.compile(r"_1[a-z][0-9a-f]{16,}$")


class _BreatheRefidResolver(SphinxPostTransform):
    # Breathe renders a reference inside Doxygen comment prose as a std
    # ":ref:" keyed by the raw Doxygen refid, which the std domain cannot
    # resolve, so it falls back to plain text.  Run before Sphinx's
    # ReferencesResolver (priority 10), decode each such refid back to its C++
    # name, and resolve it through the cpp domain so the comment prose links to
    # the generated class and member reference like the rest of the page.
    default_priority = 5

    @classmethod
    def _demangle_doxy_name(cls, mangled):
        # Reverse Doxygen's refid encoding for a namespaced C++ name.
        # Doxygen writes "::" as "_1_1" (each "_1" is a ":"), a literal "_"
        # as "__", and an uppercase letter as "_" plus its lowercase form,
        # so "solvcon_1_1_r_scene" decodes to "solvcon::RScene".
        out = []
        i, n = 0, len(mangled)
        while i < n:
            c = mangled[i]
            if c == "_" and i + 1 < n:
                nxt = mangled[i + 1]
                if nxt == "1":
                    out.append(":")
                elif nxt == "_":
                    out.append("_")
                elif nxt.isalpha() and nxt.islower():
                    out.append(nxt.upper())
                else:
                    out.append(c)
                    i += 1
                    continue
                i += 2
                continue
            out.append(c)
            i += 1
        return "".join(out)

    @classmethod
    def _cpp_name_from_refid(cls, refid, text):
        # Decode a Doxygen compound or member refid into a C++ qualified
        # name the cpp domain can resolve: "classsolvcon_1_1_r_scene" ->
        # "solvcon::RScene"; a member refid keeps the class and takes the
        # member name from the displayed text
        # ("solvcon::RScene::extendBoundingBox").
        match = _doxy_compound_re.match(refid)
        if not match:
            return None
        body = match.group(1)
        member = _doxy_member_re.search(body)
        if member:
            owner = cls._demangle_doxy_name(body[:member.start()])
            return "%s::%s" % (owner, text.rstrip("()")) if text else None
        return cls._demangle_doxy_name(body)

    def run(self, **kwargs):
        cpp = self.env.get_domain("cpp")
        for node in list(self.document.findall(addnodes.pending_xref)):
            if node.get("refdomain") != "std" or node.get("reftype") != "ref":
                continue
            text = node.astext()
            name = self._cpp_name_from_refid(node.get("reftarget", ""), text)
            if not name:
                continue
            contnode = nodes.inline("", text)
            try:
                results = cpp.resolve_any_xref(self.env, self.env.docname,
                                               self.app.builder, name, node,
                                               contnode)
            except Exception:
                results = None
            if results:
                node.replace_self(results[0][1])


def setup(app):
    app.connect("missing-reference", _resolve_qt_reference)
    app.add_post_transform(_BreatheRefidResolver)


# vim: set ft=python ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:

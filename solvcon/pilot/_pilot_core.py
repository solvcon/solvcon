# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Probe the pilot once, and provide its C++ names and the lazy-export helper.
"""

# Use flake8 http://flake8.pycqa.org/en/latest/user/error-codes.html

import importlib
import sys

# The pilot needs two things: (1) the C++ widgets, which only a build with Qt
# (BUILD_QT=ON) compiles into _solvcon, and (2) PySide6, which the Python
# widgets import. Probe both once and keep the failure so require() can report
# it.
try:
    import _solvcon as _ext
except ImportError:
    from .. import _solvcon as _ext

enable = False
import_error = None
try:
    _pilot_impl = getattr(_ext, 'pilot', None)
    if _pilot_impl is None:
        raise ImportError('_solvcon was built without Qt (BUILD_QT=OFF)')
    from PySide6 import QtWidgets  # noqa: F401
    enable = True
except ImportError as e:
    import_error = e


def require(feature):
    """Raise ImportError, chained to the probe failure, without the pilot."""
    if not enable:
        raise ImportError(
            f'{feature} needs solvcon built with Qt (BUILD_QT=ON) and PySide6'
        ) from import_error


def lazy_exports(package, modules, ungated=()):
    """Resolve the public names of a package on first use (PEP 562).

    A name imports its module when first read, and names only, never
    modules, are bound, so a real submodule import is not shadowed.

    :param package: The ``__name__`` of the package that binds the result.
    :type package: str
    :param modules:
        Maps each private module of ``package``, by its name relative to
        ``package``, to the public names it provides.
    :type modules: dict[str, list[str]]
    :param ungated:
        Modules whose names need PySide6 but not the pilot. The names of any
        other module raise the ImportError of require() without the pilot.
    :type ungated: tuple[str, ...]
    :returns:
        ``__all__``, ``__getattr__``, and ``__dir__`` for the package to bind.
    :rtype: tuple[list[str], callable, callable]
    """
    owners = {name: mod for mod, names in modules.items() for name in names}

    def __getattr__(name):
        if name not in owners:
            raise AttributeError(
                f'module {package!r} has no attribute {name!r}')
        feature = f'{package}.{name}'
        if owners[name] not in ungated:
            require(feature)
        try:
            module = importlib.import_module(f'{package}.{owners[name]}')
        except (ImportError, AttributeError) as e:
            # An AttributeError raised while the module runs would otherwise
            # escape __getattr__ and read as a missing name to hasattr().
            raise ImportError(f'cannot import {feature}: {e}') from e
        value = getattr(module, name)
        setattr(sys.modules[package], name, value)
        return value

    def __dir__():
        return sorted(set(vars(sys.modules[package])) | set(owners))

    return list(owners), __getattr__, __dir__


# pilot directory symbols organized by source files

# RDomainWidget.hpp/.cpp
list_of_rdomainwidget = [
    'RDomainWidget',
]

# R2DWidget.hpp/.cpp
list_of_r2dwidget = [
    'R2DWidget',
    'Overlay2dOptions',
]

# RManager.hpp/.cpp
list_of_rmanager = [
    'RManager',
    'mgr',
]

# RThreadManager.hpp/.cpp
list_of_rthreadmanager = [
    'WorkflowState',
    'Error',
    'Succeeded',
    'Failed',
    'Cancelled',
    'TaskContext',
    'WorkflowContext',
    'WorkflowHandle',
    'RThreadManager',
]

# RPythonConsoleDockWidget.hpp/.cpp
list_of_rpythonconsole = [
    'RPythonConsoleDockWidget',
]

# RPythonTerminalDockWidget.hpp/.cpp
list_of_rpythonterminal = [
    'RPythonTerminalDockWidget',
]

# DrawTool.hpp/.cpp
list_of_drawtool = [
    'draw_tool_names',
    'default_draw_tool_name',
]

# PlotLimits2d.hpp
list_of_plotlimits = [
    'PlotLimits2d',
]

# plot_style.hpp/.cpp
list_of_plot_style = [
    'PlotColor',
    'plot_color_cycle',
    'plot_cycle_color',
]

# RPlotSeries.hpp/.cpp
list_of_rplotseries = [
    'RPlotSeries',
]

# RPlotDecimator.hpp/.cpp
list_of_rplotdecimator = [
    'RPlotDecimator',
]

# RPlotModel.hpp/.cpp
list_of_rplotmodel = [
    'RPlotModel',
]

# RPlotTicker.hpp/.cpp
list_of_rplotticker = [
    'RPlotTicker',
]


_from_impl = (  # noqa: F822
    list_of_rdomainwidget +
    list_of_r2dwidget +
    list_of_rmanager +
    list_of_rthreadmanager +
    list_of_rpythonconsole +
    list_of_rpythonterminal +
    list_of_drawtool +
    list_of_plotlimits +
    list_of_plot_style +
    list_of_rplotseries +
    list_of_rplotdecimator +
    list_of_rplotmodel +
    list_of_rplotticker
)

__all__ = _from_impl + [  # noqa: F822
    'enable',
    'import_error',
    'lazy_exports',
    'require',
]


if enable:
    for _name in _from_impl:
        globals()[_name] = getattr(_pilot_impl, _name)
    del _name

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

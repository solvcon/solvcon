/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <pybind11/stl.h> // Must be the first include.

#include <solvcon/pilot/wrap_pilot.hpp> // Must be the first include but give way to above.
#include <solvcon/python/common.hpp>

#include <solvcon/pilot/R2DWidget.hpp>
#include <solvcon/pilot/pilot.hpp>

#include <QClipboard>
#include <QMenu>
#include <QPixmap>
#include <QPointer>
#include <QString>

// Usually SOLVCON_PYSIDE6_FULL is not defined unless for debugging.
#ifdef SOLVCON_PYSIDE6_FULL
#include <pyside.h>
#else // SOLVCON_PYSIDE6_FULL
namespace PySide
{
// The prototypes are taken from pyside.h
PyTypeObject * getTypeForQObject(const QObject * cppSelf);
PyObject * getWrapperForQObject(QObject * cppSelf, PyTypeObject * sbk_type);
QObject * convertToQObject(PyObject * object, bool raiseError);
} // end namespace PySide
#endif // SOLVCON_PYSIDE6_FULL

namespace pybind11
{

namespace detail
{

template <typename type>
struct qt_type_caster
{
    // Adapted from PYBIND11_TYPE_CASTER.
protected:
    type * value;

public:
    template <typename T_, enable_if_t<std::is_same<type, remove_cv_t<T_>>::value, int> = 0>
    static handle cast(T_ * src, return_value_policy policy, handle parent)
    {
        if (!src)
            return none().release();
        if (policy == return_value_policy::take_ownership)
        {
            auto h = cast(std::move(*src), policy, parent);
            delete src;
            return h;
        }
        else
        {
            return cast(*src, policy, parent);
        }
    }
    operator type *() { return value; } /* NOLINT(bugprone-macro-parentheses) */
    operator type &() { return *value; } /* NOLINT(bugprone-macro-parentheses) */
    // Disable: operator type &&() && { return std::move(*value); } /* NOLINT(bugprone-macro-parentheses) */
    template <typename T_>
    using cast_op_type = pybind11::detail::movable_cast_op_type<T_>;
    // End adaptation from PYBIND11_TYPE_CASTER.

    bool load(handle src, bool)
    {
        if (!src)
        {
            return false;
        }

        QObject * q = PySide::convertToQObject(src.ptr(), /* raiseError */ true);
        if (!q)
        {
            return false;
        }

        value = qobject_cast<type *>(q);
        return true;
    }

    static handle cast(type * src, return_value_policy /* policy */, handle /* parent */)
    {
        PyObject * p = nullptr;
        PyTypeObject * to = PySide::getTypeForQObject(src);
        if (to)
        {
            p = PySide::getWrapperForQObject(src, to);
        }
        return pybind11::handle(p);
    }
};

#define QT_TYPE_CASTER(type, py_name)                      \
    template <>                                            \
    struct type_caster<type> : public qt_type_caster<type> \
    {                                                      \
        static constexpr auto name = py_name;              \
    }

QT_TYPE_CASTER(QWidget, _("QWidget"));
QT_TYPE_CASTER(QMenu, _("QMenu"));
QT_TYPE_CASTER(QCoreApplication, _("QCoreApplication"));
QT_TYPE_CASTER(QApplication, _("QApplication"));
QT_TYPE_CASTER(QMainWindow, _("QMainWindow"));
QT_TYPE_CASTER(QMdiArea, _("QMdiArea"));
QT_TYPE_CASTER(QMdiSubWindow, _("QMdiSubWindow"));

} /* end namespace detail */

} /* end namespace pybind11 */

PYBIND11_DECLARE_HOLDER_TYPE(T, QPointer<T>);

namespace solvcon
{

namespace python
{

/// Convert a length-3 Python sequence to a QVector3D.
static QVector3D seq_to_vec3(pybind11::sequence const & s)
{
    return QVector3D(
        s[0].cast<float>(), s[1].cast<float>(), s[2].cast<float>());
}

/// Convert a pick result to a Python dict, or None on a miss.
static pybind11::object pick_to_py(RDomainWidget::PickResult const & r)
{
    namespace py = pybind11;
    if (!r.hit())
    {
        return py::none();
    }
    py::dict d;
    d["kind"] = r.kind;
    d["id"] = r.id;
    d["type"] = r.type;
    d["measure"] = r.measure;
    d["centroid"] = py::make_tuple(r.centroid.x(), r.centroid.y(), r.centroid.z());
    return d;
}

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapRDomainWidget
    : public WrapBase<WrapRDomainWidget, RDomainWidget, QPointer<RDomainWidget>>
{

    friend root_base_type;

    WrapRDomainWidget(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {

        namespace py = pybind11;

        (*this)
            .def(py::init(
                []()
                {
                    return new RDomainWidget();
                }))
            .def(
                "resize",
                [](wrapped_type & self, int w, int h)
                {
                    self.resize(w, h);
                },
                py::arg("w"),
                py::arg("h"))
            .def_property_readonly("mesh", &wrapped_type::mesh)
            .def("updateMesh", &wrapped_type::updateMesh, py::arg("mesh"))
            .def("showMesh", &wrapped_type::showMesh, py::arg("show"))
            .def("setMeshOpacity", &wrapped_type::setMeshOpacity, py::arg("opacity"))
            .def("setFieldOpacity", &wrapped_type::setFieldOpacity, py::arg("opacity"))
            .def(
                "showMeshStyle",
                &wrapped_type::showMeshStyle,
                py::arg("name"),
                py::arg("show"),
                "Show or hide one mesh style (\"surface\", \"wireframe\", or "
                "\"points\") independently, so any combination can be drawn.")
            .def(
                "meshStyleShown",
                &wrapped_type::meshStyleShown,
                py::arg("name"),
                "Whether the named mesh style is currently shown.")
            .def(
                "updateColorField",
                &wrapped_type::updateColorField,
                py::arg("vertices"),
                py::arg("colors"),
                py::arg("indices"))
            .def(
                "updateScalarField",
                &wrapped_type::updateScalarField,
                py::arg("vertices"),
                py::arg("scalars"),
                py::arg("indices"))
            .def_property(
                "colormap",
                [](wrapped_type & self)
                {
                    return self.colormap();
                },
                [](wrapped_type & self, std::string const & name)
                {
                    self.setColormap(name);
                },
                "Named colormap for the scalar field and the scalar bar: "
                "\"viridis\", \"coolwarm\", \"jet\", or \"grayscale\".")
            .def(
                "setScalarRange",
                &wrapped_type::setScalarRange,
                py::arg("lo"),
                py::arg("hi"))
            .def_property_readonly(
                "scalarRange",
                [](wrapped_type & self)
                {
                    auto const range = self.scalarRange();
                    return py::make_tuple(range.first, range.second);
                })
            .def("showScalarBar", &wrapped_type::showScalarBar, py::arg("show"))
            .def(
                "setScalarBarTitle",
                &wrapped_type::setScalarBarTitle,
                py::arg("title"))
            .def("showBoundary", &wrapped_type::showBoundary, py::arg("ibc"), py::arg("show"))
            .def("showFeatureEdges", &wrapped_type::showFeatureEdges, py::arg("show"))
            .def("showNormals", &wrapped_type::showNormals, py::arg("show"))
            .def(
                "pickCell",
                [](wrapped_type & self, int x, int y)
                {
                    return pick_to_py(self.pickCell(x, y));
                },
                py::arg("x"),
                py::arg("y"),
                "Pick the cell under the pixel; a dict with id, type, measure, "
                "and centroid, or None on a miss.")
            .def(
                "pickNode",
                [](wrapped_type & self, int x, int y)
                {
                    return pick_to_py(self.pickNode(x, y));
                },
                py::arg("x"),
                py::arg("y"),
                "Pick the nearest node to the pixel ray; a dict or None.")
            .def(
                "pickFace",
                [](wrapped_type & self, int x, int y)
                {
                    return pick_to_py(self.pickFace(x, y));
                },
                py::arg("x"),
                py::arg("y"),
                "Pick the boundary face under the pixel (3D only); a dict or "
                "None.")
            .def("clearSelection", &wrapped_type::clearSelection)
            .def_property_readonly("hasSelection", &wrapped_type::hasSelection)
            .def(
                "measureDistance",
                [](wrapped_type & self, py::sequence p0, py::sequence p1)
                {
                    return self.measureDistance(seq_to_vec3(p0), seq_to_vec3(p1));
                },
                py::arg("p0"),
                py::arg("p1"),
                "Measure and draw the distance between two (x, y, z) points.")
            .def(
                "measureAngle",
                [](wrapped_type & self, py::sequence p0, py::sequence p1, py::sequence p2)
                {
                    return self.measureAngle(
                        seq_to_vec3(p0), seq_to_vec3(p1), seq_to_vec3(p2));
                },
                py::arg("p0"),
                py::arg("p1"),
                py::arg("p2"),
                "Measure and draw the angle (degrees) at p1 between the arms "
                "to p0 and p2.")
            .def("clearMeasurements", &wrapped_type::clearMeasurements)
            .def(
                "colorByCellType",
                &wrapped_type::colorByCellType,
                "Color the mesh by the cell element type through a discrete "
                "colormap with a legend.")
            .def(
                "colorByCellGroup",
                &wrapped_type::colorByCellGroup,
                "Color the mesh by the cell group (clgrp) through a discrete "
                "colormap with a legend.")
            .def(
                "colorByBoundary",
                &wrapped_type::colorByBoundary,
                "Color the mesh by boundary set through a discrete colormap "
                "with a legend.")
            .def(
                "clearCellColoring",
                &wrapped_type::clearCellColoring,
                "Remove the categorical cell coloring and its legend.")
            .def(
                "colorByQuality",
                &wrapped_type::colorByQuality,
                py::arg("metric"),
                "Color the mesh by a per-cell quality metric (\"volume\", "
                "\"aspect_ratio\", \"skewness\", \"min_angle\", \"max_angle\") "
                "through the continuous colormap and scalar bar.")
            .def(
                "qualityRange",
                [](wrapped_type & self, std::string const & metric)
                {
                    auto const range = self.qualityRange(metric);
                    return py::make_tuple(range.first, range.second);
                },
                py::arg("metric"),
                "The (min, max) range of the named quality metric over the "
                "cells.")
            .def("showAxis", &wrapped_type::showAxis, py::arg("show"))
            .def("fitCameraToScene", &wrapped_type::fitCameraToScene)
            .def(
                "setView",
                &wrapped_type::setView,
                py::arg("name"),
                "Point the camera along a view preset and frame the scene: "
                "\"front\", \"back\", \"left\", \"right\", \"top\", "
                "\"bottom\", \"+x\"..\"-z\", or \"iso\".")
            .def_property(
                "projection",
                [](wrapped_type & self)
                {
                    return self.projection();
                },
                [](wrapped_type & self, std::string const & name)
                {
                    self.setProjection(name);
                },
                "Projection: \"auto\" (orthographic 2D, perspective 3D), "
                "\"parallel\", or \"perspective\".")
            .def_property(
                "orbitStyle",
                [](wrapped_type & self)
                {
                    return self.orbitStyle();
                },
                [](wrapped_type & self, std::string const & name)
                {
                    self.setOrbitStyle(name);
                },
                "Orbit style: \"turntable\" (up axis fixed) or \"trackball\" "
                "(free tumble that can roll the horizon).")
            .def(
                "setOrbitStyle",
                &wrapped_type::setOrbitStyle,
                py::arg("name"),
                "Select the orbit style: \"turntable\" or \"trackball\".")
            .def(
                "setPivot",
                &wrapped_type::setPivot,
                py::arg("x"),
                py::arg("y"),
                py::arg("z"),
                "Set the orbit pivot, the point the orbit swings the eye "
                "around.")
            .def(
                "frameSelected",
                &wrapped_type::frameSelected,
                "Frame the current selection, or the whole scene when nothing "
                "is selected.")
            .def(
                "zoomToSelection",
                &wrapped_type::zoomToSelection,
                "Frame the current selection (from a pick), or the whole scene "
                "when nothing is selected.")
            .def(
                "resetCamera",
                &wrapped_type::resetCamera,
                "Reset the camera to the fit-to-scene default.")
            .def_property(
                "navigationMapping",
                [](wrapped_type & self)
                {
                    return self.navigationMapping();
                },
                [](wrapped_type & self, std::string const & name)
                {
                    self.setNavigationMapping(name);
                },
                "Mouse navigation mapping: \"default\" (left rotates) or "
                "\"blender\" (middle orbits; Shift/Ctrl/Alt+middle pan/zoom/"
                "pivot; Alt+left aliases middle).")
            .def(
                "setNavigationMapping",
                &wrapped_type::setNavigationMapping,
                py::arg("name"))
            .def(
                "setOrbitSensitivity",
                &wrapped_type::setOrbitSensitivity,
                py::arg("factor"),
                "Scale the orbit/look drag speed.")
            .def(
                "orbitStep",
                &wrapped_type::orbitStep,
                py::arg("yaw_deg"),
                py::arg("pitch_deg"),
                "Orbit by a fixed number of degrees (a discrete step).")
            .def_property(
                "cameraMode",
                [](wrapped_type & self)
                {
                    return self.cameraMode();
                },
                [](wrapped_type & self, std::string const & name)
                {
                    self.setCameraMode(name);
                },
                "Navigation mode: \"pan\" (2D pan/zoom), \"fps\" (3D "
                "first-person), or \"orbit\" (3D orbit around the target).")
            .def(
                "rotateCamera",
                [](wrapped_type & self, float dx, float dy)
                {
                    self.rotateCamera(dx, dy);
                },
                py::arg("dx"),
                py::arg("dy"))
            .def(
                "panCamera",
                [](wrapped_type & self, float dx, float dy)
                {
                    self.panCamera(dx, dy);
                },
                py::arg("dx"),
                py::arg("dy"))
            .def(
                "zoomCamera",
                [](wrapped_type & self, float steps)
                {
                    self.zoomCamera(steps);
                },
                py::arg("steps"))
            .def(
                "pinchCamera",
                [](wrapped_type & self, float factor)
                {
                    self.pinchCamera(factor);
                },
                py::arg("factor"))
#define MM_DECL_CAMERA_VECTOR(NAME, GETTER, SETTER)            \
    .def_property(                                             \
        NAME,                                                  \
        [](wrapped_type & self)                                \
        {                                                      \
            QVector3D const v = self.GETTER();                 \
            return py::make_tuple(v.x(), v.y(), v.z());        \
        },                                                     \
        [](wrapped_type & self, std::vector<double> const & v) \
        {                                                      \
            self.SETTER(QVector3D(v.at(0), v.at(1), v.at(2))); \
        })
            // clang-format off
            MM_DECL_CAMERA_VECTOR("cameraPosition", cameraPosition, setCameraPosition)
            MM_DECL_CAMERA_VECTOR("cameraTarget", cameraTarget, setCameraTarget)
            MM_DECL_CAMERA_VECTOR("cameraUp", cameraUp, setCameraUp)
        // clang-format on
#undef MM_DECL_CAMERA_VECTOR
            .def(
                "saveImage",
                [](wrapped_type & self, std::string const & filename)
                {
                    self.grabImage().save(filename.c_str());
                },
                py::arg("filename"))
            .def(
                "clipImage",
                [](wrapped_type & self)
                {
                    QClipboard * clipboard = QGuiApplication::clipboard();
                    clipboard->setPixmap(QPixmap::fromImage(self.grabImage()));
                })
            //
            ;
    }

}; /* end class WrapRDomainWidget */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapR2DWidget
    : public WrapBase<WrapR2DWidget, R2DWidget, QPointer<R2DWidget>>
{

    friend root_base_type;

    WrapR2DWidget(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .def_property_readonly(
                "viewTransform",
                &wrapped_type::viewTransform,
                py::return_value_policy::copy)
            .def("setViewTransform", &wrapped_type::setViewTransform, py::arg("v"))
            .def("resetView", &wrapped_type::resetView)
            .def("updateWorld", &wrapped_type::updateWorld, py::arg("world"))
            .def("requestRepaint", &wrapped_type::requestRepaint)
            .def("setDrawTool", &wrapped_type::setDrawTool, py::arg("name"))
            .def_property_readonly("drawTool", &wrapped_type::drawTool)
            .def_property_readonly("selectedShape", &wrapped_type::selectedShape)
            .def_property_readonly(
                "rotateHandleScreen",
                [](wrapped_type & self)
                {
                    // TODO: add pybind for small_vector to avoid this copy.
                    auto const h = self.rotateHandleScreen();
                    return std::vector<double>(h.begin(), h.end());
                })
            //
            ;

        (*this)
            .def("clipImage", [](wrapped_type & self)
                 {
                     QClipboard * clipboard = QGuiApplication::clipboard();
                     clipboard->setPixmap(self.grab()); })
            .def("saveImage", [](wrapped_type & self, std::string const & filename)
                 { self.grab().save(filename.c_str()); },
                 py::arg("filename"))
            //
            ;
    }

}; /* end class WrapR2DWidget */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapRPythonConsoleDockWidget
    : public WrapBase<WrapRPythonConsoleDockWidget, RPythonConsoleDockWidget>
{

    friend root_base_type;

    WrapRPythonConsoleDockWidget(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {

        namespace py = pybind11;

        (*this)
            .def("writeToHistory", &wrapped_type::writeToHistory)
            .def_property(
                "command",
                [](wrapped_type const & self)
                {
                    return self.command().toStdString();
                },
                [](wrapped_type & self, std::string const & command)
                {
                    return self.setCommand(QString::fromStdString(command));
                })
            .def_property(
                "python_redirect",
                [](wrapped_type const & self)
                {
                    return self.hasPythonRedirect();
                },
                [](wrapped_type & self, bool enabled)
                {
                    self.setPythonRedirect(enabled);
                })
            //
            ;
    }

}; /* end class WrapRPythonConsoleDockWidget */

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapRManager
    : public WrapBase<WrapRManager, RManager>
{

    friend root_base_type;

    WrapRManager(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {

        namespace py = pybind11;

        (*this)
            .def_property_readonly_static(
                "instance",
                [](py::object const &) -> wrapped_type &
                {
                    return RManager::instance();
                })
            .def_property_readonly_static(
                "core",
                [](py::object const &) -> QCoreApplication *
                {
                    return RManager::instance().core();
                })
            .def("setUp", &RManager::setUp)
            .def(
                "exec",
                [](wrapped_type & self)
                {
                    return self.core()->exec();
                })
            .wrap_widget()
            .wrap_mainWindow()
            //
            ;
    }

    wrapper_type & wrap_widget()
    {
        namespace py = pybind11;

        (*this)
            .def_property_readonly(
                "pycon",
                [](wrapped_type & self)
                {
                    return self.pycon();
                })
            .def(
                "add3DWidget",
                [](wrapped_type & self)
                {
                    return self.add3DWidget();
                })
            .def(
                "add2DWidget",
                [](wrapped_type & self)
                {
                    return self.add2DWidget();
                })
            .def(
                "currentR3DWidget",
                [](wrapped_type & self)
                {
                    return self.currentR3DWidget();
                })
            .def(
                "currentR2DWidget",
                [](wrapped_type & self)
                {
                    return self.currentR2DWidget();
                })
            .def(
                "list2DWidgets",
                [](wrapped_type & self)
                {
                    return self.list2DWidgets();
                })
            .def("setDrawTool", &wrapped_type::setDrawTool, py::arg("name"))
            .def_property_readonly("drawTool", &wrapped_type::drawTool)
            .def_property_readonly(
                "mdiArea",
                [](wrapped_type & self)
                {
                    return self.mdiArea();
                })
            .def(
                "toggleConsole",
                [](wrapped_type & self)
                {
                    return self.toggleConsole();
                })
            //
            ;

        return *this;
    }

    wrapper_type & wrap_mainWindow()
    {
        namespace py = pybind11;

        (*this)
            .def_property_readonly(
                "mainWindow",
                [](wrapped_type & self) -> QMainWindow *
                {
                    return self.mainWindow();
                })
            .def_property_readonly("fileMenu", &wrapped_type::fileMenu)
            .def_property_readonly("viewMenu", &wrapped_type::viewMenu)
            .def_property_readonly("oneMenu", &wrapped_type::oneMenu)
            .def_property_readonly("meshMenu", &wrapped_type::meshMenu)
            .def_property_readonly("canvasMenu", &wrapped_type::canvasMenu)
            .def_property_readonly("profilingMenu", &wrapped_type::profilingMenu)
            .def_property_readonly("windowMenu", &wrapped_type::windowMenu)
            .def(
                "quit",
                [](wrapped_type & self)
                {
                    self.quit();
                })
            .def(
                "show",
                [](wrapped_type & self)
                {
                    self.mainWindow()->show();
                })
            .def(
                "resize",
                [](wrapped_type & self, int w, int h)
                {
                    self.mainWindow()->resize(w, h);
                },
                py::arg("w"),
                py::arg("h"))
            .def(
                "addSubWindow",
                [](wrapped_type & self, QWidget * widget)
                {
                    QMdiSubWindow * subwin = self.addSubWindow(widget);
                    subwin->resize(300, 200);
                    subwin->setAttribute(Qt::WA_DeleteOnClose);
                    return subwin;
                },
                py::arg("widget"))
            .def_property(
                "windowTitle",
                [](wrapped_type & self)
                {
                    return self.mainWindow()->windowTitle().toStdString();
                },
                [](wrapped_type & self, std::string const & name)
                {
                    self.mainWindow()->setWindowTitle(QString::fromStdString(name));
                })
            //
            ;

        return *this;
    }

}; /* end class WrapRManager */

struct RManagerProxy
{
};

class SOLVCON_PYTHON_WRAPPER_VISIBILITY WrapRManagerProxy
    : public WrapBase<WrapRManagerProxy, RManagerProxy>
{

    friend root_base_type;

    WrapRManagerProxy(pybind11::module & mod, char const * pyname, char const * pydoc)
        : root_base_type(mod, pyname, pydoc)
    {
        namespace py = pybind11;

        (*this)
            .def(
                "__getattr__",
                [](wrapped_type &, char const * name)
                {
                    py::object obj = py::cast(RManager::instance());
                    obj = obj.attr(name);
                    return obj;
                })
            //
            ;
    }

}; /* end class WrapRManagerProxy */

void wrap_pilot(pybind11::module & mod)
{
    namespace py = pybind11;

    WrapRDomainWidget::commit(
        mod,
        "RDomainWidget",
        "Interactive QRhi viewer for 2D and 3D unstructured-mesh domains and "
        "fields. Drive it with updateMesh / showMesh / showMeshStyle / "
        "setMeshOpacity, "
        "updateColorField / setFieldOpacity, "
        "showBoundary, showFeatureEdges, showNormals, and showAxis; navigate "
        "with cameraMode, the "
        "cameraPosition / cameraTarget / cameraUp pose, rotateCamera / "
        "panCamera / zoomCamera / pinchCamera, and fitCameraToScene; capture "
        "frames with "
        "saveImage / clipImage.");
    WrapR2DWidget::commit(mod, "R2DWidget", "R2DWidget");
    WrapRPythonConsoleDockWidget::commit(mod, "RPythonConsoleDockWidget", "RPythonConsoleDockWidget");
    WrapRManager::commit(mod, "RManager", "RManager");
    WrapRManagerProxy::commit(mod, "RManagerProxy", "RManagerProxy");

    // The C++ tool registry is the single source of truth for drawing tools.
    mod.def("draw_tool_names", &draw_tool_names);
    mod.def("default_draw_tool_name", &default_draw_tool_name);

    mod.attr("mgr") = RManagerProxy();

    try
    {
        // Creating module level variable to handle Qt MainWindow which is
        // created by c++ and registered it to Shiboken6 to prevent runtime
        // error occured.
        // RuntimeError:
        // Internal C++ object (PySide6.QtGui.QWindow) already deleted.
        // py::module::import("PySide6.QtWidgets");
        py::globals()["_mainWindow"] = RManager::instance().mainWindow();
    }
    catch (const pybind11::error_already_set & e)
    {
        std::cerr << e.what() << std::endl;
    }

    // Register a Python atexit handler to destroy QApplication while Python
    // is still fully alive.
    try
    {
        py::module_::import("atexit").attr("register")(
            py::cpp_function([]()
                             { RManager::instance().reset(); }));
    }
    catch (const pybind11::error_already_set & e)
    {
        std::cerr << e.what() << std::endl;
    }
}

struct view_pymod_tag;

template <>
OneTimeInitializer<view_pymod_tag> & OneTimeInitializer<view_pymod_tag>::me()
{
    static OneTimeInitializer<view_pymod_tag> instance;
    return instance;
}

void initialize_pilot(pybind11::module & mod)
{
    auto initialize_impl = [](pybind11::module & mod)
    {
        wrap_pilot(mod);
    };

    if (Toggle::instance().solid().use_pyside())
    {
        try
        {
            // Before using PySide6 api, the function signature need
            // to be imported or will get type error:
            // TypeError: Unable to convert function return value to a Python type!
            pybind11::module::import("PySide6.QtWidgets");
        }
        catch (const pybind11::error_already_set & e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

    OneTimeInitializer<view_pymod_tag>::me()(mod, initialize_impl);
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

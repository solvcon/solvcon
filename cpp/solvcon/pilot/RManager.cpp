/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RManager.hpp> // Must be the first include.

#include <functional>
#include <stdexcept>
#include <vector>

#include <solvcon/pilot/DrawTool.hpp>
#include <solvcon/pilot/RAction.hpp>
#include <solvcon/pilot/RMenuModel.hpp>
#include <Qt>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QActionGroup>
#include <QKeySequence>
#include <QVBoxLayout>
#include <QWidget>

namespace solvcon
{

RManager & RManager::instance()
{
    static RManager ret;
    return ret;
}

RManager::RManager()
    : QObject()
{
    m_core.reset(QApplication::instance());
    static int argc = 1;
    static char exename[] = "pilot";
    static char * argv[] = {exename};
    if (!m_core)
    {
        m_core.reset(new QApplication(argc, argv));
    }

    m_mainWindow = new QMainWindow;
    m_mainWindow->setWindowIcon(QIcon(QString(":/icon.ico")));
    // Do not call setUp() from the constructor.  Windows may crash with
    // "exited with code -1073740791".  The reason is not yet clarified.
}

RManager & RManager::setUp()
{
    if (!m_already_setup)
    {
        this->setUpConsole();
        this->setUpCentral();
        this->primeRhiComposition();
        this->setUpMenu();

        m_already_setup = true;
    }
    return *this;
}

void RManager::reset()
{
    // Nullify all child pointers before destroying QApplication.
    // Qt's parent-child mechanism will delete widget children; we only
    // need to ensure our raw pointers don't dangle after the call.
    m_already_setup = false;
    // Empty the menu model while the application (and its widgets) is still
    // alive, so a repeated setUp on the singleton starts from a clean bar.
    if (m_menuModel)
    {
        m_menuModel->clear();
    }
    m_core.reset();
    m_mainWindow = nullptr;
    m_fileMenu = nullptr;
    m_editMenu = nullptr;
    m_viewMenu = nullptr;
    m_oneMenu = nullptr;
    m_meshMenu = nullptr;
    m_canvasMenu = nullptr;
    m_profilingMenu = nullptr;
    m_windowMenu = nullptr;
    m_menuModel = nullptr;
    m_pycon = nullptr;
    m_mdiArea = nullptr;
    m_rhi_primer = nullptr;
}

RManager::~RManager()
{
    reset();
}

RDomainWidget * RManager::add3DWidget()
{
    RDomainWidget * viewer = nullptr;
    if (m_mdiArea)
    {
        // A QRhiWidget cannot be the direct child of a QMdiSubWindow: nested
        // there it never reaches a QRhi-flushed backing store, so it logs
        // "QRhiWidget: No QRhi" and draws nothing, and a second viewer brings
        // the swapchain down with it (seen on macOS). Host the viewer inside a
        // plain container widget, which composites correctly and lets several
        // viewers coexist.
        auto * host = new QWidget;
        host->setWindowTitle("Domain viewer");
        auto * layout = new QVBoxLayout(host);
        layout->setContentsMargins(0, 0, 0, 0);
        viewer = new RDomainWidget(/*parent*/ host);
        layout->addWidget(viewer);
        // Associate the Escape reset shortcut with this viewer; a
        // Qt::WidgetShortcut fires only while a widget it was added to has
        // focus. The hidden RHI primer is created elsewhere and never reaches
        // here, so it stays unbound.
        if (m_menuModel)
        {
            if (QAction * reset = m_menuModel->action("camera.reset"))
            {
                viewer->addAction(reset);
            }
        }
        auto * subwin = this->addSubWindow(host);
        subwin->resize(400, 300);
    }
    return viewer;
}

R2DWidget * RManager::add2DWidget()
{
    R2DWidget * viewer = nullptr;
    if (m_mdiArea)
    {
        viewer = new R2DWidget(/*parent*/ m_mdiArea);
        viewer->setWindowTitle("2D canvas");
        viewer->show();
        auto * subwin = this->addSubWindow(viewer);
        subwin->resize(400, 300);
        viewer->resize(400, 300);
    }
    return viewer;
}

RDomainWidget * RManager::currentR3DWidget()
{
    if (m_mdiArea == nullptr)
    {
        return nullptr;
    }

    return domainWidgetOf(m_mdiArea->currentSubWindow());
}

/// The RDomainWidget hosted by @p subwin, or nullptr when @p subwin holds
/// some other widget. The viewer sits inside a plain container widget, so
/// reach through the subwindow's direct child to find it.
RDomainWidget * RManager::domainWidgetOf(QMdiSubWindow * subwin)
{
    if (subwin == nullptr)
    {
        return nullptr;
    }
    QWidget * host = subwin->widget();
    if (host == nullptr)
    {
        return nullptr;
    }
    // The host is the container; the viewer is its child. Guard the host
    // itself too, in case an unwrapped viewer is ever added directly.
    if (auto * viewer = dynamic_cast<RDomainWidget *>(host))
    {
        return viewer;
    }
    return host->findChild<RDomainWidget *>();
}

R2DWidget * RManager::currentR2DWidget()
{
    if (m_mdiArea == nullptr)
    {
        return nullptr;
    }

    const auto * subwin = m_mdiArea->currentSubWindow();
    if (subwin == nullptr)
    {
        return nullptr;
    }

    return dynamic_cast<R2DWidget *>(subwin->widget());
}

std::vector<R2DWidget *> RManager::list2DWidgets()
{
    std::vector<R2DWidget *> widgets;
    if (m_mdiArea == nullptr)
    {
        return widgets;
    }

    for (auto subwin : m_mdiArea->subWindowList())
    {
        auto * viewer = dynamic_cast<R2DWidget *>(subwin->widget());

        if (viewer == nullptr)
            continue;

        widgets.push_back(viewer);
    }
    return widgets;
}

void RManager::setDrawTool(std::string const & name)
{
    // Validate eagerly so an unknown tool is rejected even when no 2D
    // canvas is focused to surface the error.
    if (!is_draw_tool(name))
    {
        throw std::invalid_argument("RManager::setDrawTool: unknown tool '" + name + "'");
    }
    m_draw_tool = name;
    applyDrawTool();
    // Keep the draw-tool action group in step, so scripting setDrawTool from
    // the console checks the matching radio item and toolbox button. The
    // action group's triggered wiring is unaffected because setChecked emits
    // toggled, not triggered.
    if (m_menuModel)
    {
        if (QAction * action = m_menuModel->action("draw.tool." + name))
        {
            action->setChecked(true);
        }
    }
}

void RManager::applyDrawTool()
{
    if (R2DWidget * canvas = currentR2DWidget())
    {
        canvas->setDrawTool(m_draw_tool);
    }
}

void RManager::undoCanvas() const
{
    auto const * subwin = m_mdiArea ? m_mdiArea->currentSubWindow() : nullptr;
    auto * canvas = subwin ? dynamic_cast<R2DWidget *>(subwin->widget()) : nullptr;
    if (canvas == nullptr || canvas->world() == nullptr)
    {
        return;
    }
    canvas->world()->undo();
    canvas->requestRepaint();
}

void RManager::redoCanvas() const
{
    auto const * subwin = m_mdiArea ? m_mdiArea->currentSubWindow() : nullptr;
    auto * canvas = subwin ? dynamic_cast<R2DWidget *>(subwin->widget()) : nullptr;
    if (canvas == nullptr || canvas->world() == nullptr)
    {
        return;
    }
    canvas->world()->redo();
    canvas->requestRepaint();
}

void RManager::toggleConsole()
{
    if (m_pycon)
    {
        if (m_pycon->isVisible())
        {
            m_pycon->hide();
        }
        else
        {
            m_pycon->show();
        }
    }
}

void RManager::setUpConsole()
{
    m_pycon = new RPythonConsoleDockWidget(QString("Console"), m_mainWindow);
    m_pycon->setAllowedAreas(Qt::AllDockWidgetAreas);
    m_mainWindow->addDockWidget(Qt::BottomDockWidgetArea, m_pycon);
}

void RManager::setUpCentral()
{
    m_mdiArea = new QMdiArea(m_mainWindow);
    m_mdiArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_mdiArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_mainWindow->setCentralWidget(m_mdiArea);
    // Keep the focused 2D canvas in step with the active draw tool, so the
    // single Painter toolbox always drives whichever canvas has focus.
    QObject::connect(m_mdiArea, &QMdiArea::subWindowActivated, m_mdiArea, [this](QMdiSubWindow *)
                     { applyDrawTool(); });
}

void RManager::primeRhiComposition()
{
    // A QRhiWidget renders to a texture. The first one to appear in the main
    // window forces Qt to recreate the top-level native window so its backing
    // store can flush through QRhi; on macOS that tears down and rebuilds every
    // existing sub-window and dock, so the GUI looks like it restarts and any
    // already-open viewer vanishes. Trigger that switch once, before any user
    // content exists, by parking a hidden RDomainWidget in the window. The
    // recreation then happens with nothing to lose, and later viewers reuse the
    // same native window. The widget tree owns the primer.
    m_rhi_primer = new RDomainWidget(m_mdiArea);
    m_rhi_primer->resize(0, 0);
    m_rhi_primer->hide();
}

void RManager::setUpMenu()
{
    m_mainWindow->setMenuBar(new QMenuBar(nullptr));
    // NOTE: All menus need to be populated or Windows may crash with
    // "exited with code -1073740791".  The reason is not yet clarified.

    // Parent the model to the main window so the widget tree owns it.
    m_menuModel = new RMenuModel(m_mainWindow, m_mainWindow);

    // Seed the bar from one weighted table, the single source of truth for
    // its order. The weight bands leave room to slot a menu between two
    // others without renumbering. The members hold the model's menus until
    // the getters become adapters over the model.
    m_fileMenu = m_menuModel->menu("File", 0);
    m_editMenu = m_menuModel->menu("Edit", 10);
    m_viewMenu = m_menuModel->menu("View", 20);
    m_oneMenu = m_menuModel->menu("One", 30);
    m_meshMenu = m_menuModel->menu("Mesh", 40);
    m_canvasMenu = m_menuModel->menu("Canvas", 50);
    m_profilingMenu = m_menuModel->menu("Profiling", 60);
    m_windowMenu = m_menuModel->menu("Window", 70);

    setUpEditMenuItems();
    // Code for controlling camera is not exposed to Python yet.
    setUpCameraControllersMenuItems();
    setUpCameraMovementMenuItems();
}

void RManager::setUpEditMenuItems() const
{
    // Parent the actions to the model so they die with it instead of leaking.
    auto * undo_action = new RAction(
        QString("Undo"),
        QString("Undo the last change on the focused 2D canvas"),
        [this]()
        { undoCanvas(); },
        m_menuModel);
    undo_action->setObjectName("edit.undo");
    undo_action->setShortcut(QKeySequence::Undo);

    auto * redo_action = new RAction(
        QString("Redo"),
        QString("Redo the last undone change on the focused 2D canvas"),
        [this]()
        { redoCanvas(); },
        m_menuModel);
    redo_action->setObjectName("edit.redo");
    redo_action->setShortcut(QKeySequence::Redo);

    m_menuModel->place("Edit", undo_action, 10);
    m_menuModel->place("Edit", redo_action, 20);
}

void RManager::setUpCameraControllersMenuItems() const
{
    auto set_mode = [this](std::string const & mode)
    {
        for (auto subwin : m_mdiArea->subWindowList())
        {
            if (auto * viewer = domainWidgetOf(subwin))
            {
                viewer->setCameraMode(mode);
            }
        }
    };

    struct CameraMode
    {
        char const * id;
        QString text;
        QString tip;
        std::string mode;
    };
    std::vector<CameraMode> const modes = {
        {"camera.mode_orbit", QString("Orbit camera (3D)"), QString("Orbit the domain around its center"), "orbit"},
        {"camera.mode_fps", QString("First-person camera (3D)"), QString("Fly through the domain in first person"), "fps"},
        {"camera.mode_pan", QString("Pan / zoom camera (2D)"), QString("Pan and zoom the domain in the plane"), "pan"},
    };

    // The group owns the exclusive mode, held by the model under a group id
    // so Python can query the checked action.
    auto * cameraGroup = m_menuModel->group("camera.mode");
    int weight = 10;
    for (auto const & item : modes)
    {
        auto * action = new RAction(
            item.text, item.tip, [set_mode, mode = item.mode]()
            { set_mode(mode); },
            m_menuModel);
        action->setObjectName(item.id);
        action->setCheckable(true);
        cameraGroup->addAction(action);
        m_menuModel->place("View/Camera", action, weight);
        weight += 10;
    }
    if (QAction * orbit = m_menuModel->action("camera.mode_orbit"))
    {
        orbit->setChecked(true);
    }
}

void RManager::setUpCameraMovementMenuItems() const
{
    static constexpr float pan_step = 40.0f;
    static constexpr float rotate_step = 30.0f;
    static constexpr float zoom_step = 1.0f;

    struct CameraMove
    {
        char const * id;
        QString text;
        std::function<void(RDomainWidget *)> act;
    };
    // The keyboard hints the labels used to carry were wrong: those keys are
    // handled in RDomainWidget::keyPressEvent with different behavior, so they
    // are dropped here rather than advertised.
    std::vector<CameraMove> const moves = {
        {"camera.reset", QString("Reset camera"), [](RDomainWidget * v)
         { v->fitCameraToScene(); }},
        {"camera.move_up", QString("Move camera up"), [](RDomainWidget * v)
         { v->panCamera(0.0f, pan_step); }},
        {"camera.move_down", QString("Move camera down"), [](RDomainWidget * v)
         { v->panCamera(0.0f, -pan_step); }},
        {"camera.move_right", QString("Move camera right"), [](RDomainWidget * v)
         { v->panCamera(-pan_step, 0.0f); }},
        {"camera.move_left", QString("Move camera left"), [](RDomainWidget * v)
         { v->panCamera(pan_step, 0.0f); }},
        {"camera.move_forward", QString("Move camera forward"), [](RDomainWidget * v)
         { v->zoomCamera(zoom_step); }},
        {"camera.move_backward", QString("Move camera backward"), [](RDomainWidget * v)
         { v->zoomCamera(-zoom_step); }},
        {"camera.yaw_positive", QString("Rotate camera positive yaw"), [](RDomainWidget * v)
         { v->rotateCamera(rotate_step, 0.0f); }},
        {"camera.yaw_negative", QString("Rotate camera negative yaw"), [](RDomainWidget * v)
         { v->rotateCamera(-rotate_step, 0.0f); }},
        {"camera.pitch_positive", QString("Rotate camera positive pitch"), [](RDomainWidget * v)
         { v->rotateCamera(0.0f, rotate_step); }},
        {"camera.pitch_negative", QString("Rotate camera negative pitch"), [](RDomainWidget * v)
         { v->rotateCamera(0.0f, -rotate_step); }},
    };

    int weight = 10;
    for (auto const & item : moves)
    {
        auto * action = new RAction(
            item.text, item.text, createCameraMovementItemHandler(item.act), m_menuModel);
        action->setObjectName(item.id);
        m_menuModel->place("View/Camera move", action, weight);
        weight += 10;
    }

    // Escape resets the camera; add3DWidget attaches this action to each
    // viewer so its Qt::WidgetShortcut context can fire.
    if (QAction * reset = m_menuModel->action("camera.reset"))
    {
        reset->setShortcut(QKeySequence(Qt::Key_Escape));
        reset->setShortcutContext(Qt::WidgetShortcut);
    }
}

std::function<void()> RManager::createCameraMovementItemHandler(const std::function<void(RDomainWidget *)> & func) const
{
    return [this, func]()
    {
        if (m_mdiArea == nullptr)
        {
            return;
        }
        if (auto * viewer = domainWidgetOf(m_mdiArea->currentSubWindow()))
        {
            func(viewer);
        }
    };
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

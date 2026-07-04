/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/pilot/RManager.hpp> // Must be the first include.

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

    // The live menu model shares the same bar; consumers migrate onto it in
    // later steps. Parent it to the main window so the widget tree owns it.
    m_menuModel = new RMenuModel(m_mainWindow, m_mainWindow);

    m_fileMenu = m_mainWindow->menuBar()->addMenu(QString("File"));
    m_editMenu = m_mainWindow->menuBar()->addMenu(QString("Edit"));
    setUpEditMenuItems();
    m_viewMenu = m_mainWindow->menuBar()->addMenu(QString("View"));
    {
        // Code for controlling camera is not exposed to Python yet
        setUpCameraControllersMenuItems();
        setUpCameraMovementMenuItems();
    }
    m_oneMenu = m_mainWindow->menuBar()->addMenu(QString("One"));
    m_meshMenu = m_mainWindow->menuBar()->addMenu(QString("Mesh"));
    m_canvasMenu = m_mainWindow->menuBar()->addMenu(QString("Canvas"));
    m_profilingMenu = m_mainWindow->menuBar()->addMenu(QString("Profiling"));
    m_windowMenu = m_mainWindow->menuBar()->addMenu(QString("Window"));
}

void RManager::setUpEditMenuItems() const
{
    auto * undo_action = new RAction(
        QString("Undo"),
        QString("Undo the last change on the focused 2D canvas"),
        [this]()
        { undoCanvas(); });
    undo_action->setShortcut(QKeySequence::Undo);

    auto * redo_action = new RAction(
        QString("Redo"),
        QString("Redo the last undone change on the focused 2D canvas"),
        [this]()
        { redoCanvas(); });
    redo_action->setShortcut(QKeySequence::Redo);

    m_editMenu->addAction(undo_action);
    m_editMenu->addAction(redo_action);
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

    auto * use_pan_camera = new RAction(
        QString("Pan / zoom camera (2D)"),
        QString("Pan and zoom the domain in the plane"),
        [set_mode]()
        { set_mode("pan"); });

    auto * use_fps_camera = new RAction(
        QString("First-person camera (3D)"),
        QString("Fly through the domain in first person"),
        [set_mode]()
        { set_mode("fps"); });

    auto * use_orbit_camera = new RAction(
        QString("Orbit camera (3D)"),
        QString("Orbit the domain around its center"),
        [set_mode]()
        { set_mode("orbit"); });

    auto * cameraGroup = new QActionGroup(m_mainWindow);
    cameraGroup->addAction(use_orbit_camera);
    cameraGroup->addAction(use_fps_camera);
    cameraGroup->addAction(use_pan_camera);

    use_orbit_camera->setCheckable(true);
    use_fps_camera->setCheckable(true);
    use_pan_camera->setCheckable(true);
    use_orbit_camera->setChecked(true);

    auto * cameraMenu = m_viewMenu->addMenu(QString("Camera"));
    cameraMenu->addAction(use_orbit_camera);
    cameraMenu->addAction(use_fps_camera);
    cameraMenu->addAction(use_pan_camera);
}

void RManager::setUpCameraMovementMenuItems() const
{
    constexpr float pan_step = 40.0f;
    constexpr float rotate_step = 30.0f;
    constexpr float zoom_step = 1.0f;

    auto * reset_camera = new RAction(
        QString("Reset (esc)"),
        QString("Reset (esc)"),
        createCameraMovementItemHandler([](RDomainWidget * viewer)
                                        { viewer->fitCameraToScene(); }));

    auto * move_camera_up = new RAction(
        QString("Move camera up (W/UP)"),
        QString("Move camera up (W/UP)"),
        createCameraMovementItemHandler([](RDomainWidget * viewer)
                                        { viewer->panCamera(0.0f, pan_step); }));

    auto * move_camera_down = new RAction(
        QString("Move camera down (S/DOWN)"),
        QString("Move camera down (S/DOWN)"),
        createCameraMovementItemHandler([](RDomainWidget * viewer)
                                        { viewer->panCamera(0.0f, -pan_step); }));

    auto * move_camera_right = new RAction(
        QString("Move camera right (D/RIGHT)"),
        QString("Move camera right (D/RIGHT)"),
        createCameraMovementItemHandler([](RDomainWidget * viewer)
                                        { viewer->panCamera(-pan_step, 0.0f); }));

    auto * move_camera_left = new RAction(
        QString("Move camera left (A/LEFT)"),
        QString("Move camera left (A/LEFT)"),
        createCameraMovementItemHandler([](RDomainWidget * viewer)
                                        { viewer->panCamera(pan_step, 0.0f); }));

    auto * move_camera_forward = new RAction(
        QString("Move camera forward (Ctrl+W/UP)"),
        QString("Move camera forward (Ctrl+W/UP)"),
        createCameraMovementItemHandler([](RDomainWidget * viewer)
                                        { viewer->zoomCamera(zoom_step); }));

    auto * move_camera_backward = new RAction(
        QString("Move camera backward (Ctrl+S/DOWN)"),
        QString("Move camera backward (Ctrl+S/DOWN)"),
        createCameraMovementItemHandler([](RDomainWidget * viewer)
                                        { viewer->zoomCamera(-zoom_step); }));

    auto * rotate_camera_positive_yaw = new RAction(
        QString("Rotate camera positive yaw"),
        QString("Rotate camera positive yaw"),
        createCameraMovementItemHandler([](RDomainWidget * viewer)
                                        { viewer->rotateCamera(rotate_step, 0.0f); }));

    auto * rotate_camera_negative_yaw = new RAction(
        QString("Rotate camera negative yaw"),
        QString("Rotate camera negative yaw"),
        createCameraMovementItemHandler([](RDomainWidget * viewer)
                                        { viewer->rotateCamera(-rotate_step, 0.0f); }));

    auto * rotate_camera_positive_pitch = new RAction(
        QString("Rotate camera positive pitch"),
        QString("Rotate camera positive pitch"),
        createCameraMovementItemHandler([](RDomainWidget * viewer)
                                        { viewer->rotateCamera(0.0f, rotate_step); }));

    auto * rotate_camera_negative_pitch = new RAction(
        QString("Rotate camera negative pitch"),
        QString("Rotate camera negative pitch"),
        createCameraMovementItemHandler([](RDomainWidget * viewer)
                                        { viewer->rotateCamera(0.0f, -rotate_step); }));

    reset_camera->setShortcut(QKeySequence(Qt::Key_Escape));
    reset_camera->setShortcutContext(Qt::WidgetShortcut);

    auto cameraMoveSubmenu = m_viewMenu->addMenu("Camera move");
    cameraMoveSubmenu->addAction(reset_camera);
    cameraMoveSubmenu->addAction(move_camera_up);
    cameraMoveSubmenu->addAction(move_camera_down);
    cameraMoveSubmenu->addAction(move_camera_right);
    cameraMoveSubmenu->addAction(move_camera_left);
    cameraMoveSubmenu->addAction(move_camera_forward);
    cameraMoveSubmenu->addAction(move_camera_backward);
    cameraMoveSubmenu->addAction(rotate_camera_positive_yaw);
    cameraMoveSubmenu->addAction(rotate_camera_negative_yaw);
    cameraMoveSubmenu->addAction(rotate_camera_positive_pitch);
    cameraMoveSubmenu->addAction(rotate_camera_negative_pitch);
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

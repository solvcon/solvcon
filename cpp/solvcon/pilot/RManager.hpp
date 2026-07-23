#pragma once

/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

/**
 * @file
 * Singleton manager that owns the pilot main window and its child widgets.
 *
 * @ingroup group_domain
 */

#include <solvcon/pilot/common/common_detail.hpp> // Must be the first include.

#include <solvcon/pilot/DrawTool.hpp>
#include <solvcon/pilot/visual/RDomainWidget.hpp>
#include <solvcon/pilot/R2DWidget.hpp>
#include <solvcon/pilot/RAction.hpp>
#include <solvcon/pilot/console/RPythonConsoleDockWidget.hpp>
#include <solvcon/pilot/console/RPythonTerminalDockWidget.hpp>

#include <vector>

#include <QApplication>
#include <QMainWindow>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <Qt>

namespace solvcon
{

class RMenuModel;
class RThemeManager;
class RShortcutManager;

/**
 * @brief Singleton that owns the pilot main window and coordinates its
 * widgets, menus, and the active canvas drawing tool.
 *
 * The manager holds the QCoreApplication, the QMainWindow with its menus and
 * MDI area, and the Python console dock. It creates and tracks the 2D and 3D
 * domain widgets, exposes the currently focused canvas, and routes the active
 * drawing tool to it. Access the single instance through instance().
 *
 * @ingroup group_domain
 */
class RManager
    : public QObject
{
    Q_OBJECT

public:

    ~RManager() override;

    RManager & setUp();

    static RManager & instance();

    QCoreApplication * core() { return m_core; }

    RDomainWidget * add3DWidget();
    R2DWidget * add2DWidget();
    RDomainWidget * currentR3DWidget();
    R2DWidget * currentR2DWidget();
    std::vector<RDomainWidget *> list3DWidgets();
    std::vector<R2DWidget *> list2DWidgets();

    /// Name of the active canvas drawing tool.
    std::string drawTool() const { return m_draw_tool; }

    /// Select the active drawing tool and apply it to the focused 2D canvas.
    void setDrawTool(std::string const & name);

    RPythonConsoleDockWidget * pycon() { return m_pycon; }

    RPythonTerminalDockWidget * pyterm() { return m_pyterm; }

    QMainWindow * mainWindow() { return m_mainWindow; }

    QMdiArea * mdiArea() { return m_mdiArea; }

    template <typename... Args>
    QMdiSubWindow * addSubWindow(Args &&... args);

    /// The live model of the menu bar, addressable by path from Python.
    RMenuModel * menuModel() { return m_menuModel; }

    /// The application-wide theme controller, scriptable from Python.
    RThemeManager * themeManager() { return m_themeManager; }

    /// Keyboard-shortcut resolver. C++ uses applyTo; Python uses apply_shortcut.
    RShortcutManager * shortcutManager() { return m_shortcutManager; }

    void quit() { m_core->quit(); }

    /// Only call reset() when the program is to be stopped.
    void reset();

    void toggleConsole();
    void toggleTerminal();

private:

    RManager();

    void setUpConsole();
    void setUpTerminal();
    void setUpCentral();
    void setUpMenu();

    /**
     * Park a hidden QRhiWidget in the main window so the top-level adopts
     * render-to-texture composition once, up front, before any user content.
     */
    void primeRhiComposition();

    /**
     * Push the active draw tool onto the focused 2D canvas, if any. A
     * no-op when the focused subwindow is not a 2D canvas.
     */
    void applyDrawTool();

    void setUpEditMenuItems() const;
    void connectThemeMenuSync() const;
    void setUpCameraControllersMenuItems() const;
    void setUpCameraMovementMenuItems() const;

    /**
     * Undo or redo the most recent shape change on the focused 2D canvas,
     * then repaint it. A no-op when no 2D canvas is focused.
     */
    void undoCanvas() const;
    void redoCanvas() const;

    std::function<void()> createCameraMovementItemHandler(const std::function<void(RDomainWidget *)> &) const;

    static RDomainWidget * domainWidgetOf(QMdiSubWindow * subwin);

    bool m_already_setup = false;

    /// Non-owning handle, possibly borrowed from another runtime (PySide6).
    QCoreApplication * m_core = nullptr;

    /// Owns the application only when the pilot created it, null otherwise.
    std::unique_ptr<QApplication> m_owned_core = nullptr;

    QMainWindow * m_mainWindow = nullptr;

    /// Live menu model owned by the widget tree (parented to the main window).
    RMenuModel * m_menuModel = nullptr;

    /**
     * Application-wide theme controller, parented to the manager so it and its
     * OS color-scheme connection outlive each rebuild of the main window.
     */
    RThemeManager * m_themeManager = nullptr;

    /**
     * Keyboard-shortcut resolver, parented to the manager so it outlives each
     * rebuild of the main window like the theme controller.
     */
    RShortcutManager * m_shortcutManager = nullptr;

    RPythonConsoleDockWidget * m_pycon = nullptr;
    RPythonTerminalDockWidget * m_pyterm = nullptr;
    QMdiArea * m_mdiArea = nullptr;

    /**
     * Hidden QRhiWidget that keeps the main window in render-to-texture
     * composition mode for the whole session. Owned by the widget tree.
     */
    RDomainWidget * m_rhi_primer = nullptr;

    /**
     * Active canvas drawing tool, shared by every 2D canvas. Starts on
     * the default tool (pan navigation).
     */
    std::string m_draw_tool = default_draw_tool_name();
}; /* end class RManager */

template <typename... Args>
QMdiSubWindow * RManager::addSubWindow(Args &&... args)
{
    QMdiSubWindow * subwin = nullptr;
    if (m_mdiArea)
    {
        subwin = m_mdiArea->addSubWindow(std::forward<Args>(args)...);
        subwin->show();
        m_mdiArea->setActiveSubWindow(subwin);
    }
    return subwin;
}

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:

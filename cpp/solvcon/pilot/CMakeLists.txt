# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(SOLVCON_PILOT_PYMODHEADERS
    # common/
    ${CMAKE_CURRENT_SOURCE_DIR}/common/platform.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/render_misc.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RDomainWidget.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RScene.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RCameraController.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RDrawable.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RMeshFrame.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RField.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RColormap.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RScalarField.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RScalarBar.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RBoundary.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RFeatureEdges.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RNormals.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RAxisGizmo.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RMaterial.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/R2DWidget.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DrawTool.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RWorldRenderer2d.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RManager.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RMenuModel.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/theme.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RThemeBackend.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RMacThemeBackend.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RWindowsThemeBackend.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RLinuxThemeBackend.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RThemeManager.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/keymap.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RShortcutManager.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RPythonConsoleDockWidget.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RPythonTerminalDockWidget.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RPythonConsoleHistory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RPythonSyntaxHighlighter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RPythonSyntaxRules.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RAction.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pilot.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wrap_pilot.hpp
    CACHE FILEPATH "" FORCE
)

set(SOLVCON_PILOT_PYMODSOURCES
    # common/
    ${CMAKE_CURRENT_SOURCE_DIR}/common/render_misc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RDomainWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RScene.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RCameraController.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RDrawable.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RMeshFrame.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RField.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RColormap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RScalarField.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RScalarBar.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RBoundary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RFeatureEdges.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RNormals.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RSegments.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RTextOverlay.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RAxisGizmo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RMaterial.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/R2DWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DrawTool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RWorldRenderer2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RMenuModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/theme.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RThemeBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RMacThemeBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RWindowsThemeBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RLinuxThemeBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RThemeManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/keymap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RShortcutManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RPythonConsoleDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RPythonTerminalDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RPythonConsoleHistory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RPythonSyntaxHighlighter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RPythonSyntaxRules.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RAction.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wrap_pilot.cpp
    CACHE FILEPATH "" FORCE
)

set(SOLVCON_PILOT_FILES
    ${SOLVCON_PILOT_PYMODHEADERS}
    ${SOLVCON_PILOT_PYMODSOURCES}
    CACHE FILEPATH "" FORCE)

# GLSL shader sources baked to .qsb by the qsb tool and embedded in the Qt
# resource system. Consumed by qt6_add_shaders() in cpp/solvcon/CMakeLists.txt.
set(SOLVCON_PILOT_SHADERDIR
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders
    CACHE PATH "" FORCE)

set(SOLVCON_PILOT_SHADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/color.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/color.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vcolor.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/scalar.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/scalar.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/texture.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/texture.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/lit.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/lit.frag
    CACHE FILEPATH "" FORCE)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 sts=4:

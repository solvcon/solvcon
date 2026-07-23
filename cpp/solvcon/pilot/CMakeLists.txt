# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

cmake_minimum_required(VERSION 4.0.1)

set(SOLVCON_PILOT_PYMODHEADERS
    # common/
    ${CMAKE_CURRENT_SOURCE_DIR}/common/platform.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common/render_misc.hpp
    # theme/
    ${CMAKE_CURRENT_SOURCE_DIR}/theme/theme.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/theme/RThemeBackend.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/theme/RMacThemeBackend.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/theme/RWindowsThemeBackend.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/theme/RLinuxThemeBackend.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/theme/RThemeManager.hpp
    # console/
    ${CMAKE_CURRENT_SOURCE_DIR}/console/RPythonConsoleDockWidget.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/console/RPythonTerminalDockWidget.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/console/RPythonConsoleHistory.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/console/RPythonSyntaxHighlighter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/console/RPythonSyntaxRules.hpp
    # visual/
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RDomainWidget.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RScene.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RCameraController.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RDrawable.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RMeshFrame.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RField.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RColormap.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RScalarField.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RScalarBar.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RBoundary.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RFeatureEdges.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RNormals.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RAxisGizmo.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RMaterial.hpp
    # canvas/
    ${CMAKE_CURRENT_SOURCE_DIR}/canvas/R2DWidget.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/canvas/DrawTool.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/canvas/RWorldRenderer2d.hpp
    # app/
    ${CMAKE_CURRENT_SOURCE_DIR}/app/RManager.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/app/RMenuModel.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/app/keymap.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/app/RShortcutManager.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/app/RAction.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pilot.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/wrap_pilot.hpp
    CACHE FILEPATH "" FORCE
)

set(SOLVCON_PILOT_PYMODSOURCES
    # common/
    ${CMAKE_CURRENT_SOURCE_DIR}/common/render_misc.cpp
    # theme/
    ${CMAKE_CURRENT_SOURCE_DIR}/theme/theme.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/theme/RThemeBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/theme/RMacThemeBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/theme/RWindowsThemeBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/theme/RLinuxThemeBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/theme/RThemeManager.cpp
    # console/
    ${CMAKE_CURRENT_SOURCE_DIR}/console/RPythonConsoleDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/console/RPythonTerminalDockWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/console/RPythonConsoleHistory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/console/RPythonSyntaxHighlighter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/console/RPythonSyntaxRules.cpp
    # visual/
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RDomainWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RScene.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RCameraController.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RDrawable.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RMeshFrame.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RField.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RColormap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RScalarField.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RScalarBar.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RBoundary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RFeatureEdges.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RNormals.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RSegments.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RAxisGizmo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/visual/RMaterial.cpp
    # canvas/
    ${CMAKE_CURRENT_SOURCE_DIR}/canvas/RTextOverlay.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/canvas/R2DWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/canvas/DrawTool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/canvas/RWorldRenderer2d.cpp
    # app/
    ${CMAKE_CURRENT_SOURCE_DIR}/app/RManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/app/RMenuModel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/app/keymap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/app/RShortcutManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/app/RAction.cpp
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

#!/usr/bin/env python
#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
tube_0 = XMLUnstructuredGridReader(FileName=['../result/tube_0000.vtu', '../result/tube_0200.vtu', '../result/tube_0400.vtu', '../result/tube_0600.vtu'])
tube_0.CellArrayStatus = ['M', 'T', 'ke', 'p', 'rho', 'sch', 'soln[0]', 'soln[1]', 'soln[2]', 'soln[3]', 'soln[4]', 'v']

# get animation scene
animationScene1 = GetAnimationScene()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# Properties modified on tube_0
tube_0.CellArrayStatus = ['p']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [906, 804]

# show data in view
tube_0Display = Show(tube_0, renderView1)
# trace defaults for the display properties.
tube_0Display.ColorArrayName = [None, '']
tube_0Display.GlyphType = 'Arrow'
tube_0Display.ScalarOpacityUnitDistance = 0.06298890955886752

# reset view to fit data
renderView1.ResetCamera()

# set scalar coloring
ColorBy(tube_0Display, ('CELLS', 'p'))

# rescale color and/or opacity maps used to include current data range
tube_0Display.RescaleTransferFunctionToDataRange(True)

# show color bar/color legend
tube_0Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'p'
pLUT = GetColorTransferFunction('p')
pLUT.RGBPoints = [0.039999999999999994, 0.231373, 0.298039, 0.752941, 0.21999999999999997, 0.865003, 0.865003, 0.865003, 0.3999999999999999, 0.705882, 0.0156863, 0.14902]
pLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'p'
pPWF = GetOpacityTransferFunction('p')
pPWF.Points = [0.039999999999999994, 0.0, 0.5, 0.0, 0.3999999999999999, 1.0, 0.5, 0.0]
pPWF.ScalarRangeInitialized = 1

# change representation type
tube_0Display.SetRepresentationType('Wireframe')

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [-1.5246746258669206, -1.1658697596319207, 0.8504380295196987]
renderView1.CameraFocalPoint = [1.7157778085307534e-17, 8.217522903875317e-17, 0.5]
renderView1.CameraViewUp = [-0.5973587129868047, 0.7996275200836257, 0.06130576762148048]
renderView1.CameraParallelScale = 0.5049752469181039

#### uncomment the following to render all views
#RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
animationScene1.PlayMode = 'Real Time'
animationScene1.Play()

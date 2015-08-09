#!~/source/ParaView-3.12.0-Linux-x86_64/bin/pvpython

from paraview.simple import *

restratification_after_oodc_ = XMLPartitionedUnstructuredGridReader( FileName=['/tamay/mensa/fluidity/mli_250m_4/mli_0.pvtu'])

AnimationScene2 = GetAnimationScene()
restratification_after_oodc_.PointArrayStatus = ['Pressure', 'Density', 'Temperature', 'DG_CourantNumber', 'DistanceToTop', 'DistanceToBottom', 'Time', 'Velocity', 'GravityDirection', 'TemperatureDiffusivity', 'Viscosity']
restratification_after_oodc_.CellArrayStatus = []

AnimationScene2.EndTime = 1.0
AnimationScene2.PlayMode = 'Snap To TimeSteps'

RenderView2 = GetRenderView()
restratification_after_oodc_.PointArrayStatus = ['Temperature']

AnimationScene2.AnimationTime = 1.0

RenderView2.ViewTime = 1.0

a1_Temperature_PVLookupTable = GetLookupTableForArray( "Temperature", 1, NanColor=[0.498039, 0.498039, 0.498039], RGBPoints=[2.9569335125755933, 1.0, 0.0, 0.0, 4.768769211443129, 0.0, 0.0, 1.0], VectorMode='Component', ColorSpace='HSV', ScalarRangeInitialized=1.0 )

a1_Temperature_PiecewiseFunction = CreatePiecewiseFunction()

print 'show'

DataRepresentation2 = Show()
DataRepresentation2.EdgeColor = [0.0, 0.0, 0.5019607843137255]
DataRepresentation2.ScalarOpacityFunction = a1_Temperature_PiecewiseFunction
DataRepresentation2.ColorArrayName = 'Temperature'
DataRepresentation2.ScalarOpacityUnitDistance = 606.2976320627819
DataRepresentation2.LookupTable = a1_Temperature_PVLookupTable
DataRepresentation2.Representation = 'Surface'

RenderView2.CameraClippingRange = [134255.03795004194, 139917.94294878037]

print 'render'
Render()

print 'save'
WriteImage("./image.png")



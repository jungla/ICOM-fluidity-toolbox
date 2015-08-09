#!~/source/ParaView-3.12.0-Linux-x86_64/bin/pvpython

from paraview.simple import *

mli_ = XMLPartitionedUnstructuredGridReader( FileName=['/nethome/jmensa/fluidity/examples/mli_500m/mli_0.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_1.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_2.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_3.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_4.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_5.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_6.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_7.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_8.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_9.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_10.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_11.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_12.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_13.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_14.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_15.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_16.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_17.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_18.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_19.pvtu', '/nethome/jmensa/fluidity/examples/mli_500m/mli_20.pvtu'] )

AnimationScene2 = GetAnimationScene()
mli_.PointArrayStatus = ['Pressure', 'Density', 'Temperature', 'DG_CourantNumber', 'DistanceToTop', 'DistanceToBottom', 'Time', 'Velocity', 'GravityDirection', 'TemperatureDiffusivity', 'Viscosity']
mli_.CellArrayStatus = []

AnimationScene2.EndTime = 20.0
AnimationScene2.PlayMode = 'Snap To TimeSteps'

mli_.PointArrayStatus = ['Temperature']

RenderView2 = GetRenderView()
Contour1 = Contour( PointMergeMethod="Uniform Binning" )

SetActiveSource(mli_)
DataRepresentation3 = Show()
DataRepresentation3.ScalarOpacityUnitDistance = 154.8669039335148
DataRepresentation3.Representation = 'Outline'
DataRepresentation3.EdgeColor = [0.0, 0.0, 0.5000076295109483]

RenderView2.CameraClippingRange = [40084.72263717625, 42233.276239125145]

Contour1.PointMergeMethod = "Uniform Binning"
Contour1.ContourBy = ['POINTS', 'Temperature']
Contour1.Isosurfaces = [2.5]

Contour1.Isosurfaces = [2.0, 2.111111111111111, 2.2222222222222223, 2.3333333333333335, 2.4444444444444446, 2.5, 2.5555555555555554, 2.666666666666667, 2.7777777777777777, 2.888888888888889, 3.0]

SetActiveSource(Contour1)
DataRepresentation4 = Show()
DataRepresentation4.EdgeColor = [0.0, 0.0, 0.5000076295109483]

a1_Temperature_PVLookupTable = GetLookupTableForArray( "Temperature", 1, NanColor=[0.498039, 0.498039, 0.498039], RGBPoints=[2.111111111111111, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.0], VectorMode='Component', ColorSpace='HSV', ScalarRangeInitialized=1.0 )

a1_Temperature_PiecewiseFunction = CreatePiecewiseFunction()

DataRepresentation4.ColorArrayName = 'Temperature'
DataRepresentation4.LookupTable = a1_Temperature_PVLookupTable

Render()

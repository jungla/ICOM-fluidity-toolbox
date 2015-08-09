try: paraview.simple
except: from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

ring_2_pvtu = XMLPartitionedUnstructuredGridReader( FileName=['/tamay2/mensa/fluidity/r_5k_A_0/ring_20.pvtu'] )

ring_2_pvtu.PointArrayStatus = ['Velocity_CG']
#ring_2_pvtu.PointArrayStatus = ['Temperature_CG', 'Density_CG', 'Salinity_CG', 'Time', 'Velocity_CG']
ring_2_pvtu.CellArrayStatus = []

RenderView1 = GetRenderView()
DataRepresentation1 = Show()
DataRepresentation1.ScaleFactor = 50000.0
DataRepresentation1.ScalarOpacityUnitDistance = 6504.9353949593215
DataRepresentation1.SelectionPointFieldDataArrayName = 'Velocity_CG'
DataRepresentation1.EdgeColor = [0.0, 0.0, 0.50000762951094835]


RenderView1.CenterOfRotation = [0.0, 0.0, -500.0]

Plane1 = Plane()

RenderView1.CameraPosition = [0.0, 0.0, 1365526.7698091595]
RenderView1.CameraFocalPoint = [0.0, 0.0, -500.0]
RenderView1.CameraClippingRange = [1351371.5021110678, 1387769.6713562971]
RenderView1.CameraParallelScale = 353553.74414648756

Delete(Plane1)
SetActiveSource(ring_2_pvtu)
Transform1 = Transform( Transform="Transform" )

Transform1.Transform = "Transform"

active_objects.source.SMProxy.InvokeEvent('UserEvent', 'ShowWidget')


DataRepresentation2 = Show()
DataRepresentation2.ScaleFactor = 50000.0
DataRepresentation2.ScalarOpacityUnitDistance = 7066.1614643302582
DataRepresentation2.SelectionPointFieldDataArrayName = 'Velocity_CG'
DataRepresentation2.EdgeColor = [0.0, 0.0, 0.50000762951094835]

DataRepresentation1.Visibility = 0

Transform1.Transform.Scale = [1.0, 1.0, 300.0]

Plane2 = Plane()

RenderView1.CameraClippingRange = [1201871.5021110678, 1914009.6713562971]

DataRepresentation3 = Show()
DataRepresentation3.ScaleFactor = 30000.0
DataRepresentation3.SelectionPointFieldDataArrayName = 'Normals'
DataRepresentation3.EdgeColor = [0.0, 0.0, 0.50000762951094835]

Plane2.Origin = [-100000.0, 0.0, 0.0]
Plane2.Point1 = [100000.0, 0.0, 0.0]
Plane2.Point2 = [-100000.0, 0.0, -300000.0]

RenderView1.CameraViewUp = [-0.049004711592166503, 0.11616161404440234, 0.99202067401056115]
RenderView1.CameraPosition = [-7261.8989273561956, -1356779.0959423478, 158014.62226899868]
RenderView1.CameraClippingRange = [1333845.0671552576, 1449059.4157374003]
RenderView1.CameraFocalPoint = [8.6775824959637285e-16, -8.7932835959099119e-14, -499.99999999999994]

DataRepresentation2.Visibility = 0

Plane2.YResolution = 10
Plane2.XResolution = 3

DataRepresentation3.Representation = 'Wireframe'

SetActiveSource(Transform1)
StreamTracerWithCustomSource1 = StreamTracerWithCustomSource(SeedSource=Plane2 , InitialStepLength = 0.8, MaximumSteps=int(1e8), MaximumStreamlineLength=int(1e8))

StreamTracerWithCustomSource1.Vectors = ['POINTS', 'Velocity_CG']

DataRepresentation4 = Show()
DataRepresentation4.SelectionCellFieldDataArrayName = 'ReasonForTermination'
DataRepresentation4.ScaleFactor = 30000.062054383758
DataRepresentation4.SelectionPointFieldDataArrayName = 'AngularVelocity'
DataRepresentation4.EdgeColor = [0.0, 0.0, 0.50000762951094835]

StreamTracerWithCustomSource1.IntegrationDirection = 'FORWARD'
StreamTracerWithCustomSource1.IntegratorType = 'Runge-Kutta 2'

RenderView1.CameraClippingRange = [1135783.5527039804, 1697299.1273428979]
RenderView1.WriteImage('r_5k_0.png', 'vtkPNGWriter')
Render()

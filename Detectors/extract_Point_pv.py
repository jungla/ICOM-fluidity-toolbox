try: paraview.simple
except: from paraview.simple import *

#mli_0_pvtu = XMLPartitionedUnstructuredGridReader( FileName=['/Users/jeanmensa/Dropbox/temp/m_50_small_2/mli_0.pvtu'] )

mli_0_pvtu = XMLPartitionedUnstructuredGridReader( FileName=['./mli_0.pvtu'] )

mli_0_pvtu.PointArrayStatus = ['Temperature_CG', 'Density_CG', 'Time', 'Velocity_CG', 'TemperatureDiffusivity', 'Viscosity']
mli_0_pvtu.CellArrayStatus = []

ProbeLocation2 = ProbeLocation( ProbeType="Fixed Radius Point Source" )

ProbeLocation2.ProbeType.Center = [500.0, 500.0, -15.0]
ProbeLocation2.ProbeType = "Fixed Radius Point Source"

t = ProbeLocation2.PointData['Tracer_1_CG'].GetRange()
print t

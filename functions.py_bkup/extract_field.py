#!/usr/bin/python
import sys
import libvtkCommon as vtktools

sys.path.append('/share/apps/paraview/3.14/lib/paraview-3.14/')

data = vtktools.vtu('../mli_4.pvtu')
uvw = data.GetVectorField('Velocity')
u = uvw[:,0]

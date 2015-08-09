import os, sys
import fio
import vtktools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import myfun
import pyvtk

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days
label = sys.argv[1]
dayi  = int(sys.argv[2])
dayf  = int(sys.argv[3])
days  = int(sys.argv[4])

#label = 'm_250_8c_str'
#dayi  = 150
#dayf  = 200
#days  = 5

path = '/tamay/mensa/fluidity/'+label+'/'

try: os.stat('./output/'+label)
except OSError: os.mkdir('./output/'+label)

xstep = 150
ystep = 150
zstep = -1

lat = 15000
lon = 7500
depth = -30

Xlist = np.arange(0.0,lon+xstep,xstep)# x co-ordinates of the desired array shape
Ylist = np.arange(0.0,lat+ystep,ystep)# y co-ordinates of the desired array shape
Zlist = np.arange(0.0,depth+zstep,zstep)# y co-ordinates of the desired array shape
[X,Y,Z] = myfun.meshgrid2(Xlist,Ylist,Zlist)
Y = np.reshape(Y,(np.size(Y),))
X = np.reshape(X,(np.size(X),))
Z = np.reshape(Z,(np.size(Z),))

Tref = [0.2, 0.3]

for time in range(dayi,dayf,days):
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file0 = 'mli_' + str(time) + '.pvtu'
 filepath = path+file0
 file1 = label+'_' + tlabel
 fileout  = path + file1
 #
 print 'opening: ', filepath
 #
 data = vtktools.vtu(filepath)
 #
 print 'fields: ', data.GetFieldNames()
 #
 Ts = np.reshape(data.ProbeData(vtktools.arr(zip(X,Y,Z)), 'Temperature'),[len(Zlist),len(Ylist),len(Xlist)]) 
 triangles = myfun.pts2trs(len(Xlist),len(Ylist))
 #
 #
 for t in range(len(Tref)):
  points = np.zeros([len(Ylist)*len(Xlist),3])
  mld = np.zeros(len(Ylist)*len(Xlist))
  id = 0
  #
  for j in xrange(len(Ylist)):
   for i in xrange(len(Xlist)):
    for k in xrange(len(Zlist)):
     if (Ts[k,j,i] - Ts[0,j,i] <= -Tref[t]):
      points[id,:] = Xlist[i],Ylist[j],Zlist[k]
      mld[id] = Zlist[k]
      id = id + 1
      break
   
  vtk = pyvtk.VtkData(pyvtk.UnstructuredGrid(points,triangle=triangles),pyvtk.PointData(pyvtk.Scalars(mld,'MLD')),'Unstructured Grid Example')
  print './output/'+label+'/ML_'+myfun.digit(Tref[t],3)+'_'+file1
  vtk.tofile('./output/'+label+'/ML_'+myfun.digit(Tref[t],3)+'_'+file1)

 del data

import os, sys
import gc
import fio, myfun
import vtktools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.mlab import griddata

gc.enable()

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

label = sys.argv[1]
basename = sys.argv[2]
dayi  = int(sys.argv[3])
dayf  = int(sys.argv[4])
days  = int(sys.argv[5])

path = '/tamay2/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

file0 = basename+'_' + str(1) + '.pvtu'
filepath = path+file0
#
data = vtktools.vtu(filepath)
coords = data.GetLocations()
del data
depths = sorted(list(set(coords[:,2])))

X = np.linspace(np.min(coords[:,0]),np.max(coords[:,0]),250)
Y = np.linspace(np.min(coords[:,1]),np.max(coords[:,1]),250)
f = np.zeros([250, 250])
f = f + 0.00073
v = np.linspace(-1, 2, 50, endpoint=True)
vl = np.linspace(-1, 2, 5, endpoint=True)

for time in range(dayi,dayf,days):
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file0 = basename+'_' + str(time) + '.pvtu'
 filepath = path+file0
 file1 = label+'_' + tlabel
 fileout  = path + file1
 #
 print 'opening: ', filepath
 #
 #
 data = vtktools.vtu(filepath)
 print 'fields: ', data.GetFieldNames()
 print 'extract V'
 ddc = data.GetDerivative('Velocity_CG')
 Zc = ddc[:,2]-ddc[:,4]
 data.AddField('Vorticity', Zc)
 data.CellDataToPointData()
 Zp = data.GetScalarField('Vorticity')
 del data
# V = data.GetVectorField('Velocity_CG')
 for d in range(len(depths)):
  Zl = Zp[coords[:,2] == depths[d]]
  Xl = coords[coords[:,2] == depths[d],0]
  Yl = coords[coords[:,2] == depths[d],1]
  Zi = griddata(Xl,Yl,Zl,X,Y,interp='nn') 
  #
  fig = plt.figure()
#  plt.contourf(Zi/f,v,extend='both',cmap=plt.cm.PiYG)
#  plt.contourf(Zi/f,v,extend='both',cmap=plt.cm.PiYG)
  plt.contourf(Zi/f,12,extend='both',cmap=plt.cm.PiYG)
#  plt.colorbar(ticks=vl)
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')
  # plt.xlim([-0.002, 0.002])
  # plt.xticks([-0.002, 0.002])
  plt.savefig('./plot/'+label+'/Z_'+file1+'_'+str(abs(depths[d]))+'.eps',bbox_inches='tight')
  plt.close()

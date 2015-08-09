from memory_profiler import memory_usage
from matplotlib.colors import LinearSegmentedColormap
import os, sys
import gc
import fio, myfun
import vtktools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
gc.enable()

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

label = 'r_3k_B_1F0'
basename = 'ring'
dayi = 100
dayf = 101
days = 1

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
print 'reading',filepath
#
data = vtktools.vtu(filepath)
coords = data.GetLocations()
del data
gc.collect()
print memory_usage(-1, interval=.2, timeout=.2)

depths = sorted(list(set(coords[:,2])))

f = 0.00073
g = 9.81
r0 = 1027.0

xn = 150
tn = 24

Xr = np.linspace(0.0,np.max(coords[:,0]),xn)
del coords
gc.collect()
Tr = np.linspace(0,2.0*np.pi,tn)

pts = []

for t in Tr:
 for r in Xr:
  for z in depths:
   pts.append([r*np.cos(t),r*np.sin(t),z])

pts = np.asarray(pts)

print 'looping'  
print memory_usage(-1, interval=.2, timeout=.2)

d = 4

for time in range(dayi,dayf,days):
 PVt = np.zeros((d,len(depths),xn))
 Rhot = np.zeros((d,len(depths),xn))
 tlabel = str(time)
 file1 = label+'_' + tlabel
 print 'day',tlabel
 #
 for stime in range(d):
  stlabel = str(time+stime)
  while len(stlabel) < 3: stlabel = '0'+stlabel
  file0 = basename+'_' + str(time+stime) + '.pvtu'
  filepath = path+file0
  #
  print 'opening: ', filepath
  #
  #
  data = vtktools.vtu(filepath)
  print 'fields: ', data.GetFieldNames()
  print 'extract V'
  ddc = data.GetDerivative('Velocity_CG')
  Zc = ddc[:,1]-ddc[:,3]
  del ddc
  gc.collect()
  print memory_usage(-1, interval=.2, timeout=.2)
  ddr = data.GetDerivative('Density_CG')
  gR = ddr[:,2]
  del ddr
  gc.collect()
  print memory_usage(-1, interval=.2, timeout=.2)
  pZ = (Zc+f)*(gR)*-g/r0
  del Zc, gR
  gc.collect()
  print memory_usage(-1, interval=.2, timeout=.2)
  data.AddField('PVorticity', pZ)
  del pZ
  gc.collect()
  print memory_usage(-1, interval=.2, timeout=.2)
 # data.AddField('Vorticity', Zc)
 # data.AddField('gradRho', gR)
  data.CellDataToPointData()
 # Z = data.ProbeData(pts,'Vorticity')
  PV = data.ProbeData(pts,'PVorticity')
  Rho = data.ProbeData(pts,'Density_CG')
  del data
  gc.collect()
  print memory_usage(-1, interval=.2, timeout=.2)
  # gRho = data.ProbeData(pts,'gradRho')
   #
   # Azimuthal  average
  PVr = np.reshape(PV,(tn,xn,len(depths)))
  # Zr = np.reshape(Z,(tn,xn,len(depths)))
  Rhor = np.reshape(Rho,(tn,xn,len(depths)))
  # gRhor = np.reshape(gRho,(tn,xn,len(depths)))
  # gRhot = np.zeros((len(depths),xn))
  # Zt = np.zeros((len(depths),xn))
  for r in range(len(Xr)):
   for z in range(len(depths)):
    PVt[stime,z,r] = np.mean(PVr[:,r,z])
   #   Zt[z,r] = np.mean(Zr[:,r,z])
    Rhot[stime,z,r] = np.mean(Rhor[:,r,z])
   #   gRhot[z,r] = np.mean(gRhor[:,r,z])
     #
  gc.collect()
 
 cdict1 = {'red':   ((0.0, 0.0, 0.0),
                   (0.05, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.05, 1.0, 1.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 1.0),
                   (0.05, 1.0, 1.0),
                   (1.0, 0.0, 0.0))
        }

 blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)

 v = np.linspace(-0.1e-8, 2e-8, 50, endpoint=True)
 vl = np.linspace(-0.1e-8, 2e-8, 5, endpoint=True)
 fig = plt.figure()
 #plt.contour(Xr,depths,np.mean(PVt,0),colors='k',levels=v)
 plt.contourf(Xr/1000,depths,np.mean(PVt,0),v,extend='both',cmap=blue_red1) # plt.cm.PiYG)
 plt.colorbar(ticks=vl)
 # plt.contour(Xr,depths,np.mean(Rhot,0),colors='k')
 plt.xlabel('radius [Km]')
 plt.ylabel('depth [m]')
 plt.savefig('./plot/'+label+'/PV_'+file1+'.eps',bbox_inches='tight')
 plt.close()
 del fig
 gc.collect()
 print memory_usage(-1, interval=.2, timeout=.2)
 print 'saving', './plot/'+label+'/PV_'+file1+'.eps'

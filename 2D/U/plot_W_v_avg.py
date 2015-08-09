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
 Wt = np.zeros((d,len(depths),xn))
 Rt = np.zeros((d,len(depths),xn))
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
 # Z = data.ProbeData(pts,'Vorticity')
  V = data.ProbeData(pts,'Velocity_CG')
  Rho = data.ProbeData(pts,'Density_CG')
  del data
  gc.collect()
  print memory_usage(-1, interval=.2, timeout=.2)
   # Azimuthal  average
  Vr = np.reshape(V,(tn,xn,len(depths),3))
  Rr = np.reshape(Rho,(tn,xn,len(depths)))
  for r in range(len(Xr)):
   for z in range(len(depths)):
    Wt[stime,z,r] = np.mean(Vr[:,r,z,2])
    Rt[stime,z,r] = np.mean(Rr[:,r,z])
     #
  gc.collect()

 v = np.linspace(-1e-6, 1e-6, 50, endpoint=True)
 vl = np.linspace(-1e-6, 1e-6, 5, endpoint=True)
 fig = plt.figure()
 #plt.contour(Xr,depths,np.mean(PVt,0),colors='k',levels=v)
 plt.contourf(Xr/1000,depths,np.mean(Wt,0),v,extend='both',cmap=plt.cm.PiYG)
 plt.colorbar(ticks=vl)
 plt.contour(Xr/1000,depths,np.mean(Rt,0),20,colors='k',linewidth='0.7')
 # plt.contour(Xr,depths,np.mean(Rhot,0),colors='k')
 plt.xlabel('radius [Km]')
 plt.ylabel('depth [m]')
 plt.savefig('./plot/'+label+'/W_'+file1+'_v_avg.eps',bbox_inches='tight')
 plt.close()
 del fig
 gc.collect()
 print memory_usage(-1, interval=.2, timeout=.2)
 print 'saving', './plot/'+label+'/W_'+file1+'_v_avg.eps'

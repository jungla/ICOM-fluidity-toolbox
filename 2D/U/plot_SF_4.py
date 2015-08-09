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
from fipy import *
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

file0 = basename+'_' + str(100) + '.pvtu'
filepath = path+file0
print 'reading',filepath
#
data = vtktools.vtu(filepath)
#data.Crop(-3000,3000,-3000,3000,-900,0)
coords = data.GetLocations()
del data
gc.collect()
print memory_usage(-1, interval=.2, timeout=.2)

depths = sorted(list(set(coords[:,2])))

print depths

f = 0.0001
g = 9.81
r0 = 1027.0

xn = 150
zn = 200
tn = 48

Xr = np.linspace(0.0,np.max(coords[:,0]),xn)
Xrt = np.linspace(0,2*np.max(coords[:,0]),xn*2)
Zr = np.linspace(0.0,np.max(coords[:,2]),zn) 

del coords
gc.collect()
Tr = np.linspace(0,2.0*np.pi,tn)

pts = []
ptsc = []

for t in Tr:
 for r in Xr:
  for z in depths:
   pts.append([r*np.cos(t),r*np.sin(t),z])
   ptsc.append([r,t,z])

pts = np.asarray(pts)
ptsc = np.asarray(ptsc)

print 'looping'  
print memory_usage(-1, interval=.2, timeout=.2)

d = 4

for time in range(dayi,dayf,days):
 mVrt = np.zeros((d,xn,len(depths)))
 mWt = np.zeros((d,xn,len(depths)))
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
  Vt = data.ProbeData(pts,'Velocity_CG')
  Vr = Vt[:,1]*np.cos(ptsc[:,1])+Vt[:,0]*np.sin(ptsc[:,1])
  rVr = np.reshape(Vr,(tn,xn,len(depths)))
 # ptsr = np.reshape(pts,(tn,xn,len(depths),3))
  rW = np.reshape(Vt[:,2],(tn,xn,len(depths)))
  del data
  gc.collect()
  print memory_usage(-1, interval=.2, timeout=.2)
  #
  mVrt[stime,:,:]=np.mean(rVr,0)
  mWt[stime,:,:]=np.mean(rW,0)

  #depths = sorted(depths,reverse=True)
 mW = np.transpose(np.mean(mWt,0))
 mVr= np.transpose(np.mean(mVrt,0))

 mW = np.hstack((np.fliplr(mW),mW))
 mVr = np.hstack((np.fliplr(mVr),mVr))

 dWdR = np.zeros(mW.shape)
 dVdZ = np.zeros(mW.shape)

 depths = np.asarray(depths)

 for k in range(len(depths)):
  dWdR[k,:] = np.gradient(mW[k,:])/np.gradient(Xrt)
 
 for i in range(len(Xrt)):
  dVdZ[:,i] = np.gradient(mVr[:,i])/np.gradient(depths)  

 Z = dVdZ - dWdR

 nx = len(depths)
 dx = 0.1
 ny = len(Xrt)
 dy = 0.1
 
 mesh = Grid2D(dx=dx,dy=dy,nx=nx,ny=ny)
 
 vorticity = CellVariable(mesh=mesh, name='Z')
 vorticity.setValue(np.reshape(Z,nx*ny,1))
 
 potential = CellVariable(mesh=mesh, name='potential', value=0.)
 potential.equation = (DiffusionTerm(coeff = 1.) - vorticity == 0.)
  
 bcs = (
     FixedValue(value=0,faces=mesh.getFacesLeft()  ),
     FixedValue(value=0,faces=mesh.getFacesRight() ),
     FixedValue(value=0,faces=mesh.getFacesTop() ),
     FixedValue(value=0,faces=mesh.getFacesBottom() ),
 )
 
 potential.equation.solve(var=potential, boundaryConditions=bcs)
 
 
 Phi = np.reshape(potential.value,(len(Xrt),len(depths)))
  
 # plot SF
 v = np.linspace(-2e-5, 2e-4, 50, endpoint=True)
 vl = np.linspace(-2e-5, 2e-4, 5, endpoint=True)
 fig = plt.figure()
 #plt.streamplot(Xr,Zr,mWri,mVri)
 #plt.quiver(Xr,Zr,mWri,mVri)
 #plt.colorbar()
 #plt.colorbar(ticks=vl)
 # plt.contour(Xr,depths,np.mean(Rhot,0),colors='k')
 #plt.contour(np.fliplr(np.transpose(Phi)),colors='k')
 #plt.contourf(Phi,50,extend='both',cmap=plt.cm.PiYG)
 plt.contourf(Xr/1000,depths,np.transpose(Phi[len(Xr):,:]),50,extend='both',cmap=plt.cm.PiYG)
 #plt.contour(Xr/1000,depths,np.transpose(Phi),10,extend='both',colors='k')
 plt.colorbar(format='%.3e')
 plt.xlabel('radius [Km]')
 plt.ylabel('depth [m]')
 plt.savefig('./plot/'+label+'/Phi_'+file1+'.eps',bbox_inches='tight')
 plt.close()
 del fig
   
 v = np.linspace(-0.0005, 0.0005, 50, endpoint=True)
 vl = np.linspace(-0.0005, 0.0005, 5, endpoint=True)
 fig = plt.figure()
 #plt.streamplot(Xr,Zr,mWri,mVri)
 #plt.quiver(Xr,Zr,mWri,mVri)
 #plt.colorbar()
 # plt.contour(Xr,depths,np.mean(Rhot,0),colors='k')
 #plt.contour(np.fliplr(np.transpose(Phi)),colors='k')
 plt.contourf(Xr/1000,depths,Z[:,len(Xr):],v,extend='both',cmap=plt.cm.PiYG)
 plt.colorbar(format='%.3e')
 plt.xlabel('radius [Km]')
 plt.ylabel('depth [m]')
 plt.savefig('./plot/'+label+'/Z_'+file1+'.eps',bbox_inches='tight')
 plt.close()


 # plot Vr
 v = np.linspace(-0.1e-8, 2e-8, 50, endpoint=True)
 vl = np.linspace(-0.1e-8, 2e-8, 5, endpoint=True)
 fig = plt.figure()
 plt.contourf(Xr/1000,depths,mVr[:,len(Xr):],50,extend='both',cmap=plt.cm.PiYG)
 plt.colorbar(format='%.3e')
 #plt.colorbar(ticks=vl)
 # plt.contour(Xr,depths,np.mean(Rhot,0),colors='k')
 plt.xlabel('radius [Km]')
 plt.ylabel('depth [m]')
 plt.savefig('./plot/'+label+'/SF_Vr_'+file1+'.eps',bbox_inches='tight')
 plt.close()
 del fig
 
 #plot Wr
 v = np.linspace(-0.1e-8, 2e-8, 50, endpoint=True)
 vl = np.linspace(-0.1e-8, 2e-8, 5, endpoint=True)
 fig = plt.figure()
 plt.contourf(Xr/1000,depths,mW[:,len(Xr):],50,extend='both',cmap=plt.cm.PiYG)
 plt.colorbar(format='%.3e')
 #plt.colorbar(ticks=vl)
 #plt.contour(Xr,depths,np.mean(Rhot,0),colors='k')
 plt.xlabel('radius [Km]')
 plt.ylabel('depth [m]')
 plt.savefig('./plot/'+label+'/SF_W_'+file1+'.eps',bbox_inches='tight')
 plt.close()
 del fig
  
# plot stream function
 gc.collect()
 print memory_usage(-1, interval=.2, timeout=.2)


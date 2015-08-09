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
zn = 200
tn = 24

Xr = np.linspace(0.0,np.max(coords[:,0]),xn)
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

d = 2

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
  ptsr = np.reshape(pts,(tn,xn,len(depths),3))
  rW = np.reshape(Vt[:,2],(tn,xn,len(depths)))
#  Rho = data.ProbeData(pts,'Density_CG')
  del data
  gc.collect()
  print memory_usage(-1, interval=.2, timeout=.2)
  #
  mVrt[stime,:,:]=np.mean(rVr,0)
  mWt[stime,:,:]=np.mean(rW,0)
#  gc.collect()

  # integration trapezodial
  # 

  #depths = sorted(depths,reverse=True)
  
  # easy BCs
  
 Phi = np.zeros((xn,len(depths))) 
 PhiW = np.zeros((xn,len(depths))) 
 PhiV = np.zeros((xn,len(depths))) 
 
 mW = np.mean(mWt,0) 
 mVr= np.mean(mVrt,0)
 
 for k in range(1,len(depths)):
  for i in range(0,xn-1):
   PhiV[i,k] =  PhiV[i,k-1] + (depths[k]-depths[k-1])*(mVr[i,k-1]*0.5+mVr[i,k]*0.5) # trapz
 #  Phi[i,k] =  Phi[i,k-1] + (depths[k]-depths[k-1])*(mVr[i,k-1]-mVr[i,k]) # euler
   PhiW[i,k] =  -Xr[i]/Xr[i+1]*(mW[i,k]*(Xr[i+1]-Xr[i]) - PhiW[i+1,k]) # euler
   Phi[i,k] = Phi[i,k-1] + (depths[k]-depths[k-1])*(mVr[i,k-1]-mVr[i,k]) + -Xr[i]/Xr[i+1]*(mW[i,k]*(Xr[i+1]-Xr[i]) - Phi[i+1,k])
  
 # plot SF
 v = np.linspace(-0.18, 0.18, 50, endpoint=True)
 vl = np.linspace(-0.18, 0.18, 5, endpoint=True)
 fig = plt.figure()
 #plt.streamplot(Xr,Zr,mWri,mVri)
 #plt.quiver(Xr,Zr,mWri,mVri)
 #plt.colorbar()
 #plt.colorbar(ticks=vl)
 # plt.contour(Xr,depths,np.mean(Rhot,0),colors='k')
 #plt.contour(np.fliplr(np.transpose(Phi)),colors='k')
 plt.contourf(Xr/1000,depths,np.transpose(Phi),v,extend='both',cmap=plt.cm.PiYG)
 plt.colorbar(ticks=vl)
 plt.xlabel('radius [Km]')
 plt.ylabel('depth [m]')
 plt.savefig('./plot/'+label+'/SF_'+file1+'.eps',bbox_inches='tight')
 plt.close()
 del fig
  
 # plot Vr
 v = np.linspace(-0.1e-8, 2e-8, 50, endpoint=True)
 vl = np.linspace(-0.1e-8, 2e-8, 5, endpoint=True)
 fig = plt.figure()
 plt.contourf(Xr/1000,depths,np.transpose(PhiV),50,extend='both',cmap=plt.cm.PiYG)
 plt.colorbar()
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
 plt.contourf(Xr/1000,depths,np.transpose(PhiW),50,extend='both',cmap=plt.cm.PiYG)
 plt.colorbar()
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


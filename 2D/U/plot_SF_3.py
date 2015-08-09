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
tn = 96

Xr = np.linspace(0.0,150000,xn)

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
ptsr = np.reshape(pts,(tn,xn,len(depths),3))
ptss = ptsr[0,:,:,:]
lx=np.reshape(ptss[:,:,0],(np.size(ptss[:,:,0]),))
lz=np.reshape(ptss[:,:,2],(np.size(ptss[:,:,2]),))

print 'looping'  
print memory_usage(-1, interval=.2, timeout=.2)

d = 1
epsilon = 0.00001

for time in range(dayi,dayf,days):
 Rhot = np.zeros((d,xn,len(depths)))
 Phit = np.zeros((d,xn,len(depths)))
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
#  Vr = Vt[:,1]*np.cos(ptsc[:,1])+Vt[:,0]*np.sin(ptsc[:,1])
#  rVr = np.reshape(Vr,(tn,xn,len(depths)))
  Rho = data.ProbeData(pts,'Density_CG')
  W = np.reshape(Vt[:,2],(tn,xn,len(depths)))
  U =  np.reshape(Vt[:,1]*np.cos(ptsc[:,1])+Vt[:,0]*np.sin(ptsc[:,1]),(tn,xn,len(depths))) 
  R = np.reshape(Rho[:],(tn,xn,len(depths)))
  Rhot[stime,:,:] = np.mean(R,0)
  B = -g*R/r0
  del data
  gc.collect()
  print memory_usage(-1, interval=.2, timeout=.2)
  #
  mB=np.mean(B,0) # azimuthal means
  mW=np.mean(W,0)
  mU=np.mean(U,0)
  #gc.collect()
  # 
  aB = np.zeros(B.shape)
  aW = np.zeros(W.shape)
  aU = np.zeros(U.shape)
  #
  for t in range(tn):
   aB[t,:,:] = B[t,:,:]-mB
   aW[t,:,:] = W[t,:,:]-mW
   aU[t,:,:] = U[t,:,:]-mU
  #
  mBW = np.mean(aB*aW,0) # mean of the anomalies
  mBU = np.mean(aB*aU,0) # mean of the anomalies
  
  # Bx is with the total buoyancy and THEN averaged in t
  
  Bx = np.zeros(B.shape)
  for t in range(tn):
   for z in range(len(depths)):
    Bx[t,:len(Xr)-1,z] = np.diff(B[t,:,z])/np.diff(Xr)
   
  for t in range(tn):
   Bx[t,len(Xr)-1,:] = (B[t,len(Xr)-1,:]-B[t,len(Xr)-2,:])/(Xr[len(Xr)-1]-Xr[len(Xr)-2])
 
  Bz = np.zeros(B.shape)
  for t in range(tn):
   for x in range(len(Xr)):
    Bz[t,x,:len(depths)-1] = np.diff(B[t,x,:])/np.diff(depths)

  for t in range(tn):
   Bz[t,:,len(depths)-1] = (B[t,:,len(depths)-1]-B[t,:,len(depths)-2])/(depths[len(depths)-1]-depths[len(depths)-2])

 
  mBx = np.mean(Bx,0)
  mBz = np.mean(Bz,0)
 
  Phit[stime,:,:] = epsilon*(epsilon*mBU/mBz-1.0/epsilon*mBW*mBx)/(mBx**2 + epsilon**2*mBz**2)

 Phi = np.mean(Phit,0)
#plt.rcParams['contour.negative_linestyle'] = 'solid'
#plt.rcParams['contour.positive_linestyle'] = 'dashed'

 v = np.linspace(np.percentile(Phi,5),np.percentile(Phi,95) , 20, endpoint=True)
 vl = np.linspace(np.percentile(Phi,5),np.percentile(Phi,95) , 5, endpoint=True)
 fig = plt.figure()

 plt.contourf(Xr/1000,depths,np.transpose(Phi),v,extend='both',cm=plt.cm.hot)
 #plt.colorbar()
 plt.colorbar(ticks=vl)
 plt.contour(Xr/1000,depths,np.transpose(np.mean(Rhot,0)),colors='k')
 plt.title('Phi')
 plt.xlabel('radius [Km]')
 plt.ylabel('depth [m]')
 plt.savefig('./plot/'+label+'/SF_'+file1+'.eps',bbox_inches='tight')
 plt.close()
 del fig

# plot stream function

 gc.collect()
 print memory_usage(-1, interval=.2, timeout=.2)

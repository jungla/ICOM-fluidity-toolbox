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

file0 = basename+'_' + str(1) + '.pvtu'
filepath = path+file0
print 'reading',filepath
#
data = vtktools.vtu(filepath)
coords = data.GetLocations()
del data
gc.collect()
print memory_usage(-1, interval=.2, timeout=.2)

depths = np.asarray(sorted(list(set(coords[:,2]))))

f = 0.00073
g = 9.81
r0 = 1027.0

xn = 150
zn = 200
tn = 48

Xr = np.linspace(10000.0,np.max(coords[:,0]),xn)
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

day = 4

for time in range(dayi,dayf,days):
 mVrt = np.zeros((day,xn,len(depths)))
 mWt = np.zeros((day,xn,len(depths)))
 mBt = np.zeros((day,xn,len(depths)))
 rVr = np.zeros((day,tn,xn,len(depths)))
 rW = np.zeros((day,tn,xn,len(depths)))
 rB = np.zeros((day,tn,xn,len(depths)))
 aW = np.zeros(rW.shape)
 aVr = np.zeros(rW.shape)
 aB = np.zeros(rW.shape)
 tlabel = str(time)
 file1 = label+'_' + tlabel
 print 'day',tlabel
 #
 for stime in range(day-1,-1,-1):
  stlabel = str(time-stime)
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
  Bt = -g/r0*data.ProbeData(pts,'Density_CG')
  Vr = Vt[:,1]*np.cos(ptsc[:,1])+Vt[:,0]*np.sin(ptsc[:,1])
  rVr[stime,:,:,:] = np.reshape(Vr,(tn,xn,len(depths)))
  rB[stime,:,:,:]  = np.reshape(Bt,(tn,xn,len(depths)))
  rW[stime,:,:,:] = np.reshape(Vt[:,2],(tn,xn,len(depths)))
  del data
  gc.collect()
  print memory_usage(-1, interval=.2, timeout=.2)
  #
  # mean along sections
  mVrt[stime,:,:]=np.mean(rVr[stime,:,:,:],0)
  mWt[stime,:,:]=np.mean(rW[stime,:,:,:],0)
  mBt[stime,:,:]=np.mean(rB[stime,:,:,:],0)
  #
  # plot snapshot Vr,W


 #  gc.collect()
 # mean in time
 mW = np.mean(mWt,0)
 mVr= np.mean(mVrt,0)
 mB = np.mean(mBt,0)

 for t in range(tn):
  for stime in range(day):
   aW[stime,t,:,:] = rW[stime,t,:,:] - mW
   aVr[stime,t,:,:] = rVr[stime,t,:,:] - mVr
   aB[stime,t,:,:] = rB[stime,t,:,:] - mB

 maWaB = np.mean(np.mean(aW*aB,0),0)
 maVraB = np.mean(np.mean(aVr*aB,0),0)

 # FIRST TERM
 
 dWBdZ = np.zeros(mW.shape)
 for i in range(len(Xr)):
  dWBdZ[i,:] = np.gradient(mW[i,:]*mB[i,:])/np.gradient(depths)  

 dVBdR = np.zeros(mW.shape)
 for k in range(len(depths)):
  dVBdR[:,k] = 1/Xr*np.gradient(Xr*mVr[:,k]*mB[:,k])/np.gradient(Xr)

 F = -dWBdZ-dVBdR
 
 # plot
# v = np.linspace(np.percentile(F,5), np.percentile(F,95), 50, endpoint=True)
# v = np.linspace(-np.max(abs(np.percentile(F,5)),abs(np.percentile(F,95))),np.max(abs(np.percentile(F,5)),abs(np.percentile(F,95))), 50, endpoint=True)
 v = np.linspace(-1.5e-6,1.5e-6, 50, endpoint=True)
 vl = np.linspace(-0.18, 0.18, 5, endpoint=True)
 fig = plt.figure()
# plt.contour(Xr/1000,depths,np.transpose(F[:,:]),v,colors='k')
 plt.contourf(Xr/1000,depths,np.transpose(F[:,:]),50,extend='both',cmap=plt.cm.PiYG)
 plt.colorbar(format='%.3e')
 plt.xlabel('radius [Km]')
 plt.ylabel('depth [m]')
 plt.title('A')
 plt.savefig('./plot/'+label+'/flux_1_'+file1+'.eps',bbox_inches='tight')
 print './plot/'+label+'/flux_1_'+file1+'.eps'
 plt.close()
 del fig
 
 # SECOND TERM

 daWaBdZ = np.zeros(mW.shape)
 for i in range(len(Xr)):
  daWaBdZ[i,:] = np.gradient(maWaB[i,:])/np.gradient(depths)


 S = -daWaBdZ

 # plot
# v = np.linspace(np.percentile(S,5), np.percentile(S,95), 50, endpoint=True)
# v = np.linspace(-np.max(abs(np.percentile(S,5)),abs(np.percentile(S,95))),np.max(abs(np.percentile(S,5)),abs(np.percentile(S,95))), 50, endpoint=True)
 v = np.linspace(-3e-11,3e-11, 50, endpoint=True)
 vl = np.linspace(-0.18, 0.18, 5, endpoint=True)
 fig = plt.figure()
# plt.contour(Xr/1000,depths,np.transpose(S[:,:]),v,colors='k')
 plt.contourf(Xr/1000,depths,np.transpose(S[:,:]),50,extend='both',cmap=plt.cm.PiYG)
 plt.colorbar(format='%.3e')
 plt.xlabel('radius [Km]')
 plt.ylabel('depth [m]')
 plt.title('B')
 plt.savefig('./plot/'+label+'/flux_2_'+file1+'.eps',bbox_inches='tight')
 print './plot/'+label+'/flux_2_'+file1+'.eps'
 plt.close()
 del fig

 # THIRD

 daVraBdX = np.zeros(mW.shape)
 for k in range(len(depths)):
  daVraBdX[:,k] = 1/Xr*np.gradient(Xr*maVraB[:,k])/np.gradient(Xr)

 T = -daVraBdX

 # plot
# v = np.linspace(np.percentile(T,5), np.percentile(T,95), 50, endpoint=True)
# v = np.linspace(-np.max(abs(np.percentile(T,5)),abs(np.percentile(T,95))),np.max(abs(np.percentile(T,5)),abs(np.percentile(T,95))), 50, endpoint=True)
 v = np.linspace(-3e-11,3e-11, 50, endpoint=True)
 vl = np.linspace(-0.18, 0.18, 5, endpoint=True)
 fig = plt.figure()
 plt.contourf(Xr/1000,depths,np.transpose(T[:,:]),50,extend='both',cmap=plt.cm.PiYG)
# plt.contour(Xr/1000,depths,np.transpose(T[:,:]),v,colors='k')
 plt.colorbar(format='%.3e')
 plt.xlabel('radius [Km]')
 plt.ylabel('depth [m]')
 plt.title('C')
 plt.savefig('./plot/'+label+'/flux_3_'+file1+'.eps',bbox_inches='tight')
 print './plot/'+label+'/flux_3_'+file1+'.eps'
 plt.close()
 del fig

 # FORTH TERM

 dBdR = np.zeros(mW.shape)
 dBdZ = np.zeros(mW.shape)
 for k in range(len(depths)):
  dBdR[:,k] = np.gradient(np.gradient(mB[:,k])/np.gradient(Xr))/np.gradient(Xr) + 1/Xr*np.gradient(mB[:,k])/np.gradient(Xr)
 
 for i in range(len(Xr)):
  dBdZ[i,:] =  np.gradient(np.gradient(mB[i,:])/np.gradient(depths))/np.gradient(depths)


 F  = dBdR*0.75 + dBdZ*1e-5

 # plot
# v = np.linspace(-np.max(abs(np.percentile(F,5)),abs(np.percentile(F,95))),np.max(abs(np.percentile(F,5)),abs(np.percentile(F,95))), 50, endpoint=True)
 v = np.linspace(-3e-12,3e-12, 50, endpoint=True)
 vl = np.linspace(-0.18, 0.18, 5, endpoint=True)
 fig = plt.figure()
 plt.contourf(Xr/1000,depths,np.transpose(F[:,:]),50,extend='both',cmap=plt.cm.PiYG)
# plt.contour(Xr/1000,depths,np.transpose(F[:,:]),v,colors='k')
 plt.colorbar(format='%.3e')
 plt.xlabel('radius [Km]')
 plt.ylabel('depth [m]')
 plt.title('D')
 plt.savefig('./plot/'+label+'/flux_4_'+file1+'.eps',bbox_inches='tight')
 print './plot/'+label+'/flux_4_'+file1+'.eps'
 plt.close()
 del fig

 
 gc.collect()
 print memory_usage(-1, interval=.2, timeout=.2)


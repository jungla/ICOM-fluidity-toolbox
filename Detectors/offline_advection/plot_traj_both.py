#!~/python
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import myfun
import numpy as np
import os
import lagrangian_stats
from scipy.interpolate import interp1d
import advect_functions

# read offline
print 'reading offline'
exp = 'm_25_1'
filename0 = 'traj_m_25_2_512_0_48.csv' 
tt = 48 # IC + 24-48 included
#x0 = range(0,2000,50)
#y0 = range(0,2000,50)
#z0 = range(0,50,2)
x0 = range(3000,4000,50)
y0 = range(2000,3000,50)
z0 = range(0,30,1)
xp = len(x0)
yp = len(y0)
zp = len(z0)

pt = xp*yp*zp 
timet, par0 = advect_functions.read_particles_csv(filename0,xp,yp,zp,tt)

time0 = (timet)*1200 + 48*3600 - 1200

# read online
print 'reading online'
exp = 'm_25_2_512'

try: os.stat('./plot/'+exp)
except OSError: os.mkdir('./plot/'+exp)

filename0 = './mli_tracer.detectors'
filename0 = '/tamay2/mensa/fluidity/'+exp+'/'+filename0

time1, parT = lagrangian_stats.read_particles(filename0)

dt = 1200
pt = len(parT)
timei = np.asarray(range(0,tt,1))*dt + 86400*2.0 + dt
par1 = np.zeros((pt,3,len(timei)))

for p in range(len(par1)):
 f0 = interp1d(time1,parT[p,0,:])
 f1 = interp1d(time1,parT[p,1,:])
 f2 = interp1d(time1,parT[p,2,:])
 par1[p,0,:] = f0(timei)
 par1[p,1,:] = f1(timei)
 par1[p,2,:] = f2(timei)

time = timei

tt = len(time)

# horizontal
depths = [1, 5, 11, 17, 26]
depths = [1, 5, 17]
depths = [1] #, 17, 1]

for z in depths: 
 print 'depth', z
 par0z = np.reshape(par0,(xp,yp,zp,3,tt))
 par1z = np.reshape(par1,(xp,yp,zp,3,tt))
 par0zr = par0z[:,:,z,:,:]
 par1zr = par1z[:,:,z,:,:]
 #
 par0z = np.reshape(par0zr,(xp*yp,3,tt))
 par1z = np.reshape(par1zr,(xp*yp,3,tt))
 #
 for t in range(0,tt,1):
  print 'time', time[t]
  #
  #
  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(111, aspect='equal')
  #
  s0D = plt.scatter(par0z[:,0,t]/1000, par0z[:,1,t]/1000, marker='.', s=65, facecolor='b', lw = 0)
  s1D = plt.scatter(par1z[:,0,t]/1000, par1z[:,1,t]/1000, marker='.', s=65, facecolor='r', lw = 0)
  plt.legend([s0D,s1D],['offline','online'])
  #
  plt.xlim([0, 10])
  plt.ylim([0, 4])
  plt.xlabel('X [km]',fontsize=18)
  plt.ylabel('Y [km]',fontsize=18)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig('./plot/'+exp+'/traj_2_'+exp+'_z'+str(z)+'_'+str(time[t])+'_h.eps')
  print       './plot/'+exp+'/traj_2_'+exp+'_z'+str(z)+'_'+str(time[t])+'_h.eps'
  plt.close()



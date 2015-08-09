#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import myfun
import numpy as np
import os
#import princax
from matplotlib.patches import Ellipse
import lagrangian_stats

exp = 'm_25_1_particles'

try: os.stat('./plot/'+exp)
except OSError: os.mkdir('./plot/'+exp)

filename0 = './mli_checkpoint.detectors'
filename0 = '/tamay2/mensa/fluidity/'+exp+'/'+filename0

time, par3D = lagrangian_stats.read_particles(filename0)

tt = len(time)

# horizontal
depths = [1, 5, 11, 17, 26]
depths = [1, 5, 17]
depths = [1] #, 17, 1]

# periodicity

#par3DP = lagrangian_stats.periodicCoords(par3D,2000,2000)

# interpolate to same timeseries as offline advection
from scipy.interpolate import interp1d
dt = 600
pt = len(par3D)
timei = np.asarray(range(0,141,1))*dt + 86400*2.0 + dt
par3Di = np.zeros((pt,3,len(timei)))

for p in range(len(par3D)):
 f0 = interp1d(time,par3D[p,0,:])
 f1 = interp1d(time,par3D[p,1,:])
 f2 = interp1d(time,par3D[p,2,:])
 par3Di[p,0,:] = f0(timei)
 par3Di[p,1,:] = f1(timei)
 par3Di[p,2,:] = f2(timei)

par3D = par3Di
time = timei
tt = len(timei)

for z in depths: 
 print 'depth', z
 par3Dz = np.reshape(par3D,(40,40,25,3,tt))
 #
 par3Dzr = par3Dz[:,:,z,:,:]
 #
 par3Dz = np.reshape(par3Dzr,(1600,3,tt))
 #

 for t in range(len(timei)):
  print 'time', time[t]/3600.0
  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(111, aspect='equal')
  #
#  plt.plot((4,4),(7,8),'k')
#  plt.plot((4,5),(7,7),'k')
#  plt.plot((5,5),(7,8),'k')
#  plt.plot((4,5),(8,8),'k')
#
#  plt.plot((0,0),(0,10),'k')
#  plt.plot((0,10),(0,0),'k')
#  plt.plot((10,10),(0,10),'k')
#  plt.plot((0,10),(10,10),'k')
  #
  s3D = plt.scatter(par3Dz[:,0,t]/1000, par3Dz[:,1,t]/1000, marker='.', s=35, facecolor='r', lw = 0)
  #

  plt.xlim([-1, 3])
  plt.ylim([-1, 3])
  plt.xlabel('X [km]',fontsize=18)
  plt.ylabel('Y [km]',fontsize=18)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
#  plt.title(str(time/3)+' hr',fontsize=18)

  ax.text(1, 9, str(z)+'m, '+str(time[t]/3600)+'h', fontsize=18)

  plt.savefig('./plot/'+exp+'/traj_'+exp+'_z'+str(z)+'_'+str(time[t])+'_h.eps')
  print       './plot/'+exp+'/traj_'+exp+'_z'+str(z)+'_'+str(time[t])+'_h.eps'
  plt.close()
 
  # plot ellipse

 # vertical

  fig = plt.figure(figsize=(8,8))
  #
  s3D = plt.scatter(par3Dz[:,0,t]/1000, par3Dz[:,2,t],  marker='.', s=35, facecolor='r', lw = 0)
  #
  plt.xlim([0, 2])
  plt.ylim([-50, 0])
  #
  print 'Saving 2D to eps'
  # 
  plt.text(6, -40, str(z)+'m, '+str(time[t]/3600)+'h', fontsize=18)
  plt.savefig('./plot/'+exp+'/traj_'+exp+'_z'+str(z)+'_'+str(time[t])+'_v.eps')
  print       './plot/'+exp+'/traj_'+exp+'_z'+str(z)+'_'+str(time[t])+'_v.eps'
  plt.close()

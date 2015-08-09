#!~/python
import fluidity_tools
import matplotlib  as mpl
#mpl.use('ps')
import matplotlib.pyplot as plt
import myfun
import numpy as np
import os
import fio
import lagrangian_stats
import advect_functions

exp = 'm_25_1'
filename0 = 'traj_m_25_1_particles_240_0_2D_b.csv'
filename0 = 'traj_m_25_1_particles_0_240_2D.csv'
filename0 = 'traj_m_25_1_tracer_0_640_2D.csv'
tt = 640 # IC + 24-48 included
x0 = range(500,1510,10)
y0 = range(500,1510,10)
z0 = range(1,20,4)

xp = len(x0)
yp = len(y0)
zp = len(z0)
pt = xp*yp*zp

#timet, par0 = advect_functions.read_particles_csv(filename0,pt,tt)
#par0 = lagrangian_stats.periodicCoords(par0,2000,2000)

time0 = (timet)*360 - 360

# horizontal
depths = [1, 5, 11, 17, 26]
depths = [1, 5, 17]
#depths = [1] #, 17, 1]

# periodicity

par = np.reshape(par0,(pt,3,tt))

#par = lagrangian_stats.periodicCoords(par,2000,2000)

for z in range(len(depths)): 
 print 'depth', z
 par0z = np.reshape(par,(xp,yp,zp,3,tt))
 par0zr = par0z[:,:,z,:,:]
 #
 par0z = np.reshape(par0zr,(xp*yp,3,tt))
 #

 for t in range(0,tt-1,1):
  print 'time', time0[t]/24
  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(111, aspect='equal')
  #
#  plt.plot((4,4),(7,8),'k')
#  plt.plot((4,5),(7,7),'k')
#  plt.plot((5,5),(7,8),'k')
#  plt.plot((4,5),(8,8),'k')

#  plt.plot((0,0),(0,10),'k')
#  plt.plot((0,10),(0,0),'k')
#  plt.plot((10,10),(0,10),'k')
#  plt.plot((0,10),(10,10),'k')
  #
  s3D = plt.scatter(par0z[:,0,t]/1000, par0z[:,1,t]/1000, marker='.', s=35, facecolor='r', lw = 0)
  #
  print 'Saving 0 to eps'
  # 
#  ax.text(1, 9, str(z)+'m, '+str(time[t]*3600)+'h', fontsize=18)
  plt.xlim([-2, 4])
  plt.ylim([-2, 4])
  plt.savefig('./plot/'+exp+'/traj_'+exp+'_z'+str(z)+'_'+str(time0[t])+'_h.png')
  print       './plot/'+exp+'/traj_'+exp+'_z'+str(z)+'_'+str(time0[t])+'_h.png'
  plt.close()
 
  # plot ellipse

 # vertical

  fig = plt.figure(figsize=(8,8))

#  plt.plot((1,1),(0,-50),'k')
#  plt.plot((2,2),(0,-50),'k')
  #
  s3D = plt.scatter(par0z[:,0,t]/1000, par0z[:,2,t],  marker='.', s=35, facecolor='r', lw = 0)
  #
  plt.xlim([0, 2])
  plt.ylim([-50, 0])
  #
  print 'Saving 0 to eps'
  # 

  plt.text(6, -40, str(z)+'m, '+str(time0[t])+'h', fontsize=18)
  plt.savefig('./plot/'+exp+'/traj_'+exp+'_z'+str(z)+'_'+str(time0[t])+'_v.png')
  print       './plot/'+exp+'/traj_'+exp+'_z'+str(z)+'_'+str(time0[t])+'_v.png'
  plt.close()

#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
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
filename0 = 'traj_m_25_1_particles_960_950_2D_b.csv'
tt = 10  # IC + 24-48 included
x0 = range(0,2000,25)
y0 = range(0,2000,25)
z0 = range(50,0,-10)

dx = 5
dy = 5

xt = []
yt = []
zt = []

for z in z0:
 print z
 for x in x0:
  for y in y0:
   for p in range(3):
    if p == 0:
     xt.append(x)
     yt.append(y)
     zt.append(z)
    if p == 1:
     xt.append(x+dx)
     yt.append(y)
     zt.append(z)
    if p == 2:
     xt.append(x)
     yt.append(y+dy)
     zt.append(z)

xt = np.asarray(xt)
yt = np.asarray(yt)
zt = np.asarray(zt)

xp = len(xt)
yp = len(yt)
zp = len(zt)

pt = xp

timet, par0 = advect_functions.read_particles_csv(filename0,pt,tt)
par0 = lagrangian_stats.periodicCoords(par0,2000,2000)

time0 = (timet)*360 + 48*3600 - 360

# horizontal
depths = [1, 5, 11, 17, 26]
depths = [1, 5, 17]
#depths = [1] #, 17, 1]

# periodicity

par = np.reshape(par0,(pt,3,tt))

#par = lagrangian_stats.periodicCoords(par,2000,2000)

for z in depths: 
 print 'depth', z
 par0z = par[pt/len(z0)*z:pt/len(z0)*(z+1),:,:]
 #

 for t in range(0,tt,1):
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
  plt.xlim([0, 0.5])
  plt.ylim([0, 0.5])
  plt.savefig('./plot/'+exp+'/traj_'+exp+'_z'+str(z)+'_'+str(time0[t])+'_h.eps')
  print       './plot/'+exp+'/traj_'+exp+'_z'+str(z)+'_'+str(time0[t])+'_h.eps'
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
  plt.savefig('./plot/'+exp+'/traj_'+exp+'_z'+str(z)+'_'+str(time0[t])+'_v.eps')
  print       './plot/'+exp+'/traj_'+exp+'_z'+str(z)+'_'+str(time0[t])+'_v.eps'
  plt.close()

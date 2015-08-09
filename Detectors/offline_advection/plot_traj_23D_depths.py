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

exp = 'm_25_2b_particles'

filename2D = './traj/traj_'+exp+'_0_500_2D.csv'
filename3D = './traj/traj_'+exp+'_0_500_3D.csv'

tt = 500# IC + 24-48 included

x0 = range(3500,4510,10)
y0 = range(3500,4510,10)
z0 = [0,5,10,15]

xp = len(x0)
yp = len(y0)
zp = len(z0)
pt = xp*yp*zp

time2D, par2D = advect_functions.read_particles_csv(filename2D,pt,tt)
par2D = lagrangian_stats.periodicCoords(par2D,8000,8000)
time3D, par3D = advect_functions.read_particles_csv(filename3D,pt,tt)
par3D = lagrangian_stats.periodicCoords(par3D,8000,8000)

time2D = (time2D)*1440 
time3D = (time3D)*1440 

time0 = time2D

# horizontal
#depths = [1, 5, 11, 17, 26]
depths = [5, 15]
depthsid = [1, 3]
#depths = [1] #, 17, 1]

par2Dzr = np.zeros((xp*yp,3,tt,len(depths)))
par3Dzr = np.zeros((xp*yp,3,tt,len(depths)))

# 2D

for t in range(0,tt,5):
 print 'time', time0[t]/24
 fig = plt.figure(figsize=(8,8))
 ax = fig.add_subplot(111, aspect='equal')
 #
 plt.plot((3.5,3.5),(3.5,4.5),'k',linewidth=2)
 plt.plot((3.5,4.5),(3.5,3.5),'k',linewidth=2)
 plt.plot((4.5,4.5),(3.5,4.5),'k',linewidth=2)
 plt.plot((3.5,4.5),(4.5,4.5),'k',linewidth=2)

 for z in range(len(depths)): 
  print 'depth', z
  par2Dz = np.reshape(par2D,(xp,yp,zp,3,tt))
  par2Dzr[:,:,:,z] = np.reshape(par2Dz[:,:,depthsid[z],:,:],(xp*yp,3,tt))
  #

 s5 = plt.scatter(par2Dzr[:,0,t,0]/1000, par2Dzr[:,1,t,0]/1000, marker='.', s=35, facecolor='b', lw = 0)
 s15 = plt.scatter(par2Dzr[:,0,t,1]/1000, par2Dzr[:,1,t,1]/1000, marker='.', s=35, facecolor='r', lw = 0)
 #

 plt.legend((s5,s15),('2D 5m','2D 15m'))

 print 'Saving 0 to eps'

 plt.xlim([0, 8])
 plt.ylim([0, 8])
 plt.xlabel('X [km]',fontsize=18)
 plt.ylabel('Y [km]',fontsize=18)
 plt.xticks(fontsize=16)
 plt.yticks(fontsize=16)
 plt.xlabel('X [km]', fontsize=18)
 plt.ylabel('Y [km]', fontsize=18)
 plt.savefig('./plot/'+exp+'/traj_'+exp+'_2D_'+str(time0[t])+'_h.eps')
 print       './plot/'+exp+'/traj_'+exp+'_2D_'+str(time0[t])+'_h.eps'
 plt.close()

# 3D

 fig = plt.figure(figsize=(8,8))
 ax = fig.add_subplot(111, aspect='equal')
 #
 plt.plot((3.5,3.5),(3.5,4.5),'k',linewidth=2)
 plt.plot((3.5,4.5),(3.5,3.5),'k',linewidth=2)
 plt.plot((4.5,4.5),(3.5,4.5),'k',linewidth=2)
 plt.plot((3.5,4.5),(4.5,4.5),'k',linewidth=2)

 for z in range(len(depths)):
  print 'depth', z
  par3Dz = np.reshape(par3D,(xp,yp,zp,3,tt))
  par3Dzr[:,:,:,z] = np.reshape(par3Dz[:,:,depthsid[z],:,:],(xp*yp,3,tt))
  #

 s5 = plt.scatter(par3Dzr[:,0,t,0]/1000, par3Dzr[:,1,t,0]/1000, marker='.', s=35, facecolor='b', lw = 0)
 s15 = plt.scatter(par3Dzr[:,0,t,1]/1000, par3Dzr[:,1,t,1]/1000, marker='.', s=35, facecolor='r', lw = 0)
 #

 plt.legend((s5,s15),('3D 5m','3D 15m'))

 print 'Saving 0 to eps'

 plt.xlim([0, 8])
 plt.ylim([0, 8])
 plt.xlabel('X [km]',fontsize=18)
 plt.ylabel('Y [km]',fontsize=18)
 plt.xticks(fontsize=16)
 plt.yticks(fontsize=16)
 plt.xlabel('X [km]', fontsize=18)
 plt.ylabel('Y [km]', fontsize=18)
 plt.savefig('./plot/'+exp+'/traj_'+exp+'_3D_'+str(time0[t])+'_h.eps')
 print       './plot/'+exp+'/traj_'+exp+'_3D_'+str(time0[t])+'_h.eps'
 plt.close()

 # vertical
#
#  fig = plt.figure(figsize=(8,8))
#
#  plt.plot((3,3),(0,-50),'k')
#  plt.plot((4,4),(0,-50),'k')
#  #
#  s2D = plt.scatter(par2Dz[:,0,t]/1000, par2Dz[:,2,t],  marker='.', s=35, facecolor='b', lw = 0)
#  s3D = plt.scatter(par3Dz[:,0,t]/1000, par3Dz[:,2,t],  marker='.', s=35, facecolor='r', lw = 0)
#
#  plt.legend((s3D,s2D),('3D','2D'))
#  #
#  plt.xlim([-1, 11])
#  plt.ylim([-50, 0])
#  #
#  print 'Saving 0 to eps'
#  # 
#
# # plt.text(10, -40, str(depths[z])+'m, '+str(time0[t]/3600)+'h', fontsize=18)
#  plt.xlabel('X [km]', fontsize=18)
#  plt.ylabel('Z [m]', fontsize=18)
#  plt.savefig('./plot/'+exp+'/traj_'+exp+'_z'+str(depths[z])+'_'+str(time0[t])+'_v.eps')
#  print       './plot/'+exp+'/traj_'+exp+'_z'+str(depths[z])+'_'+str(time0[t])+'_v.eps'
#  plt.close()

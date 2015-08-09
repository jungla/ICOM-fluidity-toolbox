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
from matplotlib.patches import Ellipse

exp = 'm_25_2'
filename2D = 'traj_m_25_2_512_0_500_2D.csv'
filename3D = 'traj_m_25_2_512_0_500_3D.csv'
tt = 500 # IC + 24-48 included

#tt = 230 # IC + 24-48 included
x0 = range(3000,4010,10)
y0 = range(2000,3010,10)
#z0 = range(1,20,4)
z0 = [0,5,10,15]

xp = len(x0)
yp = len(y0)
zp = len(z0)
pt = xp*yp*zp

#time2D, par2D = advect_functions.read_particles_csv(filename2D,pt,tt)
#par2D = lagrangian_stats.periodicCoords(par2D,10000,4000)
#time3D, par3D = advect_functions.read_particles_csv(filename3D,pt,tt)
#par3D = lagrangian_stats.periodicCoords(par3D,10000,4000)
#
#time2D = (time2D)*1440 
#time3D = (time3D)*1440 

time0 = time2D

# horizontal
#depths = [1, 5, 11, 17, 26]
depths = [5, 10, 15]
depthsid = [1, 2, 3]
#depths = [1] #, 17, 1]


for z in range(len(depths)): 
 print 'depth', z
 par2Dz = np.reshape(par2D,(xp,yp,zp,3,tt))
 par2Dzr = par2Dz[:,:,depthsid[z],:,:]
 par2Dz = np.reshape(par2Dzr,(xp*yp,3,tt))

 par3Dz = np.reshape(par3D,(xp,yp,zp,3,tt))
 par3Dzr = par3Dz[:,:,depthsid[z],:,:]
 par3Dz = np.reshape(par3Dzr,(xp*yp,3,tt))
 #

 for t in range(70,75,5):
  print 'time', time0[t]/24
  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(111, aspect='equal')
  #
  plt.plot((3,3),(2,3),'k')
  plt.plot((3,4),(2,2),'k')
  plt.plot((4,4),(2,3),'k')
  plt.plot((3,4),(3,3),'k')

  plt.plot((0,0),(0,4),'k')
  plt.plot((0,10),(0,0),'k')
  plt.plot((10,10),(0,4),'k')
  plt.plot((0,10),(4,4),'k')
  #
  s3D = plt.scatter(par3Dz[:,0,t]/1000, par3Dz[:,1,t]/1000, marker='.', s=35, facecolor='r', lw = 0)
  s2D = plt.scatter(par2Dz[:,0,t]/1000, par2Dz[:,1,t]/1000, marker='.', s=35, facecolor='b', lw = 0)
  #
  plt.legend((s3D,s2D),('3D','2D'))

  print 'Saving 0 to eps'

  xt3 = par3Dz[:,0,t] - np.mean(par3Dz[:,0,t])
  yt3 = par3Dz[:,1,t] - np.mean(par3Dz[:,1,t])
  xt2 = par2Dz[:,0,t] - np.mean(par2Dz[:,0,t])
  yt2 = par2Dz[:,1,t] - np.mean(par2Dz[:,1,t])
 #
  cov3 = np.cov(xt3/1000, yt3/1000)
  lambda_3, v = np.linalg.eig(cov3)
  lambda_3 = np.sqrt(lambda_3)
  theta3 = np.rad2deg(0.5*np.arctan2(2*cov3[1,0],(cov3[0,0]-cov3[1,1])))
  theta3 = np.rad2deg(np.arcsin(v[0, 0]))
  #
  cov2 = np.cov(xt2/1000, yt2/1000)
  lambda_2, v = np.linalg.eig(cov2)
  lambda_2 = np.sqrt(lambda_2)
  theta2 = np.rad2deg(0.5*np.arctan2(2*cov2[1,0],(cov2[0,0]-cov2[1,1]))) + np.pi*0.5
  theta2 = np.rad2deg(np.arcsin(v[0, 0]))
  #
  e0 = Ellipse(xy=(np.mean(par3Dz[:,0,t])/1000,np.mean(par3Dz[:,1,t])/1000),width=4*lambda_3[1],height=4*lambda_3[0],angle=theta3)
  e1 = Ellipse(xy=(np.mean(par2Dz[:,0,t])/1000,np.mean(par2Dz[:,1,t])/1000),width=4*lambda_2[1],height=4*lambda_2[0],angle=theta2)

  ax.add_artist(e0)
  e0.set_facecolor('none')
  e0.set_edgecolor('k')
  e0.set_linewidth(2.5)

  ax.add_artist(e1)
  e1.set_facecolor('none')
  e1.set_edgecolor('k')
  e1.set_linewidth(2.5)
  e1.set_linestyle('dashed')

  plt.xlim([-1, 11])
  plt.ylim([-5, 5])
  plt.xlabel('X [km]',fontsize=18)
  plt.ylabel('Y [km]',fontsize=18)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)

  # 
#  ax.text(0, 7, str(depths[z])+'m, '+str(time0[t]/3600)+'h', fontsize=18)
  plt.xlabel('X [km]', fontsize=18)
  plt.ylabel('Y [km]', fontsize=18)
  plt.savefig('./plot/'+exp+'/traj_'+exp+'_z'+str(depths[z])+'_'+str(time0[t])+'_h.eps')
  print       './plot/'+exp+'/traj_'+exp+'_z'+str(depths[z])+'_'+str(time0[t])+'_h.eps'
  plt.close()
 
  # plot ellipse

 # vertical

  fig = plt.figure(figsize=(8,8))

  plt.plot((3,3),(0,-50),'k')
  plt.plot((4,4),(0,-50),'k')
  #
  s2D = plt.scatter(par2Dz[:,0,t]/1000, par2Dz[:,2,t],  marker='.', s=35, facecolor='b', lw = 0)
  s3D = plt.scatter(par3Dz[:,0,t]/1000, par3Dz[:,2,t],  marker='.', s=35, facecolor='r', lw = 0)

  plt.legend((s3D,s2D),('3D','2D'))
  #
  plt.xlim([-1, 11])
  plt.ylim([-50, 0])
  #
  print 'Saving 0 to eps'
  # 

 # plt.text(10, -40, str(depths[z])+'m, '+str(time0[t]/3600)+'h', fontsize=18)
  plt.xlabel('X [km]', fontsize=18)
  plt.ylabel('Z [m]', fontsize=18)
  plt.savefig('./plot/'+exp+'/traj_'+exp+'_z'+str(depths[z])+'_'+str(time0[t])+'_v.eps')
  print       './plot/'+exp+'/traj_'+exp+'_z'+str(depths[z])+'_'+str(time0[t])+'_v.eps'
  plt.close()

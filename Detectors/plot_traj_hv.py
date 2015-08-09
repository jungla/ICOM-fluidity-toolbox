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

exp = 'm_25_2_23D_particles'

try: os.stat('./plot/'+exp)
except OSError: os.mkdir('./plot/'+exp)

exp2D = 'm_25_2_2D_particles'

filename0 = './mli_checkpoint.detectors'
filename0 = '/tamay2/mensa/fluidity/'+exp2D+'/'+filename0

time2D, par2D = lagrangian_stats.read_particles(filename0)

exp3D = 'm_25_2_3D_particle'

filename0 = './mli_checkpoint.detectors'
filename0 = '/tamay2/mensa/fluidity/'+exp3D+'/'+filename0

time3D, par3D = lagrangian_stats.read_particles(filename0)

tt = min(len(time2D),len(time3D))

par3D = par3D[:,:,:tt]
par2D = par2D[:,:,:tt]

time = time2D[:tt]

# horizontal
depths = [1, 5, 11, 17, 26]
depths = [1, 5, 17]
#depths = [5] #, 17, 1]

# periodicity

par2DP = lagrangian_stats.periodicCoords(par2D,4000,10000)
par3DP = lagrangian_stats.periodicCoords(par3D,4000,10000)

for z in depths: 
 print 'depth', z
 par2Dz = np.reshape(par2DP,(20,20,30,3,tt))
 par3Dz = np.reshape(par3DP,(20,20,30,3,tt))
 #
 par2Dzr = par2Dz[:,:,z,:,:]
 par3Dzr = par3Dz[:,:,z,:,:]
 #
 par2Dz = np.reshape(par2Dzr,(400,3,tt))
 par3Dz = np.reshape(par3Dzr,(400,3,tt))
 #

 for t in range(0,tt,5):
  print 'time', time[t]/3600.0
  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(111, aspect='equal')
  #
  plt.plot((4,4),(7,8),'k')
  plt.plot((4,5),(7,7),'k')
  plt.plot((5,5),(7,8),'k')
  plt.plot((4,5),(8,8),'k')

  plt.plot((0,0),(0,10),'k')
  plt.plot((0,10),(0,0),'k')
  plt.plot((10,10),(0,10),'k')
  plt.plot((0,10),(10,10),'k')
  #
  s3D = plt.scatter(par3Dz[:,0,t]/1000, par3Dz[:,1,t]/1000, marker='.', s=35, facecolor='r', lw = 0)
  #
  s2D = plt.scatter(par2Dz[:,0,t]/1000, par2Dz[:,1,t]/1000, marker='.', s=35, facecolor='b', lw = 0)
  #
  plt.legend((s3D,s2D),('3D','2D'))
  #
  print 'Saving 2D to eps'
  # 
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

  plt.xlim([-20, 20])
  plt.ylim([-20, 20])
  plt.xlabel('X [km]',fontsize=18)
  plt.ylabel('Y [km]',fontsize=18)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
#  plt.title(str(time/3)+' hr',fontsize=18)

  ax.text(1, 9, str(z)+'m, '+str(time[t]/3600)+'h', fontsize=18)

  plt.savefig('./plot/'+exp+'/traj_'+exp+'_z'+str(z)+'_'+str(time[t]/3600)+'_h.eps')
  print       './plot/'+exp+'/traj_'+exp+'_z'+str(z)+'_'+str(time[t]/3600)+'_h.eps'
  plt.close()
 
  # plot ellipse

 # vertical

  fig = plt.figure(figsize=(8,8))

  plt.plot((1,1),(0,-50),'k')
  plt.plot((2,2),(0,-50),'k')
  #
  s3D = plt.scatter(par3Dz[:,0,t]/1000, par3Dz[:,2,t],  marker='.', s=35, facecolor='r', lw = 0)
  #
  s2D = plt.scatter(par2Dz[:,0,t]/1000, par2Dz[:,2,t], marker='.', s=35, facecolor='b', lw = 0)
  #
  plt.legend((s3D,s2D),('3D','2D'))
  plt.xlim([0, 10])
  plt.ylim([-50, 0])
  #
  print 'Saving 2D to eps'
  # 

  plt.text(6, -40, str(z)+'m, '+str(time[t]/3600)+'h', fontsize=18)
  plt.savefig('./plot/'+exp+'/traj_'+exp+'_z'+str(z)+'_'+str(time[t]/3600)+'_v.eps')
  print       './plot/'+exp+'/traj_'+exp+'_z'+str(z)+'_'+str(time[t]/3600)+'_v.eps'
  plt.close()

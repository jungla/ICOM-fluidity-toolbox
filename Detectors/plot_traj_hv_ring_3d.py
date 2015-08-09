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

exp = 'r_1k_B_1F1_particles'

filename0 = './ring_checkpoint_checkpoint.detectors'
filename1 = './ring_checkpoint_checkpoint_checkpoint.detectors'

filename0 = '/tamay2/mensa/fluidity/'+exp+'/'+filename0
filename1 = '/tamay2/mensa/fluidity/'+exp+'/'+filename1

print 'Reading ', filename1 , filename0 

#try: os.stat('./output/'+exp3D)
#except OSError: os.mkdir('./output/'+exp3D)

time0, par0 = lagrangian_stats.read_particles(filename0)
time1, par1 = lagrangian_stats.read_particles(filename1)

time = np.hstack((time0[time0<time1[0]],time1))
par = np.concatenate((par0[:,:,time0<time1[0]], par1), axis=2)

tt = len(time)

# horizontal

Zlist = [-10,-50,-100,-200,-500,-800]

for z in range(0,len(Zlist),2): 
 print 'depth', z
 par2Dz = par[z::6,:,:] 
 #
 for t in range(1,tt,5):
  print 'time', time[t]/3600.0
  fig = plt.figure(figsize=(10,7))
  ax = fig.add_subplot(111, projection='3d')
  #
  s2D = ax.scatter(par2Dz[:,0,t]/1000.0, par2Dz[:,1,t]/1000.0, par2Dz[:,2,t]-Zlist[z], c = par2Dz[:,2,t], s = 35, marker='.', lw = 0)
  #
  print 'Saving 2D to eps'
  # 
  ax.text(-120, -120, 90, str(Zlist[z])+'m', fontsize=16)
  ax.text(-120, -120, 80, str(np.round((time[t]/86400 - time[0]/86400)*10)/10.0)+'day', fontsize=16)
  #
  ax.set_xlabel('X [km]',fontsize=18)
  ax.set_ylabel('Y [km]',fontsize=18)
  ax.set_zlabel(r'$\Delta$Z [m]',fontsize=18)
  ax.tick_params(axis='x', labelsize=14)
  ax.tick_params(axis='y', labelsize=14)
  ax.tick_params(axis='z', labelsize=14)
  ax.set_xlim([-120,120])
  ax.set_ylim([-120,120])
  ax.set_zlim([-50,50])
  plt.savefig('./plot/'+exp+'/traj_3d_'+exp+'_'+str(Zlist[z])+'_'+myfun.digit(str(int(time[t]/3600)),4)+'.eps')
  print       './plot/'+exp+'/traj_3d_'+exp+'_'+str(Zlist[z])+'_'+myfun.digit(str(int(time[t]/3600)),4)+'.eps'
  plt.close()
 

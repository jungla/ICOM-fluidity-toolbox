#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import myfun
import numpy as np
from scipy import optimize
import os
import scipy.stats as sp
import scipy
import lagrangian_stats
import advect_functions
import csv

# read offline
print 'reading offline'

#exp = 'm_25_1_tracer'
#label = 'm_25_1_tracer'
#filename2D = 'traj_m_25_1_tracer_0_640_2D.csv'
#filename3D = 'traj_m_25_1_tracer_0_640_3D.csv'
#tt = 640 # IC + 24-48 included

label = 'm_25_2b_particles'
label_B = 'm_25_1b_particles'
label_BW = 'm_25_2b_particles'
filename2D_BW = './csv/CDv_'+label_BW+'.csv'
filename2D_B = './csv/CDv_'+label_B+'.csv'

tt = 500-61

time2D_B, CD_2D_B =lagrangian_stats.read_dispersion(filename2D_B)
time2D_BW, CD_2D_BW =lagrangian_stats.read_dispersion(filename2D_BW)

time = time2D_B[:]
time = np.asarray(range(tt))*1440

depths = [5, 10, 15] 
depthid = [1, 2, 3] 

nl = len(depths)


# relative D

#CD_2D_B = np.log10(CD_2D_B)
#CD_2D_B = CD_2D_B[1:]
#CD_2D_BW = np.log10(CD_2D_BW)
#CD_2D_BW = CD_2D_BW[1:]

#x = np.log10(time-time[0])
#x = x[1:]
x = (time-time[0])
#

p_B , = plt.plot(x,CD_2D_B[:,0],color=[0,0,1],linewidth=3)
#z = 1
#p_B5 , = plt.plot(x,CD_2D_B[:,z],'--',color=[0,0,1],linewidth=3)
#z = 2
#p_B17 , = plt.plot(x,CD_2D_B[:,z],'-.',color=[0,0,1],linewidth=3)


p_BW, = plt.plot(x,CD_2D_BW[:,0],color=[1,0,0],linewidth=3)
#z = 1
#p_BW5, = plt.plot(x,CD_2D_BW[:,z],'--',color=[1,0,0],linewidth=3)
#z = 2
#p_BW17, = plt.plot(x,CD_2D_BW[:,z],'-.',color=[1,0,0],linewidth=3)

#plt.gca().set_yscale('log',basey=10)
#plt.gca().set_xscale('log',basex=10)

plt.xlabel(r'Time $[hr]$',fontsize=24)
plt.ylabel(r'$\sigma^2_{C_z}$ $[m^2]$',fontsize=24)
#plt.ylabel('Relative Dispersion [m^2]')
#plt.legend((p_B,p_B5,p_B17,p_BW,p_BW5,p_BW17),('B 5m','B 10m','B 15m','BW 5m','BW 10m','BW 15m'),loc=4,fontsize=20)
plt.legend((p_B,p_BW),('B 5m','BW 5m'),loc=4,fontsize=20)
plt.xlim((0,168))
#plt.xlim((np.log10(1440),np.log10(72*3600)))
#plt.ylim((2-.5,7.5))

#ind = np.linspace(2,14,13)*86400    # the x locations for the groups
#ind = np.linspace(0,288,9)*3600
ind = np.linspace(0,168,8)+72
vind = np.linspace(0,168,8)*3600

#plt.xticks(vind,['0.4','24','48','72'],fontsize=16)
plt.xticks(vind,ind.astype(int),fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()
plt.savefig('./plot/'+label+'/CDv_'+label+'.eps')
print       './plot/'+label+'/CDv_'+label+'.eps'
plt.close()


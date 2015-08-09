#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
#filename = 'traj_m_25_1_tracer_0_640_.csv'
#filename3D = 'traj_m_25_1_tracer_0_640_3D.csv'
#tt = 640 # IC + 24-48 included

label = 'm_25_2b_particles'
label_BW = 'm_25_2b_particles'
label_B = 'm_25_1b_particles'


time = np.asarray(range(60,240,2))*1440./3600.

for tday in range(60,240,2): #240
 Diff_B = [] #np.zeros((tt_B,nl))
 Diff_BW = [] #np.zeros((tt_B,nl))

 dayi  = tday+0 #10*24*1  
 dayf  = tday+15 #10*24*4

 filename_B = './csv/CD_v_'+label_B+'_'+str(dayi)+'_'+str(dayf)+'.csv'
 filename_BW = './csv/CD_v_'+label_BW+'_'+str(dayi)+'_'+str(dayf)+'.csv'

 print filename_B
 print filename_BW

 time_B, CD_B = lagrangian_stats.read_dispersion(filename_B)
 time_BW, CD_BW = lagrangian_stats.read_dispersion(filename_BW)

 gradBW,poop = np.gradient(CD_BW)
 Diff_BW.append(0.5*gradBW/1440.)

 gradB,poop = np.gradient(CD_B)
 Diff_B.append(0.5*gradB/1440.)

 Diff_B = np.squeeze(np.asarray(Diff_B))
 Diff_BW = np.squeeze(np.asarray(Diff_BW))
 
 # relative D
 
# CD 

 plt.scatter(np.sqrt(CD_B[:,0]),Diff_B[:,0],linewidth=3)

 plt.xlabel(r'$\sigma_(C_z)$ $[m]$',fontsize=24)
 plt.ylabel(r'$\sigma_{C_z}$ $[m]$',fontsize=24)
 #plt.legend((p_5_B,p_10_B,p_15_B,p_5_BW,p_10_BW,p_15_BW),('B 5m','B 10m','B 15m','BW 5m','BW 10m','BW 15m'),loc=1,fontsize=20,ncol=2)
 #plt.legend((p_5_B,p_5_BW),('B 5m','BW 5m'),loc=1,fontsize=20,ncol=2)
 #plt.ylim((0,30))
 #plt.xlim((24,72))
 ax.set_yscale('log')
 ax.set_xscale('log')

 plt.yticks(fontsize=20)
 plt.xticks(fontsize=20)

 plt.tight_layout()

 plt.savefig('./plot/'+label+'/Diffv_'+str(tday)+'_'+label+'.eps',)
 print       './plot/'+label+'/Diffv_'+str(tday)+'_'+label+'.eps'
 plt.close()


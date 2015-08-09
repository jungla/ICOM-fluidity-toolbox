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
#filename = 'traj_m_25_1_tracer_0_640_.csv'
#filename3D = 'traj_m_25_1_tracer_0_640_3D.csv'
#tt = 640 # IC + 24-48 included

label = 'm_25_2b_particles'
label_BW = 'm_25_2b_particles'
label_B = 'm_25_1b_particles'

Diff_B = [] #np.zeros((tt_B,nl))
Diff_BW = [] #np.zeros((tt_B,nl))

time = np.asarray(range(60,240,2))*1440./3600.
depths = -1*np.asarray(range(0,52,2))

for tday in range(60,240,2): #240
 dayi  = tday+0 #10*24*1  
 dayf  = tday+15 #10*24*4

 filename_B = './csv/CD_vb_'+label_B+'_'+str(dayi)+'_'+str(dayf)+'.csv'
 filename_BW = './csv/CD_vb_'+label_BW+'_'+str(dayi)+'_'+str(dayf)+'.csv'

 print filename_B
 print filename_BW

 CD_B = [] #np.zeros((tt_B,nl))
 time_B = []

 with open(filename_B, 'r') as csvfile:
  spamreader = csv.reader(csvfile)
  spamreader.next()
  for row in spamreader:
   time_B.append(row[0])
   CD_B.append(row[1:])
 
 time_B = np.asarray(time_B).astype(float)
 CD_B = np.asarray(CD_B).astype(float)

 CD_BW = [] #np.zeros((tt_B,nl))
 time_BW = []

 with open(filename_BW, 'r') as csvfile:
  spamreader = csv.reader(csvfile)
  spamreader.next()
  for row in spamreader:
   time_BW.append(row[0])
   CD_BW.append(row[1:])

 time_BW = np.asarray(time_BW).astype(float)
 CD_BW = np.asarray(CD_BW).astype(float)

 gradBW,poop = np.gradient(CD_BW)
 Diff_BW = list(0.5*np.mean(gradBW[3:,:],0)/1440.)

 gradB,poop = np.gradient(CD_B)
 Diff_B = list(0.5*np.mean(gradB[3:,:],0)/1440.)

 p_B, = plt.plot(Diff_B,depths,'r-',linewidth=3)
 p_BW, = plt.plot(Diff_BW,depths,'b-',linewidth=3)
 
 plt.xlabel(r'Time $[hr]$',fontsize=24)
 plt.ylabel(r'$K_z$ $[m^2s^{-1}]$',fontsize=24)
 plt.legend((p_B,p_BW),('B','BW'),loc=4,fontsize=20)

 #plt.xlim((24,96))
# plt.ylim((0.0001,0.001))
 plt.xticks(np.linspace(0,0.0007,8),np.linspace(0,0.0007,8),fontsize=14)
 plt.yticks(fontsize=14)
 
 plt.tight_layout()
 plt.savefig('./plot/'+label+'/Diff_CDz_'+label+'_'+str(tday)+'.eps')
 print       './plot/'+label+'/Diff_CDz_'+label+'_'+str(tday)+'.eps'
 plt.close()


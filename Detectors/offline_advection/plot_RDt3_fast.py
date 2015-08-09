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

label = 'm_25_1b_particles'
dim = '2D'
filename2D_B = './csv/RD_'+dim+'_'+label+'.csv'

time2D_B, RD_2D_B = lagrangian_stats.read_dispersion(filename2D_B)

time = time2D_B - time2D_B[0]

depths = [5, 10, 15] 
depthid = [1, 2, 3] 

nl = len(depths)


# relative D
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

fig, ax = plt.subplots()

for d in range(3):
 RD_2D_B[:,d] = RD_2D_B[:,d]/(time**3)

RD_2D = np.log10(RD_2D_B)
RD_2D = RD_2D[1:]

x = np.log10(time)
x = x[1:]
#
# RD 2D

if dim == '2D':
 print dim
 p_2D, = plt.plot(x,RD_2D[:,0],color=[0,0,1],linewidth=2)
 z = 1
 p_2D5, = plt.plot(x,RD_2D[:,z],'--',color=[0,0,1],linewidth=2)
 z = 2
 p_2D17, = plt.plot(x,RD_2D[:,z],'-.',color=[0,0,1],linewidth=2)
 plt.legend((p_2D,p_2D5,p_2D17),(dim+' 5m',dim+' 10m',dim+' 15m'),loc=1,fontsize=16)
else:
 p_2D, = plt.plot(x,RD_2D[:,0],color=[1,0,0],linewidth=2,label=str(dim+' 5m'))
 plt.legend(loc=1,fontsize=16)

plt.xlabel(r'Time $[hr]$',fontsize=20)
plt.ylabel(r'$log(\sigma^2_D\,t^{-3})$ $[m^2s^{-3}]$',fontsize=20)
plt.xlim((np.log10(1440),np.log10(24*3*3600)))
plt.ylim(-11.5,-7)


M_xticks = []
M_xticks_labels = []
m_xticks = []

for i in range(0,24*3+2,2):
 m_xticks.append(np.log10(i*3600.))
 if i%24==0:
  M_xticks.append(np.log10(i*3600))
  M_xticks_labels.append(i+72)

m_xticks[0] = np.log10(1440)
M_xticks_labels[0] = 72.4
#M_xticks_labels = [72.4, 96, '', 144, '', 192, '']
M_xticks[0] = np.log10(1440)

# Specify tick label size
ax.tick_params(axis = 'both', which = 'major', labelsize = 16, length=10)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 0, length=7)
# Suppress minor tick labels

ax.set_xticks(M_xticks)
ax.set_xticks(m_xticks, minor = True)
ax.set_xticklabels(M_xticks_labels, minor=False)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=16)

#plt.xticks(vind,ind.astype(int))

plt.tight_layout()
plt.savefig('./plot/'+label+'/RD_'+dim+'_'+label+'.eps')
print       './plot/'+label+'/RD_'+dim+'_'+label+'.eps'
plt.close()

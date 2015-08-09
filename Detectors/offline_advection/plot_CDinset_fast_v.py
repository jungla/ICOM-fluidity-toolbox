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

label_B = 'm_25_1b_particles'
label_BW = 'm_25_2b_particles'
label = 'm_25_2b_particles'
filename_B = './csv/CDv_'+label_B+'.csv'
filename_BW = './csv/CDv_'+label_BW+'.csv'

time_B, RD_B = lagrangian_stats.read_dispersion(filename_B)
time_BW, RD_BW = lagrangian_stats.read_dispersion(filename_BW)

time = time_B[:]  - time_B[0]
time = time_BW[:]  - time_BW[0]

depths = [5, 10, 15] 
depthid = [1, 2, 3] 

nl = len(depths)

# relative D
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

fig, ax = plt.subplots()

RD_Bl = np.log10(RD_B)
RD_Bl = RD_Bl[1:]
RD_BWl = np.log10(RD_BW)
RD_BWl = RD_BWl[1:]

x = np.log10(time)
x = x[1:]

new_x = np.linspace(np.log10(35000),5.0,10)
y3 = 0.5 -3.*new_x[0] + 3.*new_x
y2 = 0.5 -2.*new_x[0] + 2.*new_x
y1 = 0.5 -1.*new_x[0] + 1.*new_x
plt.plot(new_x,y1,'k',linewidth=1.5)
plt.plot(new_x,y2,'k',linewidth=1.5)
plt.plot(new_x,y3,'k',linewidth=1.5)
plt.text(4.74,-1,'$t^3$',fontsize=20)	

p_B, = plt.plot(x,RD_Bl[:,0],color=[0,0,1],linewidth=2)
p_BW, = plt.plot(x,RD_BWl[:,0],color=[1,0,0],linewidth=2)
plt.legend([p_B,p_BW],['B 5m','BW 5m'],bbox_to_anchor=(1., 0.023),loc=4,fontsize=18)

plt.xlabel(r'Time $[hr]$',fontsize=24)
plt.ylabel(r'$log(\sigma^2_{D_z})$ $[m^2]$',fontsize=24)
plt.xlim((np.log10(1440),np.log10(24*3600*3)))
plt.ylim((-1.5,4))

ind = []
vind = []

M_xticks = []
M_xticks_labels = []
m_xticks = []

for i in range(0,24*3+2,2):
 m_xticks.append(np.log10(i*3600.))
 if i%24==0:
  M_xticks_labels.append(i+72)
  M_xticks.append(np.log10(i*3600))

m_xticks[0] = np.log10(1440)
M_xticks_labels[0] = 72.4
M_xticks[0] = np.log10(1440) 

# Specify tick label size
ax.tick_params(axis = 'both', which = 'major', labelsize = 14, length=10)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 0, length=7)
# Suppress minor tick labels

ax.set_xticks(M_xticks)
ax.set_xticks(m_xticks, minor = True)
ax.set_xticklabels(M_xticks_labels, minor=False)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=14)

############################################################### INSET

for d in range(3):
 RD_B[:,d] = RD_B[:,d]/(time**3)
 RD_BW[:,d] = RD_BW[:,d]/(time**3)

RD_B = np.log10(RD_B)
RD_B = RD_B[1:]
RD_BW = np.log10(RD_BW)
RD_BW = RD_BW[1:]

a = plt.axes([.25, .64, .4, .3])

#p_2D, = plt.plot(x,RD_2D[:,0],color=[0,0,1],linewidth=2)
#z = 1
#p_2D5, = plt.plot(x,RD_2D[:,z],'--',color=[0,0,1],linewidth=2)
#z = 2
#p_2D17, = plt.plot(x,RD_2D[:,z],'-.',color=[0,0,1],linewidth=2)
#else:
p_B, = plt.plot(x,RD_B[:,0],color=[0,0,1],linewidth=2)
p_BW, = plt.plot(x,RD_BW[:,0],color=[1,0,0],linewidth=2)

plt.xlabel(r'Time $[hr]$',fontsize=15)
plt.ylabel(r'$log(\sigma^2_D\,t^{-3})$ $[m^2s^{-3}]$',fontsize=15)
plt.xlim((np.log10(1440),np.log10(24*3*3600)))
#plt.ylim(-11.5,-7)


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
a.tick_params(axis = 'both', which = 'major', labelsize = 10, length=10)
a.tick_params(axis = 'both', which = 'minor', labelsize = 0, length=7)
# Suppress minor tick labels

a.set_xticks(M_xticks)
a.set_xticks(m_xticks, minor = True)
a.set_xticklabels(M_xticks_labels, minor=False)

a.yaxis.set_ticks_position('left')
a.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig('./plot/'+label+'/CDv_'+label+'.eps')
print       './plot/'+label+'/CDv_'+label+'.eps'
plt.close()


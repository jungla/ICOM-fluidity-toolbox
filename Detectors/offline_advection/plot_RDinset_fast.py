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

time = time2D_B[:]  - time2D_B[0]

depths = [5, 10, 15] 
depthid = [1, 2, 3] 

nl = len(depths)

# relative D
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

fig, ax = plt.subplots()

RD_2D = np.log10(RD_2D_B)
RD_2D = RD_2D[1:]

x = np.log10(time)
x = x[1:]

if dim == '2D' and label == 'm_25_1b_particles':
 print dim
 new_x = np.linspace(np.log10(40000),5.35,10)
 y = 3.7 -3.*new_x[0] + 3.*new_x
 plt.plot(new_x,y,'k',linewidth=1.5)
 plt.text(5.0,y[-4],'$t^3$',fontsize=16)

if dim == '2D' and label == 'm_25_2b_particles':
 new_x = np.linspace(np.log10(50000),5.35,10)
 y = 5.1 -3.*new_x[0] + 3.*new_x
 plt.plot(new_x,y,'k',linewidth=1.5)
 plt.text(5.0,y[-4],'$t^3$',fontsize=16)

if dim == '3D' and label == 'm_25_1b_particles':
 print dim
 new_x = np.linspace(np.log10(50000),5.35,10)
 y = 4.3 -3.*new_x[0] + 3.*new_x
 plt.plot(new_x,y,'k',linewidth=1.5)
 plt.text(5.0,y[-4],'$t^3$',fontsize=16)

if dim == '3D' and label == 'm_25_2b_particles':
 new_x = np.linspace(np.log10(85000),5.35,10)
 y = 6.2 -3.*new_x[0] + 3.*new_x
 plt.plot(new_x,y,'k',linewidth=1.5)
 plt.text(5.1,y[-4],'$t^3$',fontsize=16)


if dim == '2D':
 p_2D, = plt.plot(x,RD_2D[:,0],color=[0,0,1],linewidth=2)
 z = 1
 p_2D5, = plt.plot(x,RD_2D[:,z],'--',color=[0,0,1],linewidth=2)
 z = 2
 p_2D17, = plt.plot(x,RD_2D[:,z],'-.',color=[0,0,1],linewidth=2)
 plt.legend((p_2D,p_2D5,p_2D17),(dim+' 5m',dim+' 10m',dim+' 15m'),bbox_to_anchor=(1., 0.023),loc=4,fontsize=14)
else:
 p_2D, = plt.plot(x,RD_2D[:,0],color=[1,0,0],linewidth=2,label=str(dim+' 5m'))
 plt.legend(bbox_to_anchor=(1., 0.023),loc=4,fontsize=14)
# z = 1
# p_2D5, = plt.plot(x,RD_2D[:,z],'--',color=[1,0,0],linewidth=2)
# z = 2
# p_2D17, = plt.plot(x,RD_2D[:,z],'-.',color=[1,0,0],linewidth=2)

plt.xlabel(r'Time $[hr]$',fontsize=20)
plt.ylabel(r'$\sigma^2_D$ $[m^2]$',fontsize=20)
plt.xlim((np.log10(1440),np.log10(24*3600*3)))
plt.ylim((2-.5,8.5))

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
ax.tick_params(axis = 'both', which = 'major', labelsize = 16, length=10)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 0, length=7)
# Suppress minor tick labels

ax.set_xticks(M_xticks)
ax.set_xticks(m_xticks, minor = True)
ax.set_xticklabels(M_xticks_labels, minor=False)

ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
#plt.yticks(fontsize=16)

M_yticks = []
M_yticks_labels = []
m_yticks = []

for i in range(2,9,1):
 M_yticks_labels.append('$10^{'+str(i)+'}$')
 M_yticks.append(i)
 if i < 9:
  for j in range(1,10,1):
   m_yticks.append(np.log10(np.power(10.,i)*j))

ax.set_yticks(M_yticks)
ax.set_yticks(m_yticks, minor = True)
ax.set_yticklabels(M_yticks_labels, minor=False)
#ax.ticklabel_format(axis = 'y', style='sci', scilimits=(0,0))

############################################################### INSET

for d in range(3):
 RD_2D_B[:,d] = RD_2D_B[:,d]/(time**3)

RD_2D = np.log10(RD_2D_B)
RD_2D = RD_2D[1:]

a = plt.axes([.29, .59, .4, .35])

if dim == '2D':
 print dim
 p_2D, = plt.plot(x,RD_2D[:,0],color=[0,0,1],linewidth=2)
 z = 1
 p_2D5, = plt.plot(x,RD_2D[:,z],'--',color=[0,0,1],linewidth=2)
 z = 2
 p_2D17, = plt.plot(x,RD_2D[:,z],'-.',color=[0,0,1],linewidth=2)
else:
 p_2D, = plt.plot(x,RD_2D[:,0],color=[1,0,0],linewidth=2,label=str(dim+' 5m'))

plt.xlabel(r'Time $[hr]$',fontsize=13)
plt.ylabel(r'$log(\sigma^2_D\,t^{-3})$ $[m^2s^{-3}]$',fontsize=13)
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
a.tick_params(axis = 'both', which = 'major', labelsize = 10, length=10)
a.tick_params(axis = 'both', which = 'minor', labelsize = 0, length=7)
# Suppress minor tick labels

a.set_xticks(M_xticks)
a.set_xticks(m_xticks, minor = True)
a.set_xticklabels(M_xticks_labels, minor=False)

a.yaxis.set_ticks_position('left')
a.xaxis.set_ticks_position('bottom')
plt.yticks(fontsize=10)

plt.tight_layout()
plt.savefig('./plot/'+label+'/RD_'+dim+'_'+label+'.eps')
print       './plot/'+label+'/RD_'+dim+'_'+label+'.eps'
plt.close()


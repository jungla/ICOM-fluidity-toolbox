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

time = time2D_B[:]

depths = [5, 10, 15] 
depthid = [1, 2, 3] 

nl = len(depths)


# relative D
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

fig, ax = plt.subplots()

RD_2D = np.log10(RD_2D_B)
RD_2D = RD_2D[1:]

x = np.log10(time-time[0])
x = x[1:]
#
# RD 2D
#new_x = np.linspace(np.log10(30000),5.2,10)
#y = 3.7 -3.*new_x[0] + 3.*new_x
#plt.plot(new_x,y,'k',linewidth=1.5)
#plt.text(4.8,y[-4],'3',fontsize=16)
#y = 3.5 -2.*new_x[0] + 2.*new_x
#plt.plot(new_x,y,'k',linewidth=1.5)
#plt.text(4.3,y[-1],'2',fontsize=16)
#y = 3.5 -1.*new_x[0] + 1.*new_x
#plt.plot(new_x,y,'k',linewidth=1.5)
#plt.text(4.3,y[-1],'1',fontsize=16)

if dim == '2D' and label == 'm_25_1b_particles':
 print dim
 new_x = np.linspace(np.log10(30000),5.2,10)
 y = 3.7 -3.*new_x[0] + 3.*new_x
 plt.plot(new_x,y,'k',linewidth=1.5)
 plt.text(4.8,y[-4],'3',fontsize=16)

if dim == '2D' and label == 'm_25_2b_particles':
 new_x = np.linspace(np.log10(30000),5.2,10)
 y = 4.6 -3.*new_x[0] + 3.*new_x
 plt.plot(new_x,y,'k',linewidth=1.5)
 plt.text(4.8,y[-4],'3',fontsize=16)

if dim == '3D' and label == 'm_25_1b_particles':
 print dim
 new_x = np.linspace(np.log10(50000),5.35,10)
 y = 4.3 -3.*new_x[0] + 3.*new_x
 plt.plot(new_x,y,'k',linewidth=1.5)
 plt.text(5.0,y[-4],'3',fontsize=16)

if dim == '3D' and label == 'm_25_2b_particles':
 new_x = np.linspace(np.log10(85000),5.35,10)
 y = 6.2 -3.*new_x[0] + 3.*new_x
 plt.plot(new_x,y,'k',linewidth=1.5)
 plt.text(5.1,y[-4],'3',fontsize=16)


if dim == '2D':
 p_2D, = plt.plot(x,RD_2D[:,0],color=[0,0,1],linewidth=2)
 z = 1
 p_2D5, = plt.plot(x,RD_2D[:,z],'--',color=[0,0,1],linewidth=2)
 z = 2
 p_2D17, = plt.plot(x,RD_2D[:,z],'-.',color=[0,0,1],linewidth=2)

 plt.legend((p_2D,p_2D5,p_2D17),(dim+' 5m',dim+' 10m',dim+' 15m'),loc=4,fontsize=16)
else:
 p_2D, = plt.plot(x,RD_2D[:,0],color=[1,0,0],linewidth=2,label=str(dim+' 5m'))
 plt.legend(loc=4,fontsize=16)
# z = 1
# p_2D5, = plt.plot(x,RD_2D[:,z],'--',color=[1,0,0],linewidth=2)
# z = 2
# p_2D17, = plt.plot(x,RD_2D[:,z],'-.',color=[1,0,0],linewidth=2)

plt.xlabel(r'Time $[hr]$',fontsize=20)
plt.ylabel(r'$log(\sigma^2_D)$ $[m^2]$',fontsize=20)
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

#ind = [72.4,84,96,144]
#vind = np.log10((np.asarray(ind)-72)*3600)

#ind = np.linspace(0,72,4)*3600
#ind[0] = 1440
#vind = np.log10(ind)

#plt.xticks(vind,['0.4','24','48','72'],fontsize=16)
#plt.xticks(vind,['72.4','96','120','144'],fontsize=16)
#plt.xticks(vind,ind,fontsize=16)

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


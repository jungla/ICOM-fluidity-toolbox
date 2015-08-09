#!~/python
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import myfun
import numpy as np
import os
import lagrangian_stats
from scipy.interpolate import interp1d
import advect_functions

# read offline
print 'reading offline'
exp = 'm_25_1'
filename0 = './traj_m_25_1_particles_0_140_RK4_2t.csv'
tt = 140 # IC + 24-48 included
xp = 40
yp = 40
zp = 25
pt = xp*yp*zp 
timet, par0 = advect_functions.read_particles_csv(filename0,xp,yp,zp,tt)

time0 = (timet)*600 + 48*3600 - 600

# read online
print 'reading online'
exp = 'm_25_1_particles'

try: os.stat('./plot/'+exp)
except OSError: os.mkdir('./plot/'+exp)

filename0 = './mli_checkpoint.detectors'
filename0 = '/tamay2/mensa/fluidity/'+exp+'/'+filename0

time1, parT = lagrangian_stats.read_particles(filename0)

dt = 600
pt = len(parT)
timei = np.asarray(range(0,tt,1))*dt + 86400*2.0 + dt
par1 = np.zeros((pt,3,len(timei)))

for p in range(len(par1)):
 f0 = interp1d(time1,parT[p,0,:])
 f1 = interp1d(time1,parT[p,1,:])
 f2 = interp1d(time1,parT[p,2,:])
 par1[p,0,:] = f0(timei)
 par1[p,1,:] = f1(timei)
 par1[p,2,:] = f2(timei)

time = timei

tt = len(time)

E = np.zeros(tt)
#
for t in range(0,tt,1):
 print 'time', time[t]
 #
 E[t] = np.sum((par0[:,0,t]-par1[:,0,t])**2 + (par0[:,1,t]-par1[:,1,t])**2 + (par0[:,2,t]-par1[:,2,t])**2) 

plt.plot(time[:-1]/3600,E[:-1],linewidth=2)
plt.xlim((time[0]/3600,time[-1]/3600))
plt.xlabel('time')
plt.ylabel('RMS')
plt.savefig('./plot/'+exp+'/E_'+exp+'.eps')
print       './plot/'+exp+'/E_'+exp+'.eps'
plt.close()

#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import myfun
import numpy as np
import os
import lagrangian_stats
import advect_functions
from scipy import interpolate
import csv
import advect_functions

# read offline
print 'reading particles'

exp = 'm_25_1b'
label = 'm_25_1b'
filename2D_BW = 'traj_m_25_2b_particles_0_500_2D.csv'
filename3D_BW = 'traj_m_25_2b_particles_0_500_3D.csv'
filename2D_B = 'traj_m_25_1b_particles_0_600_2D.csv'
filename3D_B = 'traj_m_25_1b_particles_0_600_3D.csv'
tt_BW = 600 # IC + 24-48 included
tt_B = 600 # IC + 24-48 included

Xlist_BW = np.linspace(0,10000,801)
Ylist_BW = np.linspace(0,4000,321)
Xlist_B = np.linspace(0,8000,641)
Ylist_B = np.linspace(0,8000,641)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = 1.*np.cumsum(dl)

xp = 101
yp = 101
zp = 4
pt = xp*yp*zp
          
time2D_B, par2D_B = advect_functions.read_particles_csv(filename2D_B,pt,tt_B)
par2D_B = lagrangian_stats.periodicCoords(par2D_B,8000,8000)
time3D_B, par3D_B = advect_functions.read_particles_csv(filename3D_B,pt,tt_B)
par3D_B = lagrangian_stats.periodicCoords(par3D_B,8000,8000)

time2D_B = (time2D_B)*1440
time3D_B = (time3D_B)*1440 

time2D_BW, par2D_BW = advect_functions.read_particles_csv(filename2D_BW,pt,tt_BW)
par2D_BW = lagrangian_stats.periodicCoords(par2D_BW,10000,4000)
time3D_BW, par3D_BW = advect_functions.read_particles_csv(filename3D_BW,pt,tt_BW)
par3D_BW = lagrangian_stats.periodicCoords(par3D_BW,10000,4000)

time2D_BW = (time2D_BW)*1440
time3D_BW = (time3D_BW)*1440
   
time = time2D_BW[:-1]

depths = [5, 10, 15] 
depthid = [1, 2, 3] 

nl = len(depths)

RD_2D_BW = np.zeros((tt_BW,nl))
RD_3D_BW = np.zeros((tt_BW,nl))
RD_2D_B = np.zeros((tt_B,nl))
RD_3D_B = np.zeros((tt_B,nl))

#drate = np.zeros((61,36))
par3Dzi_BW = np.zeros((len(depths),xp*yp,3,tt_BW))
par2Dzi_BW = np.zeros((len(depths),xp*yp,3,tt_BW))
par3Dzi_B = np.zeros((len(depths),xp*yp,3,tt_B))
par2Dzi_B = np.zeros((len(depths),xp*yp,3,tt_B))

for z in range(len(depths)):
 print z
 print 'depth', depths[z]
 #B
 par2Dz = np.reshape(par2D_B,(xp,yp,zp,3,tt_B))
 par3Dz = np.reshape(par3D_B,(xp,yp,zp,3,tt_B))
 par2Dzr = par2Dz[:,:,depthid[z],:,:]
 par3Dzr = par3Dz[:,:,depthid[z],:,:]
 par2Dz = np.reshape(par2Dzr,(xp*yp,3,tt_B))
 par3Dz = np.reshape(par3Dzr,(xp*yp,3,tt_B))
 RD_2D_B[:,z] = lagrangian_stats.RD_t(par2Dzr,tt_B,xp-1)
 RD_3D_B[:,z] = lagrangian_stats.RD_t(par3Dzr,tt_B,xp-1)
 par3Dzi_B[z,:,:,:] = par3Dz[:,:,:]
 par2Dzi_B[z,:,:,:] = par2Dz[:,:,:]
 #BW
 par2Dz = np.reshape(par2D_BW,(xp,yp,zp,3,tt_BW))
 par3Dz = np.reshape(par3D_BW,(xp,yp,zp,3,tt_BW))
 par2Dzr = par2Dz[:,:,depthid[z],:,:]
 par3Dzr = par3Dz[:,:,depthid[z],:,:]
 par2Dz = np.reshape(par2Dzr,(xp*yp,3,tt_BW))
 par3Dz = np.reshape(par3Dzr,(xp*yp,3,tt_BW))
 RD_2D_BW[:,z] = lagrangian_stats.RD_t(par2Dzr,tt_BW,xp-1)
 RD_3D_BW[:,z] = lagrangian_stats.RD_t(par3Dzr,tt_BW,xp-1)
 par3Dzi_BW[z,:,:,:] = par3Dz[:,:,:]
 par2Dzi_BW[z,:,:,:] = par2Dz[:,:,:]


# cut particles to time of interest

timeD = np.asarray(range(0,86400,1440))
vtime = time - time[0]

ttime = vtime[(vtime > 0.2*86400) * (vtime < 86400)]
RD_2D_B = RD_2D_B[(vtime > 0.2*86400) * (vtime < 86400),:]
RD_3D_B = RD_3D_B[(vtime > 0.2*86400) * (vtime < 86400),:]
RD_2D_BW = RD_2D_BW[(vtime > 0.2*86400) * (vtime < 86400),:]
RD_3D_BW = RD_3D_BW[(vtime > 0.2*86400) * (vtime < 86400),:]

# read 3D eps and get eps at particle's location

drateD_BW = np.zeros((len(timeD),len(Zlist)))
drateD_B = np.zeros((len(timeD),len(Zlist)))

for t in range(len(timeD)):
 print 'read drate', t
 drate = np.zeros((len(Xlist_BW),len(Ylist_BW),len(Zlist)))
 # read
 with open('../../2D/U/drate_m_25_2_512_'+str(t)+'_3D.csv', 'rb') as csvfile:
  spamreader = csv.reader(csvfile)
  j = 0; k = 0
  for row in spamreader:
   j = j + 1
   if j == len(Ylist_BW): k = k + 1; j = 0
   if k == len(Zlist): k = 0
   drate[:,j,k] = row[::-1]  

 drateD_BW[t,:] = np.mean(np.mean(drate,0),0)

for t in range(len(timeD)):
 print 'read drate', t
 drate = np.zeros((len(Xlist_B),len(Ylist_B),len(Zlist)))
 # read
 with open('../../2D/U/drate_m_25_1b_particles_'+str(t)+'_3D.csv', 'rb') as csvfile:
  spamreader = csv.reader(csvfile)
  j = 0; k = 0
  for row in spamreader:
   j = j + 1
   if j == len(Ylist_B): k = k + 1; j = 0
   if k == len(Zlist): k = 0
   drate[:,j,k] = row[::-1]   

# epsilon averaged over xy (?)
 drateD_B[t,:] = np.mean(np.mean(drate,0),0)

# averaged over time as well...
drateDt_BW = np.mean(drateD_BW[(vtime > 0.2*86400) * (vtime < 86400)],0)
drateDt_B = np.mean(drateD_B[(vtime > 0.2*86400) * (vtime < 86400)],0)

# compute surface forcing


# Rich 2D-3D

fig = plt.figure(figsize=(8, 6))

ax1 = plt.subplot()

Rich = RD_2D_BW[:,0]/ttime**3/drateDt_BW[depths[0]]
print '2D 5m: mean', np.mean(Rich), 'std', np.std(Rich)
R2D1_BW, = ax1.semilogx(ttime,Rich,'r',linewidth=2)

Rich = RD_3D_BW[:,0]/ttime**3/drateDt_BW[depths[0]]
print '3D 5m: mean', np.mean(Rich), 'std', np.std(Rich)
R3D1_BW, = ax1.semilogx(ttime,Rich,'r--',linewidth=2)

Rich = RD_2D_B[:,0]/ttime**3/drateDt_B[depths[0]]
print '2D 5m: mean', np.mean(Rich), 'std', np.std(Rich)
R2D1_B, = ax1.semilogx(ttime,Rich,'b',linewidth=2)

Rich = RD_3D_B[:,0]/ttime**3/drateDt_B[depths[0]]
print '3D 5m: mean', np.mean(Rich), 'std', np.std(Rich)
R3D1_B, = ax1.semilogx(ttime,Rich,'b--',linewidth=2)


for tic in ax1.xaxis.get_minor_ticks():
    tic.tick1On = tic.tick2On = False

#plt.legend((R2D1,R3D1,R2D5,R3D5,R2D17,R3D17),('2D 5m','3D 5m','2D 10m','3D 10m','2D 15m','3D 15m'),loc=3,fontsize=16,ncol=3)
plt.legend((R2D1_BW,R3D1_BW,R2D1_B,R3D1_B),('$BW25_m$ 2D','$BW25_m$ 3D','$B25_m$ 2D','$B25_m$ 3D'),loc=2,fontsize=16,ncol=2)

plt.xlabel('Time $[hr]$',fontsize=20)
plt.ylabel('$\sigma^2_D t^{-3} \epsilon^{-1}$ ',fontsize=20)

plt.xlim((ttime[0],ttime[-1]))
plt.xticks((ttime[0],6*3600,12*3600.,18*3600,ttime[-1]),(ttime[0]/3600.+48,6+48,12+48,18+48,ttime[-1]/3600.+48),fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('./plot/'+label+'/Rich_23_BW_'+label+'.eps')
print       './plot/'+label+'/Rich_23_BW_'+label+'.eps'
plt.close()



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

dim = '3D'
label = 'm_25_2b_particles'
filename3D = './traj/traj_'+label+'_60_500_'+dim+'_big.csv'
tt = 500-60 # IC + 24-48 included

x0 = range(0,5010,10)
y0 = range(0,5010,10)
#y0 = range(0,4010,10)
z0 = [0,5,10,15]


Xlist = np.linspace(0,8000,641)
Ylist = np.linspace(0,8000,641)
#Xlist = np.linspace(0,2000,161)
#Ylist = np.linspace(0,2000,161)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = 1.*np.cumsum(dl)

maps = [Xlist,Ylist,Zlist]

lo = np.array([ 0, 0, 0])
hi = np.array([ 8000, 8000, 50])

xp = len(x0)
yp = len(y0)
zp = len(z0)
pt = xp*yp*zp
          
time3D, par3D = advect_functions.read_particles_csv(filename3D,pt,tt)
par3D = lagrangian_stats.periodicCoords(par3D,8000,8000)

time3D = (time3D)*1440 
    
time = time3D[:-1]

depths = [5, 10, 15] 
depthid = [1, 2, 3] 

nl = len(depths)

RD_3D = np.zeros((tt,nl))

#drate = np.zeros((61,36))
par3Dzi = np.zeros((len(depths),xp*yp,3,tt))

for z in range(len(depths)):
 print z
 print 'depth', depths[z]
 par3Dz = np.reshape(par3D,(xp,yp,zp,3,tt))
 par3Dzr = par3Dz[:,:,depthid[z],:,:]
 #
 par3Dz = np.reshape(par3Dzr,(xp*yp,3,tt))
 #
 RD_3D[:,z] = lagrangian_stats.RD_t(par3Dzr,tt,xp-1)
 #
 par3Dzi[z,:,:,:] = par3Dz[:,:,:]

# cut particles to time of interest

timeD = np.asarray(range(0,86400,1440))
vtime = time - time[0]

ttime = vtime[(vtime > 0.2*86400) * (vtime < 86400)]
RD_3D = RD_3D[(vtime > 0.2*86400) * (vtime < 86400),:]

# read 3D eps and get eps at particle's location

drateP3D = np.zeros((len(timeD),len(depths)))

for t in range(len(timeD)):
 print 'read drate', t
 drate = np.zeros((len(Xlist),len(Ylist),len(Zlist)))
 # read
 with open('../../2D/U/drate_'+label+'_'+str(t)+'_3D.csv', 'rb') as csvfile:
  spamreader = csv.reader(csvfile)
  j = 0; k = 0
  for row in spamreader:
   j = j + 1
   if j == len(Ylist): k = k + 1; j = 0
   if k == len(Zlist): k = 0
   drate[:,j,k] = row[::-1]  

 # plot some values
#plt.contour(Ylist/1000,Xlist/1000,np.log10(drate[:,:,0]),10)
#plt.colorbar()
#plt.axes().set_aspect('equal')
#plt.savefig('./plot/'+label+'/drate_0_'+label+'_'+str(t)+'.eps')
#print       './plot/'+label+'/drate_0_'+label+'_'+str(t)+'.eps'
#plt.close()

 # compute epsilon at particle position, averaged over all particles

 for d in range(len(depths)):
  # interpolation
  x = par3Dzi[d,:,0,t]
  y = par3Dzi[d,:,1,t]
  z = -par3Dzi[d,:,2,t]
  drateP3D[t,d] = np.mean(advect_functions.interp(x,y,z,lo,hi,maps,drate[:,:,:]))

drateP3Dt = drateP3D[(vtime > 0.2*86400) * (vtime < 86400),:]
# compute surface forcing

def forcing(time):
 if time > 0:
  t = time/3600.0%24/6
  if t >= 0 and t < 2:
   Q_0 = 0
  if t >= 2 and t < 3:
   Q_0 = (t-2)
  if t >= 3 and t < 4:
   Q_0 = 1 - (t-3)
 else:
  Q_0 = 0
# print Q_0
 Q = Q_0
 return Q

flux = []
for t in ttime:
 flux.append(forcing(t))

# Rich 3D
from matplotlib import gridspec

fig = plt.figure(figsize=(8, 4.5)) 
#gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4]) 

#ax0 = plt.subplot(gs[0])
#ax0.semilogx(ttime,flux,'k',linewidth=2)

#for tic in ax0.xaxis.get_minor_ticks():
#    tic.tick1On = tic.tick2On = False

#plt.xlim((ttime[0],ttime[-1]))
#plt.ylabel('$Q_0$ $[kWm^{2}]$',fontsize=16)

#plt.xticks((ttime[0],6*3600,12*3600.,18*3600,ttime[-1]),(ttime[0]/3600.,6,12,18,ttime[-1]/3600.),fontsize=16)

ax1 = plt.subplot()

Rich = RD_3D[:,0]/ttime**3/drateP3Dt[:,0]
print '3D 5m: mean', np.mean(Rich), 'std', np.std(Rich)
R3D1, = ax1.loglog(ttime,Rich,'k',linewidth=2)

Rich = RD_3D[:,1]/ttime**3/drateP3Dt[:,1]
print '3D 10m: mean', np.mean(Rich), 'std', np.std(Rich)
R3D5, = ax1.loglog(ttime,Rich,'k--',linewidth=2)

Rich = RD_3D[:,2]/ttime**3/drateP3Dt[:,2]
print '3D 15m: mean', np.mean(Rich), 'std', np.std(Rich)
R3D17, = ax1.loglog(ttime,Rich,'k-.',linewidth=2)

#plt.legend((R2D1,R3D1,R2D5,R3D5,R2D17,R3D17),('2D 5m','3D 5m','2D 10m','3D 10m','2D 15m','3D 15m'),loc=3,fontsize=16,ncol=3)
for tic in ax1.xaxis.get_minor_ticks():
    tic.tick1On = tic.tick2On = False

plt.legend((R3D1,R3D5,R3D17),('5m','10m','15m'),loc=2,fontsize=16,ncol=3)
plt.xlabel('Time $[hr]$',fontsize=20)
plt.ylabel('$\sigma^2_D t^{-3} \epsilon^{-1}$ ',fontsize=20)

#plt.ylim((10**-2,10**1))
plt.xlim((ttime[0],ttime[-1]))
plt.yticks(fontsize=16)
plt.xticks((ttime[0],6*3600,12*3600.,18*3600,ttime[-1]),(ttime[0]/3600.+48,6+48,12+48,18+48,ttime[-1]/3600.+48),fontsize=16)
plt.tight_layout()
plt.savefig('./plot/'+label+'/Rich_'+dim+'_'+label+'.eps')
print       './plot/'+label+'/Rich_'+dim+'_'+label+'.eps'
plt.close()


# EPS 3D

R3D1, = plt.loglog(ttime,drateP3Dt[:,0],'k',linewidth=2)
R3D5, = plt.loglog(ttime,drateP3Dt[:,1],'k--',linewidth=2)
R3D17, = plt.loglog(ttime,drateP3Dt[:,2],'k-.',linewidth=2)

plt.xlabel('Time $[hr]$',fontsize=20)
plt.ylabel('$\epsilon^{-1}$ ',fontsize=20)

#plt.ylim((10**-3,10**1))
plt.xlim((ttime[0],ttime[-1]))
plt.xticks((ttime[0],6*3600,12*3600.,18*3600,ttime[-1]),(ttime[0]/3600.,6,12,18,ttime[-1]/3600.),fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('./plot/'+label+'/Eps_'+dim+'_'+label+'.eps')
print       './plot/'+label+'/Eps_'+dim+'_'+label+'.eps'
plt.close()


# RD 3D

R3D1, = plt.loglog(ttime,RD_3D[:,0],'k',linewidth=2)
R3D5, = plt.loglog(ttime,RD_3D[:,1],'k--',linewidth=2)
R3D17, = plt.loglog(ttime,RD_3D[:,2],'k-.',linewidth=2)

plt.xlabel('Time $[hr]$',fontsize=20)
plt.ylabel('$\sigma^2_D$ ',fontsize=20)

#plt.ylim((10**-3,10**1))
plt.xlim((ttime[0],ttime[-1]))
#plt.xticks((ttime[0],6*3600,12*3600.,18*3600,ttime[-1]),(ttime[0]/3600.,6,12,18,ttime[-1]/3600.),fontsize=16)
plt.xticks((ttime[0],6*3600,12*3600.,18*3600,ttime[-1]),(ttime[0]/3600.+48,6+48,12+48,18+48,ttime[-1]/3600.+48),fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('./plot/'+label+'/RD_'+dim+'_'+label+'.eps')
print       './plot/'+label+'/RD_'+dim+'_'+label+'.eps'
plt.close()


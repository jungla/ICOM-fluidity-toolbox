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

# read offline
print 'reading offline'

exp = 'm_25_2'
label = 'm_25_2'
filename2D = 'traj_m_25_2_512_0_500_2D.csv'
filename3D = 'traj_m_25_2_512_0_300_3D_big.csv'
tt = 300 # IC + 24-48 included

#x0 = range(3000,4010,10)
#y0 = range(2000,3010,10)
z0 = [0,5,10,15] #range(1,20,4)
x0 = range(0000,6010,10)
y0 = range(0000,3010,10)


xp = len(x0)
yp = len(y0)
zp = len(z0)
pt = xp*yp*zp
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
Zlist = -1*np.cumsum(dl)
          
#time2D, par2D = advect_functions.read_particles_csv(filename2D,pt,tt)
#par2D = lagrangian_stats.periodicCoords(par2D,10000,4000)
#time3D, par3D = advect_functions.read_particles_csv(filename3D,pt,tt)
#par3D = lagrangian_stats.periodicCoords(par3D,10000,4000)

#time2D = (time2D)*1440
#time3D = (time3D)*1440 
#    
time = time2D[:-1]

depths = [5, 10, 15] 
depthid = [1, 2, 3] 

nl = len(depths)

RD_2D = np.zeros((tt,nl))
RD_3D = np.zeros((tt,nl))

#def RD_t(par2Dz,tt):
# pt = len(par2Dz)
## RD_2D=[]
# RD_2D = np.zeros((pt,tt))
# for i in xrange(pt):
#  print i,pt
#  for j in xrange(i,pt-1):
##   print np.mean(((par2Dz[i,0,:] - par2Dz[j,0,:])**2 + (par2Dz[i,1,:] - par2Dz[j,1,:])**2),0)
#   RD_2D[i,:] = (par2Dz[i,0,:] - par2Dz[j,0,:])**2 + (par2Dz[i,1,:] - par2Dz[j,1,:])**2
## print RD_2D
# return np.mean(RD_2D,0)

par3Dzi = np.zeros((len(depths),xp*yp,tt))
par2Dzi = np.zeros((len(depths),xp*yp,tt))
drate = np.zeros((61,36))

for z in range(len(depths)):
 print z
 print 'depth', depths[z]
 par2Dz = np.reshape(par2D,(xp,yp,zp,3,tt))
 par3Dz = np.reshape(par3D,(xp,yp,zp,3,tt))
 par2Dzr = par2Dz[:,:,depthid[z],:,:]
 par3Dzr = par3Dz[:,:,depthid[z],:,:]
 #
 par2Dz = np.reshape(par2Dzr,(xp*yp,3,tt))
 par3Dz = np.reshape(par3Dzr,(xp*yp,3,tt))
 #
 RD_2D[:,z] = lagrangian_stats.RD_t(par2Dzr,tt,xp-1)
 RD_3D[:,z] = lagrangian_stats.RD_t(par3Dzr,tt,xp-1)
 #
 ## RICHARDSON Plot
 par3Dzi[z,:,:] = par3Dz[:,2,:]
 par2Dzi[z,:,:] = par2Dz[:,2,:]

#read eps for scaling

timeD = []

with open('drate_m_25_2_512.csv', 'rb') as csvfile:
 spamreader = csv.reader(csvfile)
 t = 0
 for row in spamreader:
  t = t + 1
  if t > 1:
   timeD.append(float(row[0]))
   drate[t-2,:] = row[1:]  

timeD = np.asarray(timeD)
timeD = timeD - timeD[0]

vtime = time - time[0]

ttime = vtime[(vtime > 0.2*86400) * (vtime < 86400)]
RD_2D = RD_2D[(vtime > 0.2*86400) * (vtime < 86400),:]
RD_3D = RD_3D[(vtime > 0.2*86400) * (vtime < 86400),:]

#vtime = vtime[5:]

drateT = np.zeros((len(ttime),len(Zlist)))
drateP3D = np.zeros((len(ttime),len(depths)))
drateP2D = np.zeros((len(ttime),len(depths)))

for d in range(len(Zlist)):
 f = interpolate.interp1d(timeD, drate[:,d])
 drateT[:,d] = f(ttime)   # use interpolation function returned by `interp1d`

# plot drate with 3D particles average location...

plt.contourf(ttime,np.sort(Zlist),np.rot90(drateT),30)
plt.colorbar()
for k in range(len(depths)):
 plt.plot(ttime,np.mean(par3Dzi[k,:,(vtime > 0.2*86400) * (vtime < 86400)],1),'k',linewidth=2)
 plt.plot(ttime,np.mean(par3Dzi[k,:,(vtime > 0.2*86400) * (vtime < 86400)],1)+np.std(par3Dzi[k,:,(vtime > 0.2*86400) * (vtime < 86400)],1),'--k',linewidth=2)
 plt.plot(ttime,np.mean(par3Dzi[k,:,(vtime > 0.2*86400) * (vtime < 86400)],1)-np.std(par3Dzi[k,:,(vtime > 0.2*86400) * (vtime < 86400)],1),'--k',linewidth=2)

plt.tight_layout()
plt.savefig('./plot/'+label+'/drate_3_trajM_'+label+'.eps')
print       './plot/'+label+'/drate_3_trajM_'+label+'.eps'
plt.close()

# plot some trajs in the vertical

plt.contourf(ttime,np.sort(Zlist),np.rot90(drateT),30)

plt.colorbar()
k = 1
plt.plot(ttime,par3Dzi[k,0:xp*yp:100,(vtime > 0.2*86400) * (vtime < 86400)],'k')

plt.tight_layout()
plt.savefig('./plot/'+label+'/drate_3_traj_'+label+'.eps')
print       './plot/'+label+'/drate_3_traj_'+label+'.eps'
plt.close()


# drate for 3D particles is interpolated at particle depth

for z in range(len(depths)):
 par3Di = par3Dzi[z,:,(vtime > 0.2*86400) * (vtime < 86400)]
 par2Di = par2Dzi[z,:,(vtime > 0.2*86400) * (vtime < 86400)]

 for t in range(len(ttime)):
  dratePt = 0
  f = interpolate.interp1d(Zlist,drateT[t,:])
  drateP3D[t,z] = np.mean(f(par3Di[t,:]))  # use interpolation function returned by `interp1d`
  drateP2D[t,z] = np.mean(f(par2Di[t,:]))  # use interpolation function returned by `interp1d`



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

Rich = RD_3D[:,0]/ttime**3/drateP3D[:,0]
print '3D 5m: mean', np.mean(Rich), 'std', np.std(Rich)
R3D1, = ax1.loglog(ttime,Rich,'k',linewidth=2)

Rich = RD_3D[:,1]/ttime**3/drateP3D[:,1]
print '3D 10m: mean', np.mean(Rich), 'std', np.std(Rich)
R3D5, = ax1.loglog(ttime,Rich,'k--',linewidth=2)

Rich = RD_3D[:,2]/ttime**3/drateP3D[:,2]
print '3D 15m: mean', np.mean(Rich), 'std', np.std(Rich)
R3D17, = ax1.loglog(ttime,Rich,'k-.',linewidth=2)

#plt.legend((R2D1,R3D1,R2D5,R3D5,R2D17,R3D17),('2D 5m','3D 5m','2D 10m','3D 10m','2D 15m','3D 15m'),loc=3,fontsize=16,ncol=3)
for tic in ax1.xaxis.get_minor_ticks():
    tic.tick1On = tic.tick2On = False

plt.legend((R3D1,R3D5,R3D17),('5m','10m','15m'),loc=2,fontsize=16,ncol=3)
plt.xlabel('Time $[hr]$',fontsize=20)
plt.ylabel('$\sigma^2_D t^{-3} \epsilon^{-1}$ ',fontsize=20)

plt.ylim((10**-2,10**1))
plt.xlim((ttime[0],ttime[-1]))
plt.yticks(fontsize=16)
plt.xticks((ttime[0],6*3600,12*3600.,18*3600,ttime[-1]),(ttime[0]/3600.+48,6+48,12+48,18+48,ttime[-1]/3600.+48),fontsize=16)
plt.tight_layout()
plt.savefig('./plot/'+label+'/Rich_3_'+label+'.eps')
print       './plot/'+label+'/Rich_3_'+label+'.eps'
plt.close()

# Rich 2D

fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])

ax0 = plt.subplot(gs[0])
ax0.semilogx(ttime,flux,'k',linewidth=2)

for tic in ax0.xaxis.get_minor_ticks():
    tic.tick1On = tic.tick2On = False

#plt.xticks((ttime[0],6*3600,12*3600.,18*3600,ttime[-1]),(ttime[0]/3600.,6,12,18,ttime[-1]/3600.),fontsize=16)
plt.xticks((ttime[0],6*3600,12*3600.,18*3600,ttime[-1]),(ttime[0]/3600.+48,6+48,12+48,18+48,ttime[-1]/3600.+48),fontsize=16)
plt.xlim((ttime[0],ttime[-1]))
plt.ylabel('$Q_0$ $[kWm^{2}]$',fontsize=16)
#ax0.set_title('Sharing X axis')

ax1 = plt.subplot(gs[1])

Rich = RD_2D[:,0]/ttime**3/drateP2D[:,0]
print '2D 5m: mean', np.mean(Rich), 'std', np.std(Rich)
R3D1, = ax1.loglog(ttime,Rich,'k',linewidth=2)

Rich = RD_2D[:,1]/ttime**3/drateP2D[:,1]
print '2D 10m: mean', np.mean(Rich), 'std', np.std(Rich)
R3D5, = ax1.loglog(ttime,Rich,'k--',linewidth=2)

Rich = RD_2D[:,2]/ttime**3/drateP2D[:,2]
print '2D 15m: mean', np.mean(Rich), 'std', np.std(Rich)
R3D17, = ax1.loglog(ttime,Rich,'k-.',linewidth=2)

for tic in ax1.xaxis.get_minor_ticks():
    tic.tick1On = tic.tick2On = False

#plt.legend((R2D1,R3D1,R2D5,R3D5,R2D17,R3D17),('2D 5m','3D 5m','2D 10m','3D 10m','2D 15m','3D 15m'),loc=3,fontsize=16,ncol=3)
plt.legend((R3D1,R3D5,R3D17),('5m','10m','15m'),loc=2,fontsize=16,ncol=3)

plt.xlabel('Time $[hr]$',fontsize=20)
plt.ylabel('$\sigma^2_D t^{-3} \epsilon^{-1}$ ',fontsize=20)

plt.ylim((10**-2,10**1))
plt.xlim((ttime[0],ttime[-1]))
#plt.xticks((ttime[0],6*3600,12*3600.,18*3600,ttime[-1]),(ttime[0]/3600.,6,12,18,ttime[-1]/3600.),fontsize=16)
plt.xticks((ttime[0],6*3600,12*3600.,18*3600,ttime[-1]),(ttime[0]/3600.+48,6+48,12+48,18+48,ttime[-1]/3600.+48),fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('./plot/'+label+'/Rich_2_'+label+'.eps')
print       './plot/'+label+'/Rich_2_'+label+'.eps'
plt.close()


# EPS 3D

R3D1, = plt.loglog(ttime,drateP3D[:,0],'k',linewidth=2)
R3D5, = plt.loglog(ttime,drateP3D[:,1],'k--',linewidth=2)
R3D17, = plt.loglog(ttime,drateP3D[:,2],'k-.',linewidth=2)

plt.xlabel('Time $[hr]$',fontsize=20)
plt.ylabel('$\epsilon^{-1}$ ',fontsize=20)

#plt.ylim((10**-3,10**1))
plt.xlim((ttime[0],ttime[-1]))
plt.xticks((ttime[0],6*3600,12*3600.,18*3600,ttime[-1]),(ttime[0]/3600.,6,12,18,ttime[-1]/3600.),fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('./plot/'+label+'/Eps_3_'+label+'.eps')
print       './plot/'+label+'/Eps_3_'+label+'.eps'
plt.close()


# EPS 2D

R3D1, = plt.loglog(ttime,drateP2D[:,0],'k',linewidth=2)
R3D5, = plt.loglog(ttime,drateP2D[:,1],'k--',linewidth=2)
R3D17, = plt.loglog(ttime,drateP2D[:,2],'k-.',linewidth=2)

plt.xlabel('Time $[hr]$',fontsize=20)
plt.ylabel('$\epsilon^{-1}$ ',fontsize=20)

#plt.ylim((10**-3,10**1))
plt.xlim((ttime[0],ttime[-1]))
plt.xticks((ttime[0],6*3600,12*3600.,18*3600,ttime[-1]),(ttime[0]/3600.,6,12,18,ttime[-1]/3600.),fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('./plot/'+label+'/Eps_2_'+label+'.eps')
print       './plot/'+label+'/Eps_2_'+label+'.eps'
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
plt.savefig('./plot/'+label+'/RD_3_'+label+'.eps')
print       './plot/'+label+'/RD_3_'+label+'.eps'
plt.close()

# RD 2D

R3D1, = plt.loglog(ttime,RD_2D[:,0],'k',linewidth=2)
R3D5, = plt.loglog(ttime,RD_2D[:,1],'k--',linewidth=2)
R3D17, = plt.loglog(ttime,RD_2D[:,2],'k-.',linewidth=2)

plt.xlabel('Time $[hr]$',fontsize=20)
plt.ylabel('$\sigma^2_D$ ',fontsize=20)

#plt.ylim((10**-3,10**1))
plt.xlim((ttime[0],ttime[-1]))
plt.xticks((ttime[0],6*3600,12*3600.,18*3600,ttime[-1]),(ttime[0]/3600.,6,12,18,ttime[-1]/3600.),fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('./plot/'+label+'/RD_2_'+label+'.eps')
print       './plot/'+label+'/RD_2_'+label+'.eps'
plt.close()




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

def RD_t(par2Dzr,tt,px,py):
 RD_2Dm = [] #np.zeros((px+py,tt))
 for i in range(px):
  RD_2Dm.append(np.mean(((par2Dzr[i+1,:,0,:] - par2Dzr[i,:,0,:])**2 + (par2Dzr[i+1,:,1,:] - par2Dzr[i,:,1,:])**2),0))
 for j in range(py):
  RD_2Dm.append(np.mean(((par2Dzr[:,j+1,0,:] - par2Dzr[:,j,0,:])**2 + (par2Dzr[:,j+1,1,:] - par2Dzr[:,j,1,:])**2),0))
 return np.mean(RD_2Dm,0)


# read offline
print 'reading particles'

exp = 'm_25_1b'
label = 'm_25_1b'
filename2D_BW = 'traj_m_25_2_512_0_500_2D_big.csv'
filename2D_B = 'traj_m_25_1b_particles_0_500_2D_big.csv'
tt_BW = 500 # IC + 24-48 included
tt_B = 500 # IC + 24-48 included

Xlist_BW = np.linspace(0,10000,801)
Ylist_BW = np.linspace(0,4000,321)
Xlist_B = np.linspace(0,8000,641)
Ylist_B = np.linspace(0,8000,641)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = 1.*np.cumsum(dl)

x0 = range(0,7010,10)
y0 = range(0,4010,10)
z0 = [0,5,10,15]

xp_B = len(x0)
yp_B = len(y0)
zp_B = len(z0)
pt_B = xp_B*yp_B*zp_B
    
x0 = range(0,9010,10)
y0 = range(0,3010,10)
z0 = [0,5,10,15]

xp_BW = len(x0)
yp_BW = len(y0)
zp_BW = len(z0)
pt_BW = xp_BW*yp_BW*zp_BW

#time2D_B, par2D_B = advect_functions.read_particles_csv(filename2D_B,pt_B,tt_B)
#par2D_B = lagrangian_stats.periodicCoords(par2D_B,8000,8000)
#time2D_B = (time2D_B)*1440
##
#time2D_BW, par2D_BW = advect_functions.read_particles_csv(filename2D_BW,pt_BW,tt_BW)
#par2D_BW = lagrangian_stats.periodicCoords(par2D_BW,10000,4000)
#time2D_BW = (time2D_BW)*1440

time = time2D_BW[:-1]

depths = [5, 10, 15] 
depthid = [1, 2, 3] 

nl = len(depths)

RD_2D_BW = np.zeros((tt_BW,nl))
RD_2D_B = np.zeros((tt_B,nl))

#drate = np.zeros((61,36))

for z in range(len(depths)):
 print z
 print 'depth', depths[z]
 #B
 par2Dz = np.reshape(par2D_B,(xp_B,yp_B,zp_B,3,tt_B))
 par2Dzr = par2Dz[:,:,depthid[z],:,:]
 par2Dz = np.reshape(par2Dzr,(xp_B*yp_B,3,tt_B))
 RD_2D_B[:,z] = RD_t(par2Dzr,tt_B,xp_B-1,yp_B-1)
 #BW
 par2Dz = np.reshape(par2D_BW,(xp_BW,yp_BW,zp_BW,3,tt_BW))
 par2Dzr = par2Dz[:,:,depthid[z],:,:]
 par2Dz = np.reshape(par2Dzr,(xp_BW*yp_BW,3,tt_BW))
 RD_2D_BW[:,z] = RD_t(par2Dzr,tt_BW,xp_BW-1,yp_BW-1)

#del par2D_B, par2D_BW

# cut particles to time of interest

timeD = np.asarray(range(0,86400,1440))
vtime = time - time[0]


# read 3D eps and get eps at particle's location

drateD_BW = np.zeros((len(timeD),len(Zlist)))
drateD_B = np.zeros((len(timeD),len(Zlist)))

for t in range(len(timeD)):
 print 'read drate', t
 drate = np.zeros((len(Xlist_BW),len(Ylist_BW),len(Zlist)))
 # read
 with open('../../2D/U/drate/drate_m_25_2_512_'+str(t)+'_3D.csv', 'rb') as csvfile:
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
 with open('../../2D/U/drate/drate_m_25_1b_particles_'+str(t)+'_3D.csv', 'rb') as csvfile:
  spamreader = csv.reader(csvfile)
  j = 0; k = 0
  for row in spamreader:
   j = j + 1
   if j == len(Ylist_B): k = k + 1; j = 0
   if k == len(Zlist): k = 0
   drate[:,j,k] = row[::-1]   

 drateD_B[t,:] = np.mean(np.mean(drate,0),0)

drateDt_BW = np.mean(drateD_BW[(vtime > 0.5*86400) * (vtime < 86400)],0)
drateDt_B = np.mean(drateD_B[(vtime > 0.5*86400) * (vtime < 86400)],0)
ttime = vtime[(vtime > 0.5*86400) * (vtime < 86400)]
RD_2D_Bt = RD_2D_B[(vtime > 0.5*86400) * (vtime < 86400),:]
RD_2D_BWt = RD_2D_BW[(vtime > 0.5*86400) * (vtime < 86400),:]

# normalized RD
fig = plt.figure(figsize=(8, 6))

R2D5_B, = plt.loglog(time2D_B[:-1],RD_2D_B[:,0]/time2D_B[:-1]**3,'b',linewidth=1)
R2D10_B, = plt.loglog(time2D_B[:-1],RD_2D_B[:,1]/time2D_B[:-1]**3,'b--',linewidth=1)
R2D15_B, = plt.loglog(time2D_B[:-1],RD_2D_B[:,2]/time2D_B[:-1]**3,'b-.',linewidth=1)
R2D5_BW, = plt.loglog(time2D_BW[:-1],RD_2D_BW[:,0]/time2D_BW[:-1]**3,'r',linewidth=1)
R2D10_BW, = plt.loglog(time2D_BW[:-1],RD_2D_BW[:,1]/time2D_BW[:-1]**3,'r--',linewidth=1)
R2D15_BW, = plt.loglog(time2D_BW[:-1],RD_2D_BW[:,2]/time2D_BW[:-1]**3,'r-.',linewidth=1)

#B
intm = 0.5*86400; intM = 1.5*86400; interval = (vtime > intm) * (vtime < intM)
R2D5_B, = plt.loglog(time2D_B[interval],RD_2D_B[interval,0]/time2D_B[interval]**3,'b',linewidth=3.5)

intm = 0.6*86400; intM = 2*86400; interval = (vtime > intm) * (vtime < intM)
R2D10_B, = plt.loglog(time2D_B[interval],RD_2D_B[interval,1]/time2D_B[interval]**3,'b--',linewidth=3.5)

intm = 0.7*86400; intM = 3*86400; interval = (vtime > intm) * (vtime < intM)
R2D15_B, = plt.loglog(time2D_B[interval],RD_2D_B[interval,2]/time2D_B[interval]**3,'b-.',linewidth=3.5)

#BW
intm = 0.15*86400; intM = 1.3*86400; interval = (vtime > intm) * (vtime < intM)
R2D5_BW, = plt.loglog(time2D_BW[interval],RD_2D_BW[interval,0]/time2D_BW[interval]**3,'r',linewidth=3.5)

intm = 0.2*86400; intM = 2*86400; interval = (vtime > intm) * (vtime < intM)
R2D10_BW, = plt.loglog(time2D_BW[interval],RD_2D_BW[interval,1]/time2D_BW[interval]**3,'r--',linewidth=3.5)

intm = 0.3*86400; intM = 3*86400; interval = (vtime > intm) * (vtime < intM)
R2D15_BW, = plt.loglog(time2D_BW[interval],RD_2D_BW[interval,2]/time2D_BW[interval]**3,'r-.',linewidth=3.5)

plt.legend((R2D5_BW,R2D10_BW,R2D15_BW,R2D5_B,R2D10_B,R2D15_B),('$BW25_m$ 5m','$BW25_m$ 10m','$BW25_m$ 15m','$B25_m$ 5m','$B25_m$ 10m','$B25_m$ 15m'),    loc=1,fontsize=16,ncol=2)

plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.tight_layout()
plt.savefig('./plot/'+label+'/RDt3_2_BW_'+label+'.eps')
print       './plot/'+label+'/RDt3_2_BW_'+label+'.eps'
plt.close()

# Rich 2D-3D

fig = plt.figure(figsize=(8, 6))
intm = 0.5*86400; intM = 1.5*86400; interval = (vtime > intm) * (vtime < intM)
Rich = RD_2D_BW[interval,0]/time2D_BW[interval]**3/drateD_BW[interval,depths[0]]
#print '2D 5m: mean', np.mean(Rich), 'std', np.std(Rich)
R2D5_BW, = plt.semilogx(time2D_BW[interval],Rich,'r',linewidth=2)

Rich = RD_2D_BWt[:,1]/ttime**3/drateDt_BW[depths[1]]
#print '2D 10m: mean', np.mean(Rich), 'std', np.std(Rich)
R2D10_BW, = plt.semilogx(ttime,Rich,'r--',linewidth=2)

Rich = RD_2D_BWt[:,2]/ttime**3/drateDt_BW[depths[2]]
#print '2D 15m: mean', np.mean(Rich), 'std', np.std(Rich)
R2D15_BW, = plt.semilogx(ttime,Rich,'r-.',linewidth=2)

Rich = RD_2D_Bt[:,0]/ttime**3/drateDt_B[depths[0]]
#print '2D 5m: mean', np.mean(Rich), 'std', np.std(Rich)
R2D5_B, = plt.semilogx(ttime,Rich,'b',linewidth=2)

Rich = RD_2D_Bt[:,1]/ttime**3/drateDt_B[depths[1]]
#print '2D 10m: mean', np.mean(Rich), 'std', np.std(Rich)
R2D10_B, = plt.semilogx(ttime,Rich,'b--',linewidth=2)

Rich = RD_2D_Bt[:,2]/ttime**3/drateDt_B[depths[2]]
#print '2D 15m: mean', np.mean(Rich), 'std', np.std(Rich)
R2D15_B, = plt.semilogx(ttime,Rich,'b-.',linewidth=2)

#for tic in plt.xaxis.get_minor_ticks():
#    tic.tick1On = tic.tick2On = False

#plt.legend((R2D1,R3D1,R2D5,R3D5,R2D17,R3D17),('2D 5m','3D 5m','2D 10m','3D 10m','2D 15m','3D 15m'),loc=3,fontsize=16,ncol=3)
plt.legend((R2D5_BW,R2D10_BW,R2D15_BW,R2D5_B,R2D10_B,R2D15_B),('$BW25_m$ 5m','$BW25_m$ 10m','$BW25_m$ 15m','$B25_m$ 5m','$B25_m$ 10m','$B25_m$ 15m'),loc=2,fontsize=16,ncol=2)

plt.xlabel('Time $[hr]$',fontsize=20)
plt.ylabel('$\sigma^2_D t^{-3} \epsilon^{-1}$ ',fontsize=20)
plt.ylim(0.03,0.13)
plt.xlim((ttime[0],ttime[-1]))
#plt.xticks((ttime[0],6*3600,12*3600.,18*3600,ttime[-1]),(ttime[0]/3600.+48,6+48,12+48,18+48,ttime[-1]/3600.+48),fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('./plot/'+label+'/Rich_2_BW_'+label+'.eps')
print       './plot/'+label+'/Rich_2_BW_'+label+'.eps'
plt.close()


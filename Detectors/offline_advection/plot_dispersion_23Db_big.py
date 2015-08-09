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
from scipy.interpolate import interp1d

# read offline
print 'reading offline'

#exp = 'm_25_1_tracer'
#label = 'm_25_1_tracer'
#filename2D = 'traj_m_25_1_tracer_0_640_2D.csv'
#filename3D = 'traj_m_25_1_tracer_0_640_3D.csv'
#tt = 640 # IC + 24-48 included

exp = 'm_25_2b_particles'
label = 'm_25_2b_particles'
filename2D = 'traj_m_25_2b_particles_0_500_2D.csv'
filename3D = 'traj_m_25_2b_particles_0_180_3D_big.csv'
tt =  180 # IC + 24-48 included

x0 = range(0,7010,10)
y0 = range(0,4010,10)
z0 = [0, 5, 10, 15] #range(1,20,4)



xp = len(x0)
yp = len(y0)
zp = len(z0)
pt = xp*yp*zp
          
#time2D, par2D = advect_functions.read_particles_csv(filename2D,pt,tt)
#par2D = lagrangian_stats.periodicCoords(par2D,8000,8000)
#time3D, par3D = advect_functions.read_particles_csv(filename3D,pt,tt)
#par3D = lagrangian_stats.periodicCoords(par3D,8000,8000)
#
#time2D = (time2D)*1440
#time3D = (time3D)*1440

def RD_t(par2Dzr,tt,px):
 RD_2Dm = []
 for i in range(px):
  RD_2Dm.append(np.mean(((par2Dzr[i+1,:,0,:] - par2Dzr[i,:,0,:])**2 + (par2Dzr[i+1,:,1,:] - par2Dzr[i,:,1,:])**2),0))
# for j in range(py):
#  RD_2Dm.append(np.mean(((par2Dzr[:,j+1,0,:] - par2Dzr[:,j,0,:])**2 + (par2Dzr[:,j+1,1,:] - par2Dzr[:,j,1,:])**2),0))
 return np.mean(RD_2Dm,0)

    
time = time2D[:-1]


xm = 2.5 
xM = 7.5
ym = 2
yM = 7

depths = [5, 10, 15] 
depthid = [1, 2, 3] 

nl = len(depths)

AD_2D = np.zeros((tt,nl))
AD_3D = np.zeros((tt,nl))
RD_2D = np.zeros((tt,nl))
RD_3D = np.zeros((tt,nl))
ED_2D = np.zeros((tt,nl))
ED_3D = np.zeros((tt,nl))
CD_2D = np.zeros((tt,nl))
CD_3D = np.zeros((tt,nl))

import scipy.optimize as optimize
import csv

for z in range(len(depths)):
 print z
 print 'depth', depths[z]
 par2Dz = np.reshape(par2D,(xp,yp,zp,3,tt))
 par3Dz = np.reshape(par3D,(xp,yp,zp,3,tt))
 par2Dzr = par2Dz[:,:,depthid[z],:,:]
 par3Dzr = par3Dz[:,:,depthid[z],:,:]

 par2Dz = np.reshape(par2Dzr,(xp*yp,3,tt))
 par3Dz = np.reshape(par3Dzr,(xp*yp,3,tt))

 AD_2D[:,z] = lagrangian_stats.AD_t(par2Dz,tt)
 AD_3D[:,z] = lagrangian_stats.AD_t(par3Dz,tt)
 CD_2D[:,z] = lagrangian_stats.CD_t(par2Dz,tt)
 CD_3D[:,z] = lagrangian_stats.CD_t(par3Dz,tt)
 ED_2D[:,z] = lagrangian_stats.ED_t(par2Dz,tt)
 ED_3D[:,z] = lagrangian_stats.ED_t(par3Dz,tt)
 RD_2D[:,z] = lagrangian_stats.RD_t(par2Dzr,tt,xp-1,yp-1)
 RD_3D[:,z] = lagrangian_stats.RD_t(par3Dzr,tt,xp-1,yp-1)
 #RD_2D[:,z] = RD_t(par2Dzr,tt,xp-1)
 #RD_3D[:,z] = RD_t(par3Dzr,tt,xp-1)

# plottting all depths

# absolute D

p_2D, = plt.plot(time/86400,AD_2D[:,0],color=[0,0,1],linewidth=2)
p_3D, = plt.plot(time/86400,AD_3D[:,0],color=[1,0,0],linewidth=2)
#plt.gca().set_yscale('log',basey=10,basey=10)
#plt.gca().set_xscale('log')

z = 1
p_2D5, = plt.plot(time/86400,AD_3D[:,z],'--',color=[0,0,1],linewidth=2)
p_3D5, = plt.plot(time/86400,AD_2D[:,z],'--',color=[1,0,0],linewidth=2)
#z = 2
#p_2D11, = plt.plot(time/86400,AD_2D[:,z],color=[0,0,0],linewidth=2)
#p_3D11, = plt.plot(time/86400,AD_3D[:,z],color=[0,0,1],linewidth=2)
z = 2
p_2D17, = plt.plot(time/86400,AD_3D[:,z],'-.',color=[0,0,1],linewidth=2)
p_3D17, = plt.plot(time/86400,AD_2D[:,z],'-.',color=[1,0,0],linewidth=2)
#z = 4
#p_2D26, = plt.plot(time/86400,AD_2D[:,z],color=[0,0,0],linewidth=2)
#p_3D26, = plt.plot(time/86400,AD_3D[:,z],color=[0,0,1],linewidth=2)

plt.gca().set_yscale('log',basey=10)

plt.xlabel(r'Time $[days]$',fontsize=20)
plt.ylabel(r'$\sigma^2_A$ $[m^2]$',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend((p_2D,p_2D5,p_2D17,p_3D,p_3D5,p_3D17),('2D 10m','2D 10m','2D 15m','3D 10m','3D 10m','3D 15m'),loc=4,fontsize=16)
#plt.legend((p_2D,p_2D5,p_2D11,p_2D17,p_2D26,p_3D,p_3D5,p_3D11,p_3D17,p_3D26),('2D 5m','2D 10m','on 15m','2D 15m','on 26m','3D 5m','3D 10m','off 15m','3D 15m','off 26m'),loc=2,fontsize=12)
plt.ylim((10**3,3*10**7))
#plt.xlim((2,5.5))
plt.xlim((2,14))

plt.savefig('./plot/'+label+'/AD_'+label+'.eps')
print       './plot/'+label+'/AD_'+label+'.eps'

plt.close()


# relative D

def f_exp(t,a,b):
    return a + b*np.exp(t)

def f_lin(t,a):
    return a + t

def f_rich(t,a):
    return a + 3*t

def f_bal(t,a):
    return a + 2*t

RD_2D = np.log10(RD_2D)
RD_3D = np.log10(RD_3D)
RD_2D = RD_2D[1:]
RD_3D = RD_3D[1:]

x = np.log10(time-time[0])
x = x[1:]
#x = x - x[0]
cD_2D = RD_2D[~np.isnan(RD_2D[:,0]),0]
cD_3D = RD_3D[~np.isnan(RD_3D[:,0]),0]

# t < t0 diffusive
# t1< t < t2 richardson
# t2 < t < end diffusive

t0 = np.log10(.5*86400)
t1 = 4.2 #np.log10(1*86400)
t2 = 5. #np.log10(3*86400)

# exp
#
#vtime = x[(x > 3.5) * (x < 4.2)]
#Val = cD_3D[(x > 3.5) * (x < 4.2)]
#
#out,cov = optimize.curve_fit(f_exp, vtime, Val, [0, 0], maxfev=100000)
#y = out[0] + out[1]*np.exp(x)
#plt.plot(x,y,'--k',linewidth=2)
#
# richardson

vtime = x[(x < t2) * (x > t1)]
Val = cD_3D[(x < t2) * (x > t1)]

out,cov = optimize.curve_fit(f_rich, vtime, Val, [0], maxfev=100000)
y = out[0] + 3.*x
#plt.plot(x,y,'.k',linewidth=2)


# bal

vtime = x[(x < 5.3) * (x > t2)]
Val = cD_3D[(x < 5.3) * (x > t2)]

out,cov = optimize.curve_fit(f_bal, vtime, Val, [0], maxfev=100000)
y = out[0] + 2.*x
#plt.plot(x,y,'-k',linewidth=1)


# diffusive

#vtime = x[(x > t1)]
#Val = cD_3D[x > t1]
vtime = x[(x < 5.4) * (x > t2)]
Val = cD_3D[(x < 5.4) * (x > t2)]
vtime = vtime[:-1]
Val = Val[:-1]

out,cov = optimize.curve_fit(f_lin, vtime, Val, [0], maxfev=100000)
y = out[0] + x
#plt.plot(x,y,'-.k',linewidth=2)

new_x = np.linspace(np.log10(2000),4.2,10)
y = 3.5 -3.*new_x[0] + 3.*new_x
plt.plot(new_x,y,'k',linewidth=1.5)
plt.text(4.3,y[-1],'3',fontsize=16)
y = 3.5 -2.*new_x[0] + 2.*new_x
plt.plot(new_x,y,'k',linewidth=1.5)
plt.text(4.3,y[-1],'2',fontsize=16)
y = 3.5 -1.*new_x[0] + 1.*new_x
plt.plot(new_x,y,'k',linewidth=1.5)
plt.text(4.3,y[-1],'1',fontsize=16)


p_3D, = plt.plot(x,RD_3D[:,0],color=[1,0,0],linewidth=2)

z = 1
p_3D5, = plt.plot(x,RD_3D[:,z],'--',color=[1,0,0],linewidth=2)
z = 2
p_3D17, = plt.plot(x,RD_3D[:,z],'-.',color=[1,0,0],linewidth=2)
#
#plt.gca().set_yscale('log',basey=10)
#plt.gca().set_xscale('log',basex=10)

plt.xlabel(r'Time $[hr]$',fontsize=20)
plt.ylabel(r'$log(\sigma^2_D)$ $[m^2]$',fontsize=20)
#plt.ylabel('Relative Dispersion [m^2]')
plt.legend((p_3D,p_3D5,p_3D17),('3D 5m','3D 10m','3D 15m'),loc=4,fontsize=16)
#plt.xlim((0,5))
#plt.xlim((0.1,6))
plt.xlim((np.log10(1440),np.log10(72*3600)))
plt.ylim((2-.5,7.5))

#ind = np.linspace(2,14,13)*86400    # the x locations for the groups
#ind = np.linspace(0,288,9)*3600
ind = np.linspace(0,72,4)*3600
#vind = np.log10(ind-ind[0])
ind[0] = 1440
vind = np.log10(ind)

#ind = ind/86400
#ind = ind-ind[0]
#ind = ind/3600.
#ind = ind
#vind[0]=x[0]
#vind[0]=np.log10(360)

#plt.xticks(vind,[0,'','','3','','','6','','','9','','','12'],fontsize=16)
#plt.xticks(vind,['0.4','36','72','','144','','','','288'],fontsize=16)
#plt.xticks(vind,['0.4','36','72','','144'],fontsize=16)
plt.xticks(vind,['0.4','24','48','72'],fontsize=16)
plt.xticks(vind,['48.4','72','96','120'],fontsize=16)

plt.yticks(fontsize=16)

#plt.xticks(vind,ind.astype(int))

plt.tight_layout()
plt.savefig('./plot/'+label+'/RD_3_'+label+'.eps')
print       './plot/'+label+'/RD_3_'+label+'.eps'
plt.close()

# RD 2D

# exp
#
#vtime = x[(x > 3.5) * (x < 4.2)]
#Val = cD_2D[(x > 3.5) * (x < 4.2)]
#
#out,cov = optimize.curve_fit(f_exp, vtime, Val, [0, 0], maxfev=100000)
#y = out[0] + out[1]*np.exp(x)
#plt.plot(x,y,'--k',linewidth=2)
#
# richardson

vtime = x[(x < t2) * (x > t1)]
Val = cD_2D[(x < t2) * (x > t1)]

out,cov = optimize.curve_fit(f_rich, vtime, Val, [0], maxfev=100000)
y = out[0] + 3.*x
#plt.plot(x,y,'.k',linewidth=2)

# bal

vtime = x[(x < 5.3) * (x > t2)]
Val = cD_2D[(x < 5.3) * (x > t2)]

out,cov = optimize.curve_fit(f_bal, vtime, Val, [0], maxfev=100000)
y = out[0] + 2.*x
#plt.plot(x,y,'-k',linewidth=1)

# diffusive

#vtime = x[(x > t1)]
#Val = cD_3D[x > t1]
vtime = x[(x < 5.4) * (x > 5.0)]
Val = cD_2D[(x < 5.4) * (x > 5.0)]
vtime = vtime[:-1]
Val = Val[:-1]

out,cov = optimize.curve_fit(f_lin, vtime, Val, [0], maxfev=100000)
y = out[0] + x
#plt.plot(x,y,'-.k',linewidth=2)

new_x = np.linspace(np.log10(2000),4.2,10)
y = 3.5 -3.*new_x[0] + 3.*new_x
plt.plot(new_x,y,'k',linewidth=1.5)
plt.text(4.3,y[-1],'3',fontsize=16)
y = 3.5 -2.*new_x[0] + 2.*new_x
plt.plot(new_x,y,'k',linewidth=1.5)
plt.text(4.3,y[-1],'2',fontsize=16)
y = 3.5 -1.*new_x[0] + 1.*new_x
plt.plot(new_x,y,'k',linewidth=1.5)
plt.text(4.3,y[-1],'1',fontsize=16)

p_2D, = plt.plot(x,RD_2D[:,0],color=[0,0,1],linewidth=2)

z = 1
p_2D5, = plt.plot(x,RD_2D[:,z],'--',color=[0,0,1],linewidth=2)
z = 2
p_2D17, = plt.plot(x,RD_2D[:,z],'-.',color=[0,0,1],linewidth=2)
#
#plt.gca().set_yscale('log',basey=10)
#plt.gca().set_xscale('log',basex=10)

plt.xlabel(r'Time $[hr]$',fontsize=20)
plt.ylabel(r'$log(\sigma^2_D)$ $[m^2]$',fontsize=20)
#plt.ylabel('Relative Dispersion [m^2]')
plt.legend((p_2D,p_2D5,p_2D17),('2D 5m','2D 10m','2D 15m'),loc=4,fontsize=16)
#plt.xlim((2,5))
plt.xlim((np.log10(1440),np.log10(72*3600)))
plt.ylim((2-.5,7.5))

#ind = np.linspace(2,14,13)*86400    # the x locations for the groups
#ind = np.linspace(0,288,9)*3600
ind = np.linspace(0,72,4)*3600
ind[0] = 1440
#vind = np.log10(ind-ind[0])
vind = np.log10(ind)

#ind = ind/86400
#ind = ind-ind[0]
#ind = ind/3600.
#ind = ind
#vind[0]=x[0]
#vind[0]=np.log10(360)

#plt.xticks(vind,[0,'','','3','','','6','','','9','','','12'],fontsize=16)
#plt.xticks(vind,['0.4','36','72','','144','','','','288'],fontsize=16)
#plt.xticks(vind,['0.4','24','48','72'],fontsize=16)
plt.xticks(vind,['48.4','72','96','120'],fontsize=16)
plt.yticks(fontsize=16)

#plt.xticks(vind,ind.astype(int))

plt.tight_layout()
plt.savefig('./plot/'+label+'/RD_2_'+label+'.eps')
print       './plot/'+label+'/RD_2_'+label+'.eps'
plt.close()

import csv
drate = np.zeros((31,36))
timeD = []
with open('drate_m_25_1b_particles.csv', 'rb') as csvfile:
 spamreader = csv.reader(csvfile)
 t = 0
 for row in spamreader:
  t = t + 1
  if t > 1:
   timeD.append(float(row[0]))
   drate[t-2,:] = row[1:]

timeD = np.asarray(timeD)
timeD = timeD - timeD[0]

from scipy import interpolate



## RICHARDSON Plot
vtime = time - time[0]
vtime = vtime[5:]
vtime = vtime[(vtime > 0.2*86400) * (vtime < 86400)]

cD_2D = np.power(10,RD_2D[:,0])
Val = cD_2D[(vtime > 0.2*86400) * (vtime < 86400)]
f = interpolate.interp1d(timeD, drate[:,1])
drateT = f(vtime)   # use interpolation function returned by `interp1d`
Rich = Val/vtime**3/drateT
print '2D 5m: mean', np.mean(Rich), 'std', np.std(Rich)
R2D1, = plt.loglog(vtime,Rich,'b',linewidth=2)

cD_2D = np.power(10,RD_2D[:,1])
Val = cD_2D[(vtime > 0.2*86400) * (vtime < 86400)]
f = interpolate.interp1d(timeD, drate[:,5])
drateT = f(vtime)   # use interpolation function returned by `interp1d`
Rich = Val/vtime**3/drateT
print '2D 10m: mean', np.mean(Rich), 'std', np.std(Rich)
R2D5, = plt.loglog(vtime,Rich,'b--',linewidth=2)

cD_2D = np.power(10,RD_2D[:,2])
Val = cD_2D[(vtime > 0.2*86400) * (vtime < 86400)]
f = interpolate.interp1d(timeD, drate[:,17])
drateT = f(vtime)   # use interpolation function returned by `interp1d`
Rich = Val/vtime**3/drateT
print '2D 15m: mean', np.mean(Rich), 'std', np.std(Rich)
R2D17, = plt.loglog(vtime,Rich,'b-.',linewidth=2)

cD_3D = np.power(10,RD_3D[:,0])
Val = cD_3D[(vtime > 0.2*86400) * (vtime < 86400)]
f = interpolate.interp1d(timeD, drate[:,1])
drateT = f(vtime)   # use interpolation function returned by `interp1d`
Rich = Val/vtime**3/drateT
print '3D 5m: mean', np.mean(Rich), 'std', np.std(Rich)
R3D1, = plt.loglog(vtime,Rich,'r',linewidth=2)

cD_3D = np.power(10,RD_3D[:,1])
Val = cD_3D[(vtime > 0.2*86400) * (vtime < 86400)]
f = interpolate.interp1d(timeD, drate[:,5])
drateT = f(vtime)   # use interpolation function returned by `interp1d`
Rich = Val/vtime**3/drateT
print '3D 10m: mean', np.mean(Rich), 'std', np.std(Rich)
R3D5, = plt.loglog(vtime,Rich,'r--',linewidth=2)

cD_3D = np.power(10,RD_3D[:,2])
Val = cD_3D[(vtime > 0.2*86400) * (vtime < 86400)]
f = interpolate.interp1d(timeD, drate[:,17])
drateT = f(vtime)   # use interpolation function returned by `interp1d`
Rich = Val/vtime**3/drateT
print '3D 15m: mean', np.mean(Rich), 'std', np.std(Rich)
R3D17, = plt.loglog(vtime,Rich,'r-.',linewidth=2)

plt.plot([vtime[0],vtime[-1]],[0.5, 0.5],'k',linewidth=1.5)

#plt.xlim(vtime[0],vtime[-1])
#plt.ylim((10**-11,10**-7))

plt.legend((R2D1,R3D1,R2D5,R3D5,R2D17,R3D17),('2D 5m','3D 5m','2D 10m','3D 10m','2D 15m','3D 15m'),loc=3,fontsize=16,ncol=3)

plt.xlabel('Time $[hr]$',fontsize=20)
plt.ylabel('$\sigma^2_D t^{-3} \epsilon^{-1}$ ',fontsize=20)
plt.ylim((10**-3,10**1))
plt.xlim((vtime[0],vtime[-1]))
plt.xticks((vtime[0],vtime[-1]),(vtime[0]/3600.,vtime[-1]/3600.),fontsize=16)
plt.yticks(fontsize=16)
#plt.xticks(np.linspace(vtime[0],vtime[-1],14),['','','6','','','','','14','','','','','22',''])
plt.tight_layout()
plt.savefig('./plot/'+label+'/Rich_23_'+label+'.eps')
print       './plot/'+label+'/Rich_23_'+label+'.eps'
plt.close()





# Ellipses D

p_2D, = plt.plot(time/86400,ED_2D[:,0],color=[0,0,1],linewidth=2)
p_3D, = plt.plot(time/86400,ED_3D[:,0],color=[1,0,0],linewidth=2)

z = 1
p_2D5, = plt.plot(time/86400,ED_2D[:,z],'--',color=[0,0,1],linewidth=2)
p_3D5, = plt.plot(time/86400,ED_3D[:,z],'--',color=[1,0,0],linewidth=2)
#z = 2
#p_2D11, = plt.plot(time/86400,ED_2D[:,z],color=[0,0,0],linewidth=2)
#p_3D11, = plt.plot(time/86400,ED_3D[:,z],color=[0,0,1],linewidth=2)
z = 2
p_2D17, = plt.plot(time/86400,ED_2D[:,z],'-.',color=[0,0,1],linewidth=2)
p_3D17, = plt.plot(time/86400,ED_3D[:,z],'-.',color=[1,0,0],linewidth=2)
#z = 4
#p_2D26, = plt.plot(time/86400,ED_2D[:,z],color=[0,0,0],linewidth=2)
#p_3D26, = plt.plot(time/86400,ED_3D[:,z],color=[0,0,1],linewidth=2)

plt.gca().set_yscale('log',basey=10)
#plt.gca().set_xscale('log')

plt.xlabel(r'Time $[days]$',fontsize=20)
plt.ylabel(r'$\sigma^2_E$ $[m^2]$',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#plt.ylabel('Dispersion [m^2]')
#plt.ylim((3*10**4,10**6))
#plt.legend((p_2D,p_2D5,p_2D17,p_3D,p_3D5,p_3D17),('2D 5m','2D 10m','2D 15m','3D 5m','3D 10m','3D 15m'),loc=4,fontsize=12)
plt.legend((p_2D,p_2D5,p_2D17,p_3D,p_3D5,p_3D17),('2D 5m','2D 10m','2D 15m','3D 5m','3D 10m','3D 15m'),loc=4,fontsize=16)
#plt.legend((p_2D,p_3D),('on','off'))
plt.xlim((2,14))
plt.ylim((10**5,6*10**7))

plt.savefig('./plot/'+label+'/ED_'+label+'.eps')
print       './plot/'+label+'/ED_'+label+'.eps' 
plt.close()


# cloud D

p_2D, = plt.plot(time/86400,CD_2D[:,0],'b',linewidth=2)
p_3D, = plt.plot(time/86400,CD_3D[:,0],'r',linewidth=2)
plt.gca().set_yscale('log',basey=10)

#for z in range(nl):
# plt.plot(time/86400,CD_2D[:,z],color=[0,0,0],linewidth=2)
# plt.plot(time/86400,CD_3D[:,z],color=[0,0,1],linewidth=2)

z = 1
p_2D5, = plt.plot(time/86400,CD_2D[:,z],'--',color=[0,0,1],linewidth=2)
p_3D5, = plt.plot(time/86400,CD_3D[:,z],'--',color=[1,0,0],linewidth=2)
#z = 2
#p_2D11, = plt.plot(time/86400,CD_2D[:,z],color=[0,0,0],linewidth=2)
#p_3D11, = plt.plot(time/86400,CD_3D[:,z],color=[0,0,1],linewidth=2)
z = 2
p_2D17, = plt.plot(time/86400,CD_2D[:,z],'-.',color=[0,0,1],linewidth=2)
p_3D17, = plt.plot(time/86400,CD_3D[:,z],'-.',color=[1,0,0],linewidth=2)
#z = 4
#p_2D26, = plt.plot(time/86400,CD_2D[:,z],color=[0,0,0],linewidth=2)
#p_3D26, = plt.plot(time/86400,CD_3D[:,z],color=[0,0,1],linewidth=2)

plt.xlabel(r'Time $[days]$',fontsize=20)
plt.ylabel(r'$\sigma^2_C$ $[m^2]$',fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#plt.legend((p_2D,p_2D5,p_2D17,p_3D,p_3D5,p_3D17),('2D 5m','2D 10m','2D 15m','3D 5m','3D 10m','3D 15m'),loc=4,fontsize=12)
plt.legend((p_2D,p_2D5,p_2D17,p_3D,p_3D5,p_3D17),('2D 5m','2D 10m','2D 15m','3D 5m','3D 10m','3D 15m'),loc=4,fontsize=16)
#plt.legend((p_2D,p_3D),('on','off'))
#plt.ylim((10**5,3.*10**6))
plt.xlim((2,14))
plt.ylim((10**5,6*10**7))
#
# 
plt.savefig('./plot/'+label+'/CD_'+label+'.eps')
print       './plot/'+label+'/CD_'+label+'.eps' 
plt.close()




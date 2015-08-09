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

def AD_t_v(par2Dz,tt):
 temp = np.zeros((len(par2Dz[:,0,0]),len(par2Dz[0,0,:])))
 for p in range(len(par2Dz[:,0,0])):
  temp[p,:] = (par2Dz[p,2,:] - par2Dz[p,2,0])**2
 return np.mean(temp,0)

# read offline
print 'reading offline'

#exp = 'm_25_2_512'
#label = 'm_25_2_512'
#filename3D = 'traj_m_25_2_512_0_520_3D.csv'
#tt = 520

exp = 'm_25_1_particles'
label = 'm_25_1_particles'
filename3D = 'traj_m_25_1_particles_481_3400_3D.csv'
tt = 3400-481

x0 = range(500,1510,10)
y0 = range(500,1510,10)
z0 = [5,10,15] #range(1,20,4)

#filename3D = 'traj_m_25_2_512_0_290_3D.csv'
#tt = 290 # IC + 24-48 included

#x0 = range(3000,4010,10)
#y0 = range(2000,3010,10)
#z0 = range(1,20,4)

xp = len(x0)
yp = len(y0)
zp = len(z0)
pt = xp*yp*zp

#time3D, par3Dtemp = advect_functions.read_particles_csv(filename3D,pt,tt)

#par3D = lagrangian_stats.periodicCoords(par3D,10000,4000)
#time3D = (time3D)*1200 + 48*3600

#par3D = lagrangian_stats.periodicCoords(par3D,2000,2000)
#time3D = (time3D)*360 + 48*3600
    
#time = time3D[:-1] - 48*3600

xm = 2.5 
xM = 7.5
ym = 2
yM = 7

#depths = [ 5, 17] 
#depthid = [ 2, 3] 
depths = [ 5, 10] 
depthid = [ 0, 1] 

nl = len(depths)

AD_3D = np.zeros((tt,nl))
Diff_AD_3D = np.zeros((tt,nl))

for z in range(len(depths)):
 print z
 print 'depth', depths[z]
 par3Dz = np.reshape(par3D,(xp,yp,zp,3,tt))
 par3Dzr = par3Dz[:,:,depthid[z],:,:]

 par3Dz = np.reshape(par3Dzr,(xp*yp,3,tt))

 #AD_3D[:,z] = lagrangian_stats.AD_t_v(par3Dz,tt)
 AD_3D[:,z] = AD_t_v(par3Dz,tt)

 Diff_AD_3D[:,z] = 0.25*np.gradient(AD_3D[:,z])/np.gradient(time)



plt.subplots(figsize=(6,6))

p_3D5, = plt.plot(time/86400,AD_3D[:,0],color=[0,0,0],linewidth=2)
p_3D17, = plt.plot(time/86400,AD_3D[:,1],'--',color=[0,0,0],linewidth=2)

plt.gca().set_yscale('log',basey=10)

plt.xlabel(r'Time $[days]$',fontsize=20)
plt.ylabel(r'$\sigma^2_A$ $[m^2]$',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim((2,14))
plt.legend((p_3D5,p_3D17),('5m','17m'),loc=4,fontsize=18)
#plt.ylim((10**2,10**6))
plt.tight_layout()
plt.savefig('./plot/'+label+'/AD_v_'+label+'.eps')
print       './plot/'+label+'/AD_v_'+label+'.eps'
plt.close()


# plotting all depths

# absolute D

plt.subplots(figsize=(6,6))

p_3D5, = plt.plot(time/86400,Diff_AD_3D[:,0],color=[0,0,0],linewidth=2)
z = 1
p_3D17, = plt.plot(time/86400,Diff_AD_3D[:,z],'--',color=[0,0,0],linewidth=2)

plt.gca().set_yscale('log',basey=10)

plt.xlabel(r'Time $[days]$',fontsize=20)
plt.ylabel(r"$K'_z$ $[m^2s^{-1}]$",fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend((p_3D5,p_3D17),('5m','17m'),loc=4,fontsize=18)
plt.tight_layout()
plt.xlim((2,14))
plt.savefig('./plot/'+label+'/Diff_AD_v_'+label+'.eps')
print       './plot/'+label+'/Diff_AD_v_'+label+'.eps'
plt.close()


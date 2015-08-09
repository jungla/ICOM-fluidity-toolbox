#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import myfun
import numpy as np
import os
import advect_functions

def AD_t_v(par2Dz,tt):
 temp = np.zeros((len(par2Dz[:,0,0]),len(par2Dz[0,0,:])))
 for p in range(len(par2Dz[:,0,0])):
  temp[p,:] = (par2Dz[p,2,:] - par2Dz[p,2,0])**2
 return np.mean(temp,0)


##<(z(t)-z(0))^2>- - <z-z(0)>^2
#def AD_t_v(par2Dz,tt):
# temp2 = np.zeros((len(par2Dz[:,0,0]),len(par2Dz[0,0,:])))
# temp = np.zeros((len(par2Dz[:,0,0]),len(par2Dz[0,0,:])))
# for p in range(len(par2Dz[:,0,0])):
#  temp2[p,:] = (par2Dz[p,2,:] - par2Dz[p,2,0])**2
#  temp[p,:] = (par2Dz[p,2,:] - par2Dz[p,2,0])
# return np.mean(temp2,0) - np.mean(temp,0)**2


# read offline
print 'reading offline'

exp = 'm_25_2_512'
label_B = 'm_25_1b_particles'
filename_B = 'traj_m_25_1b_particles_0_600_3Dv.csv'
tt_B = 600

label_BW = 'm_25_2b'
filename_BW = 'traj_m_25_2b_particles_0_500_3Dv.csv'
tt_BW = 500

x0_B = range(0,8000,100)
y0_B = range(0,8000,100)
z0_B = range(0,52,2)

x0_BW = range(0,8000,50)
y0_BW = range(0,8000,50)
z0_BW = range(0,52,2)

xp_B = len(x0_B)
yp_B = len(y0_B)
zp_B = len(z0_B)
pt_B = xp_B*yp_B*zp_B

xp_BW = len(x0_BW)
yp_BW = len(y0_BW)
zp_BW = len(z0_BW)
pt_BW = xp_BW*yp_BW*zp_BW

time_B, par_B = advect_functions.read_particles_csv(filename_B,pt_B,tt_B)
#par_B = lagrangian_stats.periodicCoords(par_B,2000,2000)
time_B = time_B[:-1]*1440 + 86400*2

time_BW, par_BW = advect_functions.read_particles_csv(filename_BW,pt_BW,tt_BW)
#par_BW = lagrangian_stats.periodicCoords(par_BW,10000,4000)
time_BW = time_BW[:-1]*1440 + 86400*2
    
xm = 2.5 
xM = 7.5
ym = 2
yM = 7

depths = [5] 
depthid_B = [1] 
depthid_BW = [1] 

nl = len(depths)

AD_B = np.zeros((tt_B,nl))
Diff_AD_B = np.zeros((tt_B,nl))
AD_BW = np.zeros((tt_BW,nl))
Diff_AD_BW = np.zeros((tt_BW,nl))

for z in range(len(depths)):
 print z
 print 'depth', depths[z]
 par_Bz = np.reshape(par_B,(xp_B,yp_B,zp_B,3,tt_B))
 par_Bzr = par_Bz[:,:,depthid_B[z],:,:]
 par_Bz = np.reshape(par_Bzr,(xp_B*yp_B,3,tt_B))

 AD_B[:,z] = AD_t_v(par_Bz,tt_B)
 Diff_AD_B[:,z] = 0.25*np.gradient(AD_B[:,z])/np.gradient(time_B)

 par_BWz = np.reshape(par_BW,(xp_BW,yp_BW,zp_BW,3,tt_BW))
 par_BWzr = par_BWz[:,:,depthid_BW[z],:,:]
 par_BWz = np.reshape(par_BWzr,(xp_BW*yp_BW,3,tt_BW))

 AD_BW[:,z] = AD_t_v(par_BWz,tt_B)
 Diff_AD_BW[:,z] = 0.25*np.gradient(AD_BW[:,z])/np.gradient(time_BW)

plt.subplots(figsize=(6,6))

p_B, = plt.plot((time_B[:-1])/86400,AD_B[:-1,0],color=[0,0,0],linewidth=2)
p_BW, = plt.plot((time_BW[:-1])/86400,AD_BW[:-1,0],'--',color=[0,0,0],linewidth=2)

plt.gca().set_yscale('log',basey=10)
#plt.gca().set_xscale('log',basey=10)

x0 = 0.1
x1 = 1
y0 = 0.1
y1 = 1

#plt.plot([x0,x1],[y0,10],'k',linewidth=1.5)
#plt.plot([x0,x1],[y0,y1/2.],'k',linewidth=1.5)
##plt.plot([1,x],[1,x],'k',linewidth=1.5)
##plt.plot([x0,x1],[x0,x1**0.5],'k',linewidth=1.5)
#plt.text(x1+.1,10,'$t^2$',fontsize=18)
##plt.text(x+1,x,'$t$',fontsize=16)
#plt.text(x1+.1,y1/2.,'$t^{0.5}$',fontsize=18)

plt.xlabel(r'Time $[days]$',fontsize=20)
plt.ylabel(r'$\sigma^2_A$ $[m^2]$',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.xlim((10**-2,10))
plt.xlim((2,4))
plt.legend((p_B,p_BW),('$B25_m$','$BW25_m$'),loc=4,fontsize=18)
#plt.ylim((10**2,10**6))
plt.tight_layout()
plt.savefig('./plot/'+exp+'/AD_v_'+exp+'.eps')
print       './plot/'+exp+'/AD_v_'+exp+'.eps'
plt.close()


# plotting all depths

# absolute D

plt.subplots(figsize=(6,6))

p_B, = plt.plot((time_B[:])/86400,Diff_AD_B[:,0],color=[0,0,0],linewidth=2)
#p_BW, = plt.plot(time_B/86400,Diff_AD_B[:,0],color=[0,0,0],linewidth=2)
p_BW, = plt.plot((time_BW[:])/86400,Diff_AD_BW[:,0],'--',color=[0,0,0],linewidth=2)

plt.gca().set_yscale('log',basey=10)
#plt.gca().set_xscale('log',basey=10)

plt.xlabel(r'Time $[days]$',fontsize=20)
plt.ylabel(r"$K'_z$ $[m^2s^{-1}]$",fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend((p_B,p_BW),('$B25_m$','$BW25_m$'),loc=4,fontsize=18,ncol=1)
plt.tight_layout()
#plt.xlim((10**-2,10))
plt.xlim((2,4))
#plt.ylim((10**-6,10**-2))
plt.savefig('./plot/'+exp+'/Diff_AD_v_'+exp+'.eps')
print       './plot/'+exp+'/Diff_AD_v_'+exp+'.eps'
plt.close()


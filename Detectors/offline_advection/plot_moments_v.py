#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import myfun
import numpy as np
import os
import advect_functions

def AD_t_v(parz,tt):
 temp = np.zeros((len(parz[:,0,0]),len(parz[0,0,:])))
 for p in range(len(parz[:,0,0])):
  temp[p,:] = (parz[p,2,:] - parz[p,2,0])**2
 return np.mean(temp,0)

def CD_t_v(parz,tt):
 temp = np.zeros((len(parz[:,0,0]),len(parz[0,0,:])))
 temp[:,:] = (parz[:,2,:] - np.mean(parz[:,2,:],0))**2
 return np.mean(temp,0)

def CM_t_v(parz,tt):
 return np.mean(parz[:,2,:],0)


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

#time_B, par_B = advect_functions.read_particles_csv(filename_B,pt_B,tt_B)
##par_B = lagrangian_stats.periodicCoords(par_B,2000,2000)
#time_B = time_B[:-1]*1440 + 86400*2
#
#time_BW, par_BW = advect_functions.read_particles_csv(filename_BW,pt_BW,tt_BW)
##par_BW = lagrangian_stats.periodicCoords(par_BW,10000,4000)
#time_BW = time_BW[:-1]*1440 + 86400*2
    
xm = 2.5 
xM = 7.5
ym = 2
yM = 7

depths = [2] 
depthid_B = [1] 
depthid_BW = [1] 

nl = len(depths)

AD_B = np.zeros((tt_B,nl))
AD_BW = np.zeros((tt_BW,nl))
CD_B = np.zeros((tt_B,nl))
CD_BW = np.zeros((tt_BW,nl))
CM_B = np.zeros((tt_B,nl))
CM_BW = np.zeros((tt_BW,nl))

for z in range(len(depths)):
 print z
 print 'depth', depths[z]
 par_Bz = np.reshape(par_B,(xp_B,yp_B,zp_B,3,tt_B))
 par_Bzr = par_Bz[:,:,depthid_B[z],:,:]
 par_Bz = np.reshape(par_Bzr,(xp_B*yp_B,3,tt_B))

 AD_B[:,z] = AD_t_v(par_Bz,tt_B)
 CD_B[:,z] = CD_t_v(par_Bz,tt_B)
 CM_B[:,z] = CM_t_v(par_Bz,tt_B)

 par_BWz = np.reshape(par_BW,(xp_BW,yp_BW,zp_BW,3,tt_BW))
 par_BWzr = par_BWz[:,:,depthid_BW[z],:,:]
 par_BWz = np.reshape(par_BWzr,(xp_BW*yp_BW,3,tt_BW))

 AD_BW[:,z] = AD_t_v(par_BWz,tt_B)
 CD_BW[:,z] = CD_t_v(par_BWz,tt_B)
 CM_BW[:,z] = CM_t_v(par_BWz,tt_B)


# plot mean particle postion with sd lines...



# AD

plt.subplots(figsize=(6,6))

p_B, = plt.plot((time_B[:-1])/86400,AD_B[:-1,0],color=[0,0,0],linewidth=2)
p_BW, = plt.plot((time_BW[:-1])/86400,AD_BW[:-1,0],'--',color=[0,0,0],linewidth=2)

plt.gca().set_yscale('log',basey=10)
#plt.gca().set_xscale('log',basey=10)

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

# CD

plt.subplots(figsize=(6,6))

p_B, = plt.plot((time_B[:-1])/86400,CD_B[:-1,0],color=[0,0,0],linewidth=2)
p_BW, = plt.plot((time_BW[:-1])/86400,CD_BW[:-1,0],'--',color=[0,0,0],linewidth=2)

plt.gca().set_yscale('log',basey=10)
#plt.gca().set_xscale('log',basey=10)

plt.xlabel(r'Time $[days]$',fontsize=20)
plt.ylabel(r'$\sigma^2_C$ $[m^2]$',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.xlim((10**-2,10))
plt.xlim((2,4))
plt.legend((p_B,p_BW),('$B25_m$','$BW25_m$'),loc=4,fontsize=18)
#plt.ylim((10**2,10**6))
plt.tight_layout()
plt.savefig('./plot/'+exp+'/CD_v_'+exp+'.eps')
print       './plot/'+exp+'/CD_v_'+exp+'.eps'
plt.close()

# CM

plt.subplots(figsize=(6,6))

p_B, = plt.plot((time_B[:-1])/86400,CM_B[:-1,0],color=[0,0,0],linewidth=2)
p_BW, = plt.plot((time_BW[:-1])/86400,CM_BW[:-1,0],'--',color=[0,0,0],linewidth=2)
plt.plot((time_B[:-1])/86400,np.mean(par_Bz[:,2,:-1],0)+np.std(par_Bz[:,2,:-1],0),color=[0.5,0.5,0.5],linewidth=2)
plt.plot((time_B[:-1])/86400,np.mean(par_Bz[:,2,:-1],0)-np.std(par_Bz[:,2,:-1],0),color=[0.5,0.5,0.5],linewidth=2)
plt.plot((time_BW[:-1])/86400,np.mean(par_BWz[:,2,:-1],0)+np.std(par_BWz[:,2,:-1],0),'--',color=[0.5,0.5,0.5],linewidth=2)
plt.plot((time_BW[:-1])/86400,np.mean(par_BWz[:,2,:-1],0)-np.std(par_BWz[:,2,:-1],0),'--',color=[0.5,0.5,0.5],linewidth=2)


plt.xlabel(r'Time $[days]$',fontsize=20)
plt.ylabel(r'center of mass $[m]$',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.xlim((10**-2,10))
plt.xlim((2,4))
plt.legend((p_B,p_BW),('$B25_m$','$BW25_m$'),loc=3,fontsize=18)
#plt.ylim((10**2,10**6))
plt.tight_layout()
plt.savefig('./plot/'+exp+'/CM_v_'+exp+'.eps')
print       './plot/'+exp+'/CM_v_'+exp+'.eps'
plt.close()


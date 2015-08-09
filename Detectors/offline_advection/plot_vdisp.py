#!~/python
import fluidity_tools
import matplotlib  as mpl
#mpl.use('ps')
import matplotlib.pyplot as plt
import myfun
import numpy as np
import os
import fio
import advect_functions

exp = 'm_25_2_512'
filename0 = 'traj_m_25_2_512_0_500_3Dv.csv'
tt = 500# IC + 24-48 included
x0 = range(0,10000,50)
y0 = range(0,4000,50)
z0 = range(0,52,2)
#x0 = range(1000,3050,50)
#y0 = range(1000,3050,50)
#z0 = range(-50,2,2) #range(0,52,2)
xp = len(x0)
yp = len(y0)
zp = len(z0)
pt = xp*yp*zp
#timet, par0 = advect_functions.read_particles_csv(filename0,pt,tt)

time0 = (timet[:-1])*1440 + 48*3600

par = np.reshape(par0,(pt,3,tt))

# FSLE
di = 2
time = time0 - time0[0]
import scipy

for r in [2]:
 parL = range(pt-1)
 fsle  = np.zeros(pt)*np.nan
 fslec = np.zeros((pt,3))
 df=r*di # separation distance
 # 
 # loop triplets in time
 #
 for t in range(tt-1):
  for p in parL:
#    print t,len(parL)
 # loop particles
    dr = np.linalg.norm(par[p,2,t]-par[p+1,2,t])
    if (dr > df and np.isnan(fsle[p])):
     fsle[p]  = np.log(r)/(time[t]/3600)   # fsle has the dimension of the first triplet
     fslec[p,:] = par[p,:,0]  # coords of the starting point
     parL.remove(p)
 #
 # plot fsle
 # 3D arrays of fsle and fslec
 #
 fsler = np.reshape(fsle,(xp,yp,zp))
 fsler[:,:,-1] = 0
 # fslexr = np.reshape(fslec[:,0],(nlat,nlon))
 # fsleyr = np.reshape(fslec[:,1],(nlat,nlon))
 # fslezr = np.reshape(fslec[:,2],(nlat,nlon))
 #

 plt.subplots(figsize=(9,7))
 #plt.contourf(np.asarray(x0)/1000.,z0,np.rot90(scipy.stats.nanmean(fsler[:,:,:],0)),np.linspace(0,np.percentile(fsle[~np.isnan(fsle)],80),30),vmin=0,extend='both')
 plt.contourf(np.asarray(y0)/1000.,z0,np.rot90(scipy.stats.nanmean(fsler[:,:,:],0)),np.linspace(0,0.08,30),vmin=0,extend='both')
 plt.ylabel('Depth [m]', fontsize=26)
 plt.xlabel('Y [km]', fontsize=26)
 plt.yticks(np.linspace(0,50,5),np.linspace(-50,0,5),fontsize=24)
 plt.xticks(fontsize=24)
 cb = plt.colorbar(ticks=np.linspace(0,0.08,5))
 cb.ax.tick_params(labelsize=24)
 # plt.title(r'\lambda')
 plt.tight_layout()
 plt.savefig('./plot/'+exp+'/fsle_'+exp+'_'+str(r)+'.eps')
 print       './plot/'+exp+'/fsle_'+exp+'_'+str(r)+'.eps'
 plt.close()

## PDF Vertical Displacement
#
#bins = np.linspace(-50,0,50)
#values = np.zeros((len(bins)-1,len(time0)))
#
#for t in range(0,tt-1,1):
#
# values[:,t], bins = np.histogram(par[:,2,t],bins)
#
#fig = plt.figure(figsize=(12,8))
## ax = fig.add_subplot(111, aspect='equal')
#plt.pcolor(time0/3600,bins,np.log(values),vmin=4)
#plt.colorbar()
#plt.xlim(time0[0]/3600,time0[-1]/3600)
#plt.xlabel('time')
#plt.ylabel('# particles')
#print 'Saving 0 to eps'
# #
##  ax.text(1, 9, str(z)+'m, '+str(time[t]*3600)+'h', fontsize=18)
#plt.savefig('./plot/'+exp+'/vdisp_'+exp+'_'+str(tt)+'.eps')
#print       './plot/'+exp+'/vdisp_'+exp+'_'+str(tt)+'.eps'
#plt.close()
#


# TRAJECTORIES
plt.subplots(figsize=(8,7))

for p in range(0,10000,103):
 plt.plot(time0/86400,par[p,2,:],color='0.5')

plt.plot(time0/86400,par[32,2,:],'r', linewidth=4)
#plt.plot(time0/3600,par[200,2,:-1],'g', linewidth=4)
plt.plot(time0/86400,par[51,2,:],'b', linewidth=4)

plt.xticks(np.linspace(0,10,6),np.linspace(0,10,6).astype(int),fontsize=24)
plt.yticks(fontsize=24)
plt.ylim((-50,0))
plt.xlim(2,9)
plt.xlabel('Time [days]', fontsize=26)
plt.ylabel('Depth [m]', fontsize=26)
plt.tight_layout()
plt.savefig('./plot/'+exp+'/vtraj_'+exp+'_'+str(tt)+'.eps')
print       './plot/'+exp+'/vtraj_'+exp+'_'+str(tt)+'.eps'
plt.close()

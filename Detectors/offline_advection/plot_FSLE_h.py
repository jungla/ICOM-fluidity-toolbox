#!~/python
import lagrangian_stats
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import myfun
import numpy as np
import os
import fio
import advect_functions

exp = 'm_25_1'
filename0 = 'traj_m_25_1_particles_960_480_2D.csv'
filename0 = 'traj_m_25_1_particles_480_960_2D.csv'
filename0 = 'traj_m_25_1_particles_999_0_2D.csv'
tt = 999# IC + 24-48 included
x0 = range(0,2000,25)
y0 = range(0,2000,25)
z0 = range(0,-50,-10)
 
xp = len(x0)
yp = len(y0)
zp = len(z0)
pt = xp*yp*zp
pz = xp*yp

timet, par0 = advect_functions.read_particles_csv(filename0,xp,yp,zp,tt)
par0p = lagrangian_stats.periodicCoords(par0,2000,2000)
time0 = (timet)*360 + 48*3600 - 360
time0 = np.asarray(range(len(timet)))*360 + 48*3600 - 360

par = np.reshape(par0p,(pt,3,tt))

# FSLE
di = 25
time = time0 - time0[0]

rad = [2,5,10]

depths = [0, -10, -20] # z0

def dist(x0,x1,y0,y1,z0,z1):
 dr = np.sqrt((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)
 return dr

for z in range(len(depths)):
 print 'depth', z0[z]
 fsler = np.zeros((xp,yp,zp,len(rad)))*np.nan
 par0z = np.reshape(par,(xp,yp,zp,3,tt))
# par0z[-1,:,:,:,:] = np.nan
# par0z[:,-1,:,:,:] = np.nan
 par0z = par0z[:,:,z,:,:]
 par0z = np.reshape(par0z,(xp*yp,3,tt))

 for r in range(len(rad)):
  parL = range(xp*yp-xp)
  fsle  = np.zeros(pz)
  fslec  = np.zeros((pz,3))*np.nan
  df=rad[r]*di # separation distance
  # 
  # loop triplets in time
  #
  for t in range(tt):
   for p in parL:
 #   print t,len(parL)
  # loop particles
#    for ps in [1,xp]:
     dx = dist(par0z[p,0,t],par0z[p+1 ,0,t],par0z[p,1,t],par0z[p+1 ,1,t],par0z[p,2,t],par0z[p+1 ,2,t])
     dy = dist(par0z[p,0,t],par0z[p+xp,0,t],par0z[p,1,t],par0z[p+xp,1,t],par0z[p,2,t],par0z[p+xp,2,t])
#     print np.round(dx),np.round(dy),t,p
     dr = max(dx,dy)
#     print dr

#     fsle[p] = dr
     fslec[p] = par0z[p,:,0]
     if (dr > df and fsle[p]==0):
#      print np.log(dr),df,time[t]
      fsle[p]  = np.log(rad[r])/(time[t]/3600.)   #  in hours

#     fsle[p]  = dr   # fsle has the dimension of the first triplet

      parL.remove(p)
  #
  # plot fsle
  # 3D arrays of fsle and fslec
  #
  fsler[:,:,z,r] = np.reshape(fsle,(xp,yp))
  # fsler[:,:,-1] = 0
#  fslexr = np.reshape(fslec[:,0],(xp,yp))
#  fsleyr = np.reshape(fslec[:,1],(xp,yp))
  #
  plt.figure()
#  for p in parL:
#   plt.plot((par0z[p,0,0],par0z[p,0,t]),(par0z[p,1,0],par0z[p,1,t]))
  v = np.linspace(np.percentile(fsle[np.where(~np.isnan(fsle))],5),np.percentile(fsle[np.where(~np.isnan(fsle))],95),30)
  plt.contourf(x0,y0,np.transpose(fsler[:,:,z,r]),v,extend='both')
  plt.colorbar()
#  plt.scatter(par0z[:,0,t-1],par0z[:,1,t-1],marker='.', s=35, facecolor='k', lw = 0)
#  plt.contourf(fslexr,fsleyr,fsler[:,:,z,r],30)
  plt.ylabel('Latitude [m]', fontsize=18)
  plt.xlabel('Longitude [m]', fontsize=18)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  #plt.xlim(0,2000)
  #plt.ylim(0,2000)
  plt.savefig('./plot/'+exp+'/fsle_h_'+exp+'_r'+str(rad[r])+'_'+str(z0[z])+'.eps',bbox_inches='tight')
  print       './plot/'+exp+'/fsle_h_'+exp+'_r'+str(rad[r])+'_'+str(z0[z])+'.eps'
plt.close()

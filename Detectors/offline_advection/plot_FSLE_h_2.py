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
filename0 = 'traj_m_25_1_particles_960_0_2D.csv'
tt = 960# IC + 24-48 included
x0 = range(0,2000,25)
y0 = range(0,2000,25)
z0 = range(0,-50,-10)

xn = len(x0)
yn = len(y0)
zn = len(z0)
pn = xn*yn*zn

dx = 5
dy = 5

xt = []
yt = []
zt = []

for z in z0:
 print z
 for x in x0:
  for y in y0:
   for p in range(3):
    if p == 0:
     xt.append(x)
     yt.append(y)
     zt.append(z)
    if p == 1:
     xt.append(x+dx)
     yt.append(y)
     zt.append(z)
    if p == 2:
     xt.append(x)
     yt.append(y+dy)
     zt.append(z)

xt = np.asarray(xt)
yt = np.asarray(yt)
zt = np.asarray(zt)

#x0 = range(3000,4000,50)
#y0 = range(2000,3000,50)
#z0 = range(0,30,1)
xp = len(xt)
yp = len(yt)
zp = len(zt)

pt = xp
timet, par0 = advect_functions.read_particles_csv(filename0,pt,tt)
par0p = lagrangian_stats.periodicCoords(par0,2000,2000)
time0 = (timet)*360 + 48*3600 - 360
time0 = np.asarray(range(len(timet)))*360 + 48*3600 - 360

par = np.reshape(par0p,(pt,3,tt))

# FSLE
di = 5
time = time0 - time0[0]

rad = [5,10,20]

depths = [0, -10, -20] # z0

def dist(x0,x1,y0,y1,z0,z1):
 dr = np.sqrt((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)
 return dr

def dist_2D(x0,x1,y0,y1):
 dr = np.sqrt((x0-x1)**2 + (y0-y1)**2)
 return dr

for z in range(len(depths)):
 print 'depth', z0[z]
 fsler = np.zeros((xn,yn,zn,len(rad)))*np.nan
 par0z = par[pt/len(z0)*z:pt/len(z0)*(z+1),:,:]

 for r in range(len(rad)):
  parL = range(0,xn*yn*3,3)
  fsle  = np.zeros(xn*yn)
  df=rad[r]*di # separation distance
  # 
  # loop triplets in time
  #
  for t in range(tt):
   for p in parL:
 #   print t,len(parL)
  # loop particles
#    for ps in [1,xp]:
     d0 = dist_2D(par0z[p,0,t],par0z[p+1,0,t],par0z[p,1,t],par0z[p+1,1,t])
     d1 = dist_2D(par0z[p,0,t],par0z[p+2,0,t],par0z[p,1,t],par0z[p+2,1,t])
#     print np.round(dx),np.round(dy),t,p
     dr = max(d1,d0)
#     print dr
     if (dr > df and fsle[p/3]==0):
#      print np.log(dr),df,time[t]
      fsle[p/3]  = np.log(rad[r])/(time[t]/3600.)   #  in hours

#     fsle[p]  = dr   # fsle has the dimension of the first triplet

      parL.remove(p)
  #
  # plot fsle
  # 3D arrays of fsle and fslec
  #
  fsler[:,:,z,r] = np.transpose(np.reshape(fsle,(xn,yn)))
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

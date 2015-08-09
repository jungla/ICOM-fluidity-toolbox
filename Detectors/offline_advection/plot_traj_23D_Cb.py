#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import myfun
import numpy as np
import os
import fio
import lagrangian_stats
import advect_functions
from matplotlib.patches import Ellipse
import scipy.stats

#def RD_t(par2Dz,tt):
# pt = len(par2Dz)
## RD_2D=[]
# RD_2D = np.zeros((pt/10,tt))
# for i in xrange(0,pt,10):
#  print i,pt
##   print np.mean(((par2Dz[i,0,:] - par2Dz[j,0,:])**2 + (par2Dz[i,1,:] - par2Dz[j,1,:])**2),0)
#   RD_2D[100] = ,np.histogram(np.sqrt((par2Dz[:,0,t] - par2Dz[j,0,t])**2 + (par2Dz[:,1,:] - par2Dz[j,1,:])**2))
## print RD_2D
# return np.mean(RD_2D,0)

exp = 'm_25_1b_particles'

filename2D = 'traj_m_25_1b_particles_0_600_2D.csv' 
filename3D = 'traj_m_25_1b_particles_0_600_3D.csv'

tt = 600# IC + 24-48 included

x0 = range(3500,4510,10)
y0 = range(3500,4510,10)
z0 = [0,5,10,15] #range(2,5,1)
#x0 = range(5000,4010,10)
#y0 = range(2000,3010,10)
#z0 = range(1,20,4)

#tt = 230 # IC + 24-48 included
#x0 = range(5000,4010,10)
#y0 = range(2000,3010,10)
#z0 = range(1,20,4)

xp = len(x0)
yp = len(y0)
zp = len(z0)
pt = xp*yp*zp

#time2D, par2D = advect_functions.read_particles_csv(filename2D,pt,tt)
#par2D = lagrangian_stats.periodicCoords(par2D,8000,8000)
#time3D, par3D = advect_functions.read_particles_csv(filename3D,pt,tt)
#par3D = lagrangian_stats.periodicCoords(par3D,8000,8000)
##
#time2D = (time2D)*1440
#time3D = (time3D)*1440
##
time0 = time2D

# horizontal
depths = [1,2,3] #[1, 5, 11, 17, 26]
#depths = [5]
#depths = [1] #, 17, 1]


#K2xt=[]
#K3xt=[]
#K2yt=[]
#K3yt=[]
K2t=[]
K3t=[]

steps = 20

hist_2D = np.zeros((steps,tt))
hist_3D = np.zeros((steps,tt))


for z in depths: #range(len(depths)): 
 print 'depth', z
 par2Dz = np.reshape(par2D,(xp,yp,zp,3,tt))
 par2Dzr = par2Dz[:,:,z,:,:]
 par2Dz = np.reshape(par2Dzr,(xp*yp,3,tt))

 par3Dz = np.reshape(par3D,(xp,yp,zp,3,tt))
 par3Dzr = par3Dz[:,:,z,:,:]
 par3Dz = np.reshape(par3Dzr,(xp*yp,3,tt))
 #

# K2x=[]
# K3x=[]
# K2y=[]
# K3y=[]
 K2=[]
 K3=[]
 
 time = []
 for t in range(0,tt,1):
  time.append(time0[t])
  print 'time', time0[t]/24

#  fig = plt.figure(figsize=(8,8))
#  ax = fig.add_subplot(111, aspect='equal')
#  #
#  plt.plot((3.5,3.5),(3.5,4.5),'k',linewidth=2)
#  plt.plot((3.5,4.5),(3.5,3.5),'k',linewidth=2)
#  plt.plot((4.5,4.5),(3.5,4.5),'k',linewidth=2)
#  plt.plot((3.5,4.5),(4.5,4.5),'k',linewidth=2)
#
#  plt.plot((0,0),(0,8),'k',linewidth=2)
#  plt.plot((0,8),(0,0),'k',linewidth=2)
#  plt.plot((8,8),(0,8),'k',linewidth=2)
#  plt.plot((0,8),(8,8),'k',linewidth=2)
#  #
#  s3D = plt.scatter(par3Dz[:,0,t]/1000, par3Dz[:,1,t]/1000, marker='.', s=35, facecolor='r', lw = 0)
#  s2D = plt.scatter(par2Dz[:,0,t]/1000, par2Dz[:,1,t]/1000, marker='.', s=35, facecolor='b', lw = 0)
#  #
#  plt.legend((s3D,s2D),('3D','2D'))
#
#  print 'Saving 0 to eps'
#
#  xt3 = par3Dz[:,0,t] - np.mean(par3Dz[:,0,t])
#  yt3 = par3Dz[:,1,t] - np.mean(par3Dz[:,1,t])
#  xt2 = par2Dz[:,0,t] - np.mean(par2Dz[:,0,t])
#  yt2 = par2Dz[:,1,t] - np.mean(par2Dz[:,1,t])
# #
#  cov3 = np.cov(xt3/1000, yt3/1000)
#  lambda_3, v = np.linalg.eig(cov3)
#  lambda_3 = np.sqrt(lambda_3)
#  theta3 = np.rad2deg(0.5*np.arctan2(2*cov3[1,0],(cov3[0,0]-cov3[1,1])))
#  theta3 = np.rad2deg(np.arcsin(v[0, 0]))
#  #
#  cov2 = np.cov(xt2/1000, yt2/1000)
#  lambda_2, v = np.linalg.eig(cov2)
#  lambda_2 = np.sqrt(lambda_2)
#  theta2 = np.rad2deg(0.5*np.arctan2(2*cov2[1,0],(cov2[0,0]-cov2[1,1]))) + np.pi*0.5
#  theta2 = np.rad2deg(np.arcsin(v[0, 0]))
#  #
#  e0 = Ellipse(xy=(np.mean(par3Dz[:,0,t])/1000,np.mean(par3Dz[:,1,t])/1000),width=4*lambda_3[1],height=4*lambda_3[0],angle=theta3)
#  e1 = Ellipse(xy=(np.mean(par2Dz[:,0,t])/1000,np.mean(par2Dz[:,1,t])/1000),width=4*lambda_2[1],height=4*lambda_2[0],angle=theta2)
#
#  ax.add_artist(e0)
#  e0.set_facecolor('none')
#  e0.set_edgecolor('k')
#  e0.set_linewidth(2.5)
#
#  ax.add_artist(e1)
#  e1.set_facecolor('none')
#  e1.set_edgecolor('k')
#  e1.set_linewidth(2.5)
#  e1.set_linestyle('dashed')
#
# # 
##  ax.text(0, 3.5, str(z0[z])+'m, '+str(time0[t]/3600)+'h', fontsize=18)
#  plt.xlabel('X [km]', fontsize=18)
#  plt.ylabel('Y [km]', fontsize=18)
#  plt.xticks(fontsize=16)
#  plt.yticks(fontsize=16)
#  plt.xlim([-1 , 9])
#  plt.ylim([-1 , 9])
#  plt.savefig('./plot/'+exp+'/traj_'+exp+'_z'+str(z0[z])+'_'+str(time0[t])+'_h.eps')
#  print       './plot/'+exp+'/traj_'+exp+'_z'+str(z0[z])+'_'+str(time0[t])+'_h.eps'
#  plt.close()
# 
#  # plot ellipse
#
## vertical
#
#  fig = plt.figure(figsize=(8,8))
#
#  plt.plot((3.5,3.5),(0,-50),'k',linewidth=2)
#  plt.plot((4.5,4.5),(0,-50),'k',linewidth=2)
#  #plt.plot((3,3),(0,-50),'k')
#  #plt.plot((4,4),(0,-50),'k')
#  #
#  s2D = plt.scatter(par2Dz[:,0,t]/1000, par2Dz[:,2,t],  marker='.', s=35, facecolor='b', lw = 0)
#  s3D = plt.scatter(par3Dz[:,0,t]/1000, par3Dz[:,2,t],  marker='.', s=35, facecolor='r', lw = 0)
#
#  plt.legend((s3D,s2D),('3D','2D'),loc=4)
#  #
#  plt.xlim([-1, 9])
#  plt.ylim([-50, 0])
#  #
#  print 'Saving 0 to eps'
#  # 
#
##  plt.text(-2, -40, str(z0[z])+'m, '+str(time0[t]/3600)+'h', fontsize=18)
#  plt.xlabel('X [km]', fontsize=18)
#  plt.ylabel('Z [m]', fontsize=18)
#  plt.xticks(fontsize=16)
#  plt.yticks(fontsize=16)
#  plt.savefig('./plot/'+exp+'/traj_'+exp+'_z'+str(z0[z])+'_'+str(time0[t])+'_v.eps')
#  print       './plot/'+exp+'/traj_'+exp+'_z'+str(z0[z])+'_'+str(time0[t])+'_v.eps'
#  plt.close()
#
#
#  # PDF
#  fig = plt.figure(figsize=(8,6))
#  h2d,poop = np.histogram(par2Dz[:,0,t]-np.mean(par2Dz[:,0,t]), steps, (-5000,5000),normed=True)
#  h3d,poop = np.histogram(par3Dz[:,0,t]-np.mean(par3Dz[:,0,t]), steps, (-5000,5000),normed=True)
#  vals = np.linspace(-5,5,steps)
#  s2D, = plt.step(vals,h2d,'b',linewidth=2)
#  s3D, = plt.step(vals,h3d,'r',linewidth=2)
#  plt.legend((s3D,s2D),('3D','2D'))
#  plt.xlabel('X [km]', fontsize=18)
#  plt.ylabel('normalized pdf', fontsize=18)
#  plt.xticks(fontsize=16)
#  plt.yticks(fontsize=16)
#  plt.tight_layout()
#  plt.savefig('./plot/'+exp+'/traj_'+exp+'_z'+str(z0[z])+'_'+str(time0[t])+'_histx.eps')
#  print       './plot/'+exp+'/traj_'+exp+'_z'+str(z0[z])+'_'+str(time0[t])+'_histx.eps'
## 
#  fig = plt.figure(figsize=(8,6))
#  h2d,poop = np.histogram(par2Dz[:,1,t]-np.mean(par2Dz[:,1,t]), steps, (-5000,5000),normed=True)
#  h3d,poop = np.histogram(par3Dz[:,1,t]-np.mean(par3Dz[:,1,t]), steps, (-5000,5000),normed=True)
#  vals = np.linspace(-5,5,steps)
#  s2D, = plt.step(vals,h2d,'b',linewidth=2)
#  s3D, = plt.step(vals,h3d,'r',linewidth=2)
#  plt.legend((s3D,s2D),('3D','2D'))
#  plt.xlabel('Y [km]', fontsize=18)
#  plt.ylabel('normalized pdf', fontsize=18)
#  plt.xticks(fontsize=16)
#  plt.yticks(fontsize=16)
#  plt.tight_layout()
#  plt.savefig('./plot/'+exp+'/traj_'+exp+'_z'+str(z0[z])+'_'+str(time0[t])+'_histy.eps')
#  print       './plot/'+exp+'/traj_'+exp+'_z'+str(z0[z])+'_'+str(time0[t])+'_histy.eps'
#
##  fig = plt.figure(figsize=(8,6))
###  h2d,poop = np.histogram(((par2Dz[:,0,t]-np.mean(par2Dz[:,0,t]))**2+(par2Dz[:,1,t]-np.mean(par2Dz[:,1,t]))**2)**0.5, 100, (0,10000),normed=True)
###  h3d,poop = np.histogram(((par3Dz[:,0,t]-np.mean(par3Dz[:,0,t]))**2+(par3Dz[:,1,t]-np.mean(par3Dz[:,1,t]))**2)**0.5, 100, (0,10000),normed=True)
##  h2d,poop = np.histogram(((par2Dz[:,0,t]-np.mean(par2Dz[:,0,t]))**2+(par2Dz[:,1,t]-np.mean(par2Dz[:,1,t]))**2)**0.5, 100, (0,10000),normed=True)
##  h3d,poop = np.histogram(((par3Dz[:,0,t]-np.mean(par3Dz[:,0,t]))**2+(par3Dz[:,1,t]-np.mean(par3Dz[:,1,t]))**2)**0.5, 100, (0,10000),normed=True)
##  vals = np.linspace(0,10,100)
##  s2D, = plt.step(vals,h2d,'b',linewidth=2)
##  s3D, = plt.step(vals,h3d,'r',linewidth=2)
##  plt.legend((s3D,s2D),('3D','2D'))
##  plt.xlabel('X [km]', fontsize=18)
##  plt.ylabel('normalized pdf', fontsize=18)
##  plt.xticks(fontsize=16)
##  plt.yticks(fontsize=16)
##  plt.tight_layout()
##  plt.savefig('./plot/'+exp+'/traj_'+exp+'_z'+str(z0[z])+'_'+str(time0[t])+'_hist.eps')
##  print       './plot/'+exp+'/traj_'+exp+'_z'+str(z0[z])+'_'+str(time0[t])+'_hist.eps'
##
#  # compute Kurtosis in time
#
#
  pt = len(par2Dz)
#
  for j in xrange(0,pt,10):
 #   print np.mean(((par2Dz[i,0,:] - par2Dz[j,0,:])**2 + (par2Dz[i,1,:] - par2Dz[j,1,:])**2),0)
   out,bins = np.histogram(np.sqrt((par2Dz[:,0,t] - par2Dz[j,0,t])**2 + (par2Dz[:,1,t] - par2Dz[j,1,t])**2), steps,(0,10000),normed=True)
   hist_2D[:,t] = hist_2D[:,t] + out
   out,bins = np.histogram(np.sqrt((par3Dz[:,0,t] - par3Dz[j,0,t])**2 + (par3Dz[:,1,t] - par3Dz[j,1,t])**2), steps,(0,10000),normed=True)
   hist_3D[:,t] = hist_3D[:,t] + out

  hist_2D[:,t] = hist_2D[:,t]/pt*10
  hist_3D[:,t] = hist_3D[:,t]/pt*10

 
  fig = plt.figure(figsize=(8,6))
  vals = np.linspace(0,10,steps)
  s2D, = plt.step(vals,hist_2D[:,t],'b',linewidth=2)
  s2D, = plt.step(-vals,hist_2D[:,t],'b',linewidth=2)
  s3D, = plt.step(vals,hist_3D[:,t],'r',linewidth=2)
  s3D, = plt.step(-vals,hist_3D[:,t],'r',linewidth=2)
  plt.legend((s3D,s2D),('3D','2D'))
  plt.xlabel('relative distance [km]', fontsize=18)
  plt.ylabel('normalized pdf', fontsize=18)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.tight_layout()
  plt.savefig('./plot/'+exp+'/traj_'+exp+'_z'+str(z0[z])+'_'+str(time0[t])+'_hist.eps')
  print       './plot/'+exp+'/traj_'+exp+'_z'+str(z0[z])+'_'+str(time0[t])+'_hist.eps'
  
  K2.append(scipy.stats.kurtosis(np.hstack((hist_2D[:,t-1],hist_2D[::-1,t-1])), fisher=False, bias=False))
  K3.append(scipy.stats.kurtosis(np.hstack((hist_3D[:,t-1],hist_3D[::-1,t-1])), fisher=False, bias=False))
  
 K2t.append(K2)
 K3t.append(K3)
#  K2x.append(scipy.stats.kurtosis(par2Dz[:,0,t]-np.mean(par2Dz[:,0,t]), fisher=True, bias=False))
#  K3x.append(scipy.stats.kurtosis(par3Dz[:,0,t]-np.mean(par3Dz[:,0,t]), fisher=True, bias=False))
#  K2y.append(scipy.stats.kurtosis(par2Dz[:,1,t]-np.mean(par2Dz[:,1,t]), fisher=True, bias=False))
#  K3y.append(scipy.stats.kurtosis(par3Dz[:,1,t]-np.mean(par3Dz[:,1,t]), fisher=True, bias=False))
#
# K2xt.append(K2x)
# K3xt.append(K3x)
# K2yt.append(K2y)
# K3yt.append(K3y)
 # plot Kurtisis
 time = np.asarray(time)

# # Y
#
# plt.subplots(figsize=(8,6)) 
## pK2x, = plt.plot(time/3600,K2x,'r-',linewidth=2)
## pK2y, = plt.plot(time/3600,K2y,'r--',linewidth=2)
# pK2, = plt.plot(time/3600,K2x,'b',linewidth=2)
# pK3, = plt.plot(time/3600,K3x,'r',linewidth=2)
# 
# plt.xlabel('Time [hr]', fontsize=18)
# plt.ylabel('Kurtosis', fontsize=18)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
#
# plt.legend([pK2,pK3],['$\gamma_{2D}$','$\gamma_{3D}$'],loc=1)
# #plt.xlim((time[0],time[-1]))
# plt.savefig('./plot/'+exp+'/traj_'+exp+'_z'+str(z0[z])+'_kurt_x.eps')
# print       './plot/'+exp+'/traj_'+exp+'_z'+str(z0[z])+'_kurt_x.eps'
# plt.close()
#
# # Y
#
# plt.subplots(figsize=(8,6))
## pK2x, = plt.plot(time/3600,K2x,'r-',linewidth=2)
## pK2y, = plt.plot(time/3600,K2y,'r--',linewidth=2)
# pK2, = plt.plot(time/3600,K2y,'b',linewidth=2)
# pK3, = plt.plot(time/3600,K3y,'r',linewidth=2)
#
# plt.xlabel('Time [hr]', fontsize=18)
# plt.ylabel('Kurtosis', fontsize=18)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
#
# plt.legend([pK2,pK3],['$\gamma_{2D}$','$\gamma_{3D}$'],loc=1)
# #plt.xlim((time[0],time[-1]))
# plt.savefig('./plot/'+exp+'/traj_'+exp+'_z'+str(z0[z])+'_kurt_y.eps')
# print       './plot/'+exp+'/traj_'+exp+'_z'+str(z0[z])+'_kurt_y.eps'
# plt.close()
#
#
#K2xt = np.asarray(K2xt)
#K3xt = np.asarray(K3xt)
#K2yt = np.asarray(K2yt)
#K3yt = np.asarray(K3yt)

K3t = np.asarray(K3t)
K2t = np.asarray(K2t)

# plot Kurtisis
time = np.asarray(time)

#
#plt.subplots(figsize=(8,6))
#
#pK20, = plt.plot(time/86400,K2xt[0,:],'b',linewidth=2)
#pK30, = plt.plot(time/86400,K3xt[0,:],'r',linewidth=2)
#pK21, = plt.plot(time/86400,K2xt[1,:],'b--',linewidth=2)
#pK31, = plt.plot(time/86400,K3xt[1,:],'r--',linewidth=2)
#pK22, = plt.plot(time/86400,K2xt[2,:],'b-.',linewidth=2)
#pK32, = plt.plot(time/86400,K3xt[2,:],'r-.',linewidth=2)
#
#plt.xlabel('Time [days]', fontsize=18)
#plt.ylabel('Kurtosis X', fontsize=18)
#plt.xticks(fontsize=16)
#plt.yticks(fontsize=16)
#plt.tight_layout()
#plt.legend([pK20,pK30,pK21,pK31,pK22,pK32],['$\gamma_{2D}$ 5m','$\gamma_{3D}$ 5m','$\gamma_{2D}$ 10m','$\gamma_{3D}$ 10m','$\gamma_{2D}$ 17m','$\gamma_{3D}$ 17m'],loc=1,ncol=3)
##plt.xticks(range(int(time[0]/86400),int(time[-1]/86400),12),range(int(time[0]/3600),int(time[-1]/3600),12))
#plt.xticks(np.linspace(0,9,10),np.linspace(0,9,10).astype(int))
#plt.xlim((time[0]/86400,9))
##plt.ylim((1,6))
#plt.savefig('./plot/'+exp+'/traj_'+exp+'_kurt_x.eps')
#print       './plot/'+exp+'/traj_'+exp+'_kurt_x.eps'
#plt.close()
#
#
#
#plt.subplots(figsize=(8,6))
#
#pK20, = plt.plot(time/86400,K2yt[0,:],'b',linewidth=2)
#pK30, = plt.plot(time/86400,K3yt[0,:],'r',linewidth=2)
#pK21, = plt.plot(time/86400,K2yt[1,:],'b--',linewidth=2)
#pK31, = plt.plot(time/86400,K3yt[1,:],'r--',linewidth=2)
#pK22, = plt.plot(time/86400,K2yt[2,:],'b-.',linewidth=2)
#pK32, = plt.plot(time/86400,K3yt[2,:],'r-.',linewidth=2)
#
#plt.xlabel('Time [days]', fontsize=18)
#plt.ylabel('Kurtosis Y', fontsize=18)
#plt.xticks(fontsize=16)
#plt.yticks(fontsize=16)
#plt.tight_layout()
#plt.legend([pK20,pK30,pK21,pK31,pK22,pK32],['$\gamma_{2D}$ 5m','$\gamma_{3D}$ 5m','$\gamma_{2D}$ 10m','$\gamma_{3D}$ 10m','$\gamma_{2D}$ 17m','$\gamma_{3D}$ 17m'],loc=1,ncol=3)
##plt.xticks(range(int(time[0]/86400),int(time[-1]/86400),12),range(int(time[0]/3600),int(time[-1]/3600),12))
#plt.xticks(np.linspace(0,9,10),np.linspace(0,9,10).astype(int))
#plt.xlim((time[0]/86400,time[-1]/86400))
##plt.ylim((1,7))
#plt.savefig('./plot/'+exp+'/traj_'+exp+'_kurt_y.eps')
#print       './plot/'+exp+'/traj_'+exp+'_kurt_y.eps'
#plt.close()
#

plt.subplots(figsize=(8,6))

pK20, = plt.plot(time/86400,K2t[0,:],'b',linewidth=2)
pK30, = plt.plot(time/86400,K3t[0,:],'r',linewidth=2)
pK21, = plt.plot(time/86400,K2t[1,:],'b--',linewidth=2)
pK31, = plt.plot(time/86400,K3t[1,:],'r--',linewidth=2)
pK22, = plt.plot(time/86400,K2t[2,:],'b-.',linewidth=2)
pK32, = plt.plot(time/86400,K3t[2,:],'r-.',linewidth=2)

plt.xlabel('Time [days]', fontsize=18)
plt.ylabel('Kurtosis', fontsize=18)
plt.ylim(-2,10)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.legend([pK20,pK30,pK21,pK31,pK22,pK32],['$\gamma_{2D}$ 5m','$\gamma_{3D}$ 5m','$\gamma_{2D}$ 10m','$\gamma_{3D}$ 10m','$\gamma_{2D}$ 17m','$\gamma_{3D}$ 17m'],loc=4,ncol=3)
#plt.xticks(range(int(time[0]/86400),int(time[-1]/86400),12),range(int(time[0]/3600),int(time[-1]/3600),12))
plt.xticks(np.linspace(0,9,10),np.linspace(0,9,10).astype(int))
plt.xlim((time[0]/86400,time[-1]/86400))
#plt.ylim((1,7))
plt.savefig('./plot/'+exp+'/traj_'+exp+'_kurt.eps')
print       './plot/'+exp+'/traj_'+exp+'_kurt.eps'
plt.close()




#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import myfun
import numpy as np
from scipy import optimize
import os
import scipy.stats as sp
import scipy
import lagrangian_stats
import advect_functions
import csv

# read offline
print 'reading offline'

#exp = 'm_25_1_tracer'
#label = 'm_25_1_tracer'
#filename = 'traj_m_25_1_tracer_0_640_.csv'
#filename3D = 'traj_m_25_1_tracer_0_640_3D.csv'
#tt = 640 # IC + 24-48 included

label = 'm_25_2b_particles'
label_BW = 'm_25_2b_particles'
label_B = 'm_25_1b_particles'

filename_BW = './csv/CDv_'+label_BW+'.csv'
filename_B = './csv/CDv_'+label_B+'.csv'

tt = 500-61

depths = ['5','10','15']

time_B, CD_B =lagrangian_stats.read_dispersion(filename_B)
time_BW, CD_BW =lagrangian_stats.read_dispersion(filename_BW)

Diff_B = [] #np.zeros((tt_B,nl))
Diff_BW = [] #np.zeros((tt_B,nl))

gradBW,poop = np.gradient(CD_BW)
Diff_BW.append(0.5*gradBW/1440.)

gradB,poop = np.gradient(CD_B)
Diff_B.append(0.5*gradB/1440.)

Diff_B = np.squeeze(np.asarray(Diff_B))
Diff_BW = np.squeeze(np.asarray(Diff_BW))

# relative D

# CD 

fig = plt.figure(figsize=(8,7))

pB5 = plt.scatter(np.log10(3*np.sqrt(CD_B[1:200,0])),np.log10(Diff_B[1:200,0]),s=60,color=[1,0,0],marker='o')
pB10 = plt.scatter(np.log10(3*np.sqrt(CD_B[1:200,1])),np.log10(Diff_B[1:200,1]),s=60,color=[0,1,0],marker='o')
pB15 = plt.scatter(np.log10(3*np.sqrt(CD_B[1:200,2])),np.log10(Diff_B[1:200,2]),s=60,color=[0,0,1],marker='o')

OKx = np.linspace(1,5)
OKy = 0.0103*OKx**1.15
Rcy = 10**-4.0*OKx**1.33 #33
OK, = plt.plot(np.log10(OKx),np.log10(Rcy),'k-',linewidth=2)
#Rch, = plt.plot(OKx,Rcy,'k--',linewidth=2)
#plt.xlim(0,np.log10(100))

plt.ylabel(r'$log(K^\prime_z)$ $[m^2\,s^{-2}]$',fontsize=24)
plt.xlabel(r'$3\sigma_{C_z}$ $[m]$',fontsize=24)
plt.legend((OK,pB5,pB10,pB15),('Richardson','B 5m','B 10m','B 15m'),loc=4,fontsize=16,ncol=2)
#ax.set_yscale('log')
#ax.set_xscale('log')

plt.ylim(-4.8,-2.8)
v = np.asarray([1,5,10,20,50]) #np.linspace(5,50,10)
vl = np.log10(v)
plt.yticks(fontsize=20)
plt.xticks(vl, v,fontsize=20)

plt.savefig('./plot/'+label_B+'/Diffv_'+label_B+'.eps',bbox_inches='tight')
print       './plot/'+label_B+'/Diffv_'+label_B+'.eps'
plt.close()


fig = plt.figure(figsize=(8,7))

pB5 = plt.scatter(np.log10(3*np.sqrt(CD_BW[1:200,0])),np.log10(Diff_BW[1:200,0]),s=60,color=[1,0,0],marker='o')
pB10 = plt.scatter(np.log10(3*np.sqrt(CD_BW[1:200,1])),np.log10(Diff_BW[1:200,1]),s=60,color=[0,1,0],marker='o')
pB15 = plt.scatter(np.log10(3*np.sqrt(CD_BW[1:200,2])),np.log10(Diff_BW[1:200,2]),s=60,color=[0,0,1],marker='o')

OKx = np.linspace(1,5)
OKy = 0.0103*OKx**1.15
Rcy = 10**-4.0*OKx**1.33 #33
OK, = plt.plot(np.log10(OKx),np.log10(Rcy),'k-',linewidth=2)
#Rch, = plt.plot(OKx,Rcy,'k--',linewidth=2)
#plt.xlim(0,np.log10(100))

plt.ylabel(r'$log(K^\prime_z)$ $[m^2\,s^{-2}]$',fontsize=24)
plt.xlabel(r'$3\sigma_{C_z}$ $[m]$',fontsize=24)
plt.legend((OK,pB5,pB10,pB15),('Richardson','B 5m','B 10m','B 15m'),loc=4,fontsize=16,ncol=2)


plt.ylabel(r'$log(K^\prime_z)$ $[m^2\,s^{-2}]$',fontsize=24)
plt.xlabel(r'$3\sigma_{C_z}$ $[m]$',fontsize=24)
plt.legend((pB5,pB10,pB15),('BW 5m','BW 10m','BW 15m'),loc=4,fontsize=16,ncol=3)
#ax.set_yscale('log')
#ax.set_xscale('log')
plt.ylim(-4.8,-2.8)

v = np.asarray([1,5,10,20,50]) #np.linspace(5,50,10)
vl = np.log10(v)
plt.yticks(fontsize=20)
plt.xticks(vl, v.astype(int),fontsize=20)

plt.savefig('./plot/'+label_BW+'/Diffv_'+label_BW+'.eps',bbox_inches='tight')
print       './plot/'+label_BW+'/Diffv_'+label_BW+'.eps'
plt.close()


# depths

for d in range(len(depths)): 
 fig = plt.figure(figsize=(8,7))

 print np.min(Diff_B[1:200,d])
 print np.max(Diff_B[1:200,d])
 print np.min(Diff_BW[1:200,d])
 print np.max(Diff_BW[1:200,d])
 
 pB5 = plt.scatter(np.log10(3*np.sqrt(CD_B[1:200,d])),np.log10(Diff_B[1:200,d]),s=60,color=[0,0,1],marker='o')
 pBW5 = plt.scatter(np.log10(3*np.sqrt(CD_BW[1:200,d])),np.log10(Diff_BW[1:200,d]),s=60,color=[1,0,0],marker='o')
 
 OKx = np.linspace(1,5)
 OKy = 0.0103*OKx**1.15
 Rcy = 10**-4.0*OKx**1.33 #33
 OK, = plt.plot(np.log10(OKx),np.log10(Rcy),'k-',linewidth=2)
 #Rch, = plt.plot(OKx,Rcy,'k--',linewidth=2)
 #plt.xlim(0,np.log10(100))
 
 plt.ylabel(r'$log(K^\prime_z)$ $[m^2\,s^{-2}]$',fontsize=24)
 plt.xlabel(r'$3\sigma_{C_z}$ $[m]$',fontsize=24)
 plt.legend((OK,pB5,pBW5),('Richardson','B 5m','BW 5m'),loc=3,fontsize=16,ncol=3)
 
 #ax.set_yscale('log')
 #ax.set_xscale('log')
 plt.ylim(-4.8,-2.8)
 
 v = np.asarray([1,5,10,25,50]) #np.linspace(5,50,10)
 vl = np.log10(v)
 plt.yticks(fontsize=20)
 plt.xticks(vl, v.astype(int),fontsize=20)
 
 plt.savefig('./plot/'+label+'/Diffv_'+label+'_'+depths[d]+'.eps',bbox_inches='tight')
 print       './plot/'+label+'/Diffv_'+label+'_'+depths[d]+'.eps'
 plt.close()






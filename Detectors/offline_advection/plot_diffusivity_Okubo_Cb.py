#!~/python
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import myfun
import numpy as np
import os, csv
import advect_functions
import lagrangian_stats

# read offline
print 'reading offline'

exp = 'm_25_1b_particles'
label = 'm_25_1b_particles'
filename2D = 'traj_m_25_1b_particles_0_600_2D.csv'
filename3D = 'traj_m_25_1b_particles_0_600_3D.csv'
tt =  600

dayi = 0
dayf = tt
days = 1

x0 = range(3500,4510,10)
y0 = range(3500,4510,10)
z0 = [0, 5, 10, 15]

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
#
time0 = time2D[:-1] - time2D[0] #+ 5*3600

depths = [1, 2, 3]

nl = len(depths)

CD_2D = np.zeros((tt,nl))
CD_3D = np.zeros((tt,nl))
CD_Tr = np.zeros((tt,nl))

vi = 10
vf = -1
vfp = 500

time = time0[vi:vf]
timep = time0[vi:vfp]

for z in range(len(depths)):
 print z
 print 'depth', depths[z]
 par2Dz = np.reshape(par2D,(xp,yp,zp,3,tt))
 par3Dz = np.reshape(par3D,(xp,yp,zp,3,tt))
 par2Dzr = par2Dz[:,:,depths[z],:,:]
 par3Dzr = par3Dz[:,:,depths[z],:,:]
 #
 par2Dz = np.reshape(par2Dzr,(xp*yp,3,tt))
 par3Dz = np.reshape(par3Dzr,(xp*yp,3,tt))
 #
 CD_2D[:,z] = 100**2*lagrangian_stats.RD_t(par2Dzr,tt,100)
 CD_3D[:,z] = 100**2*lagrangian_stats.RD_t(par3Dzr,tt,100)


depths = [1,5, 17]
Trid = [1,2, 4]


xm = 2*10**3
xM = 5*10**6
ym = 10**1
yM = 3*10**5

OKx = np.linspace(xm,xM)
OKy = 0.0103*OKx**1.15
Rcy = 0.009*OKx**1.33

# B  only
fig = plt.figure(figsize=(8,7))
ax = plt.gca()
z = 0
p2D0 = ax.scatter(3*np.sqrt(CD_2D[vi:vf,z]),0.25*1./time*CD_2D[vi:vf,z],s=60,color=[1,0,0],marker='o') # color=[z/float(nl),z/float(nl),1          ])
z = 1
p2D1 = ax.scatter(3*np.sqrt(CD_2D[vi:vf,z]),0.25*1./time*CD_2D[vi:vf,z],s=60,color=[0,1,0],marker='o') # color=[z/float(nl),z/float(nl),1          ])
z = 2
p2D2 = ax.scatter(3*np.sqrt(CD_2D[vi:vf,z]),0.25*1./time*CD_2D[vi:vf,z],s=60,color=[0,0,1],marker='o') # color=[z/float(nl),z/float(nl),1          ])

xrefid, = np.where(3*np.sqrt(CD_2D[:,0])>10**5)
xref = 3*np.sqrt(CD_2D[xrefid[0],0])
print 'diff 2D z0 vs OK at 10^5', 0.0103*xref**1.15/(0.25*1./time0[xrefid[0]]*CD_2D[xrefid[0],0]) 

xrefid, = np.where(3*np.sqrt(CD_2D[:,1])>10**5)
xref = 3*np.sqrt(CD_2D[xrefid[0],1])
print 'diff 2D z0 vs OK at 10^5', 0.0103*xref**1.15/(0.25*1./time0[xrefid[0]]*CD_2D[xrefid[0],1])

xrefid, = np.where(3*np.sqrt(CD_2D[:,2])>10**5)
xref = 3*np.sqrt(CD_2D[xrefid[0],2])
print 'diff 2D z0 vs OK at 10^5', 0.0103*xref**1.15/(0.25*1./time0[xrefid[0]]*CD_2D[xrefid[0],2])

OK, = plt.plot(OKx,OKy,'k-',linewidth=2)
Rch, = plt.plot(OKx,Rcy,'k--',linewidth=2)
plt.legend([OK,Rch,p2D0,p2D1,p2D2],['Okubo','Richardson','2D 5m','2D 10m','2D 15m'],loc=4,fontsize=20)

ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel(r'$3\sigma_D$ $[cm]$',fontsize=28)
#plt.ylabel(r'$\frac{\sigma^2_D}{4t}$ $[cm^2s^{-1}$]',fontsize=21)
plt.ylabel(r'$k_D$ $[cm^2s^{-1}]$',fontsize=28)
plt.xlim([xm,xM])
plt.ylim([ym,yM])
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)
plt.tight_layout()
plt.savefig('./plot/'+label+'/Diff_O_'+label+'_2D.eps')
print './plot/'+label+'/Diff_O_'+label+'_2D.eps'
plt.close()


# BW only

fig = plt.figure(figsize=(8,7))
ax = plt.gca()
z = 0 
p3D0 = ax.scatter(3*np.sqrt(CD_3D[vi:vf,z]),0.25*1./time*CD_3D[vi:vf,z],s=60,color=[1,0,0],marker='o') # color=[z/float(nl),z/float(nl),1          ])
z = 1 
p3D1 = ax.scatter(3*np.sqrt(CD_3D[vi:vf,z]),0.25*1./time*CD_3D[vi:vf,z],s=60,color=[0,1,0],marker='o') # color=[z/float(nl),z/float(nl),1          ])
z = 2
p3D2 = ax.scatter(3*np.sqrt(CD_3D[vi:vf,z]),0.25*1./time*CD_3D[vi:vf,z],s=60,color=[0,0,1],marker='o') # color=[z/float(nl),z/float(nl),1          ])

# average differences
print 'mean difference 3D z0-z1:', np.mean(0.25*1./timep*CD_3D[vi:vfp,0] - 0.25*1./timep*CD_3D[vi:vfp,1])
print 'mean difference 3D z0-z2:', np.mean(0.25*1./timep*CD_3D[vi:vfp,0] - 0.25*1./timep*CD_3D[vi:vfp,2])
print 'mean difference 3D z1-z2:', np.mean(0.25*1./timep*CD_3D[vi:vfp,1] - 0.25*1./timep*CD_3D[vi:vfp,2])

OK, = plt.plot(OKx,OKy,'k-',linewidth=2)
Rch, = plt.plot(OKx,Rcy,'k--',linewidth=2)
plt.legend([OK,Rch,p3D0,p3D1,p3D2],['Okubo','Richardson','3D 5m','3D 10m','3D 15m'],loc=4,fontsize=20)

ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel(r'$3\sigma_D$ $[cm]$',fontsize=28)
#plt.ylabel(r'$\frac{\sigma^2_D}{4t}$ $[cm^2s^{-1}$]',fontsize=21)
plt.ylabel(r'$k_D$ $[cm^2s^{-1}]$',fontsize=28)
plt.xlim([xm,xM])
plt.ylim([ym,yM])
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)
plt.tight_layout()
plt.savefig('./plot/'+label+'/Diff_O_'+label+'_3D.eps')
print './plot/'+label+'/Diff_O_'+label+'_3D.eps'
plt.close()

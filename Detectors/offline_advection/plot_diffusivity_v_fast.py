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

label = 'm_25_2b_particles'
filename2D = './csv/RD_2D_'+label+'.csv'
filename3D = './csv/RD_3D_'+label+'.csv'
tt =  439

dayi = 0
dayf = tt
days = 1

z0 = [0, 5, 10, 15]

depths = [1, 2, 3]

nl = len(depths)

CD_2D = np.zeros((tt,nl))
CD_3D = np.zeros((tt,nl))

vi = 10
vf = -1
vfp = 500

time2D, RD_2D =lagrangian_stats.read_dispersion(filename2D)
time3D, RD_3D =lagrangian_stats.read_dispersion(filename3D)

CD_2D[:,:] = 100**2*RD_2D
CD_3D[:,:] = 100**2*RD_3D

time0 = time2D - time2D[0] #+ 5*3600
time = time0[vi:vf]
timep = time0[vi:vfp]

xm = 2*10**3
xM = 5*10**6
ym = 10**1
yM = 10**6

OKx = np.linspace(xm,xM)
OKy = 0.0103*OKx**1.15
Rcy = 0.009*OKx**1.33

# 2D only
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


# 3D only

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
plt.ylabel(r'$k_D$ $[cm^2s^{-1}]$',fontsize=28)
plt.xlim([xm,xM])
plt.ylim([ym,yM])
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)
plt.tight_layout()
plt.savefig('./plot/'+label+'/Diff_O_'+label+'_3D.eps')
print './plot/'+label+'/Diff_O_'+label+'_3D.eps'
plt.close()


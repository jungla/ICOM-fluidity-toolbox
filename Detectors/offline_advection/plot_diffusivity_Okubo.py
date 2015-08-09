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

exp = 'm_25_2'
label = 'm_25_2'
filename2D = 'traj_m_25_2_512_0_500_2D.csv'
filename3D = 'traj_m_25_2_512_0_500_3D.csv'
tt = 500 # IC + 24-48 included

x0 = range(3000,4010,10)
y0 = range(2000,3010,10)
z0 = [0,5,10,15] #range(1,20,4)
#z0 = range(1,20,4)


xp = len(x0)
yp = len(y0)
zp = len(z0)
pt = xp*yp*zp
#
#time2D, par2D = advect_functions.read_particles_csv(filename2D,pt,tt)
#par2D = lagrangian_stats.periodicCoords(par2D,10000,4000)
#time3D, par3D = advect_functions.read_particles_csv(filename3D,pt,tt)
#par3D = lagrangian_stats.periodicCoords(par3D,10000,4000)
#
#time2D = (time2D)*1200 + 48*3600 - 1200
#time3D = (time3D)*1200 + 48*3600 - 1200
##
time0 = time2D[:-1] - time2D[0]

depths = [1, 2, 3]

nl = len(depths)

CD_2D = np.zeros((tt,nl))
CD_3D = np.zeros((tt,nl))
CD_Tr = np.zeros((tt,nl))

vi = 10
vip = 10
vf = -1
vfp = 70

time = time0[vi:vf]
timep = time0[vip:vfp]

def ED2_t(par2Dz,tt):
 D_2D = np.zeros(tt)
 for t in range(tt):
  x2 = par2Dz[:,0,t]
  y2 = par2Dz[:,1,t]
  #
  #  x2 = x2[~np.isnan(x2)]
  #  y2 = y2[~np.isnan(y2)]
  #     
  if len(x2) > 1 and len(y2) > 1:
   xt2 = x2 - np.mean(x2)
   yt2 = y2 - np.mean(y2)
   cov = np.cov(xt2, yt2)
   if ~np.isnan(cov).any():
    term1 = (cov[0,0]+cov[1,1]) 
    term2 = np.sqrt((cov[0,0]-cov[1,1])**2 + 4*cov[1,0]**2) 
    lambda_1   = np.sqrt(.5*(term1+term2)) 
    lambda_2   = np.sqrt(.5*(term1-term2)) 
    #    lambda_2, v = np.linalg.eig(cov2)
    #    lambda_2 = np.sqrt(lambda_2)
    D_2D[t] = 2*lambda_1*lambda_2
   else:
    D_2D[t] = np.nan
  else:
   D_2D[t] = np.nan
 return D_2D

for z in range(len(depths)):
 print z
 print 'depth', depths[z]
 par2Dz = np.reshape(par2D,(xp,yp,zp,3,tt))
 par3Dz = np.reshape(par3D,(xp,yp,zp,3,tt))
 par2Dzr = par2Dz[:,:,depths[z],:,:]
 par3Dzr = par3Dz[:,:,depths[z],:,:]

 par2Dz = np.reshape(par2Dzr,(xp*yp,3,tt))
 par3Dz = np.reshape(par3Dzr,(xp*yp,3,tt))

 CD_2D[:,z] = 100**2*lagrangian_stats.RD_t(par2Dzr,tt,100)
 CD_3D[:,z] = 100**2*lagrangian_stats.RD_t(par3Dzr,tt,100)
 #CD_2D[:,z] = 100**2*ED2_t(par2Dz,tt)
 #CD_3D[:,z] = 100**2*ED2_t(par3Dz,tt)

## Tracer second moment
#
#depths = [1,5, 17]
#Trid = [1,2, 4]
#
## Tracer second moment
#
#for z in range(len(depths)):
# print z
# f0 = open('D_Tracer_'+str(Trid[z])+'_CG_'+label+'.csv','r')
# r0 = csv.reader(f0)
# vals = []
# for row in r0:
#  bogusTime,val = row[0].split(', ')
#  vals.append(float(val))
# CD_Tr[:,z] = 0 #100**2*np.asarray(vals[dayi:dayf:days])
# f0.close()

# all on same plot

#xm = 10**5
#xM = 5*10**6
#ym = 10**3
#yM = 3*10**5

xm = 2*10**3
xM = 5*10**6
ym = 10**1
yM = 3*10**5

OKx = np.linspace(xm,xM)
OKy = 0.0103*OKx**1.15
Rcy = 0.009*OKx**1.33
#
#ax = plt.gca()
#s3D = ax.scatter(3*np.sqrt(CD_3D[vi:vf,z]),(0.25/time)*CD_3D[vi:vf,z],color=[1,0,0])
#s2D = ax.scatter(3*np.sqrt(CD_2D[vi:vf,z]),(0.25/time)*CD_2D[vi:vf,z],color=[0,0,1])
#sTr = ax.scatter(3*np.sqrt(CD_Tr[vi:vf,z]),(0.25/time)*CD_Tr[vi:vf,z],color=[1,1,1])
#
#for z in range(nl):
# ax.scatter(3*np.sqrt(CD_3D[vi:vf,z]),0.25/time*CD_3D[vi:vf,z],color=[1          ,z/float(nl),z/float(nl)])
# ax.scatter(3*np.sqrt(CD_2D[vi:vf,z]),0.25/time*CD_2D[vi:vf,z],color=[z/float(nl),z/float(nl),1          ])
# ax.scatter(3*np.sqrt(CD_Tr[vi:vf,z]),0.25/time*CD_Tr[vi:vf,z],color=[z/float(nl),z/float(nl),z/float(nl)])
#
#OK, = plt.plot(OKx,OKy,'k-',linewidth=2)
#plt.legend([OK,s3D,s2D],['Okubo','3D','2D'],loc=4)
#
#ax.set_yscale('log',ybase=10)
#ax.set_xscale('log',xbase=10)
#plt.xlabel(r'$3\sigma_{D}$ $[cm]$')
#plt.ylabel(r'$\frac{\sigma^2_{D}}{4t}$ $[cm^2s^{-1}]$')
#plt.xlim([xm,xM])
#plt.ylim([ym,yM])
#
#plt.savefig('./plot/'+label+'/Diff_O_'+label+'_23D.eps')
#print './plot/'+label+'/Diff_O_'+label+'_23D.eps'
#plt.close()
#
# 2D only
fig = plt.figure(figsize=(8,7))
ax = plt.gca()
z = 0 
p2D0 = ax.scatter(3*np.sqrt(CD_2D[vi:vf,z]),0.25*1./time*CD_2D[vi:vf,z],s=60,color=[1,0,0],marker='o') # color=[z/float(nl),z/float(nl),1          ])
z = 1 
p2D1 = ax.scatter(3*np.sqrt(CD_2D[vi:vf,z]),0.25*1./time*CD_2D[vi:vf,z],s=60,color=[0,1,0],marker='o') # color=[z/float(nl),z/float(nl),1          ])
z = 2
p2D2 = ax.scatter(3*np.sqrt(CD_2D[vi:vf,z]),0.25*1./time*CD_2D[vi:vf,z],s=60,color=[0,0,1],marker='o') # color=[z/float(nl),z/float(nl),1          ])

xrefid0, = np.where(3*np.sqrt(CD_2D[:,0])>10**5)
xrefid1, = np.where(3*np.sqrt(CD_2D[:,1])>10**5)
xrefid2, = np.where(3*np.sqrt(CD_2D[:,2])>10**5)
#plt.scatter(xref,0.0103*xref**1.15)
#plt.scatter(xref,0.25*1./time0[xrefid[0]]*CD_2D[xrefid[0],0])
#print 'diff 2D z1 vs z5 at 10^5', (0.25*1./time0[xrefid0[0]]*CD_2D[xrefid0[0],0])/(0.25*1./time0[xrefid1[0]]*CD_2D[xrefid1[0],1])
#print 'diff 2D z5 vs z17 at 10^5', (0.25*1./time0[xrefid1[0]]*CD_2D[xrefid1[0],0])/(0.25*1./time0[xrefid2[0]]*CD_2D[xrefid2[0],2])
#print 'diff 2D z1 vs z17 at 10^5', (0.25*1./time0[xrefid0[0]]*CD_2D[xrefid0[0],0])/(0.25*1./time0[xrefid2[0]]*CD_2D[xrefid2[0],2])

#for z in range(nl):
# ax.scatter(3*np.sqrt(CD_2D[vi:vf,z]),0.25*1./time*CD_2D[vi:vf,z],color=[z/float(nl),z/float(nl),1          ])
# par = np.polyfit(np.log10(3*np.sqrt(CD_2D[vip:vfp,z])), np.log10(0.25*1./timep*CD_2D[vip:vfp,z]), 1)
# y = x*par[0]+par[1]
# plt.plot(np.power(10,x),np.power(10,y),'k')
# print z, par[0],np.power(10,par[1])

OK, = plt.plot(OKx,OKy,'k-',linewidth=2)
Rch, = plt.plot(OKx,Rcy,'k--',linewidth=2)
plt.legend([OK,Rch,p2D0,p2D1,p2D2],['Okubo','Richardson','2D 5m','2D 10m','2D 15m'],loc=4,fontsize=20)

ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel(r'$3\sigma_{D}$ $[cm]$',fontsize=28)
#plt.ylabel(r'$\frac{\sigma^2_{D}}{4t}$ $[cm^2s^{-1}]$',fontsize=28)
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

#OK, = plt.plot([10**4,10**7],[10,10**6],'k-',linewidth=)
OK, = plt.plot(OKx,OKy,'k-',linewidth=2)
Rch, = plt.plot(OKx,Rcy,'k--',linewidth=2)
plt.legend([OK,Rch,p3D0,p3D1,p3D2],['Okubo','Richardson','3D 5m','3D 10m','3D 15m'],loc=4,fontsize=20)
#plt.legend([OK,s3D,sD,sTr],['Okubo','3D','D','Tr'])

ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel(r'$3\sigma_{D}$ $[cm]$',fontsize=28)
#plt.ylabel(r'$\frac{\sigma^2_{D}}{4t}$ $[cm^2s^{-1}]$',fontsize=28)
plt.ylabel(r'$k_D$ $[cm^2s^{-1}]$',fontsize=28)
plt.xlim([xm,xM])
plt.ylim([ym,yM])
plt.yticks(fontsize=24)
plt.xticks(fontsize=24)
plt.tight_layout()
plt.savefig('./plot/'+label+'/Diff_O_'+label+'_3D.eps')
print './plot/'+label+'/Diff_O_'+label+'_3D.eps'
plt.close()



## Tr only
#
#ax = plt.gca()
#sTr = ax.scatter(3*np.sqrt(CD_Tr[vi:vf,z]),0.25*1./time*CD_Tr[vi:vf,z],color=[1,1,1])
#
#for z in range(nl):
# ax.scatter(3*np.sqrt(CD_Tr[vi:vf,z]),0.25*1./time*CD_Tr[vi:vf,z],color=[z/float(nl),z/float(nl),z/float(nl)])
#
#OK, = plt.plot(OKx,OKy,'k-',linewidth=2)
##plt.legend([OK,s3D,sD,sTr],['Okubo','3D','D','Tr'])
#
#ax.set_yscale('log')
#ax.set_xscale('log')
#plt.xlabel('D [cm]')
#plt.ylabel('k [cm^/s]')
#plt.xlim([xm,xM])
#plt.ylim([ym,yM])
#
#plt.savefig('./plot/'+label+'/Diff_O_'+label+'_Tr.eps')
#print './plot/'+label+'/Diff_O_'+label+'_Tr.eps'
#plt.close()

#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import myfun
import numpy as np
import os
import scipy.stats as sp
import scipy.interpolate
import advect_functions
import lagrangian_stats
import csv

# read offline
print 'reading offline'

exp = 'm_25_2_512'
label = 'm_25_2_512'
filename2D = 'traj_m_25_2_512_0_230_2D.csv'
filename3D = 'traj_m_25_2_512_0_230_3D.csv'
tt = 230 # IC + 24-48 included
dayi = 0
dayf = tt 
days = 1

x0 = range(3000,4050,50)
y0 = range(2000,3050,50)
#x0 = range(3000,4050,50)
#y0 = range(0,4000,50)
z0 = range(0,30,1)


xp = len(x0)
yp = len(y0)
zp = len(z0)
pt = xp*yp*zp

#time2D, par2D = advect_functions.read_particles_csv(filename2D,xp,yp,zp,tt)
#par2D = lagrangian_stats.periodicCoords(par2D,10000,4000)
#time3D, par3D = advect_functions.read_particles_csv(filename3D,xp,yp,zp,tt)
#par3D = lagrangian_stats.periodicCoords(par3D,10000,4000)
#par2D = np.reshape(par2D,(pt,3,tt))
#par3D = np.reshape(par3D,(pt,3,tt))

#time2D = (time2D)*1200 + 48*3600 - 1200
#time3D = (time3D)*1200 + 48*3600 - 1200

time = time2D

depths = [1, 5, 17]

nl = len(depths)

CD_2D = np.zeros((tt,nl))
CD_3D = np.zeros((tt,nl))

for z in range(len(depths)):
 print z
 print 'depth', depths[z]
 par2Dz = np.reshape(par2D,(xp,yp,zp,3,tt))
 par3Dz = np.reshape(par3D,(xp,yp,zp,3,tt))
 par2Dzr = par2Dz[:,:,depths[z],:,:]
 par3Dzr = par3Dz[:,:,depths[z],:,:]

 par2Dz = np.reshape(par2Dzr,(xp*yp,3,tt))
 par3Dz = np.reshape(par3Dzr,(xp*yp,3,tt))

 CD_2D[:,z] = 100*lagrangian_stats.RD_t(par2Dzr,tt,20)
 CD_3D[:,z] = 100*lagrangian_stats.RD_t(par3Dzr,tt,20)

# Tracer second moment

timeTr = time #(time)*1200 + 48*3600 - 1200

depths = [1,5,17]
Trid = [1,2,4]

CD_Tr = np.zeros([len(time),len(depths)])
Diff_CD_3D = np.zeros([len(time),len(depths)])
Diff_CD_2D = np.zeros([len(time),len(depths)])
Diff_CD_Tr = np.zeros([len(time),len(depths)])

timei = np.linspace(3*86400, time[-1], tt*3)

from scipy import interpolate

CD_3Di = np.zeros([len(timei),len(depths)])
CD_2Di = np.zeros([len(timei),len(depths)])
CD_Tri = np.zeros([len(timei),len(depths)])
Diff_CD_3Di = np.zeros([len(timei),len(depths)])
Diff_CD_2Di = np.zeros([len(timei),len(depths)])
Diff_CD_Tri = np.zeros([len(timei),len(depths)])

# Tracer second moment

for z in range(len(depths)):
 print z
 f0 = open('D_Tracer_'+str(Trid[z])+'_CG_'+label+'.csv','r')
 r0 = csv.reader(f0)
 vals = []
 for row in r0:
  bogusTime,val = row[0].split(', ')
  vals.append(float(val))
 CD_Tr[:,z] = 100*np.asarray(vals[dayi:dayf:days])
 f0.close()

# interpolate and smooth data

 f = interpolate.interp1d(time, CD_2D[:,z], kind='slinear')
 CD_2Di[:,z] = f(timei)
 f = interpolate.interp1d(time, CD_3D[:,z], kind='slinear')
 CD_3Di[:,z] = f(timei)
 f = interpolate.interp1d(time, CD_Tr[:,z], kind='slinear')
 CD_Tri[:,z] = f(timei)


# Diff
# Diff_CD_3D[:,z] = 0.5*np.gradient(CD_3D[:,z])/np.gradient(time)
# Diff_CD_2D[:,z] = 0.5*np.gradient(CD_2D[:,z])/np.gradient(time)
# Diff_CD_Tr[:,z] = 0.5*np.gradient(CD_Tr[:,z])/np.gradient(time)

# Diff_CD_3D[Diff_CD_3D[:,z]<=0,z] = np.nan
# Diff_CD_2D[Diff_CD_2D[:,z]<=0,z] = np.nan
# Diff_CD_Tr[Diff_CD_Tr[:,z]<=0,z] = np.nan

 Diff_CD_3Di[:,z] = 0.5*np.gradient(CD_3Di[:,z])/np.gradient(timei)
 Diff_CD_2Di[:,z] = 0.5*np.gradient(CD_2Di[:,z])/np.gradient(timei)
 Diff_CD_Tri[:,z] = 0.5*np.gradient(CD_Tri[:,z])/np.gradient(timei)

 Diff_CD_3Di[Diff_CD_3Di[:,z]<=0,z] = np.nan
 Diff_CD_2Di[Diff_CD_2Di[:,z]<=0,z] = np.nan
 Diff_CD_Tri[Diff_CD_Tri[:,z]<=0,z] = np.nan

# all on same plot
# params

xm = 5*10**3
xM = 10**5
ym = 10
yM = 10**4

OKx = np.linspace(xm,xM)
OKy = 0.0103*OKx**1.15

ax = plt.gca()
s3D = ax.scatter(3.*np.sqrt(CD_3Di[:,z]),Diff_CD_3Di[:,z],color=[1,0,0])
s2D = ax.scatter(3.*np.sqrt(CD_2Di[:,z]),Diff_CD_2Di[:,z],color=[0,0,1])
sTr = ax.scatter(3.*np.sqrt(CD_Tri[:,z]),Diff_CD_Tri[:,z],color=[0,0,0])

for z in range(nl):
 ax.scatter(3.*np.sqrt(CD_3Di[:,z]),Diff_CD_3Di[:,z],color=[1,z/float(nl),z/float(nl)])
 ax.scatter(3.*np.sqrt(CD_2Di[:,z]),Diff_CD_2Di[:,z],color=[z/float(nl),z/float(nl),1])
 ax.scatter(3.*np.sqrt(CD_Tri[:,z]),Diff_CD_Tri[:,z],color=[z/float(nl),z/float(nl),z/float(nl)])

OK, = plt.plot(OKx,OKy,'k-',linewidth=2)
plt.legend([OK,s3D,s2D,sTr],['Okubo','3D','2D','Tr'],loc=4)

ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel(r'$3\sigma_{C_x}$ $[cm]$')
plt.ylabel(r'$k$ $[cm^2/s]$')
plt.xlim([10**4,2*10**5])
plt.ylim([10**2,10**5])

plt.savefig('./plot/'+label+'_23D/Diff_T_'+label+'_23D.eps')
print './plot/'+label+'_23D/Diff_T_'+label+'_23D.eps'
plt.close()

# 2D only

fig = plt.figure()
ax = plt.gca()
s2D = ax.scatter(3.*np.sqrt(CD_2Di[:,z]),Diff_CD_2Di[:,z],color=[0,0,1])

for z in range(nl):
 ax.scatter(3.*np.sqrt(CD_2Di[:,z]),Diff_CD_2Di[:,z],color=[z/float(nl),z/float(nl),1])

OK, = plt.plot(OKx,OKy,'k-',linewidth=2)
plt.legend([OK,s2D],['Okubo','2D'],loc=4)
#plt.legend([OK,PJ,s2D],['Okubo','Poje','2D'])

ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel(r'$3\sigma_{C_x}$ $[cm]$')
plt.ylabel(r'$k$ $[cm^2/s]$')
plt.xlim([xm,xM])
plt.ylim([ym,yM])

plt.savefig('./plot/'+label+'_23D/Diff_T_'+label+'_2D.eps')
print './plot/'+label+'_23D/Diff_T_'+label+'_2D.eps'
plt.close()

# 3D only

fig = plt.figure()
ax = plt.gca()
s3D = ax.scatter(3.*np.sqrt(CD_3Di[:,z]),Diff_CD_3Di[:,z],color=[1,0,0])

for z in range(nl):
 ax.scatter(3.*np.sqrt(CD_3Di[:,z]),Diff_CD_3Di[:,z],color=[1,z/float(nl),z/float(nl)])

OK, = plt.plot(OKx,OKy,'k-',linewidth=2)
#OK, = plt.plot([10**4,10**7],[10**2,10**6],'k-',linewidth=2)
plt.legend([OK,s3D],['Okubo','3D'],loc=4)

ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel(r'$3\sigma_{C_x}$ $[cm]$')
plt.ylabel(r'$k$ $[cm^2/s]$')
plt.xlim([xm,xM])
plt.ylim([ym,yM])

plt.savefig('./plot/'+label+'_23D/Diff_T_'+label+'_3D.eps')
print './plot/'+label+'_23D/Diff_T_'+label+'_3D.eps'
plt.close()

# Tr only

ax = plt.gca()
s3D = ax.scatter(3.*np.sqrt(CD_Tri[:,z]),Diff_CD_Tri[:,z],color=[0,0,0])

for z in range(nl):
 ax.scatter(3.*np.sqrt(CD_Tri[:,z]),Diff_CD_Tri[:,z],color=[z/float(nl),z/float(nl),z/float(nl)])

OK, = plt.plot(OKx,OKy,'k-',linewidth=2)
#OK, = plt.plot([10**4,10**7],[10**2,10**6],'k-',linewidth=2)
plt.legend([OK,s3D],['Okubo','Tr'],loc=4)

ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel(r'$3\sigma_{C_x}$ $[cm]$')
plt.ylabel(r'$k$ $[cm^2/s]$')
plt.xlim([xm,xM])
plt.ylim([ym,yM])

plt.savefig('./plot/'+label+'_23D/Diff_T_'+label+'_Tr.eps')
print './plot/'+label+'_23D/Diff_T_'+label+'_Tr.eps'
plt.close()

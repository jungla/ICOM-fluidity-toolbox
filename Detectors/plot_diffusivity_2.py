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
import lagrangian_stats

def f_exp(t,a,b,c):
    return a + b*np.exp(c*t)


exp = 'm_50_7_23Dc_particles'
label = 'm_50_7_23Dc'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

exp2D = 'm_50_7_2Dc_particles'

filename0 = './mli_checkpoint.detectors'
filename0 = '/tamay2/mensa/fluidity/'+exp2D+'/'+filename0

#time2D, par2D = lagrangian_stats.read_particles(filename0)

exp3D = 'm_50_7_3Dc_particles'

filename0 = './mli_checkpoint.detectors'
filename0 = '/tamay2/mensa/fluidity/'+exp3D+'/'+filename0

#time3D, par3D = lagrangian_stats.read_particles(filename0)

tt = min(len(time2D),len(time3D))

par3D = par3D[:,:,:tt]
par2D = par2D[:,:,:tt]

#par2DP = lagrangian_stats.periodicCoords(par2D,10000,10000)
#par3DP = lagrangian_stats.periodicCoords(par3D,10000,10000)

time = time2D[:tt]

# horizontal
depths = [1, 5, 11, 17, 26]
depths = [1, 5, 17]

pd = range(1,depth,3)
pd = [1, 5, 11, 17, 26]

nl = len(pd)

D_ED_3D = np.zeros((tt,nl))
D_ED_2D = np.zeros((tt,nl))
Diff_ED_3D = np.zeros((tt,nl))
Diff_ED_2D = np.zeros((tt,nl))
D_CD_3D = np.zeros((tt,nl))
D_CD_2D = np.zeros((tt,nl))
Diff_CD_3D = np.zeros((tt,nl))
Diff_CD_2D = np.zeros((tt,nl))

# Tracer Dispersion

#import csv
#path = './D2_1200.csv'
#timef = 100

lagrangian_stats.read_Tracer(filepath,pd,zn,yn,xn,timeTr)

# interpolate tracer to same scale

timeTri = time[time <= np.max(timeTr)]

D_Tr = np.zeros((len(timeTri),nl))
Diff_Tr = np.zeros((len(timeTri),nl))

for z in range(nl): 
 f = scipy.interpolate.interp1d(timeTr,Dt[:,z],kind='cubic')
 D_Tr[:,z] = f(timeTri)

#fig = plt.figure()

#for z in range(nl):
# plt.plot(timeTri,D_Tr[:,z],'b')
# plt.plot(timeTr,Dt[:,z],'k')

#plt.savefig('./plot/interpolated_Tr_disp.eps')
#plt.close()

for z in range(nl):
 print pd[z]
 lambda_3t = np.zeros([tt,2])
 lambda_2t = np.zeros([tt,2])

 # Extract particles at depth z

 par2Dz = np.reshape(par2D,(20,20,30,3,tt))
 par3Dz = np.reshape(par3D,(20,20,40,3,tt))

 par2Dzr = par2Dz[:,:,pd[z],:,:]
 par3Dzr = par3Dz[:,:,pd[z],:,:]

 par2Dz = np.reshape(par2Dzr,(400,3,tt))
 par3Dz = np.reshape(par3Dzr,(400,3,tt))

 D_ED_2D[120:,0] = np.nan
 D_ED_3D[120:,0] = np.nan

# Diff

 Diff_ED_3D[:,z] = 0.5*np.gradient(D_ED_3D[:,z]*np.pi)/np.gradient(time)
 Diff_ED_2D[:,z] = 0.5*np.gradient(D_ED_2D[:,z]*np.pi)/np.gradient(time)
 Diff_Tr[:,z] = 0.5*np.gradient(D_Tr[:,z])/np.gradient(timeTri)

 Diff_ED_3D[Diff_ED_3D[:,z]<=0,z] = np.nan
 Diff_ED_2D[Diff_ED_2D[:,z]<=0,z] = np.nan
 Diff_Tr[Diff_Tr[:,z]<=0,z] = np.nan

 fig = plt.figure()
 ax = plt.gca()
 s3D = ax.scatter(np.sqrt(lambda_3t[:,0]*lambda_3t[:,1]*np.pi)*100,Diff_ED_3D[:,z]*100**2,color='k')  # 100 is m -> cm
 s2D = ax.scatter(np.sqrt(lambda_2t[:,0]*lambda_2t[:,1]*np.pi)*100,Diff_ED_2D[:,z]*100**2,color='b')  #
 ax.set_yscale('log')
 ax.set_xscale('log')
 plt.xlabel('D [cm]')
 plt.ylabel('k [cm^2/s]')
# plt.legend([s3D,s2D],['3D','2D'])
 #Okubo K=10^3 cm^2/s at r=10^4 cm to K=10^5 cm^2/s at r=10^7
 OK, = plt.plot([10**4,10**7],[10**2,10**6],'k-',linewidth=2)
 PJ, = plt.plot([10**4,10**7],[10**4,10**8],'k--',linewidth=2)
 plt.legend([OK,PJ,s3D,s2D],['Okubo','Poje','3D','2D'])
 plt.xlim([2*10**3,2*10**5])
 plt.ylim([10**1,10**6])
 print './plot/'+label+'_23D/Diff_'+label+'_23D_z'+str(pd[z])+'.eps'
 plt.savefig('./plot/'+label+'_23D/Diff_'+label+'_23D_z'+str(pd[z])+'.eps')
 plt.close()

# all on same plot

ax = plt.gca()
s3D = ax.scatter(np.sqrt(D_ED_3D[:,z])*100,Diff_ED_3D[:,z]*100**2,color=[0,0,0])
s2D = ax.scatter(np.sqrt(D_ED_2D[:,z])*100,Diff_ED_2D[:,z]*100**2,color=[0,0,1])
sTr = ax.scatter(np.sqrt(D_Tr[:,z])*100,Diff_Tr[:,z]*100**2,color=[0,1,0])

for z in range(nl):
 ax.scatter(np.sqrt(D_ED_3D[:,z])*100,Diff_ED_3D[:,z]*100**2,color=[z/float(nl),z/float(nl),z/float(nl)])
 ax.scatter(np.sqrt(D_ED_2D[:,z])*100,Diff_ED_2D[:,z]*100**2,color=[z/float(nl),z/float(nl),1])
 ax.scatter(np.sqrt(D_Tr[:,z])*100,Diff_Tr[:,z]*100**2,color=[z/float(nl),1,z/float(nl)])

OK, = plt.plot([10**4,10**7],[10**2,10**6],'k-',linewidth=2)
PJ, = plt.plot([10**4,10**7],[10**4,10**8],'k--',linewidth=2)
plt.legend([OK,PJ,s3D,s2D,sTr],['Okubo','Poje','3D','2D','Tr'])

ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('D [cm]')
plt.ylabel('k [cm^2/s]')
plt.xlim([10**4,10**5])

plt.savefig('./plot/'+label+'_23D/Diff_'+label+'_23D.eps')
print './plot/'+label+'_23D/Diff_'+label+'_23D.eps'
plt.close()

# 2D only

fig = plt.figure()
ax = plt.gca()
s2D = ax.scatter(np.sqrt(D_ED_2D[:,z])*100,Diff_ED_2D[:,z]*100**2,color=[0,0,1])

for z in range(nl):
 ax.scatter(np.sqrt(D_ED_2D[:,z])*100,Diff_ED_2D[:,z]*100**2,color=[z/float(nl),z/float(nl),1])

OK, = plt.plot([10**4,10**7],[10**2,10**6],'k-',linewidth=2)
PJ, = plt.plot([10**4,10**7],[10**4,10**8],'k--',linewidth=2)
plt.legend([OK,PJ,s2D],['Okubo','Poje','2D'])

ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('D [cm]')
plt.ylabel('k [cm^2/s]')
plt.xlim([10**4,10**5])

plt.savefig('./plot/'+label+'_23D/Diff_'+label+'_2D.eps')
print './plot/'+label+'_23D/Diff_'+label+'_2D.eps'
plt.close()

# 3D only

fig = plt.figure()
ax = plt.gca()
s3D = ax.scatter(np.sqrt(D_ED_3D[:,z])*100,Diff_ED_3D[:,z]*100**2,color=[0,0,0])

for z in range(nl):
 ax.scatter(np.sqrt(D_ED_3D[:,z])*100,Diff_ED_3D[:,z]*100**2,color=[z/float(nl),z/float(nl),z/float(nl)])

OK, = plt.plot([10**4,10**7],[10**2,10**6],'k-',linewidth=2)
PJ, = plt.plot([10**4,10**7],[10**4,10**8],'k--',linewidth=2)
plt.legend([OK,PJ,s3D],['Okubo','Poje','3D'])

ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('D [cm]')
plt.ylabel('k [cm^2/s]')
plt.xlim([10**4,10**5])

plt.savefig('./plot/'+label+'_23D/Diff_'+label+'_3D.eps')
print './plot/'+label+'_23D/Diff_'+label+'_3D.eps'
plt.close()

# 3D only

ax = plt.gca()
s3D = ax.scatter(np.sqrt(D_Tr[:,z])*100,Diff_Tr[:,z]*100**2,color=[0,0,0])

for z in range(nl):
 ax.scatter(np.sqrt(D_Tr[:,z])*100,Diff_Tr[:,z]*100**2,color=[z/float(nl),z/float(nl),z/float(nl)])

OK, = plt.plot([10**4,10**7],[10**2,10**6],'k-',linewidth=2)
PJ, = plt.plot([10**4,10**7],[10**4,10**8],'k--',linewidth=2)
plt.legend([OK,PJ,s3D],['Okubo','Poje','Tr'])

ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('D [cm]')
plt.ylabel('k [cm^2/s]')
plt.xlim([10**4,10**6])

plt.savefig('./plot/'+label+'_23D/Diff_'+label+'_Tr.eps')
print './plot/'+label+'_23D/Diff_'+label+'_Tr.eps'
plt.close()

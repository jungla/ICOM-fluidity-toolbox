#!~/python
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import myfun
import numpy as np
from scipy import optimize
import os
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

 CD_2D[:,z] = lagrangian_stats.CDx_t(par2Dz,tt)
 CD_3D[:,z] = lagrangian_stats.CDx_t(par3Dz,tt)

# Tracer second moment

timeTr = time #(time)*1200 + 48*3600 - 1200

depths = [1,5,17]
Trid = [1,2,4]

D_Tr = np.zeros([len(timeTr),len(depths)])

# Tracer second moment

for z in range(len(depths)):
 print z
 f0 = open('D_Tracer_'+str(Trid[z])+'_CG_'+label+'.csv','r')
 r0 = csv.reader(f0)
 vals = []
 for row in r0:
  bogusTime,val = row[0].split(', ')
  vals.append(float(val))
 D_Tr[:,z] = np.asarray(vals[dayi:dayf:days])
 f0.close()


# cloud D

p3D, = plt.plot(time/86400,CD_3D[:,0],'r',linewidth=2)
p2D, = plt.plot(time/86400,CD_2D[:,0],'b',linewidth=2)
pTr, = plt.plot(timeTr/86400,D_Tr[:,0],'k',linewidth=2)

z = 1
p3D5, = plt.plot(time/86400,CD_3D[:,z],color=[1,z/float(nl),z/float(nl)],linewidth=2)
p2D5, = plt.plot(time/86400,CD_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
pTr5, = plt.plot(timeTr/86400,D_Tr[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)

z = 2
p3D17, = plt.plot(time/86400,CD_3D[:,z],color=[1,z/float(nl),z/float(nl)],linewidth=2)
p2D17, = plt.plot(time/86400,CD_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
pTr17, = plt.plot(timeTr/86400,D_Tr[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)

plt.gca().set_yscale('log')

plt.xlabel('Time [days]')
plt.ylabel('Dispersion [m]')
plt.legend((p3D,p3D5,p3D17,p2D,p2D5,p2D17,pTr,pTr5,pTr17),('3D 1m','3D 5m','3D 17m','2D 1m','2D 5m','2D 17m','Tr 1m','Tr 5m','Tr 17m'),loc=4,fontsize=12)
#plt.legend((p3D,p2D,pTr),('3D','2D','Tr'))
#plt.ylim((3*10**2,10**3))
#plt.xlim((160000,280000))
plt.ylim((0.5*10**5,10**7))
plt.xlim((2,5))
#
print 'Saving CD'
# 
plt.savefig('./plot/'+label+'_23D/CD_Tr_'+label+'_23D.eps')
print       './plot/'+label+'_23D/CD_Tr_'+label+'_23D.eps' 
plt.close()


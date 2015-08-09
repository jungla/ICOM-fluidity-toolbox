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

exp3D = 'm_50_5_3D_particles'
exp2D = 'm_50_5_2D_particles'

filename3D = '/tamay2/mensa/fluidity/'+exp3D+'/mli_particles.detectors'
filename2D = '/tamay2/mensa/fluidity/'+exp2D+'/mli_particles.detectors'

print 'Reading ', filename2D, filename3D

#try: os.stat('./output/'+exp3D)
#except OSError: os.mkdir('./output/'+exp3D)

#det3D = fluidity_tools.stat_parser(filename3D)
#det2D = fluidity_tools.stat_parser(filename2D)

# pt same for 3D and 2D
pt = int(os.popen('grep position '+filename3D+'| wc -l').read()) # read the number of particles grepping all the positions in the file
time3D = det3D['ElapsedTime']['value']
time2D = det2D['ElapsedTime']['value']

tt = min(len(time2D),len(time3D))
tt = 267
#tt = 120

if len(time2D) < len(time3D):
 time = time2D[:tt]
else:
 time = time3D[:tt]

print 'particles:',pt
print 'timesteps:',tt

par3D = np.zeros((pt,3,tt))
par2D = np.zeros((pt,3,tt))

for d in xrange(pt):
 temp3D = det3D['particles_'+myfun.digit(d+1,len(str(pt)))]['position']
 par3D[d,:,:] = temp3D[:,:tt]
 temp2D = det2D['particles_'+myfun.digit(d+1,len(str(pt)))]['position']
 par2D[d,:,:] = temp2D[:,:tt]

# horizontal
depth = 11 #11

pd = range(1,depth,3)

nl = len(pd)

A_3D = np.zeros((tt,nl))
A_2D = np.zeros((tt,nl))
D_3D = np.zeros((tt,nl))
D_2D = np.zeros((tt,nl))
Cd_3D = np.zeros((tt,nl))
Cd_2D = np.zeros((tt,nl))


# for fitting

import scipy.optimize as optimize

def f_exp(t,a,b,c):
    return a + b*np.exp(c*t)
#def f_exp_r(a,b,c,t,Val):
#    return Val - a*np.exp(b*t)+c

def f_t1(t,a,b):
    return a + b*t
#def f_t1_r(a,b,t,Val):
#    return Val - a*t + b

def f_t2(t,a,b):
    return a + b*(t**2)
#def f_t2_r(a,b,t,Val):
#    return Val - a*t**2 + b

def f_t3(t,a,b):
    return a + b*(t**3)

#def f_t3_r(a,b,t,Val):
#    return Val - a*t**3 + b

for z in range(nl):
 print pd[z]

 # Extract particles at depth z

 for t in range(tt):
  par3D[par3D[:,0,t] < 100,:,t:tt] = np.nan
  par3D[par3D[:,0,t] > 4900,:,t:tt] = np.nan
  par3D[par3D[:,1,t] < 100,:,t:tt] = np.nan
  par3D[par3D[:,1,t] > 4900,:,t:tt] = np.nan
  par2D[par2D[:,0,t] < 100,:,t:tt] = np.nan
  par2D[par2D[:,0,t] > 4900,:,t:tt] = np.nan
  par2D[par2D[:,1,t] < 100,:,t:tt] = np.nan
  par2D[par2D[:,1,t] > 4900,:,t:tt] = np.nan

 par2Dz = np.reshape(par2D,(20,20,50,3,tt))
 par3Dz = np.reshape(par3D,(20,20,50,3,tt))

 par2Dzr = par2Dz[:,:,pd[z],:,:]
 par3Dzr = par3Dz[:,:,pd[z],:,:]

 par2Dz = np.reshape(par2Dzr,(400,3,tt))
 par3Dz = np.reshape(par3Dzr,(400,3,tt))
 
 #
 #
 # Absolute Dispersion in Time
 #
 P0 = np.mean([500,1500]),np.mean([3500,4500])
 #
 A_3D[:,z] = np.sqrt(np.mean((par3Dz[:,0,:] - P0[0])**2 + (par3Dz[:,1,:] - P0[1])**2,0))
 A_2D[:,z] = np.sqrt(np.mean((par2Dz[:,0,:] - P0[0])**2 + (par2Dz[:,1,:] - P0[1])**2,0)) 
 #
 # Relative disperions
 #
 D_3Dm = np.zeros((19,tt))
 D_2Dm = np.zeros((19,tt))

 for i in range(19):
  D_3Dm[i,:] = np.mean((par3Dzr[i+1,:,0,:] - par3Dzr[i,:,0,:])**2 + (par3Dzr[i+1,:,1,:] - par3Dzr[i,:,1,:])**2,0) 
  D_2Dm[i,:] = np.mean((par2Dzr[i+1,:,0,:] - par2Dzr[i,:,0,:])**2 + (par2Dzr[i+1,:,1,:] - par2Dzr[i,:,1,:])**2,0) 

 D_3D[:,z] = np.mean(D_3Dm,0)
 D_2D[:,z] = np.mean(D_2Dm,0)
 #
 # Cloud dispersion
 #
 #
 Pt3D = np.zeros((2,tt))
 Pt2D = np.zeros((2,tt))
 #
 Pt3D[:] = np.mean(par3Dz[:,0,:],0),np.mean(par3Dz[:,1,:],0)
 Pt2D[:] = np.mean(par2Dz[:,0,:],0),np.mean(par2Dz[:,1,:],0)
 # 
 Cd_3D[:,z] = np.sqrt(np.mean((par3Dz[:,0,:] - Pt3D[0,:])**2 + (par3Dz[:,1,:] - Pt3D[1,:])**2,0))
 Cd_2D[:,z] = np.sqrt(np.mean((par2Dz[:,0,:] - Pt2D[0,:])**2 + (par2Dz[:,1,:] - Pt2D[1,:])**2,0))
 
 # plotting

 # abosolute D

 p3D, = plt.plot(time/86400,A_3D[:,z],'k',linewidth=2)
 p2D, = plt.plot(time/86400,A_2D[:,z],'b',linewidth=2)
 plt.gca().set_yscale('log')
 
 plt.xlabel('Time [days]')
 plt.ylabel('Absolute Dispersion [m]')
# plt.ylim((500,2000))
 plt.legend((p3D,p2D),('3D','2D'))
 #
 print 'Saving AD'
 # 
 plt.savefig('./plot/m_50_5_23D/AD_m_50_5_23D_z'+str(pd[z])+'.eps')
 plt.close()

 # relative D
# plt.plot(time/86400,1000.0*np.exp(time/86400)+D_3D[0,z],'--k')
# plt.plot(time/86400,1000.0*np.exp(2*time/86400)+D_3D[0,z],'--k')
# plt.plot(time/86400,1000.0*np.exp(3*time/86400)+D_3D[0,z],'--k')

 # fitting for 3D
 
 cD_3D = D_3D[~np.isnan(D_3D[:,z]),z]
 cD_2D = D_2D[~np.isnan(D_2D[:,z]),z]
 ctime3 = time[~np.isnan(D_3D[:,z])]
 ctime2 = time[~np.isnan(D_2D[:,z])]

 vtime = ctime3[ctime3<200000]
 Val = cD_3D[ctime3<200000]

 out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
 print out
 plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),color=[0.5, 0.5, 0.5],linewidth=2)

 vtime = ctime2[ctime2<200000]
 Val = cD_2D[ctime2<200000]

 out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
 print out
 plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),color=[0.5, 0.5, 0.5],linewidth=2)

 plt.plot([10**5,10**6],[10**4,10**5],'k')
 plt.plot([10**5,10**6],[10**4,10**6],'k')
 plt.plot([10**5,10**6],[10**4,10**7],'k')
 
 p3D, = plt.plot(time,D_3D[:,z],'k',linewidth=2)
 p2D, = plt.plot(time,D_2D[:,z],'b',linewidth=2)

 plt.gca().set_yscale('log')
 plt.gca().set_xscale('log')

 plt.xlabel('Time [days]')
 plt.ylabel('Relative Dispersion [m^2]')
 plt.legend((p3D,p2D),('3D','2D'))
 plt.ylim((10**3,10**6))
 plt.xlim((160000,250000))
 #
 print 'Saving RD'
 # 
 plt.savefig('./plot/m_50_5_23D/RD_m_50_5_23D_z'+str(pd[z])+'.eps')
 plt.close()

 # cloud D
# plt.plot(time/86400,(time/86400)+Cd_3D[0,z],'--k')
# plt.plot(time/86400,(time/86400)**2+Cd_3D[0,z],'--k')
# plt.plot(time/86400,(time/86400)**3+Cd_3D[0,z],'--k')

 p3D, = plt.plot(time/86400,Cd_3D[:,z],'k',linewidth=2)
 p2D, = plt.plot(time/86400,Cd_2D[:,z],'b',linewidth=2)
 plt.gca().set_yscale('log')

 plt.xlabel('Time [days]')
 plt.ylabel('Cloud Dispersion [m]')
 plt.legend((p3D,p2D),('3D','2D'))
 #plt.ylim((500,2000))
 #
 print 'Saving Cd'
 # 
 plt.savefig('./plot/m_50_5_23D/CD_m_50_5_23D_z'+str(pd[z])+'.eps')
 plt.close()

# plotting all depths


# abosolute D

p3D, = plt.plot(time,A_3D[:,0],color=[0,0,0],linewidth=2)
p2D, = plt.plot(time,A_2D[:,0],color=[0,0,1],linewidth=2)
#plt.gca().set_yscale('log')
#plt.gca().set_xscale('log')

for z in range(nl):
 plt.plot(time,A_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
 plt.plot(time,A_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)

#plt.plot([10**5,10**6],[10**4,10**5],'k')
#plt.plot([10**5,10**6],[10**4,10**6],'k')
#plt.plot([10**5,10**6],[10**4,10**7],'k')

plt.gca().set_yscale('log')
plt.gca().set_xscale('log')
plt.gca().set_xticks(np.linspace(86400.0*2,86400*3.0,6))
plt.gca().xaxis.set_ticklabels(np.linspace(86400.0*2/86400.0,86400*3.0/86400.0,6))
plt.gca().set_yticks(np.linspace(3*10**2,4*10**3,5))
plt.gca().yaxis.set_ticklabels(np.linspace(3*10**2,4*10**3,5))

plt.xlabel('Time [days]')
plt.ylabel('Absolute Dispersion [m]')
plt.legend((p3D,p2D),('3D','2D'))
plt.ylim((3*10**2,4*10**3))
plt.xlim((160000,280000))

#
print 'Saving AD'
# 
plt.savefig('./plot/m_50_5_23D/AD_m_50_5_23D.eps')
plt.close()

plt.contourf(time/86400,range(nl),np.transpose(np.log(A_2D)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Absolute Dispersion')
plt.savefig('./plot/m_50_5_23D/AD_2D_m_50_5_23D_c.eps')
plt.close()

plt.contourf(time/86400,range(nl),np.transpose(np.log(A_3D)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Absolute Dispersion')
plt.savefig('./plot/m_50_5_23D/AD_3D_m_50_5_23D_c.eps')
plt.close()


# relative D
#plt.plot(time/86400,1000.0*np.exp(time/86400)+D_3D[0,z],'--k')
#plt.plot(time/86400,1000.0*np.exp(2*time/86400)+D_3D[0,z],'--k')
#plt.plot(time/86400,1000.0*np.exp(3*time/86400)+D_3D[0,z],'--k')
p3D, = plt.plot(time,D_3D[:,0],color=[0,0,0],linewidth=2)
p2D, = plt.plot(time,D_2D[:,0],color=[0,0,1],linewidth=2)

for z in range(nl):

 cD_3D = D_3D[~np.isnan(D_3D[:,z]),z]
 cD_2D = D_2D[~np.isnan(D_2D[:,z]),z]
 ctime3 = time[~np.isnan(D_3D[:,z])]
 ctime2 = time[~np.isnan(D_2D[:,z])]

 vtime = ctime3[ctime3<200000]
 Val = cD_3D[ctime3<200000]

 out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
 print out
 plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),'--k',linewidth=1)

 vtime = ctime2[ctime2<200000]
 Val = cD_2D[ctime2<200000]

 out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
 print out
 plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),'--k',linewidth=1)

 plt.plot(time,D_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
 plt.plot(time,D_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)

plt.plot([10**5,10**6],[10**4,10**5],'k')
plt.plot([10**5,10**6],[10**4,10**6],'k')
plt.plot([10**5,10**6],[10**4,10**7],'k')

plt.gca().set_yscale('log')
plt.gca().set_xscale('log')
plt.gca().set_xticks(np.linspace(86400.0*2,86400*3.0,6))
plt.gca().xaxis.set_ticklabels(np.linspace(86400.0*2/86400.0,86400*3.0/86400.0,6))
plt.gca().set_yticks(np.linspace(10**3,10**6,5))
plt.gca().yaxis.set_ticklabels(np.linspace(10**3,10**6,5))

plt.xlabel('Time [days]')
plt.ylabel('Relative Dispersion [m^2]')
plt.legend((p3D,p2D),('3D','2D'))
plt.ylim((10**3,10**6))
plt.xlim((160000,280000))

plt.savefig('./plot/m_50_5_23D/RD_m_50_5_23D.eps')
plt.close()

plt.contourf(time/86400,range(nl),np.transpose(np.log(D_2D)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Relative Dispersion')
plt.savefig('./plot/m_50_5_23D/RD_2D_m_50_5_23D_c.eps')
plt.close()

plt.contourf(time/86400,range(nl),np.transpose(np.log(D_3D)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Relative Dispersion')
plt.savefig('./plot/m_50_5_23D/RD_3D_m_50_5_23D_c.eps')
plt.close()


# cloud D

p3D, = plt.plot(time/86400,Cd_3D[:,0],'k',linewidth=2)
p2D, = plt.plot(time/86400,Cd_2D[:,0],'b',linewidth=2)
plt.gca().set_yscale('log')

for z in range(nl):
 plt.plot(time,Cd_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
 plt.plot(time,Cd_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)

plt.gca().set_yscale('log')
plt.gca().set_xscale('log')
plt.gca().set_xticks(np.linspace(86400.0*2,86400*3.0,6))
plt.gca().xaxis.set_ticklabels(np.linspace(86400.0*2/86400.0,86400*3.0/86400.0,6))

plt.xlabel('Time [days]')
plt.ylabel('Cloud Dispersion [m]')
plt.legend((p3D,p2D),('3D','2D'))
plt.ylim((3*10**2,10**3))
plt.xlim((160000,280000))

#
print 'Saving Cd'
# 
plt.savefig('./plot/m_50_5_23D/CD_m_50_5_23D.eps')
plt.close()


plt.contourf(time/86400,range(nl),np.transpose(np.log(Cd_2D)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Relative Dispersion')
plt.savefig('./plot/m_50_5_23D/CD_2D_m_50_5_23D_c.eps')
plt.close()

plt.contourf(time/86400,range(nl),np.transpose(np.log(Cd_3D)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Relative Dispersion')
plt.savefig('./plot/m_50_5_23D/CD_3D_m_50_5_23D_c.eps')
plt.close()

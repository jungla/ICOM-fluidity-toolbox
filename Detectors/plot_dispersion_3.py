#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import myfun
import numpy as np
from scipy import optimize
import os
import scipy.stats as sp
import scipy
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

time2D, par2D = lagrangian_stats.read_particles(filename0)

exp3D = 'm_50_7_3Dc_particles'

filename0 = './mli_checkpoint.detectors'
filename0 = '/tamay2/mensa/fluidity/'+exp3D+'/'+filename0

time3D, par3D = lagrangian_stats.read_particles(filename0)

tt = min(len(time2D),len(time3D))

par3D = par3D[:,:,:tt]
par2D = par2D[:,:,:tt]

par2DP = lagrangian_stats.periodicCoords(par2D,10000,10000)
par3DP = lagrangian_stats.periodicCoords(par3D,10000,10000)

time = time2D[:tt]

# horizontal
depths = [1, 5, 11, 17, 26]
depths = [1, 5, 17]

# periodicity

par2D = np.reshape(par2DP,(20,20,30,3,tt))
par3D = np.reshape(par3DP,(20,20,30,3,tt))

nl = len(depths)

AD_3D = np.zeros((tt,nl))
AD_2D = np.zeros((tt,nl))
RD_3D = np.zeros((tt,nl))
RD_2D = np.zeros((tt,nl))
ED_3D = np.zeros((tt,nl))
ED_2D = np.zeros((tt,nl))
CD_3D = np.zeros((tt,nl))
CD_2D = np.zeros((tt,nl))


for z in range(len(depths)):
 print z
 print 'depth', depths[z]

 par2Dzr = par2D[:,:,depths[z],:,:]
 par3Dzr = par3D[:,:,depths[z],:,:]

 par2Dz = np.reshape(par2Dzr,(400,3,tt))
 par3Dz = np.reshape(par3Dzr,(400,3,tt))

 AD_2D[:,z] = lagrangian_stats.AD_t(par2Dz,tt)
 AD_3D[:,z] = lagrangian_stats.AD_t(par3Dz,tt)
 CD_2D[:,z] = lagrangian_stats.CD_t(par2Dz,tt)
 CD_3D[:,z] = lagrangian_stats.CD_t(par3Dz,tt)
 ED_2D[:,z] = lagrangian_stats.ED_t(par2Dz,tt)
 ED_3D[:,z] = lagrangian_stats.ED_t(par3Dz,tt)
 RD_2D[:,z] = lagrangian_stats.RD_t(par2Dzr,tt,19)
 RD_3D[:,z] = lagrangian_stats.RD_t(par3Dzr,tt,19)

 # abosolute D

 p3D, = plt.plot(time/86400,AD_3D[:,z],'k',linewidth=2)
 p2D, = plt.plot(time/86400,AD_2D[:,z],'b',linewidth=2)
 plt.gca().set_yscale('log')

 plt.xlabel('Time [days]')
 plt.ylabel('Absolute Dispersion [m]')
# plt.ylim((500,2000))
 plt.legend((p3D,p2D),('3D','2D'))
 #
 print 'Saving AD'
 # 
 plt.savefig('./plot/'+label+'/AD_'+label+'_z'+str(depths[z])+'.eps')
 print       './plot/'+label+'/AD_'+label+'_z'+str(depths[z])+'.eps' 
 plt.close()

 # ellipses D

 cD_3D = ED_3D[~np.isnan(ED_3D[:,z]),z]
 cD_2D = ED_2D[~np.isnan(ED_2D[:,z]),z]
 ctime3 = time[~np.isnan(ED_3D[:,z])]
 ctime2 = time[~np.isnan(ED_2D[:,z])]

 vtime = ctime3[ctime3<200000]
 Val = cD_3D[ctime3<200000]

# out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
# print out
# plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),color=[0.5, 0.5, 0.5],linewidth=2)

 vtime = ctime2[ctime2<200000]
 Val = cD_2D[ctime2<200000]

# out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
# print out
# plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),color=[0.5, 0.5, 0.5],linewidth=2)

# plt.plot([10**5,10**6],[10**4,10**5],'k')
# plt.plot([10**5,10**6],[10**4,10**6],'k')
# plt.plot([10**5,10**6],[10**4,10**7],'k')

 p3D, = plt.plot(time/86400,ED_3D[:,z],'k',linewidth=2)
 p2D, = plt.plot(time/86400,ED_2D[:,z],'b',linewidth=2)

 plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')

 plt.xlabel('Time [days]')
 plt.ylabel('Dispersion [m^2]')
 plt.legend((p3D,p2D),('3D','2D'))
# plt.ylim((3*10**4,10**6))
# plt.xlim((160000,250000))
 #
 print 'Saving ellipses D'
 # 
 plt.savefig('./plot/'+label+'/ED_'+label+'_z'+str(depths[z])+'.eps')
 print       './plot/'+label+'/ED_'+label+'_z'+str(depths[z])+'.eps' 
 plt.close()

 # relative D
 
 cRD_3D = RD_3D[~np.isnan(RD_3D[:,z]),z]
 cRD_2D = RD_2D[~np.isnan(RD_2D[:,z]),z]
 ctime3 = time[~np.isnan(RD_3D[:,z])]
 ctime2 = time[~np.isnan(RD_2D[:,z])]
 
 vtime = ctime3[ctime3<200000]
 Val = cRD_3D[ctime3<200000]

# out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
# print out 
# plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),color=[0.5, 0.5, 0.5],linewidth=2)
 
 vtime = ctime2[ctime2<200000]
 Val = cRD_2D[ctime2<200000]
 
# out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
# print out
# plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),color=[0.5, 0.5, 0.5],linewidth=2)
 
# plt.plot([10**5,10**6],[10**4,10**5],'k')
# plt.plot([10**5,10**6],[10**4,10**6],'k')
# plt.plot([10**5,10**6],[10**4,10**7],'k')

 p3D, = plt.plot(time/86400,RD_3D[:,z],'k',linewidth=2)
 p2D, = plt.plot(time/86400,RD_2D[:,z],'b',linewidth=2)

 plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')

 plt.xlabel('Time [days]')
 plt.ylabel('Relative Dispersion [m^2]')
 plt.legend((p3D,p2D),('3D','2D'))
# plt.ylim((10**3,10**6))
# plt.xlim((160000,250000))
 #
 print 'Saving RD'
 # 
 plt.savefig('./plot/'+label+'/RD_'+label+'_z'+str(depths[z])+'.eps')
 print       './plot/'+label+'/RD_'+label+'_z'+str(depths[z])+'.eps'
 plt.close()

 # cloud D
# plt.plot(time/86400,(time/86400)+CD_3D[0,z],'--k')
# plt.plot(time/86400,(time/86400)**2+CD_3D[0,z],'--k')
# plt.plot(time/86400,(time/86400)**3+CD_3D[0,z],'--k')

 p3D, = plt.plot(time/86400,CD_3D[:,z],'k',linewidth=2)
 p2D, = plt.plot(time/86400,CD_2D[:,z],'b',linewidth=2)
 plt.gca().set_yscale('log')

 plt.xlabel('Time [days]')
 plt.ylabel('Cloud Dispersion [m]')
 plt.legend((p3D,p2D),('3D','2D'))
 #plt.ylim((500,2000))
 #
 print 'Saving CD'
 # 
 plt.savefig('./plot/'+label+'/CD_'+label+'_z'+str(depths[z])+'.eps')
 print       './plot/'+label+'/CD_'+label+'_z'+str(depths[z])+'.eps' 
 plt.close()

# plottting all depths

# absolute D

p3D, = plt.plot(time/86400,AD_3D[:,0],color=[0,0,0],linewidth=2)
p2D, = plt.plot(time/86400,AD_2D[:,0],color=[0,0,1],linewidth=2)
#plt.gca().set_yscale('log')
#plt.gca().set_xscale('log')

z = 1
p3D5, = plt.plot(time/86400,AD_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
p2D5, = plt.plot(time/86400,AD_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
#z = 2
#p3D11, = plt.plot(time/86400,AD_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
#p2D11, = plt.plot(time/86400,AD_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
z = 2
p3D17, = plt.plot(time/86400,AD_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
p2D17, = plt.plot(time/86400,AD_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
#z = 4
#p3D26, = plt.plot(time/86400,AD_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
#p2D26, = plt.plot(time/86400,AD_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)

#plt.plot([10**5,10**6],[10**4,10**5],'k')
#plt.plot([10**5,10**6],[10**4,10**6],'k')
#plt.plot([10**5,10**6],[10**4,10**7],'k')

plt.gca().set_yscale('log')
#plt.gca().set_xscale('log')
#plt.gca().set_xticks(np.linspace(86400.0*2,86400*3.0,6))
#plt.gca().xaxis.set_ticklabels(np.linspace(86400.0*2/86400.0,86400*3.0/86400.0,6))
#plt.gca().set_yticks(np.linspace(3*10**2,4*10**3,5))
#plt.gca().yaxis.set_ticklabels(np.linspace(3*10**2,4*10**3,5))

plt.xlabel('Time [days]')
plt.ylabel('Absolute Dispersion [m]')
plt.legend((p3D,p3D5,p3D17,p2D,p2D5,p2D17),('3D 1m','3D 5m','3D 17m','2D 1m','2D 5m','2D 17m'),loc=2,fontsize=12)
#plt.legend((p3D,p3D5,p3D11,p3D17,p3D26,p2D,p2D5,p2D11,p2D17,p2D26),('3D 1m','3D 5m','3D 11m','3D 17m','3D 26m','2D 1m','2D 5m','2D 11m','2D 17m','2D 26m'),loc=2,fontsize=12)
plt.ylim((10**10,10**18))
#plt.xlim((1,2.2))

#
print 'Saving AD'
# 
plt.savefig('./plot/'+label+'/AD_'+label+'.eps')
print       './plot/'+label+'/AD_'+label+'.eps' 
plt.close()

plt.contourf(time/86400,range(nl),np.transpose(np.log(AD_2D)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Absolute Dispersion')
plt.savefig('./plot/'+label+'/AD_2D_'+label+'_c.eps')
print       './plot/'+label+'/AD_2D_'+label+'_c.eps' 
plt.close()

plt.contourf(time/86400,range(nl),np.transpose(np.log(AD_3D)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Absolute Dispersion')
plt.savefig('./plot/'+label+'/AD_3D_'+label+'_c.eps')
print       './plot/'+label+'/AD_3D_'+label+'_c.eps' 
plt.close()

# relative D
#plt.plot(time/86400,1000.0*np.exp(time/86400)+RD_3D[0,z],'--k')
#plt.plot(time/86400,1000.0*np.exp(2*time/86400)+RD_3D[0,z],'--k')
#plt.plot(time/86400,1000.0*np.exp(3*time/86400)+RD_3D[0,z],'--k')
p3D, = plt.plot(time/86400,RD_3D[:,0],color=[0,0,0],linewidth=2)
p2D, = plt.plot(time/86400,RD_2D[:,0],color=[0,0,1],linewidth=2)

#for z in range(nl):

# cD_3D = D_3D[~np.isnan(RD_3D[:,z]),z]
# cD_2D = D_2D[~np.isnan(RD_2D[:,z]),z]
# ctime3 = time[~np.isnan(RD_3D[:,z])]
# ctime2 = time[~np.isnan(RD_2D[:,z])]

# vtime = ctime3[ctime3<200000]
# Val = cD_3D[ctime3<200000]

# out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
# print out
# plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),'--k',linewidth=1)

# vtime = ctime2[ctime2<200000]
# Val = cD_2D[ctime2<200000]

# out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
# print out
# plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),'--k',linewidth=1)

# plt.plot(time/86400,RD_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
# plt.plot(time/86400,RD_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)

z = 1
p3D5, = plt.plot(time/86400,RD_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
p2D5, = plt.plot(time/86400,RD_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
#z = 2
#p3D11, = plt.plot(time/86400,RD_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
#p2D11, = plt.plot(time/86400,RD_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
z = 2
p3D17, = plt.plot(time/86400,RD_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
p2D17, = plt.plot(time/86400,RD_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
#z = 4
#p3D26, = plt.plot(time/86400,RD_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
#p2D26, = plt.plot(time/86400,RD_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)

#plt.plot([10**5,10**6],[10**4,10**5],'k')
#plt.plot([10**5,10**6],[10**4,10**6],'k')
#plt.plot([10**5,10**6],[10**4,10**7],'k')

plt.gca().set_yscale('log')
#plt.gca().set_xscale('log')
#plt.gca().set_xticks(np.linspace(86400.0*2,86400*3.0,6))
#plt.gca().xaxis.set_ticklabels(np.linspace(86400.0*2/86400.0,86400*3.0/86400.0,6))
#plt.gca().set_yticks(np.linspace(10**3,10**6,5))
#plt.gca().yaxis.set_ticklabels(np.linspace(10**3,10**6,5))

plt.xlabel('Time [days]')
plt.ylabel('Relative Dispersion [m^2]')
plt.legend((p3D,p3D5,p3D17,p2D,p2D5,p2D17),('3D 1m','3D 5m','3D 17m','2D 1m','2D 5m','2D 17m'),loc=2,fontsize=12)
#plt.legend((p3D,p3D5,p3D11,p3D17,p3D26,p2D,p2D5,p2D11,p2D17,p2D26),('3D 1m','3D 5m','3D 11m','3D 17m','3D 26m','2D 1m','2D 5m','2D 11m','2D 17m','2D 26m'),loc=2,fontsize=12)
#plt.legend((p3D,p2D),('3D','2D'))
#plt.ylim((10**3,10**6))
#plt.xlim((160000,280000))

plt.savefig('./plot/'+label+'/RD_'+label+'.eps')
plt.close()

plt.contourf(time/86400,depths,np.transpose(np.log(RD_2D)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Relative Dispersion')
plt.savefig('./plot/'+label+'/RD_2D_'+label+'_c.eps')
plt.close()

plt.contourf(time/86400,depths,np.transpose(np.log(RD_3D)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Relative Dispersion')
plt.savefig('./plot/'+label+'/RD_3D_'+label+'_c.eps')
plt.close()

# Ellipses D

p3D, = plt.plot(time/86400,ED_3D[:,0],color=[0,0,0],linewidth=2)
p2D, = plt.plot(time/86400,ED_2D[:,0],color=[0,0,1],linewidth=2)

#for z in range(nl):

# cD_3D = RD_3D[~np.isnan(RD_3D[:,z]),z]
# cD_2D = RD_2D[~np.isnan(RD_2D[:,z]),z]
# ctime3 = time[~np.isnan(D_3D[:,z])]
# ctime2 = time[~np.isnan(D_2D[:,z])]

# vtime = ctime3[ctime3<200000]
# Val = cD_3D[ctime3<200000]

# out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
# print out
# plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),'--k',linewidth=1)

# vtime = ctime2[ctime2<200000]
# Val = cD_2D[ctime2<200000]

# out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
# print out
# plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),'--k',linewidth=1)

# plt.plot(time/86400,ED_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
# plt.plot(time/86400,ED_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)

#plt.plot([10**5,10**6],[10**4,10**5],'k')
#plt.plot([10**5,10**6],[10**4,10**6],'k')
#plt.plot([10**5,10**6],[10**4,10**7],'k')

z = 1
p3D5, = plt.plot(time/86400,ED_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
p2D5, = plt.plot(time/86400,ED_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
#z = 2
#p3D11, = plt.plot(time/86400,ED_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
#p2D11, = plt.plot(time/86400,ED_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
z = 2
p3D17, = plt.plot(time/86400,ED_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
p2D17, = plt.plot(time/86400,ED_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
#z = 4
#p3D26, = plt.plot(time/86400,ED_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
#p2D26, = plt.plot(time/86400,ED_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)

plt.gca().set_yscale('log')
#plt.gca().set_xscale('log')

plt.xlabel('Time [days]')
plt.ylabel('Dispersion [m^2]')
#plt.ylim((3*10**4,10**6))
plt.legend((p3D,p3D5,p3D17,p2D,p2D5,p2D17),('3D 1m','3D 5m','3D 17m','2D 1m','2D 5m','2D 17m'),loc=2,fontsize=12)
#plt.legend((p3D,p3D5,p3D11,p3D17,p3D26,p2D,p2D5,p2D11,p2D17,p2D26),('3D 1m','3D 5m','3D 11m','3D 17m','3D 26m','2D 1m','2D 5m','2D 11m','2D 17m','2D 26m'),loc=2,fontsize=12)
#plt.legend((p3D,p2D),('3D','2D'))

plt.savefig('./plot/'+label+'/ED_'+label+'.eps')
print       './plot/'+label+'/ED_'+label+'.eps' 
plt.close()

plt.contourf(time/86400,depths,np.transpose(np.log(ED_2D)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Relative Dispersion')
plt.savefig('./plot/'+label+'/ED_2D_'+label+'_c.eps')
print       './plot/'+label+'/ED_2D_'+label+'_c.eps' 
plt.close()

plt.contourf(time/86400,depths,np.transpose(np.log(ED_3D)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Relative Dispersion')
plt.savefig('./plot/'+label+'/ED_3D_'+label+'_c.eps')
print       './plot/'+label+'/ED_3D_'+label+'_c.eps' 
plt.close()


# cloud D

p3D, = plt.plot(time/86400,CD_3D[:,0],'k',linewidth=2)
p2D, = plt.plot(time/86400,CD_2D[:,0],'b',linewidth=2)
plt.gca().set_yscale('log')

#for z in range(nl):
# plt.plot(time/86400,CD_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
# plt.plot(time/86400,CD_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)

z = 1
p3D5, = plt.plot(time/86400,CD_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
p2D5, = plt.plot(time/86400,CD_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
#z = 2
#p3D11, = plt.plot(time/86400,CD_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
#p2D11, = plt.plot(time/86400,CD_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
z = 2
p3D17, = plt.plot(time/86400,CD_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
p2D17, = plt.plot(time/86400,CD_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
#z = 4
#p3D26, = plt.plot(time/86400,CD_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
#p2D26, = plt.plot(time/86400,CD_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)

#plt.gca().set_xscale('log')
#plt.gca().set_xticks(np.linspace(86400.0*2,86400*3.0,6))
#plt.gca().xaxis.set_ticklabels(np.linspace(86400.0*2/86400.0,86400*3.0/86400.0,6))

plt.xlabel('Time [days]')
plt.ylabel('Cloud Dispersion [m]')
#plt.legend((p3D,p3D5,p3D11,p3D17,p3D26,p2D,p2D5,p2D11,p2D17,p2D26),('3D 1m','3D 5m','3D 11m','3D 17m','3D 26m','2D 1m','2D 5m','2D 11m','2D 17m','2D 26m'),loc=2,fontsize=12)
plt.legend((p3D,p3D5,p3D17,p2D,p2D5,p2D17),('3D 1m','3D 5m','3D 17m','2D 1m','2D 5m','2D 17m'),loc=2,fontsize=12)
#plt.legend((p3D,p2D),('3D','2D'))
#plt.ylim((3*10**2,10**3))
#plt.xlim((1,2.2))
#
print 'Saving CD'
# 
plt.savefig('./plot/'+label+'/CD_'+label+'.eps')
print       './plot/'+label+'/CD_'+label+'.eps' 
plt.close()

plt.contourf(time/86400,depths,np.transpose(np.log(CD_2D)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Relative Dispersion')
plt.savefig('./plot/'+label+'/CD_2D_'+label+'_c.eps')
print       './plot/'+label+'/CD_2D_'+label+'_c.eps' 
plt.close()

plt.contourf(time/86400,depths,np.transpose(np.log(CD_3D)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Relative Dispersion')
plt.savefig('./plot/'+label+'/CD_3D_'+label+'_c.eps')
print       './plot/'+label+'/CD_3D_'+label+'_c.eps' 
plt.close()


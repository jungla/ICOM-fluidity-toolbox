#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import myfun
import numpy as np
from scipy import optimize
import os
import scipy.stats as sp
import scipy
import lagrangian_stats
import advect_functions
from scipy.interpolate import interp1d

# read offline
print 'reading offline'
exp = 'm_25_2'
label = 'm_25_2_512'
filename0 = 'traj_m_25_2_512_0_48.csv'
tt = 48 # IC + 24-48 included
#x0 = range(0,2000,50)
#y0 = range(0,2000,50)
#z0 = range(0,50,2)
x0 = range(3000,4000,50)
y0 = range(2000,3000,50)
z0 = range(0,30,1)
xp = len(x0)
yp = len(y0)
zp = len(z0)

pt = xp*yp*zp
timet, par0 = advect_functions.read_particles_csv(filename0,xp,yp,zp,tt)

par0 = lagrangian_stats.periodicCoords(par0,10000,4000)

time0 = (timet)*1200 + 48*3600 - 1200

# read online
print 'reading online'
exp = 'm_25_2_512'

try: os.stat('./plot/'+exp)
except OSError: os.mkdir('./plot/'+exp)

filename0 = './mli_tracer.detectors'
filename0 = '/tamay2/mensa/fluidity/'+exp+'/'+filename0

time1, parT = lagrangian_stats.read_particles(filename0)

dt = 1200
pt = len(parT)
timei = np.asarray(range(0,tt,1))*dt + 86400*2.0 + dt
par1 = np.zeros((pt,3,len(timei)))

for p in range(len(par1)):
 f0 = interp1d(time1,parT[p,0,:])
 f1 = interp1d(time1,parT[p,1,:])
 f2 = interp1d(time1,parT[p,2,:])
 par1[p,0,:] = f0(timei)
 par1[p,1,:] = f1(timei)
 par1[p,2,:] = f2(timei)

time = timei

tt = len(time)

# horizontal
depths = [1, 5, 11, 17, 26]
depths = [1, 5, 17]



paroff = np.reshape(par0,(xp,yp,zp,3,tt))
paron = np.reshape(par1,(xp,yp,zp,3,tt))

nl = len(depths)

AD_on = np.zeros((tt,nl))
AD_off = np.zeros((tt,nl))
RD_on = np.zeros((tt,nl))
RD_off = np.zeros((tt,nl))
ED_on = np.zeros((tt,nl))
ED_off = np.zeros((tt,nl))
CD_on = np.zeros((tt,nl))
CD_off = np.zeros((tt,nl))


for z in range(len(depths)):
 print z
 print 'depth', depths[z]

 paroffzr = paroff[:,:,depths[z],:,:]
 paronzr = paron[:,:,depths[z],:,:]

 paroffz = np.reshape(paroffzr,(400,3,tt))
 paronz = np.reshape(paronzr,(400,3,tt))

 AD_off[:,z] = lagrangian_stats.AD_t(paroffz,tt)
 AD_on[:,z] = lagrangian_stats.AD_t(paronz,tt)
 CD_off[:,z] = lagrangian_stats.CD_t(paroffz,tt)
 CD_on[:,z] = lagrangian_stats.CD_t(paronz,tt)
 ED_off[:,z] = lagrangian_stats.ED_t(paroffz,tt)
 ED_on[:,z] = lagrangian_stats.ED_t(paronz,tt)
 RD_off[:,z] = lagrangian_stats.RD_t(paroffzr,tt,19)
 RD_on[:,z] = lagrangian_stats.RD_t(paronzr,tt,19)

 # abosolute D

 pon, = plt.plot(time/86400,AD_on[:,z],'k',linewidth=2)
 poff, = plt.plot(time/86400,AD_off[:,z],'b',linewidth=2)
 plt.gca().set_yscale('log')

 plt.xlabel('Time [days]')
 plt.ylabel('Absolute Dispersion [m]')
# plt.ylim((500,2000))
 plt.legend((pon,poff),('on','off'))
 #
 print 'Saving AD'
 # 
 plt.savefig('./plot/'+label+'/AD_'+label+'_z'+str(depths[z])+'.eps')
 print       './plot/'+label+'/AD_'+label+'_z'+str(depths[z])+'.eps' 
 plt.close()

 # ellipses D

 cD_on = ED_on[~np.isnan(ED_on[:,z]),z]
 cD_off = ED_off[~np.isnan(ED_off[:,z]),z]
 ctime3 = time[~np.isnan(ED_on[:,z])]
 ctime2 = time[~np.isnan(ED_off[:,z])]

 vtime = ctime3[ctime3<200000]
 Val = cD_on[ctime3<200000]

# out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
# print out
# plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),color=[0.5, 0.5, 0.5],linewidth=2)

 vtime = ctime2[ctime2<200000]
 Val = cD_off[ctime2<200000]

# out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
# print out
# plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),color=[0.5, 0.5, 0.5],linewidth=2)

# plt.plot([10**5,10**6],[10**4,10**5],'k')
# plt.plot([10**5,10**6],[10**4,10**6],'k')
# plt.plot([10**5,10**6],[10**4,10**7],'k')

 pon, = plt.plot(time/86400,ED_on[:,z],'k',linewidth=2)
 poff, = plt.plot(time/86400,ED_off[:,z],'b',linewidth=2)

 plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')

 plt.xlabel('Time [days]')
 plt.ylabel('Dispersion [m^2]')
 plt.legend((pon,poff),('on','off'))
# plt.ylim((3*10**4,10**6))
# plt.xlim((160000,250000))
 #
 print 'Saving ellipses D'
 # 
 plt.savefig('./plot/'+label+'/ED_'+label+'_z'+str(depths[z])+'.eps')
 print       './plot/'+label+'/ED_'+label+'_z'+str(depths[z])+'.eps' 
 plt.close()

 # relative D
 
 cRD_on = RD_on[~np.isnan(RD_on[:,z]),z]
 cRD_off = RD_off[~np.isnan(RD_off[:,z]),z]
 ctime3 = time[~np.isnan(RD_on[:,z])]
 ctime2 = time[~np.isnan(RD_off[:,z])]
 
 vtime = ctime3[ctime3<200000]
 Val = cRD_on[ctime3<200000]

# out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
# print out 
# plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),color=[0.5, 0.5, 0.5],linewidth=2)
 
 vtime = ctime2[ctime2<200000]
 Val = cRD_off[ctime2<200000]
 
# out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
# print out
# plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),color=[0.5, 0.5, 0.5],linewidth=2)
 
# plt.plot([10**5,10**6],[10**4,10**5],'k')
# plt.plot([10**5,10**6],[10**4,10**6],'k')
# plt.plot([10**5,10**6],[10**4,10**7],'k')

 pon, = plt.plot(time/86400,RD_on[:,z],'k',linewidth=2)
 poff, = plt.plot(time/86400,RD_off[:,z],'b',linewidth=2)

 plt.gca().set_yscale('log')
# plt.gca().set_xscale('log')

 plt.xlabel('Time [days]')
 plt.ylabel('Relative Dispersion [m^2]')
 plt.legend((pon,poff),('on','off'))
# plt.ylim((10**3,10**6))
# plt.xlim((160000,250000))
 #
 print 'Saving RD'
 # 
 plt.savefig('./plot/'+label+'/RD_'+label+'_z'+str(depths[z])+'.eps')
 print       './plot/'+label+'/RD_'+label+'_z'+str(depths[z])+'.eps'
 plt.close()

 # cloud D
# plt.plot(time/86400,(time/86400)+CD_on[0,z],'--k')
# plt.plot(time/86400,(time/86400)**2+CD_on[0,z],'--k')
# plt.plot(time/86400,(time/86400)**3+CD_on[0,z],'--k')

 pon, = plt.plot(time/86400,CD_on[:,z],'k',linewidth=2)
 poff, = plt.plot(time/86400,CD_off[:,z],'b',linewidth=2)
 plt.gca().set_yscale('log')

 plt.xlabel('Time [days]')
 plt.ylabel('Cloud Dispersion [m]')
 plt.legend((pon,poff),('on','off'))
 #plt.ylim((500,2000))
 #
 print 'Saving CD'
 # 
 plt.savefig('./plot/'+label+'/CD_'+label+'_z'+str(depths[z])+'.eps')
 print       './plot/'+label+'/CD_'+label+'_z'+str(depths[z])+'.eps' 
 plt.close()

# plottting all depths

# absolute D

pon, = plt.plot(time/86400,AD_on[:,0],color=[0,0,0],linewidth=2)
poff, = plt.plot(time/86400,AD_off[:,0],color=[0,0,1],linewidth=2)
#plt.gca().set_yscale('log')
#plt.gca().set_xscale('log')

z = 1
pon5, = plt.plot(time/86400,AD_on[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
poff5, = plt.plot(time/86400,AD_off[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
#z = 2
#pon11, = plt.plot(time/86400,AD_on[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
#poff11, = plt.plot(time/86400,AD_off[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
z = 2
pon17, = plt.plot(time/86400,AD_on[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
poff17, = plt.plot(time/86400,AD_off[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
#z = 4
#pon26, = plt.plot(time/86400,AD_on[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
#poff26, = plt.plot(time/86400,AD_off[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)

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
plt.legend((pon,pon5,pon17,poff,poff5,poff17),('on 1m','on 5m','on 17m','off 1m','off 5m','off 17m'),loc=2,fontsize=12)
#plt.legend((pon,pon5,pon11,pon17,pon26,poff,poff5,poff11,poff17,poff26),('on 1m','on 5m','on 11m','on 17m','on 26m','off 1m','off 5m','off 11m','off 17m','off 26m'),loc=2,fontsize=12)
plt.ylim((10**10,10**18))
#plt.xlim((1,2.2))

#
print 'Saving AD'
# 
plt.savefig('./plot/'+label+'/AD_'+label+'.eps')
print       './plot/'+label+'/AD_'+label+'.eps' 
plt.close()

plt.contourf(time/86400,range(nl),np.transpose(np.log(AD_off)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Absolute Dispersion')
plt.savefig('./plot/'+label+'/AD_off_'+label+'_c.eps')
print       './plot/'+label+'/AD_off_'+label+'_c.eps' 
plt.close()

plt.contourf(time/86400,range(nl),np.transpose(np.log(AD_on)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Absolute Dispersion')
plt.savefig('./plot/'+label+'/AD_on_'+label+'_c.eps')
print       './plot/'+label+'/AD_on_'+label+'_c.eps' 
plt.close()

# relative D
#plt.plot(time/86400,1000.0*np.exp(time/86400)+RD_on[0,z],'--k')
#plt.plot(time/86400,1000.0*np.exp(2*time/86400)+RD_on[0,z],'--k')
#plt.plot(time/86400,1000.0*np.exp(3*time/86400)+RD_on[0,z],'--k')
pon, = plt.plot(time/86400,RD_on[:,0],color=[0,0,0],linewidth=2)
poff, = plt.plot(time/86400,RD_off[:,0],color=[0,0,1],linewidth=2)

#for z in range(nl):

# cD_on = D_on[~np.isnan(RD_on[:,z]),z]
# cD_off = D_off[~np.isnan(RD_off[:,z]),z]
# ctime3 = time[~np.isnan(RD_on[:,z])]
# ctime2 = time[~np.isnan(RD_off[:,z])]

# vtime = ctime3[ctime3<200000]
# Val = cD_on[ctime3<200000]

# out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
# print out
# plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),'--k',linewidth=1)

# vtime = ctime2[ctime2<200000]
# Val = cD_off[ctime2<200000]

# out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
# print out
# plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),'--k',linewidth=1)

# plt.plot(time/86400,RD_on[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
# plt.plot(time/86400,RD_off[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)

z = 1
pon5, = plt.plot(time/86400,RD_on[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
poff5, = plt.plot(time/86400,RD_off[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
#z = 2
#pon11, = plt.plot(time/86400,RD_on[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
#poff11, = plt.plot(time/86400,RD_off[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
z = 2
pon17, = plt.plot(time/86400,RD_on[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
poff17, = plt.plot(time/86400,RD_off[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
#z = 4
#pon26, = plt.plot(time/86400,RD_on[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
#poff26, = plt.plot(time/86400,RD_off[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)

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
plt.legend((pon,pon5,pon17,poff,poff5,poff17),('on 1m','on 5m','on 17m','off 1m','off 5m','off 17m'),loc=2,fontsize=12)
#plt.legend((pon,pon5,pon11,pon17,pon26,poff,poff5,poff11,poff17,poff26),('on 1m','on 5m','on 11m','on 17m','on 26m','off 1m','off 5m','off 11m','off 17m','off 26m'),loc=2,fontsize=12)
#plt.legend((pon,poff),('on','off'))
#plt.ylim((10**3,10**6))
#plt.xlim((160000,280000))

plt.savefig('./plot/'+label+'/RD_'+label+'.eps')
plt.close()

plt.contourf(time/86400,depths,np.transpose(np.log(RD_off)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Relative Dispersion')
plt.savefig('./plot/'+label+'/RD_off_'+label+'_c.eps')
plt.close()

plt.contourf(time/86400,depths,np.transpose(np.log(RD_on)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Relative Dispersion')
plt.savefig('./plot/'+label+'/RD_on_'+label+'_c.eps')
plt.close()

# Ellipses D

pon, = plt.plot(time/86400,ED_on[:,0],color=[0,0,0],linewidth=2)
poff, = plt.plot(time/86400,ED_off[:,0],color=[0,0,1],linewidth=2)

#for z in range(nl):

# cD_on = RD_on[~np.isnan(RD_on[:,z]),z]
# cD_off = RD_off[~np.isnan(RD_off[:,z]),z]
# ctime3 = time[~np.isnan(D_on[:,z])]
# ctime2 = time[~np.isnan(D_off[:,z])]

# vtime = ctime3[ctime3<200000]
# Val = cD_on[ctime3<200000]

# out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
# print out
# plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),'--k',linewidth=1)

# vtime = ctime2[ctime2<200000]
# Val = cD_off[ctime2<200000]

# out,cov = optimize.curve_fit(f_exp, vtime, Val, [ -3.04040627e+05,   3.19292411e+01,   5.25848896e-05], maxfev=100000)
# print out
# plt.plot(vtime,out[0] + out[1]*np.exp(vtime*out[2]),'--k',linewidth=1)

# plt.plot(time/86400,ED_on[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
# plt.plot(time/86400,ED_off[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)

#plt.plot([10**5,10**6],[10**4,10**5],'k')
#plt.plot([10**5,10**6],[10**4,10**6],'k')
#plt.plot([10**5,10**6],[10**4,10**7],'k')

z = 1
pon5, = plt.plot(time/86400,ED_on[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
poff5, = plt.plot(time/86400,ED_off[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
#z = 2
#pon11, = plt.plot(time/86400,ED_on[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
#poff11, = plt.plot(time/86400,ED_off[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
z = 2
pon17, = plt.plot(time/86400,ED_on[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
poff17, = plt.plot(time/86400,ED_off[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
#z = 4
#pon26, = plt.plot(time/86400,ED_on[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
#poff26, = plt.plot(time/86400,ED_off[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)

plt.gca().set_yscale('log')
#plt.gca().set_xscale('log')

plt.xlabel('Time [days]')
plt.ylabel('Dispersion [m^2]')
#plt.ylim((3*10**4,10**6))
plt.legend((pon,pon5,pon17,poff,poff5,poff17),('on 1m','on 5m','on 17m','off 1m','off 5m','off 17m'),loc=2,fontsize=12)
#plt.legend((pon,pon5,pon11,pon17,pon26,poff,poff5,poff11,poff17,poff26),('on 1m','on 5m','on 11m','on 17m','on 26m','off 1m','off 5m','off 11m','off 17m','off 26m'),loc=2,fontsize=12)
#plt.legend((pon,poff),('on','off'))

plt.savefig('./plot/'+label+'/ED_'+label+'.eps')
print       './plot/'+label+'/ED_'+label+'.eps' 
plt.close()

plt.contourf(time/86400,depths,np.transpose(np.log(ED_off)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Relative Dispersion')
plt.savefig('./plot/'+label+'/ED_off_'+label+'_c.eps')
print       './plot/'+label+'/ED_off_'+label+'_c.eps' 
plt.close()

plt.contourf(time/86400,depths,np.transpose(np.log(ED_on)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Relative Dispersion')
plt.savefig('./plot/'+label+'/ED_on_'+label+'_c.eps')
print       './plot/'+label+'/ED_on_'+label+'_c.eps' 
plt.close()


# cloud D

pon, = plt.plot(time/86400,CD_on[:,0],'k',linewidth=2)
poff, = plt.plot(time/86400,CD_off[:,0],'b',linewidth=2)
plt.gca().set_yscale('log')

#for z in range(nl):
# plt.plot(time/86400,CD_on[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
# plt.plot(time/86400,CD_off[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)

z = 1
pon5, = plt.plot(time/86400,CD_on[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
poff5, = plt.plot(time/86400,CD_off[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
#z = 2
#pon11, = plt.plot(time/86400,CD_on[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
#poff11, = plt.plot(time/86400,CD_off[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
z = 2
pon17, = plt.plot(time/86400,CD_on[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
poff17, = plt.plot(time/86400,CD_off[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
#z = 4
#pon26, = plt.plot(time/86400,CD_on[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
#poff26, = plt.plot(time/86400,CD_off[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)

#plt.gca().set_xscale('log')
#plt.gca().set_xticks(np.linspace(86400.0*2,86400*3.0,6))
#plt.gca().xaxis.set_ticklabels(np.linspace(86400.0*2/86400.0,86400*3.0/86400.0,6))

plt.xlabel('Time [days]')
plt.ylabel('Cloud Dispersion [m]')
#plt.legend((pon,pon5,pon11,pon17,pon26,poff,poff5,poff11,poff17,poff26),('on 1m','on 5m','on 11m','on 17m','on 26m','off 1m','off 5m','off 11m','off 17m','off 26m'),loc=2,fontsize=12)
plt.legend((pon,pon5,pon17,poff,poff5,poff17),('on 1m','on 5m','on 17m','off 1m','off 5m','off 17m'),loc=2,fontsize=12)
#plt.legend((pon,poff),('on','off'))
#plt.ylim((3*10**2,10**3))
#plt.xlim((1,2.2))
#
print 'Saving CD'
# 
plt.savefig('./plot/'+label+'/CD_'+label+'.eps')
print       './plot/'+label+'/CD_'+label+'.eps' 
plt.close()

plt.contourf(time/86400,depths,np.transpose(np.log(CD_off)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Relative Dispersion')
plt.savefig('./plot/'+label+'/CD_off_'+label+'_c.eps')
print       './plot/'+label+'/CD_off_'+label+'_c.eps' 
plt.close()

plt.contourf(time/86400,depths,np.transpose(np.log(CD_on)),50)
plt.xlabel('Time [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.title('Relative Dispersion')
plt.savefig('./plot/'+label+'/CD_on_'+label+'_c.eps')
print       './plot/'+label+'/CD_on_'+label+'_c.eps' 
plt.close()


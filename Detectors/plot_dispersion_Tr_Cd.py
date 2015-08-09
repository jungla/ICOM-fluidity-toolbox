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


def f_exp(t,a,b,c):
    return a + b*np.exp(c*t)

label = 'm_50_7'

exp3D = label+'_3D_particles'
exp2D = label+'_2D_particles'

filename3D = '/tamay2/mensa/fluidity/'+exp3D+'/mli_checkpoint.detectors'
filename2D = '/tamay2/mensa/fluidity/'+exp2D+'/mli_checkpoint.detectors'

print 'Reading ', filename2D, filename3D

try: os.stat('./plot/'+label+'_23D')
except OSError: os.mkdir('./plot/'+label+'_23D')

#det3D = fluidity_tools.stat_parser(filename3D)
#det2D = fluidity_tools.stat_parser(filename2D)

# pt same for 3D and 2D
pt3 = int(os.popen('grep position '+filename3D+'| wc -l').read()) # read the number of particles grepping all the positions in the file
pt2 = int(os.popen('grep position '+filename2D+'| wc -l').read()) # read the number of particles grepping all the positions in the file
time3D = det3D['ElapsedTime']['value']
time2D = det2D['ElapsedTime']['value']

tt = min(len(time2D),len(time3D))
#tt = 267
#tt = 120

if len(time2D) < len(time3D):
 time = time2D[:tt]
else:
 time = time3D[:tt]

print 'particles:',pt3
print 'particles:',pt2
print 'timesteps:',tt

par3D = np.zeros((pt3,3,tt))
par2D = np.zeros((pt2,3,tt))

for d in xrange(pt3):
 temp3D = det3D['particles_'+myfun.digit(d+1,len(str(pt3)))]['position']
 par3D[d,:,:] = temp3D[:,:tt]
for d in xrange(pt2):
 temp2D = det2D['particles_'+myfun.digit(d+1,len(str(pt2)))]['position']
 par2D[d,:,:] = temp2D[:,:tt]

# horizontal
depth = 11 #11

pd = range(1,depth,3)
pd = [1, 5, 11, 17, 26]
#pd = [1]

nl = len(pd)

D_3D = np.zeros((tt,nl))
D_2D = np.zeros((tt,nl))
Cd_3D = np.zeros((tt,nl))
Cd_2D = np.zeros((tt,nl))

for z in range(nl):
 print pd[z]
 print 'depth', z

 for t in range(tt):
  par3D[par3D[:,0,t] < 10,:,t:tt] = np.nan
  par3D[par3D[:,0,t] > 9990,:,t:tt] = np.nan
  par3D[par3D[:,1,t] < 10,:,t:tt] = np.nan
  par3D[par3D[:,1,t] > 9990,:,t:tt] = np.nan
  par2D[par2D[:,0,t] < 10,:,t:tt] = np.nan
  par2D[par2D[:,0,t] > 9990,:,t:tt] = np.nan
  par2D[par2D[:,1,t] < 10,:,t:tt] = np.nan
  par2D[par2D[:,1,t] > 9990,:,t:tt] = np.nan

 par2Dz = np.reshape(par2D,(20,20,30,3,tt))
 par3Dz = np.reshape(par3D,(20,20,40,3,tt))

 par2Dzr = par2Dz[:,:,z,:,:]
 par3Dzr = par3Dz[:,:,z,:,:]

 par2Dz = np.reshape(par2Dzr,(400,3,tt))
 par3Dz = np.reshape(par3Dzr,(400,3,tt))


 # Cloud dispersion
 #
 #
 Pt3D = np.zeros((2,tt))
 Pt2D = np.zeros((2,tt))
 #
 Pt3D[:] = scipy.stats.nanmean(par3Dz[:,0,:],0),scipy.stats.nanmean(par3Dz[:,1,:],0)
 Pt2D[:] = scipy.stats.nanmean(par2Dz[:,0,:],0),scipy.stats.nanmean(par2Dz[:,1,:],0)
 # 
 Cd_3D[:,z] = scipy.stats.nanmean((par3Dz[:,0,:] - Pt3D[0,:])**2,0)
# Cd_3D[:,z] = scipy.stats.nanmean((par3Dz[:,0,:] - Pt3D[0,:])**2 + (par3Dz[:,1,:] - Pt3D[1,:])**2,0)
 Cd_2D[:,z] = scipy.stats.nanmean((par2Dz[:,0,:] - Pt2D[0,:])**2,0) # + (par2Dz[:,1,:] - Pt2D[1,:])**2,0)


 #
 # Ellipses disperions
 #


 for t in range(tt):
  x2 = par2Dz[:,0,t]
  y2 = par2Dz[:,1,t]
  x3 = par3Dz[:,0,t]
  y3 = par3Dz[:,1,t]

  x2 = x2[~np.isnan(x2)]
  y2 = y2[~np.isnan(y2)]
  x3 = x3[~np.isnan(x3)]
  y3 = y3[~np.isnan(y3)]

  if len(x3) > 1 and len(y3) > 1:
   xt3 = x3 - scipy.stats.nanmean(x3)
   yt3 = y3 - scipy.stats.nanmean(y3)
   cov3 = np.cov(xt3, yt3)
   if ~np.isnan(cov3).any():
    lambda_3, v = np.linalg.eig(cov3)
    lambda_3 = np.sqrt(lambda_3)
    D_3D[t,z] = lambda_3[0]*lambda_3[1]
   else:
    D_3D[t,z] = np.nan
  else:
   D_3D[t,z] = np.nan

  if len(x2) > 1 and len(y2) > 1:
   xt2 = x2 - scipy.stats.nanmean(x2)
   yt2 = y2 - scipy.stats.nanmean(y2)
   cov2 = np.cov(xt2, yt2)
   if ~np.isnan(cov2).any():
    lambda_2, v = np.linalg.eig(cov2)
    lambda_2 = np.sqrt(lambda_2)
    D_2D[t,z] = lambda_2[0]*lambda_2[1]
   else:
    D_2D[t,z] = np.nan
  else:
   D_2D[t,z] = np.nan

 D_3D[120:,0] = np.nan
 D_2D[120:,0] = np.nan

# Tracer second moment

import csv
path = './D2_1200.csv'
timef = 100

val = np.zeros([timef,5])
timeTr = np.zeros([timef])
t = 0

with open(path, 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        #time.append() # change later
        i = 0
        for item in row:
         if i == 0:
          timeTr[t] = float(item)
         else:
          val[t,i-1] = float(item)
#         print float(item),i
         i = i+1
        t = t+1

timeTr = timeTr[timeTr > 0]
D_Tr = np.reshape(val[val > 0],[len(timeTr),5])

# plotting all together

# Ellipses D

p3D, = plt.plot(time/86400,D_3D[:,0],color=[0,0,0],linewidth=2)
p2D, = plt.plot(time/86400,D_2D[:,0],color=[0,0,1],linewidth=2)
pTr, = plt.plot(timeTr/86400,D_Tr[:,0],'--',color=[0,0,0],linewidth=2)

z = 1
p3D5, = plt.plot(time/86400,D_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
p2D5, = plt.plot(time/86400,D_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
pTr5, = plt.plot(timeTr/86400,D_Tr[:,z],'--',color=[z/float(nl),z/float(nl),z/float(nl],linewidth=2)
z = 2
p3D11, = plt.plot(time/86400,D_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
p2D11, = plt.plot(time/86400,D_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
pTr11, = plt.plot(timeTr/86400,D_Tr[:,z],'--',color=[z/float(nl),z/float(nl),z/float(nl],linewidth=2)
z = 3
p3D17, = plt.plot(time/86400,D_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
p2D17, = plt.plot(time/86400,D_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
pTr17, = plt.plot(timeTr/86400,D_Tr[:,z],'--',color=[z/float(nl),z/float(nl),z/float(nl],linewidth=2)
z = 4
p3D26, = plt.plot(time/86400,D_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
p2D26, = plt.plot(time/86400,D_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
pTr26, = plt.plot(timeTr/86400,D_Tr[:,z],'--',color=[z/float(nl),z/float(nl),z/float(nl],linewidth=2)

plt.gca().set_yscale('log')
#plt.gca().set_xscale('log')

plt.xlabel('Time [days]')
plt.ylabel('Dispersion [m^2]')
plt.ylim((3*10**4,10**6))
plt.legend((p3D,p3D5,p3D11,p3D17,p3D26,p2D,p2D5,p2D11,p2D17,p2D26,pTr,pTr5,pTr11,pTr17,pTr26),('3D 1m','3D 5m','3D 11m','3D 17m','3D 26m','2D 1m','2D 5m','2D 11m','2D 17m','2D 26m','Tr 1m','Tr 5m','Tr 11m','Tr 17m','Tr 26m'),loc=2,fontsize=12)


plt.savefig('./plot/'+label+'_23D/ED_Tr_'+label+'_23D.eps')
print       './plot/'+label+'_23D/ED_Tr_'+label+'_23D.eps' 
plt.close()


# cloud D
Cd_2D[120:,0] = np.nan
Cd_3D[120:,0] = np.nan

p3D, = plt.plot(time/86400,Cd_3D[:,0],'k',linewidth=2)
p2D, = plt.plot(time/86400,Cd_2D[:,0],'b',linewidth=2)
pTr, = plt.plot(timeTr/86400,D_Tr[:,0],'k--',linewidth=2)

z = 1
p3D5, = plt.plot(time/86400,Cd_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
p2D5, = plt.plot(time/86400,Cd_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
pTr5, = plt.plot(timeTr/86400,D_Tr[:,z],'--',color=[z/float(nl),z/float(nl),z/float(nl],linewidth=2)
z = 2
p3D11, = plt.plot(time/86400,Cd_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
p2D11, = plt.plot(time/86400,Cd_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
pTr11, = plt.plot(timeTr/86400,D_Tr[:,z],'--',color=[z/float(nl),z/float(nl),z/float(nl],linewidth=2)
z = 3
p3D17, = plt.plot(time/86400,Cd_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
p2D17, = plt.plot(time/86400,Cd_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
pTr17, = plt.plot(timeTr/86400,D_Tr[:,z],'--',color=[z/float(nl),z/float(nl),z/float(nl],linewidth=2)
z = 4
p3D26, = plt.plot(time/86400,Cd_3D[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
p2D26, = plt.plot(time/86400,Cd_2D[:,z],color=[z/float(nl),z/float(nl),1],linewidth=2)
pTr26, = plt.plot(timeTr/86400,D_Tr[:,z],'--',color=[z/float(nl),z/float(nl),z/float(nl],linewidth=2)

plt.gca().set_yscale('log')

#plt.gca().set_xscale('log')
#plt.gca().set_xticks(np.linspace(86400.0*2,86400*3.0,6))
#plt.gca().xaxis.set_ticklabels(np.linspace(86400.0*2/86400.0,86400*3.0/86400.0,6))

plt.xlabel('Time [days]')
plt.ylabel('Cloud Dispersion [m]')
plt.legend((p3D,p3D5,p3D11,p3D17,p3D26,p2D,p2D5,p2D11,p2D17,p2D26,pTr,pTr5,pTr11,pTr17,pTr26),('3D 1m','3D 5m','3D 11m','3D 17m','3D 26m','2D 1m','2D 5m','2D 11m','2D 17m','2D 26m','Tr 1m','Tr 5m','Tr 11m','Tr 17m','Tr 26m'),loc=2,fontsize=12)
#plt.legend((p3D,p2D,pTr),('3D','2D','Tr'))
#plt.ylim((3*10**2,10**3))
#plt.xlim((160000,280000))


#
print 'Saving Cd'
# 
plt.savefig('./plot/'+label+'_23D/CD_Tr_'+label+'_23D.eps')
print       './plot/'+label+'_23D/CD_Tr_'+label+'_23D.eps' 
plt.close()


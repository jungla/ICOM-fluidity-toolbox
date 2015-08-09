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

filename = './particles.detectors'

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
depth = 16

pd = range(1,depth,3)

nl = len(pd)

D_3D = np.zeros((tt,nl))
D_2D = np.zeros((tt,nl))
Diff_3D = np.zeros((tt,nl))
Diff_2D = np.zeros((tt,nl))

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
 
 # relative D
 #
 D_3Dm = np.zeros((19,tt))
 D_2Dm = np.zeros((19,tt))
 for i in range(19):
  D_3Dm[i,:] = np.mean((par3Dzr[i+1,:,0,:] - par3Dzr[i,:,0,:])**2 + (par3Dzr[i+1,:,1,:] - par3Dzr[i,:,1,:])**2,0) 
  D_2Dm[i,:] = np.mean((par2Dzr[i+1,:,0,:] - par2Dzr[i,:,0,:])**2 + (par2Dzr[i+1,:,1,:] - par2Dzr[i,:,1,:])**2,0) 

 D_3D[:,z] = np.mean(D_3Dm,0)
 D_2D[:,z] = np.mean(D_2Dm,0)
 #

 # cloud D
 #
 #Pt3D = np.zeros((2,tt))
 #Pt2D = np.zeros((2,tt))
 #
 #Pt3D[:] = np.mean(par3Dz[:,0,:],0),np.mean(par3Dz[:,1,:],0)
 #Pt2D[:] = np.mean(par3Dz[:,0,:],0),np.mean(par3Dz[:,1,:],0)
 # 
 #Cd_3D[:,z] = np.sqrt(np.mean((par3Dz[:,0,:] - Pt3D[0,:])**2 + (par3Dz[:,1,:] - Pt3D[1,:])**2,0))
 #Cd_2D[:,z] = np.sqrt(np.mean((par2Dz[:,0,:] - Pt2D[0,:])**2 + (par2Dz[:,1,:] - Pt2D[1,:])**2,0))

 Diff_3D[:,z] = 0.5*np.gradient(D_3D[:,z])/np.gradient(time)
 Diff_2D[:,z] = 0.5*np.gradient(D_2D[:,z])/np.gradient(time)

 # clean
 Diff_3D[Diff_3D[:,z]<=0,z] = np.nan
 Diff_2D[Diff_2D[:,z]<=0,z] = np.nan
 D_3D[D_3D[:,z]<=0,z] = np.nan
 D_2D[D_2D[:,z]<=0,z] = np.nan

 ax = plt.gca()
 s3D = ax.scatter(np.sqrt(D_3D[:,z])*100,Diff_3D[:,z]*100**2,color='k')
 s2D = ax.scatter(np.sqrt(D_2D[:,z])*100,Diff_2D[:,z]*100**2,color='b')
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
 plt.savefig('./plot/m_50_5_23D/Diff_m_50_5_23D_z'+str(pd[z])+'.eps')
 plt.close()

# all on same plot

ax = plt.gca()
s3D = ax.scatter(np.sqrt(D_3D[:,z])*100,Diff_3D[:,z]*100**2,color=[0,0,0])
s2D = ax.scatter(np.sqrt(D_2D[:,z])*100,Diff_2D[:,z]*100**2,color=[0,0,1])

for z in range(nl):
 ax.scatter(np.sqrt(D_3D[:,z])*100,Diff_3D[:,z]*100**2,color=[z/float(nl),z/float(nl),z/float(nl)])
 ax.scatter(np.sqrt(D_2D[:,z])*100,Diff_2D[:,z]*100**2,color=[z/float(nl),z/float(nl),1])

ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('D [cm]')
plt.ylabel('k [cm^2/s]')
plt.legend([s3D,s2D],['3D','2D'])



plt.savefig('./plot/m_50_5_23D/Diff_m_50_5_23D.eps')
plt.close()

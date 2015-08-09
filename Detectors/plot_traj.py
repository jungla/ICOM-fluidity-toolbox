#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import myfun
import numpy as np
import os

exp3D = 'm_50_5_2D_particles'
exp2D = 'm_50_5_3D_particles'

filename = './particles.detectors'

tt = 47

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

if len(time2D) < len(time3D):
 time = time2D
else:
 time = time3D

print 'particles:',pt
print 'timesteps:',tt

par3D = np.zeros((pt,3,tt))
par2D = np.zeros((pt,3,tt))

for d in xrange(pt):
 temp3D = det3D['particles_'+myfun.digit(d+1,len(str(pt)))]['position']
 par3D[d,:,:] = temp3D[:,0:tt]
 temp2D = det2D['particles_'+myfun.digit(d+1,len(str(pt)))]['position']
 par2D[d,:,:] = temp2D[:,0:tt]

# horizontal
nl = 40

for z in range(nl):
 print z
 A_3D = np.zeros((tt,nl))
 A_2D = np.zeros((tt,nl))
 D_3D = np.zeros((tt,nl))
 D_2D = np.zeros((tt,nl))
 Cd_3D = np.zeros((tt,nl))
 Cd_2D = np.zeros((tt,nl))

 # Extract particles at depth z

 par2Dz = np.reshape(par2D,(20,20,50,3,tt))
 par3Dz = np.reshape(par3D,(20,20,50,3,tt))

 par3Dzr = par3Dz[:,:,z,:,:]
 par2Dzr = par2Dz[:,:,z,:,:]

 par3Dz = np.reshape(par3Dzr,(400,3,tt))
 par2Dz = np.reshape(par2Dzr,(400,3,tt))
 
 #
 #
 # Absolute Dispersion in Time
 #
 P0 = np.mean([500,1500]),np.mean([1000,2000])
 #
 A_3D[:,z] = np.sqrt(np.mean((par3Dz[:,0,:] - P0[0])**2 + (par3Dz[:,0,:] - P0[0])**2 + (par3Dz[:,0,:] - P0[0])**2,0))
 A_2D[:,z] = np.sqrt(np.mean((par2Dz[:,0,:] - P0[0])**2 + (par2Dz[:,0,:] - P0[0])**2 + (par2Dz[:,0,:] - P0[0])**2,0))
 #
 # Relative disperions
 #
 for i in range(19):
   D_3D[:,z] = np.mean((par3Dzr[i,:,0,:] - par3Dzr[i+1,:,0,:])**2 + (par3Dzr[i,:,1,:] - par3Dzr[i+1,:,1,:])**2,0) 
   D_2D[:,z] = np.mean((par2Dzr[i,:,0,:] - par2Dzr[i+1,:,0,:])**2 + (par2Dzr[i,:,1,:] - par2Dzr[i+1,:,1,:])**2,0) 
 #
 # Cloud dispersion
 #
 #
 Pt3D = np.zeros((2,tt))
 Pt2D = np.zeros((2,tt))
 #
 Pt3D[:] = np.mean(par3Dz[:,0,:],0),np.mean(par3Dz[:,1,:],0)
 Pt2D[:] = np.mean(par3Dz[:,0,:],0),np.mean(par3Dz[:,1,:],0)
 # 
 Cd_3D[:,z] = np.sqrt(np.mean((par3Dz[:,0,:] - Pt3D[0,:])**2 + (par3Dz[:,1,:] - Pt3D[1,:])**2,0))
 Cd_2D[:,z] = np.sqrt(np.mean((par2Dz[:,0,:] - Pt2D[0,:])**2 + (par2Dz[:,1,:] - Pt2D[1,:])**2,0))

 
 # plotting

 # abosolute D

 p3D, = plt.semilogy(time/86400,A_3D[:,z],'k',linewidth=2)
 p2D, = plt.semilogy(time/86400,A_2D[:,z],'b',linewidth=2)
 
 plt.xlabel('time')
 plt.ylabel('Absolute Dispersion')
# plt.ylim((500,2000))
 plt.legend((p3D,p2D),('3D','2D'))
 #
 print 'Saving AD'
 # 
 plt.savefig('./plot/m_50_5_23D/AD_m_50_5_23D_z'+str(z)+'.eps')
 plt.close()

 # relative D

 p3D, = plt.semilogy(time/86400,D_3D[:,z],'k',linewidth=2)
 p2D, = plt.semilogy(time/86400,D_2D[:,z],'b',linewidth=2)

 plt.xlabel('time')
 plt.ylabel('Relative Dispersion')
 plt.legend((p3D,p2D),('3D','2D'))
 #plt.ylim((500,2000))
 #
 print 'Saving RD'
 # 
 plt.savefig('./plot/m_50_5_23D/RD_m_50_5_23D_z'+str(z)+'.eps')
 plt.close()

 # cloud D

 p3D, = plt.semilogy(time/86400,Cd_3D[:,z],'k',linewidth=2)
 p2D, = plt.semilogy(time/86400,Cd_2D[:,z],'b',linewidth=2)

 plt.xlabel('time')
 plt.ylabel('Cloud Dispersion')
 plt.legend((p3D,p2D),('3D','2D'))
 #plt.ylim((500,2000))
 #
 print 'Saving Cd'
 # 
 plt.savefig('./plot/m_50_5_23D/CD_m_50_5_23D_z'+str(z)+'.eps')
 plt.close()


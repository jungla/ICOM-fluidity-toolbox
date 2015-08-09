#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import myfun
import numpy as np

exp = 'm_250_8c_cp'
filename = '/nethome/jmensa/fluidity-exp/'+exp+'/mli_checkpoint_checkpoint.detectors'

det = fluidity_tools.stat_parser(filename)
keys = det.keys()				 # particles

tt = 7
pt = 450000
step = 1

nlon = 75
nlat = 150
ndepth = 10
 
par = np.zeros((pt,3,tt))

time = det['ElapsedTime']['value']

# read particles

for d in range(pt):
 temp = det['Particles_'+myfun.digit(d+1,6)]['position']
 par[d,:,:] = temp[:,0:tt]

#fsle param
di = 10 # base separation distance [m]. Taken as the distance between the particles in the triplet.

for r in range(5,25,5):
 fsle  = np.zeros(pt/4)
 fslec = np.zeros((pt/4,3))
 df=r*di # separation distance
 # 
 # loop triplets in time
 #
 for t in range(tt):
  for d in range(0,pt,4):
 # loop particles
   for p in [0,1]:
    dr = np.linalg.norm(par[d+3,:,0]-par[d+p,:,t])
    if (dr > df and fsle[d/4] == 0):
     fsle[d/4]  = np.log(r)/time[t] 	# fsle has the dimension of the first triplet
     fslec[d/4,:] = par[d,:,0] 	# coords of the starting point
 #
 # plot fsle
 # 3D arrays of fsle and fslec
 #
 fsler = np.reshape(fsle,(nlat,nlon,ndepth))
# fslexr = np.reshape(fslec[:,0],(nlat,nlon))
# fsleyr = np.reshape(fslec[:,1],(nlat,nlon))
# fslezr = np.reshape(fslec[:,2],(nlat,nlon))
 #
 plt.figure()
 plt.gca().set_aspect('equal')
 plt.contourf(fsler[:,:,0])
 plt.colorbar()
 plt.savefig('./fsle_'+exp+'_'+str(r)+'.eps',bbox_inches='tight')
 plt.close()

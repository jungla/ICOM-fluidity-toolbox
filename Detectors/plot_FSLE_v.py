#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import myfun
import numpy as np
import pyvtk
import vtktools
import copy
import os 

exp = 'r_3k_B_1F0_r'
filename = './ring_checkpoint.detectors'
filename2 = '/tamay2/mensa/fluidity/'+exp+'/ring_30.pvtu'


data = vtktools.vtu(filename2)
coords = data.GetLocations()
depths = sorted(list(set(coords[:,2])))



Xlist = np.arange(-100000,100000+10000,10000)# x co-ordinates of the desired array shape
Ylist = np.arange(0,1)*0.0
Zlist = np.arange(-10,-900,-10)# y co-ordinates of the desired array shape
[X,Y,Z] = myfun.meshgrid2(Xlist,Ylist,Zlist)
Y = np.reshape(Y,(np.size(Y),))
X = np.reshape(X,(np.size(X),))
Z = np.reshape(Z,(np.size(Z),))

pts = zip(X,Y,Z)
pts = vtktools.arr(pts)

R = data.ProbeData(pts, 'Density_CG')
rho = np.reshape(R,[len(Zlist),len(Ylist),len(Xlist)])

try: os.stat('./plot/'+exp)
except OSError: os.mkdir('./plot/'+exp)

print 'reading detectors'
det = fluidity_tools.stat_parser(filename)
keys = det.keys()				 # particles
print 'done.' 

tt = 1200
pt = 5896
step = 1

z = range(-10,-890,-10)
x = range(-100000,100000,3000)
y = 0.0

par = np.zeros((pt,3,tt))

time = range(1800,1800*(tt+1),1800)

# read particles

for d in range(pt):
 temp = det['particles_'+myfun.digit(d+1,4)]['position']
 par[d,:,:] = temp[:,0:tt]

#fsle param
di = 10 # base separation distance [m]. Taken as the distance between the particles in the triplet.

# read T from archive

for r in np.linspace(1,3):
 #print 'plotting for dr:',r*di
fsle  = np.zeros(pt)*np.nan
df = 11.0 #r*di # separation distance
 # 
 # loop triplets in time
 #
 #
for t in range(tt):
 for d in range(0,pt-len(x)):
 # loop particles
  if par[d,2,t] < 0.0 and par[d+len(x),2,t] < 0.0:
   dr = np.linalg.norm(par[d,2,t]-par[d+len(x),2,t])
#   if dr > 15.0: print dr,d,t
   if (dr > df and np.isnan(fsle[d])):
    fsle[d] = np.log(dr/di)/time[t] 

min_fsle = np.percentile(fsle,0.1)
max_fsle = 0.0000005 #np.percentile(fsle,99)
fsler = np.reshape(fsle,(len(z),len(x)))
 #
plt.figure()
v = np.linspace(1e-7,1e-6, 25, endpoint=True)
plt.contourf(x,z,fsler,v,extend='both',cmap='jet')
plt.colorbar(format='%.3e')
plt.contour(Xlist,Zlist,np.squeeze(rho),20,colors=[0.5,0.5,0.5])

plt.savefig('./plot/'+exp+'/fsle_'+exp+'_'+str(df)+'.eps',bbox_inches='tight')
plt.close()

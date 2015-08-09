#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import numpy as np
import vtktools
import myfun
import os
from scipy import interpolate
import gc

gc.enable()

exp = 'r_5k_B_d10_particles'
filename = '/tamay2/mensa/fluidity/'+exp+'/particles.detectors'
filename2 = '/tamay2/mensa/fluidity/'+exp+'/ring_10.pvtu'

data = vtktools.vtu(filename2)
coords = data.GetLocations()
depths = sorted(list(set(coords[:,2])))

Xlist = np.linspace(-180000,180000,50)# x co-ordinates of the desired array shape
Ylist = np.arange(0,1)*0.0
Zlist = np.linspace(0,-900,20)# y co-ordinates of the desired array shape
[X,Y,Z] = myfun.meshgrid2(Xlist,Ylist,Zlist)
Y = np.reshape(Y,(np.size(Y),))
X = np.reshape(X,(np.size(X),))
Z = np.reshape(Z,(np.size(Z),))

pts = vtktools.arr(zip(X,Y,Z))

R = data.ProbeData(pts, 'Density_CG')
rho = np.reshape(R,[len(Zlist),len(Ylist),len(Xlist)])

try: os.stat('./plot/'+exp)
except OSError: os.mkdir('./plot/'+exp)

print 'reading detectors'
det = fluidity_tools.stat_parser(filename)
keys = det.keys()				 # particles
print 'done.' 

pt = int(os.popen('grep position '+filename+'| wc -l').read()) # read the number of particles grepping all the positions in the file

tt = 45000

print 'particles:',pt
print 'timesteps:',tt

#z = range(-10,-890,-10)
#x = range(-100000,100000,3000)
#y = 0.0

par = np.zeros((pt,3,tt))

time = range(3600,3600*(tt+1),3600)

# read particles
print 'reading particles'

for d in range(pt):
 temp = det['particles_'+myfun.digit(d+1,3)]['position']
 par[d,:,:] = temp[:,0:tt]

point = []
for t in xrange(2,tt-2):
 for d in xrange(pt):
  if par[d,1,t]*par[d,1,t-1] < 0.0:
#   print par[d,0,t],par[d,1,t],par[d,2,t],par[d,0,t-1],par[d,1,t-1],par[d,2,t-1],
   f0 = interpolate.griddata(par[d,1,t-2:t+2],par[d,0,t-2:t+2],0.0,method='cubic')
   f2 = interpolate.griddata(par[d,1,t-2:t+2],par[d,2,t-2:t+2],0.0,method='cubic')
   point.append([float(f0),float(f2)])

apoint = np.asarray(point)
plt.figure()
plt.contour(Xlist,Zlist,np.squeeze(rho),20,colors=[0.5,0.5,0.5])
plt.scatter(apoint[:,0],apoint[:,1],marker='.', s=5, facecolor='0', lw = 0)
plt.ylim([-1000,0])
plt.xlim([-150000,150000])
#plt.scatter(par[:,0,999],par[:,2,999])

plt.savefig('./plot/'+exp+'/Poincare_'+exp+'.eps',bbox_inches='tight')
print 'saving','./plot/'+exp+'/Poincare_'+exp+'.eps'
plt.close()

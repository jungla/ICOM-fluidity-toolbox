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

exp = 'm_250_8c_h1'
filename = '/nethome/jmensa/fluidity-exp/'+exp+'/mli_checkpoint.detectors'
day = '40'

try: os.stat('./plot/'+exp)
except OSError: os.mkdir('./plot/'+exp)

print 'reading detectors'
det = fluidity_tools.stat_parser(filename)
keys = det.keys()				 # particles
print 'done.' 

tt = 80
pt = 450000
step = 1

# dimensions particles

lat = 15000
lon = 7500
depth = -50

nlat = 100
nlon = 100
ndepth = 5

depth_ml = -30

y = range(100, lat+100, nlat)
x = range(100, lon+100, nlon)
z = np.linspace(0, depth, ndepth)
[Xf,Yf,Zf] = myfun.meshgrid2(x,y,z)
Yf = np.reshape(Yf,np.size(Yf,))
Xf = np.reshape(Xf,np.size(Xf,))
Zf = np.reshape(Zf,np.size(Zf,))

# dimensions archives

xstep = 150
ystep = 150
zstep = -1

Xlist = np.arange(0.0,lon+xstep,xstep)# x co-ordinates of the desired array shape
Ylist = np.arange(0.0,lat+ystep,ystep)# y co-ordinates of the desired array shape
Zlist = np.arange(0.0,depth_ml+zstep,zstep)# y co-ordinates of the desired array shape
[X,Y,Z] = myfun.meshgrid2(Xlist,Ylist,Zlist)
Y = np.reshape(Y,(np.size(Y),))
X = np.reshape(X,(np.size(X),))
Z = np.reshape(Z,(np.size(Z),))

par = np.zeros((pt,3,tt))

time = det['ElapsedTime']['value']

# read particles

for d in range(pt):
 temp = det['Particles_'+myfun.digit(d+1,6)]['position']
 par[d,:,:] = temp[:,0:tt]

#fsle param
di = 0.5 # base separation distance [m]. Taken as the distance between the particles in the triplet.

# read ML depth from file

Tref = [0.2,0.3]
mld = []

for i in range(len(Tref)):
 Data = pyvtk.VtkData('/nethome/jmensa/scripts_fluidity/2D/ML/output/'+exp+'/ML_'+myfun.digit(Tref[i],3)+'_'+exp+'_'+day+'.vtk')
 mld.append(np.reshape(Data.point_data.data[0].scalars,[len(Ylist),len(Xlist)]))

# read T from archive

data = vtktools.vtu('/tamay/mensa/fluidity/'+exp+'/mli_'+day+'.pvtu')
Ts = np.reshape(data.ProbeData(vtktools.arr(zip(X,Y,Z)), 'Temperature'),[len(Zlist),len(Ylist),len(Xlist)])

# for future plotting
Yf, Zf = np.meshgrid(Ylist,Zlist)
Y, Z = np.meshgrid(y,z)

for r in np.linspace(2,5,3):
 #print 'plotting for dr:',r*di
 fsle  = np.zeros(pt/4)
 fslec = np.zeros((pt/4,3))
 df = r*di # separation distance
 # 
 # loop triplets in time
 #
 for t in range(tt):
  for d in range(0,pt,4):
  # loop particles
   dr = np.linalg.norm(par[d+3,2,t]-par[d+2,2,t])
   if (dr > df and fsle[d/4] == 0):
    fsle[d/4] = np.log(r)/time[t] 	# fsle has the dimension of the first triplet
   #  fslec[d/4,:] = par[d,:,0] 	# coords of the starting point
   
   # plot fsle
   # 3D arrays of fsle and fslec
   #
  # fslen = copy.deepcopy(fsle)
  # fslen[fslen>1.0e-40]=1.0e-40
 
 fsler = np.reshape(fsle,(len(y),len(x),10))
 fsler = fsler[:,:,0:5]
  
   # fslexr = np.reshape(fslec[:,0],(nlat,nlon))
   # fsleyr = np.reshape(fslec[:,1],(nlat,nlon))
   # fslezr = np.reshape(fslec[:,2],(nlat,nlon))
   
 plt.figure()
   # plt.gca().set_aspect('equal')
 plt.contourf(Y,Z,np.flipud(np.rot90(np.mean(fsler,axis=1))),10,cmap='jet')
 plt.colorbar()
 plt.contour(Yf,Zf,np.mean(Ts,axis=2),20,colors='White',linewidth=4.0)
   # plt.contourf(np.rot90(fsler[:,nlon/2,:]))
 for i in range(len(Tref)):
  mmld = np.mean(mld[i],axis=1)
  plt.plot(Ylist,mmld,color='k',linewidth=4.0)
  # plt.text(1000*(i+1),mmld[1]+1,myfun.digit(Tref[i],3))
 
 plt.savefig('./plot/'+exp+'/fsle_'+exp+'_'+day+'_'+str(df)+'.eps',bbox_inches='tight')
 plt.close()

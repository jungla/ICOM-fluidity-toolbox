import os, sys

import fio, myfun
import vtktools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy import interpolate
import gc

gc.enable()

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

label = 'r_3k_B_1F0'
basename = 'ring' 
dayi  = 50
dayf  = 51 
days  = 1

label = 'm_50_6'
basename = 'mli' 
dayi  = 0
dayf  = 24
days  = 2

label = sys.argv[1]
basename = sys.argv[2]
dayi  = int(sys.argv[3])
dayf  = int(sys.argv[4])
days  = int(sys.argv[5])

path = '/tamay2/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)


# dimensions archives

# RING

xn = 300
zn = 100

Xlist = np.linspace(-150000,150000,xn)# x co-ordinates of the desired array shape
Zlist = np.linspace(0,-900,zn)# x co-ordinates of the desired array shape

latitude = 0
delta = 3000

# MLI

#xn = 100
#zn = 100

#Xlist = np.linspace(0,5000,xn)# x co-ordinates of the desired array shape
#Zlist = np.linspace(0,-50,zn)# x co-ordinates of the desired array shape

#latitude = 2500
#delta = 200

[X,Z] = np.meshgrid(Xlist,Zlist)
X = np.reshape(X,(np.size(X),))
Z = np.reshape(Z,(np.size(Z),))

for time in range(dayi,dayf,days):
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file0 = basename + '_' + str(time) + '.pvtu'
 filepath = path+file0
 file1 = label+'_' + tlabel
 fileout  = path + file1
 #
 print 'opening: ', filepath
 #
 #
 data = vtktools.vtu(filepath)
 print 'extract V, R'
 data.Crop(np.min(Xlist),np.max(Xlist),latitude-delta,latitude+delta,np.min(Zlist),np.max(Zlist))
 coords = data.GetLocations()
 V = data.GetVectorField('Velocity_CG')
 Rho = data.GetScalarField('Density_CG')
 del data
 #
 W = V[np.around(coords[:,1])==latitude,2]
 R = Rho[np.around(coords[:,1])==latitude]
 del V
 del Rho
 Cw = coords[np.around(coords[:,1])==latitude,:]
 Wr = interpolate.griddata((Cw[:,0],Cw[:,2]),W,(X,Z),method='linear')
 Wr = np.reshape(Wr,[len(Zlist),len(Xlist)])
 Rr = interpolate.griddata((Cw[:,0],Cw[:,2]),R,(X,Z),method='linear')
 Rr = np.reshape(Rr,[len(Zlist),len(Xlist)])
 gc.collect()
 #
# mld = np.zeros([len(Xlist),len(Ylist)])
 #
# for x in range(len(Xlist)):
#  for y in range(len(Ylist)):
#   ml = rho[:,x,y]
#   mls = np.cumsum(ml)/range(1,len(ml)+1)
#   mlst, = np.where(mls>=ml)
#   mld[x,y] = ((Zlist[mlst[len(mlst)-1]]))

#plt.plot(np.mean(mld,1))
#plt.savefig('./plot/'+label+'/'+file1+'_MLD.eps',bbox_inches='tight')
#plt.close()

 # Density

 fig = plt.figure(figsize=(8, 10))
 
 # for d in depths:
 #  plt.axhline(y=d, xmin=-180000, xmax=180000,color='k',linestyle='--')
 
 v = np.linspace(-5e-5, 5e-5, 50, endpoint=True)
 vl = np.linspace(-5e-5, 5e-5, 5, endpoint=True)
 
 plt.contourf(Xlist/1000,Zlist,Wr,v,extend='both',cmap=plt.cm.PiYG)
 plt.colorbar(format='%.3e',ticks=vl)
 plt.contour(Xlist/1000,Zlist,Rr,10,colors='k')
# plt.colorbar()
 # plt.plot(Zlist)
 
 # plt.plot(Xlist,np.mean(mld,1),'r-')
 plt.xlabel('X (Km)')
 plt.ylabel('Z (m)')
  # plt.xticks(range(lati,lonf,1000),(range(0,15,1)))
  # plt.yticks(range(depthi,depthf,10),(range(0,15,1)))
 plt.title('W')
 
 plt.savefig('./plot/'+label+'/W_'+file1+'_v_snap.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/W_'+file1+'_v_snap.eps\n'

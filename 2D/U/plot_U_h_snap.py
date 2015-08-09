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
import myfun

gc.enable()

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

label = 'r_3k_B_1F0'
basename = 'ring' 
dayi  = 50
dayf  = 51 
days  = 1

label = 'm_50_7'
basename = 'mli' 
dayi  = 24
dayf  = 25 
days  = 1

#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

path = '/tamay2/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)


# dimensions archives

# Ring exp

xn = 300
zn = 100

Xlist = np.linspace(-150000,150000,xn)# x co-ordinates of the desired array shape
Zlist = np.linspace(0,-900,zn)# x co-ordinates of the desired array shape

latitude = 0

delta = 3000

# ML exp

xn = 100
zn = 1
yn = 100

Xlist = np.linspace(0,5000,xn)# x co-ordinates of the desired array shape
Zlist = np.linspace(-5,-5,zn)# x co-ordinates of the desired array shape
Ylist = np.linspace(0,5000,yn)# y co-ordinates of the desired array shape


[X,Y,Z] = myfun.meshgrid2(Xlist,Ylist,Zlist)
Y = np.reshape(Y,(np.size(Y),))
X = np.reshape(X,(np.size(X),))
Z = np.reshape(Z,(np.size(Z),))

depth = -5

delta = 3

pts = zip(X,Y,Z)
pts = vtktools.arr(pts)


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
 data.Crop(np.min(Xlist),np.max(Xlist),np.min(Ylist),np.max(Ylist),depth-delta,depth+delta)
 Vel = data.ProbeData(pts, 'Velocity_CG')
 Rho = data.ProbeData(pts, 'Density_CG')
 #
 U = np.squeeze(np.reshape(Vel[:,0],[len(Zlist),len(Xlist),len(Ylist)]))
 V = np.squeeze(np.reshape(Vel[:,1],[len(Zlist),len(Xlist),len(Ylist)]))
 W = np.squeeze(np.reshape(Vel[:,2],[len(Zlist),len(Xlist),len(Ylist)]))
 R = np.squeeze(np.reshape(Rho,[len(Zlist),len(Xlist),len(Ylist)]))
 S = np.sqrt(U**2+V**2)
 del data
 del Vel
 del Rho

 # plotting U

 fig = plt.figure(figsize=(8, 8))
 
 plt.contourf(Xlist/1000,Ylist/1000,S,50,extend='both',cmap=plt.cm.PiYG)
 plt.colorbar(format='%.3e')
 plt.contour(Xlist/1000,Ylist/1000,R,10,colors='k')
 plt.xlabel('X (km)')
 plt.ylabel('Y (km)')
 plt.title('U')
 
 plt.savefig('./plot/'+label+'/U_'+file1+'_h_snap.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/U_'+file1+'_h_snap.eps\n'

 # plotting W

 fig = plt.figure(figsize=(8, 8))
# v = np.linspace(-1e-4,1e-4,50)
# vl = np.linspace(-1e-4,1e-4,5)
 plt.contourf(Xlist/1000,Ylist/1000,W,50,extend='both',cmap=plt.cm.PiYG)
 plt.colorbar(format='%.3e')
# plt.colorbar(ticks=vl,format='%.3e')
 plt.contour(Xlist/1000,Ylist/1000,R,10,colors='k')
 plt.xlabel('X (km)')
 plt.ylabel('Y (km)')
 plt.title('W')

 plt.savefig('./plot/'+label+'/W_'+file1+'_h_snap.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/W_'+file1+'_h_snap.eps\n'


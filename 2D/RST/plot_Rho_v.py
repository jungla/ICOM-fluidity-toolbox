import os, sys

import fio, myfun
import vtktools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import gc

gc.enable()

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

label = 'r_3k_B_1F0'
basename = 'ring' 
dayi  = 50
dayf  = 51 
days  = 1

label = sys.argv[1]
basename = sys.argv[2]
dayi  = int(sys.argv[3])
dayf  = int(sys.argv[4])
days  = int(sys.argv[5])

path = '/tamay2/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

file0 = basename+'_' + str(0) + '.pvtu'
filepath = path+file0
print 'reading',filepath
#
data = vtktools.vtu(filepath)
coords = data.GetLocations()
del data
gc.collect()

depths = sorted(list(set(coords[:,2])))

# dimensions archives

xn = 200
yn = 1
zn = 90

Xlist = np.linspace(np.min(coords[:,0]),np.max(coords[:,0]),xn)# x co-ordinates of the desired array shape
Ylist = [0.0] #np.linspace(np.min(coords[:,1]),np.max(coords[:,1]),yn)# y co-ordinates of the desired array shape
Zlist = depths# y co-ordinates of the desired array shape
Zlist = np.linspace(0,-900,zn)# y co-ordinates of the desired array shape
[X,Y,Z] = myfun.meshgrid2(Xlist,Ylist,Zlist)
Y = np.reshape(Y,(np.size(Y),))
X = np.reshape(X,(np.size(X),))
Z = np.reshape(Z,(np.size(Z),))

pts = vtktools.arr(zip(X,Y,Z))


for time in range(dayi,dayf,days):
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file0 = basename + '_' + str(time) + '.pvtu'
 filepath = path+file0
 file1 = 'Rho_'+label+'_' + tlabel
 fileout  = path + file1
 #
 print 'opening: ', filepath
 #
 #
 data = vtktools.vtu(filepath)
 print 'fields: ', data.GetFieldNames()
 print 'extract V, R'
# V = data.ProbeData(pts, 'Velocity_CG')
 R = data.ProbeData(pts, 'Density_CG')
 print 'done.'
# w = np.reshape(V[:,2],[len(Zlist),len(Xlist),len(Ylist)])
 rho = np.reshape(R,[len(Zlist),len(Xlist),len(Ylist)])
 #del data
 #
 print 'max: ', (rho).max(), 'min: ', (rho).min()
# coords = data.GetLocations()
# z = coords[np.around(coords[:,1])==0.0,2] 
# x = coords[np.around(coords[:,1])==0.0,0] 
# plt.triplot(z,x)
# plt.savefig('./plot/'+label+'/'+file1+'_gridV.eps',bbox_inches='tight')
# plt.close()
# x = coords[np.around(coords[:,2])==0.0,0] 
# y = coords[np.around(coords[:,2])==0.0,1] 
# plt.triplot(x,y)
# plt.savefig('./plot/'+label+'/'+file1+'_gridH.eps',bbox_inches='tight')
# plt.close()

# print 'max: ', (w).max(), 'min: ', (w).min()
 #
 #
 # create arrays of velocity and temperature values at the desired points
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

 for d in depths:
  plt.axhline(y=d, xmin=-180000, xmax=180000,color='k',linestyle='--')

 #v = np.linspace(1018, 1031, 20, endpoint=True)
 #vl = np.linspace(1018, 1031, 5, endpoint=True)
 plt.contourf(Xlist,Zlist,np.mean(rho,2),50,extend='both',cmap=plt.cm.Greys)
# plt.colorbar(ticks=vl,orientation='horizontal')
 plt.colorbar()
# plt.plot(Zlist)

# plt.plot(Xlist,np.mean(mld,1),'r-')
 plt.xlabel('X (m)')
 plt.ylabel('Z (m)')
 # plt.xticks(range(lati,lonf,1000),(range(0,15,1)))
 # plt.yticks(range(depthi,depthf,10),(range(0,15,1)))
 plt.title('density, t=0')

 plt.savefig('./plot/'+label+'/'+file1+'_r.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/'+file1+'_r.eps\n'

import os, sys

import fio, myfun
import vtktools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

label = sys.argv[1]
dayi  = int(sys.argv[2])
dayf  = int(sys.argv[3])
days  = int(sys.argv[4])

path = '/tamay/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

Tmin = 17.00
Tmax = 21.00

lati = 0
loni = 0
depthi = 0

latf = 10000
lonf = 10000
depthf = -75

# dimensions archives

xstep = 500
ystep = 500
zstep = -2

Xlist = np.arange(loni,lonf+xstep,xstep)# x co-ordinates of the desired array shape
Ylist = np.arange(lati,latf+ystep,ystep)# y co-ordinates of the desired array shape
Zlist = np.arange(depthi,depthf+zstep,zstep)# y co-ordinates of the desired array shape
[X,Y,Z] = myfun.meshgrid2(Xlist,Ylist,Zlist)
Y = np.reshape(Y,(np.size(Y),))
X = np.reshape(X,(np.size(X),))
Z = np.reshape(Z,(np.size(Z),))

pts = zip(X,Y,Z)
pts = vtktools.arr(pts)

fd = open('./plot/'+label+'/'+label+'_rhow_z.csv','a')

for time in range(dayi,dayf,days):
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file0 = 'mli_' + str(time) + '.pvtu'
 filepath = path+file0
 file1 = 'rhow_'+label+'_' + tlabel
 fileout  = path + file1
 #
 print 'opening: ', filepath
 #
 #
 data = vtktools.vtu(filepath)
 print 'fields: ', data.GetFieldNames()
 print 'extract V, R'
 V = data.ProbeData(pts, 'Velocity_CG')
 R = data.ProbeData(pts, 'Density_CG')
 print 'done.'
 w = np.reshape(V[:,2],[len(Zlist),len(Xlist),len(Ylist)])
 rho = np.reshape(R,[len(Zlist),len(Xlist),len(Ylist)])
 #del data
 #
 print 'max: ', (w*rho).max(), 'min: ', (w*rho).min()
 #
 #
 mld = np.zeros([len(Xlist),len(Ylist)])
 #
 for x in range(len(Xlist)):
  for y in range(len(Ylist)):
   ml = rho[:,x,y]
   mls = np.cumsum(ml)/range(1,len(ml)+1)
   mlst, = np.where(mls>=ml)
   mld[x,y] = ((Zlist[mlst[len(mlst)-1]]))

 # RHO*W
 fig = plt.figure(figsize=(2,5))
 ax = fig.add_subplot(111)
 rhow_z = np.mean(np.mean(rho*w,axis=1),axis=1)
 plt.plot([0, 0], [min(Zlist), max(Zlist)], color='k', linestyle='--', linewidth=1)
 plt.plot(rhow_z,Zlist,'k')
 plt.plot([-0.15, 0.15],[np.mean(np.mean(mld,1)), np.mean(np.mean(mld,1))],'-',color='0.7')
 plt.xlabel('$<w*rho>$')
 plt.ylabel('Z (m)')
 plt.xlim([-0.15, 0.15])
 plt.xticks([-0.15, 0.15])
 plt.savefig('./plot/'+label+'/'+file1+'_z.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/'+file1+'_z.eps\n'

 # save to csv
 fd.write(str(time))
 for elem in rhow_z:
  fd.write(', '+ str(elem))
 #
 fd.write('\n')

 #dRHO*W/dZ 
 fig = plt.figure(figsize=(2,5))
 ax = fig.add_subplot(111)
 rhow_z = np.diff(np.mean(np.mean(rho*w,axis=1),axis=1))/np.diff(Zlist)
 plt.plot([0, 0], [min(Zlist[1:,]), max(Zlist[1:,])], color='k', linestyle='--', linewidth=1)
 plt.plot(rhow_z,Zlist[1:,],'k')
 plt.plot([-0.03, 0.03],[np.mean(np.mean(mld,1)), np.mean(np.mean(mld,1))],'-',color='0.7')
 plt.xlabel('$<d(w*rho)/dz>$')
 plt.ylabel('Z (m)')
 plt.xlim([-0.03, 0.03])
 plt.xticks([-0.03, 0.03])
 plt.savefig('./plot/'+label+'/'+file1+'_z_dz.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/'+file1+'_z_dz.eps\n'

 os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'_z.jpg ./plot/'+label+'/'+file1+'_z.eps')
 os.system('mogrify ./plot/'+label+'/'+file1+'_z.jpg -trim ./plot/'+label+'/'+file1+'_z.jpg')
 os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'_z_dz.jpg ./plot/'+label+'/'+file1+'_z_dz.eps')
 os.system('mogrify ./plot/'+label+'/'+file1+'_z_dz.jpg -trim ./plot/'+label+'/'+file1+'_z_dz.jpg')


fd.close()

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

lati = 0
loni = 0
depthi = 0

latf = 10000
lonf = 10000
depthf = -200

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

fd = open('./plot/'+label+'/'+label+'_temp_z.csv','a')

for time in range(dayi,dayf,days):
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file0 = 'mli_' + str(time) + '.pvtu'
 filepath = path+file0
 file1 = 'temp_'+label+'_' + tlabel
 fileout  = path + file1
 #
 print 'opening: ', filepath
 #
 #
 data = vtktools.vtu(filepath)
 print 'fields: ', data.GetFieldNames()
 print 'extract V, R'
 R = data.ProbeData(pts, 'Temperature_CG')
 print 'done.'
 temp = np.reshape(R,[len(Zlist),len(Xlist),len(Ylist)])
 #del data
 #
 print 'max: ', (temp).max(), 'min: ', (temp).min()
 #
 #
 # create arrays of velocity and temperature values at the desired points

 # RHO*W
 fig = plt.figure(figsize=(3,6))
 ax = fig.add_subplot(111)
 temp_z = np.mean(np.mean(temp,axis=1),axis=1)
 plt.plot([0, 0], [min(Zlist), max(Zlist)], color='k', linestyle='--', linewidth=1)
 plt.plot(temp_z,Zlist,'k')
 plt.xlabel('<temp>')
 plt.ylabel('Z (m)')
 plt.xlim([np.min(temp_z), np.max(temp_z)])
 plt.xticks([np.min(temp_z), np.max(temp_z)])
 plt.savefig('./plot/'+label+'/'+file1+'_z.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/'+file1+'_z.eps\n'

 # save to csv
 fd.write(str(time))
 for elem in temp_z:
  fd.write(', '+ str(elem))
 #
 fd.write('\n')

 #dRHO/dZ 
 fig = plt.figure(figsize=(3,6))
 ax = fig.add_subplot(111)
 temp_z = np.diff(np.mean(np.mean(temp,axis=1),axis=1))/np.diff(Zlist)
 plt.plot([0, 0], [min(Zlist[1:,]), max(Zlist[1:,])], color='k', linestyle='--', linewidth=1)
 plt.plot(temp_z,Zlist[1:,],'k')
 plt.xlabel('<d(temp)/dz>')
 plt.ylabel('Z (m)')
 plt.xlim([np.min(temp_z), np.max(temp_z)])
 plt.xticks([np.min(temp_z), np.max(temp_z)])
 plt.savefig('./plot/'+label+'/'+file1+'_z_dz.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/'+file1+'_z_dz.eps\n'

 os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'_z.jpg ./plot/'+label+'/'+file1+'_z.eps')
 os.system('mogrify ./plot/'+label+'/'+file1+'_z.jpg -trim ./plot/'+label+'/'+file1+'_z.jpg')
 os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'_z_dz.jpg ./plot/'+label+'/'+file1+'_z_dz.eps')
 os.system('mogrify ./plot/'+label+'/'+file1+'_z_dz.jpg -trim ./plot/'+label+'/'+file1+'_z_dz.jpg')


fd.close()

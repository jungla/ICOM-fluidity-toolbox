import os, sys

import fio, myfun
import vtktools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import matplotlib.tri as tri

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days
label = sys.argv[1]
dayi  = int(sys.argv[2])
dayf  = int(sys.argv[3])
days  = int(sys.argv[4])

path = '/tamay/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

xstep = 100
ystep = 100
zstep = -2

lati = 0
loni = 7500
depthi = 0

latf = 15000
lonf = 7500
depthf = -100

# dimensions archives

xstep = 150
ystep = 150
zstep = -1

Xlist = np.arange(loni,lonf+xstep,xstep)# x co-ordinates of the desired array shape
Ylist = np.arange(lati,latf+ystep,ystep)# y co-ordinates of the desired array shape
Zlist = np.arange(depthi,depthf+zstep,zstep)# y co-ordinates of the desired array shape
[X,Y,Z] = myfun.meshgrid2(Xlist,Ylist,Zlist)
Y = np.reshape(Y,(np.size(Y),))
X = np.reshape(X,(np.size(X),))
Z = np.reshape(Z,(np.size(Z),))

pts = zip(X,Y,Z)
pts = vtktools.arr(pts)

for time in range(dayi,dayf,days):
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file0 = 'mli_' + str(time) + '.pvtu'
 filepath = path+file0
 file1 = 'wT_'+label+'_' + tlabel
 fileout  = path + file1
 #
 print 'opening: ', filepath
 #
 data = vtktools.vtu(filepath)
 #
 print 'fields: ', data.GetFieldNames()
 print 'extract T'
 T = data.ProbeData(pts, 'Temperature')
 V = data.ProbeData(pts, 'Velocity')
 
 wT = np.reshape(T,[len(Ylist),len(Zlist)])*np.reshape(V[:,2],[len(Ylist),len(Zlist)])
 
 print 'done.'
 #Tmax = np.max(T)
 #Tmin = np.min(T)
 #
 del data
 #
 # create arrays of velocity and temperature values at the desired points
 #
 plt.figure()
 #plt.gca().set_aspect('equal')
 #plt.autoscale(enable=True, axis='both', tight=True)
 # colorNorm = mpl.colors.Normalize(vmin=10, vmax=20,clip=True)
 plt.contourf(Ylist,Zlist,wT,150)
 plt.contourf(Ylist,Zlist,wT,150)
 plt.contourf(Ylist,Zlist,wT,150)

 plt.colorbar()
 #plt.autumn()
 plt.xlabel('X (Km)')
 plt.ylabel('Z (m)')
# plt.xticks(range(lati,lonf,1000),(range(0,15,1)))
# plt.yticks(range(depthi,depthf,10),(range(0,15,1)))
 plt.title('wT,  '+str(np.trunc(time*5/24)/10.0)+' day')
 plt.savefig('./plot/'+label+'/'+file1+'.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/'+file1+'.eps\n'

# os.system('gs -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'.png ./plot/'+label+'/'+file1+'.eps')
 os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'.jpg ./plot/'+label+'/'+file1+'.eps')
 os.system('mogrify ./plot/'+label+'/'+file1+'.jpg -trim ./plot/'+label+'/'+file1+'.jpg')
# os.system('mogrify ./plot/'+label+'/'+file1+'.png -trim ./plot/'+label+'/'+file1+'.png')

print 'Making movie animation'

fr  = '10'
br  = '4096k'
crf = '24'

opts = '-y -f image2 -r '+fr+' -i ./plot/'+label+'/'+label+'_%03d.jpg -vcodec'

#ffmpeg_ogg = 'ffmpeg '+opts+' libtheora -b:v '+br+' ./plot/'+label+'/'+label+'.ogg'
ffmpeg_mp4 = 'ffmpeg '+opts+' libx264 -threads 0 -crf '+crf+' -s 1250x1620 ./plot/'+label+'/'+label+'.avi'

#print ffmpeg_ogg
#os.system(ffmpeg_ogg)

print ffmpeg_mp4
os.system(ffmpeg_mp4)

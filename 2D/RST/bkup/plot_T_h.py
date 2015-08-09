import os, sys

import fio
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

Tmin = 19.00
Tmax = 20.00

xstep = 100
ystep = 100
zstep = -2

lat = 15000
lon = 7500
depth = -100

Xlist = np.arange(0.0,lon+xstep,xstep)# x co-ordinates of the desired array shape
Ylist = np.arange(0.0,lat+ystep,ystep)# y co-ordinates of the desired array shape
Zlist = np.arange(0.0,depth+zstep,zstep)# y co-ordinates of the desired array shape
[X,Y] = np.meshgrid(Xlist,Ylist)
Y = np.reshape(Y,(np.size(Y),))
X = np.reshape(X,(np.size(X),))
Z = Y*0.0

pts = zip(X,Y,Z)
pts = vtktools.arr(pts)

for time in range(dayi,dayf,days):
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file0 = 'mli_' + str(time) + '.pvtu'
 filepath = path+file0
 file1 = 'T_'+label+'_' + tlabel
 fileout  = path + file1
 #
 print 'opening: ', filepath
 #
 data = vtktools.vtu(filepath)
 #
 print 'fields: ', data.GetFieldNames()
 print 'extract T'
 T = data.ProbeData(pts, 'Temperature')
 print 'done.'
 #Tmax = np.max(T)
 #Tmin = np.min(T)
 print 'using: ', Tmin,Tmax, ' while max is', np.max(T), ' and min is ', min(T)
 #
 del data
 #
 # create arrays of velocity and temperature values at the desired points
 #
 plt.figure()
 plt.gca().set_aspect('equal')
 plt.autoscale(enable=True, axis='both', tight=True)
 # colorNorm = mpl.colors.Normalize(vmin=10, vmax=20,clip=True)
 #plt.scatter(x,y,c=T,norm=colorNorm)
 # v1 = np.linspace(Tmin, Tmax, 50)
 # v2 = np.linspace(Tmin, Tmax, 11)
 # plt.tricontourf(x,y,T,v1)
 # plt.colorbar(ticks=v2)
 plt.contourf(Xlist,Ylist,np.reshape(T,[len(Ylist),len(Xlist)]),50)
 plt.colorbar()
 #plt.autumn()
 plt.xlabel('X (Km)')
 plt.ylabel('Y (Km)')
 plt.xticks(range(0,lon,1000),(range(0,15,1)))
 plt.yticks(range(0,lat,1000),(range(0,15,1)))
 plt.title('T,  '+str(np.trunc(time*5/24)/10.0)+' day')
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

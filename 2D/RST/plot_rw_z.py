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

path = '/nethome/jmensa/fluidity-exp/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

Tmin = 17.00
Tmax = 21.00

lati = 0
loni = 5000
depthi = 0

latf = 10000
lonf = 5000
depthf = -200

# dimensions archives

xstep = 50
ystep = 50
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
 V = data.ProbeData(pts, 'Velocity')
 R = data.ProbeData(pts, 'Density')
 print 'done.'
 w = np.reshape(V[:,2],[len(Zlist),len(Ylist)])
 rho = np.reshape(R,[len(Zlist),len(Ylist)])
 #del data
 #
 print 'max: ', (w*rho).max(), 'min: ', (w*rho).min()
 
 #
 twrho = np.reshape(w*rho,-1)
 Ptwrho = np.reshape(w*rho,-1)
 Ntwrho = np.reshape(w*rho,-1)
 #
 Ptwrho[twrho<0]=np.nan
 Ntwrho[twrho>0]=np.nan
 Pwrho = np.log(np.reshape(Ptwrho,w.shape))
 Nwrho = np.log(abs(np.reshape(Ntwrho,w.shape)))
 # 
 #
 # create arrays of velocity and temperature values at the desired points
 #
 plt.figure()
 #plt.gca().set_aspect('equal')
 #plt.autoscale(enable=True, axis='both', tight=True)
 #colorNorm = mpl.colors.Normalize(vmin=Tmin, vmax=Tmax,clip=True)
 #v = np.linspace(-2, 2, 50, endpoint=True)
 plt.contourf(Ylist,Zlist,Pwrho,50)
 plt.colorbar()
 #plt.autumn()
 plt.xlabel('X (m)')
 plt.ylabel('Z (m)')
 # plt.xticks(range(lati,lonf,1000),(range(0,15,1)))
 # plt.yticks(range(depthi,depthf,10),(range(0,15,1)))
 plt.title('positive rho*w,  '+str(np.trunc(time*5/24)/10.0)+' day')
 plt.savefig('./plot/'+label+'/'+file1+'_p.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/'+file1+'_p.eps\n'
 #
 plt.figure()
 #plt.gca().set_aspect('equal')
 #plt.autoscale(enable=True, axis='both', tight=True)
 #colorNorm = mpl.colors.Normalize(vmin=Tmin, vmax=Tmax,clip=True)
 #v = np.linspace(-2, 2, 50, endpoint=True)
 plt.contourf(Ylist,Zlist,Nwrho,50)
 plt.colorbar()
 #plt.autumn()
 plt.xlabel('X (m)')
 plt.ylabel('Z (m)')
 # plt.xticks(range(lati,lonf,1000),(range(0,15,1)))
 # plt.yticks(range(depthi,depthf,10),(range(0,15,1)))
 plt.title('negative rho*w,  '+str(np.trunc(time*5/24)/10.0)+' day')
 plt.savefig('./plot/'+label+'/'+file1+'_n.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/'+file1+'_n.eps\n'

 plt.figure()
 #plt.gca().set_aspect('equal')
 #plt.autoscale(enable=True, axis='both', tight=True)
 #colorNorm = mpl.colors.Normalize(vmin=Tmin, vmax=Tmax,clip=True)
 #v = np.linspace(-2, 2, 50, endpoint=True)
 plt.contourf(Ylist,Zlist,w*rho,50)
 plt.colorbar()
 #plt.autumn()
 plt.xlabel('X (m)')
 plt.ylabel('Z (m)')
 # plt.xticks(range(lati,lonf,1000),(range(0,15,1)))
 # plt.yticks(range(depthi,depthf,10),(range(0,15,1)))
 plt.title('negative rho*w,  '+str(np.trunc(time*5/24)/10.0)+' day')
 plt.savefig('./plot/'+label+'/'+file1+'.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/'+file1+'.eps\n'

 
 # os.system('gs -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'.png ./plot/'+label+'/'+file1+'.eps')
 os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'_p.jpg ./plot/'+label+'/'+file1+'_p.eps')
 os.system('mogrify ./plot/'+label+'/'+file1+'_p.jpg -trim ./plot/'+label+'/'+file1+'_n.jpg')
 os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'_n.jpg ./plot/'+label+'/'+file1+'_n.eps')
 os.system('mogrify ./plot/'+label+'/'+file1+'_n.jpg -trim ./plot/'+label+'/'+file1+'_n.jpg')
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

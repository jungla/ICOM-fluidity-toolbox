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

Umin = 00.00
Umax = 01.00

for time in range(dayi,dayf,days):
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel

 file0 = 'mli_' + str(time) + '.pvtu'
 filepath = path+file0
 file1 = label+'_' + tlabel
 fileout  = path + file1

 print 'opening: ', filepath
 
 data = vtktools.vtu(filepath)
 
 print 'fields: ', data.GetFieldNames()
 print 'extract coords'
 coord = data.GetLocations()
 print 'extract U'
 U = data.GetVectorField('Velocity')
 print 'done.'

 del data

 id_U = np.where(coord[:,2] == 0) # values from surface
 
 coord = coord[id_U]

 c     = fio.fluidityclean_coord(coord,0)
 Us    = fio.fluidityclean_2D(coord,U[id_U,0].flatten(),0)
 Vs    = fio.fluidityclean_2D(coord,U[id_U,1].flatten(),0)

 S = np.sqrt(Us**2 + Vs**2)

 x = c[:,0]
 y = c[:,1]

 ## PRINT
 
 #triang = tri.Triangulation(x, y)
 
 plt.figure()
 plt.gca().set_aspect('equal')
 plt.autoscale(enable=True, axis='both', tight=True)
 if time == 0:
  plt.triplot(x,y,linewidth=0.5,color='gray')
 # colorNorm = mpl.colors.Normalize(vmin=10, vmax=20,clip=True)
 #plt.scatter(x,y,c=T,norm=colorNorm)
# v1 = np.linspace(Tmin, Tmax, 50)
# v2 = np.linspace(Tmin, Tmax, 11)
# plt.tricontourf(x,y,T,v1)
# plt.colorbar(ticks=v2)
 plt.tricontourf(x,y,S)
 plt.colorbar()
 #plt.autumn()
 q = plt.quiver(x,y,Us,Vs,units='width',scale=1/0.25)
 p = plt.quiverkey(q,25000,50000,1,"1 m/s",coordinates='data')
 plt.ylabel('Longitude')
 plt.xlabel('Latitude')
 plt.xticks(range(0,int(np.max(coord[:,0])),1000),(range(0,15,1)))
 plt.yticks(range(0,int(np.max(coord[:,1])),1000),(range(0,15,1)))
 plt.title('U,  '+str(np.trunc(time*5/24)/10.0)+' day')
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

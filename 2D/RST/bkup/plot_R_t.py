import os, sys

import vtktools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from paraview.simple import *

## READ archive (too many points... somehow)

label = sys.argv[1]
dayi  = int(sys.argv[2])
dayf  = int(sys.argv[3])
days  = int(sys.argv[4])

path = '/tamay/mensa/fluidity/'+label+'/'
reader = XMLPartitionedUnstructuredGridReader()
filter = CleantoGrid(reader)
writer  = XMLPUnstructuredGridWriter(Input=filter)

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

Tmax = 1
Tmin = 0

for time in range(dayi,dayf,days):
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel

 file0 = 'mli_' + str(time) + '.pvtu'
 filepath = path+file0
 file1 = label+'_' + tlabel
 fileout  = path + file1

 filetemp = './temp_'+label+'.pvtu'

 print 'opening: ', filepath

 reader.FileName = filepath
 reader.PointArrayStatus = ['Density','Temperature']
 filter = CleantoGrid(reader)
 writer.FileName = filetemp
 writer.Input = filter
 writer.UpdatePipeline()
 
 data = vtktools.vtu(filetemp)
 
 #print 'fields: ', data.GetFieldNames()
 coord = data.GetLocations()
 R = data.GetScalarField('Temperature')
 T = data.GetScalarField('Density')
 #T = data.GetScalarField('T')
 print np.min(T), np.max(T), np.min(R), np.max(R)

 T = T - 1000

 del data

 id_T = np.asarray(np.where(coord[:,2] == 0)) # values from surface
 
 x = coord[id_T,0].flatten()
 y = coord[id_T,1].flatten()
 Ts = T[id_T].flatten()
 Ts[np.asarray(np.where(Ts > Tmax))]  = Tmax
 Ts[np.asarray(np.where(Ts < Tmin))]  = Tmin
 ## PRINT
 
 #triang = tri.Triangulation(x, y)
 
 plt.figure()
 plt.gca().set_aspect('equal')
 plt.autoscale(enable=True, axis='both', tight=True)
 if time == 0:
  plt.triplot(x,y,linewidth=0.5,color='gray')
 # colorNorm = mpl.colors.Normalize(vmin=10, vmax=20,clip=True)
 #plt.scatter(x,y,c=Ts,norm=colorNorm)
 v1 = np.linspace(Tmin, Tmax, 50, endpoint=True)
 v2 = np.linspace(Tmin, Tmax, 11, endpoint=True)
 plt.tricontourf(x,y,Ts,v1)
 plt.colorbar(ticks=v2)
 plt.ylabel('Longitude')
 plt.xlabel('Latitude')
 plt.xticks(range(0,15000,1000),(range(0,15,1)))
 plt.yticks(range(0,15000,1000),(range(0,15,1)))
 plt.title('T, 2Km, '+str(np.trunc(time*5/24)/10.0)+' day')
 plt.savefig('./plot/'+label+'/'+file1+'_R.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/'+file1+'_R.eps\n'


# os.system('gs -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'.png ./plot/'+label+'/'+file1+'.eps')
 os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'_R.jpg ./plot/'+label+'/'+file1+'_R.eps')
 os.system('mogrify ./plot/'+label+'/'+file1+'.jpg -trim ./plot/'+label+'/'+file1+'_R.jpg')
# os.system('mogrify ./plot/'+label+'/'+file1+'.png -trim ./plot/'+label+'/'+file1+'.png')

print 'Making movie animation'

fr  = '10'
br  = '4096k'
crf = '24'

opts = '-y -f image2 -r '+fr+' -i ./plot/'+label+'/'+label+'_%03d_R.jpg -vcodec'

#ffmpeg_ogg = 'ffmpeg '+opts+' libtheora -b:v '+br+' ./plot/'+label+'/'+label+'.ogg'
ffmpeg_mp4 = 'ffmpeg '+opts+' libx264 -threads 0 -crf '+crf+' -s 1250x1620 ./plot/'+label+'/'+label+'_R.avi'

#print ffmpeg_ogg
#os.system(ffmpeg_ogg)

print ffmpeg_mp4
os.system(ffmpeg_mp4)

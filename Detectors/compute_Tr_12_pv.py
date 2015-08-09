#from memory_profiler import memory_usage
import os, sys
import gc
import vtktools
import numpy as np
import myfun
from scipy import interpolate

gc.enable()

label = 'm_50_7'
label = label+'_2Db_particles'
basename = 'mli_checkpoint'
dayi = 0
dayf = 3
days = 1

time = range(dayi,dayf,days)

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days
#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

path = '/scratch/jmensa/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

xn = 200
yn = 200

Xlist = np.linspace(0,10000,xn)# x co-ordinates of the desired array shape
Ylist = np.linspace(0,10000,yn)# y co-ordinates of the desired array shape
[X,Y] = np.meshgrid(Xlist,Ylist)

gc.collect()

depths = [5]

t = 0

for tt in time:
# print memory_usage(-1, interval=.2, timeout=.2)
 tlabel = str(tt)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file0 = basename+'_' + str(tt) + '.pvtu'
 filepath = path+file0
 file1 = label+'_' + tlabel
 fileout  = path + file1
 #
 #
 #
 print 'opening ', filepath
 #
# data = vtktools.vtu(filepath)
# print memory_usage(-1, interval=.2, timeout=.2)
 if tt == time[0]:
  coords = vtktools.vtu(filepath).GetLocations()
  layers = sorted(set(coords[:,2]),reverse=True)
  print memory_usage(-1, interval=.2, timeout=.2)

 for k in range(len(depths)):
  print 'Tracer_',k+2
  C = vtktools.vtu(filepath).GetScalarField('Tracer_'+str(k+2)+'_CG')
#  print memory_usage(-1, interval=.2, timeout=.2)

  fd = open('./Tracer_'+label+'_Tr'+str(k+2)+'_'+str(tt)+'.csv','a')

  for z in range(len(layers)):
#   print memory_usage(-1, interval=.2, timeout=.2)
   print z
   coords_l = coords[coords[:,2]==layers[z],:]
   Ct = interpolate.griddata((coords_l[:,0],coords_l[:,1]),C[coords[:,2]==layers[z]],(X,Y),method='linear')
   for j in xrange(len(Ylist)):
    for i in xrange(len(Xlist)):
     fd.write(str(Ct[j,i])+', ')
    fd.write('\n')
  
   del Ct, coords_l
   gc.collect()
  fd.close()
  del C, fd
  gc.collect()
 gc.collect()
del coords

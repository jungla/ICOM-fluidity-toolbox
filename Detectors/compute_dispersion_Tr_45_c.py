import os, sys
from scipy import interpolate
import gc
import vtktools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import myfun
from scipy import interpolate

gc.enable()

label = 'm_50_7'
label = label+'_3D_particles'
basename = 'mli_checkpoint'
dayi = 0
dayf = 80
days = 1

time = range(dayi,dayf,days)

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days
#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

path = '/tamay2/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

xn = 200
yn = 200

Xlist = np.linspace(0,10000,xn)# x co-ordinates of the desired array shape
Ylist = np.linspace(0,10000,yn)# y co-ordinates of the desired array shape
[X,Y] = np.meshgrid(Xlist,Ylist)

gc.collect()

depths = [17]

t = 0

for tt in time:
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
 data = vtktools.vtu(filepath)
 coords = data.GetLocations()

 if tt == dayi:
  layers = sorted(set(coords[:,2]),reverse=True)

 for k in range(len(depths)):
  print 'Tracer_',k+1
  C = data.GetScalarField('Tracer_'+str(k+1)+'_CG')

  fd = open('./Tracer_'+label+'_'+str(k+1)+'_'+str(tt)+'_3D.csv','a')

  for z in range(len(layers)):
   #print z
   coords_l = coords[coords[:,2]==layers[z],:]
   C_l = C[coords[:,2]==layers[z]]
   Ct = interpolate.griddata((coords_l[:,0],coords_l[:,1]),C_l,(X,Y),method='cubic')

   # write
   for j in range(len(Ylist)):
    for i in range(len(Xlist)):
     fd.write(str(Ct[j,i])+', ')
    fd.write('\n')

  fd.close()

 del data
 gc.collect()

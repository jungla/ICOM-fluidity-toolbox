import os, sys
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
label = label+'_2D_particles'
basename = 'mli_checkpoint'
dayi = 0
dayf = 60
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

xn = 100
yn = 100
zn = 50

Xlist = np.linspace(0,10000,xn)# x co-ordinates of the desired array shape
Zlist = np.linspace(0,-50,zn)# x co-ordinates of the desired array shape
Ylist = np.linspace(0,10000,yn)# y co-ordinates of the desired array shape


[X,Y,Z] = myfun.meshgrid2(Xlist,Ylist,Zlist)
Yl = np.reshape(Y,(np.size(Y),))
Xl = np.reshape(X,(np.size(X),))
Zl = np.reshape(Z,(np.size(Z),))

pts = zip(Xl,Yl,Zl)
pts = vtktools.arr(pts)

gc.collect()

depths = [1, 5, 11]

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
 print 'opening ', filepath
 #
 data = vtktools.vtu(filepath)
 #
 for z in range(len(depths)):
  print 'extracting points'
  T = data.ProbeData(pts,'Tracer_'+str(z+1)+'_CG')
  gc.collect()
  #
  Tr = np.reshape(T,[len(Zlist),len(Ylist),len(Xlist)])
#  D3[t,z,:] = tracer_d3(Xlist,Ylist,Zlist,Tr)
  D2[t,z,:] = tracer_d2(Xlist,Ylist,Tr,depths[z])
 del data
 gc.collect()
 t = t + 1

time = np.asarray(time)*1200.0 + 86400.0

# save to csv

import csv
path = './D2_123.csv'

fd = open(path,'a')

for t in range(len(time)):
 fd.write(str(time[t])+', '+str(D2[t,0,0])+', '+str(D2[t,1,0])+', '+str(D2[t,2,0])+'\n')

fd.close()

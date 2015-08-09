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
dayi = 1
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
yn = 20
zn = 1

delta = 3

Xlist = np.linspace(0,10000,xn)# x co-ordinates of the desired array shape
Ylist = np.linspace(0,10000,yn)# y co-ordinates of the desired array shape


gc.collect()

depths = [1, 5]

def tracer_d2(Xlist,Ylist,Tr,id):
 S00 = 0
 S01 = 0
 S02 = 0
 N = len(Ylist)
 Xlist = Xlist - 1500.0
 for j in range(N):
  A = max(Xlist)-min(Xlist)
  S00 = S00 + np.trapz(Tr[id,j,:], Xlist, 0)/A
  S01 = S01 + np.trapz(Tr[id,j,:]*(X[id,j,:]-1500), Xlist, 0)/A
  S02 = S02 + np.trapz(Tr[id,j,:]*(X[id,j,:]-1500)**2, Xlist, 0)/A
  S00 = S00/N; S01 = S01/N; S02 = S02/N
 return (S02-S01**2)/S00

D2 = np.zeros([len(range(dayi,dayf,days)),len(depths),2])
D3 = np.zeros([len(range(dayi,dayf,days)),len(depths),2])

t = 0

import copy

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
 data = vtktools.vtu(filepath)
 print 'done.'
 for z in range(len(depths)):
  print z
  # points of interest (2D only!)
  Zlist = -np.linspace(depths[z],depths[z],1)
  [X,Y,Z] = myfun.meshgrid2(Xlist,Ylist,Zlist)
  Yl = np.reshape(Y,(np.size(Y),))
  Xl = np.reshape(X,(np.size(X),))
  Zl = np.reshape(Z,(np.size(Z),))
  #
  pts = zip(Xl,Yl,Zl)
  pts = vtktools.arr(pts)
  #
  print 'extracting points'
  T = data.ProbeData(pts,'Tracer_'+str(z+1)+'_CG')
  gc.collect()
  #
  Tr = np.reshape(T,[len(Zlist),len(Ylist),len(Xlist)])
#  D3[t,z,:] = tracer_d3(Xlist,Ylist,Zlist,Tr)
  fd = open('./T_'+label+'_'+str(z+1)+'_'+str(time[t])+'.csv','a')
  for j in range(len(Ylist)):
   for i in range(len(Xlist)):
    fd.write(str(Tr[0,j,i])+', ')
   fd.write('\n')
  fd.close()
  D2[t,z,:] = tracer_d2(Xlist,Ylist,Tr,0)
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

try: paraview.simple
except: from paraview.simple import *

import numpy as np
from mpi4py import MPI
import os
import csv
from scipy import interpolate
import gc
import sys

gc.enable()

comm = MPI.COMM_WORLD

label = 'm_25_3b'
labelo = 'm_25_3b'
basename = 'mli'

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days
#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

tt = int(sys.argv[1])
labelo = sys.argv[2]
label = sys.argv[2]
basename = sys.argv[3]
field = sys.argv[4]
t0 = int(sys.argv[5])
dimx = int(sys.argv[6])
dimy = int(sys.argv[7])
resx = int(sys.argv[8])
resy = int(sys.argv[9])

path = '/tamay2/mensa/fluidity/'+label+'/'

Xlist = np.linspace(0,dimx,resx)
Ylist = np.linspace(0,dimy,resy)
#Xlist = np.linspace(0,10000,resx)
#Ylist = np.linspace(0,4000,resy)
Zlist = np.linspace(0,-50,51)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = np.cumsum(dl)

[X,Y] = np.meshgrid(Xlist,Ylist)

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
 
nl = len(Zlist)/size
ll = len(Zlist)%size

mli_pvtu = XMLPartitionedUnstructuredGridReader( FileName=[path+'/'+basename+'_'+str(tt)+'.pvtu'] )
mli_pvtu.PointArrayStatus = [field]
 
sliceFilter = Slice(mli_pvtu)
sliceFilter.SliceType.Normal = [0,0,1]
 
if rank == 0:
 Tr = np.zeros((len(Ylist),len(Xlist),len(Zlist)))
 for n in range(nl+ll):
  layer = n+rank*nl
  print 'layer:', rank, layer 
  sliceFilter.SliceType.Origin = [0,0,-1*Zlist[layer]]
  DataSliceFile = paraview.servermanager.Fetch(sliceFilter)
  points = DataSliceFile.GetPoints()
  numPoints = DataSliceFile.GetNumberOfPoints()
  #
  data=np.zeros((numPoints))
  coords=np.zeros((numPoints,3))
  #
  for x in xrange(numPoints):
   data[x] = DataSliceFile.GetPointData().GetArray(field).GetValue(x)
   coords[x] = points.GetPoint(x)
  
  Tr[:,:,layer] = interpolate.griddata((coords[:,0],coords[:,1]),data,(X,Y),method='linear')
#  print rank, Tr[:,:,:]

if rank > 0:
 Tr = np.zeros((len(Ylist),len(Xlist),nl))
 for n in xrange(nl):
  layer = n+rank*nl
  print 'layer:', rank, layer
  sliceFilter.SliceType.Origin = [0,0,-1*Zlist[layer]]
  DataSliceFile = paraview.servermanager.Fetch(sliceFilter)
  points = DataSliceFile.GetPoints()
  numPoints = DataSliceFile.GetNumberOfPoints()
  #
  data=np.zeros((numPoints))
  coords=np.zeros((numPoints,3))
  #
  for x in xrange(numPoints):
   data[x] = DataSliceFile.GetPointData().GetArray(field).GetValue(x)
   coords[x] = points.GetPoint(x)
   
  Tr[:,:,n] = interpolate.griddata((coords[:,0],coords[:,1]),data,(X,Y),method='linear')
 #  print rank, Tr[:,:,:]
 
 comm.send(nl*rank+ll, dest=0, tag=10)
 comm.send(Tr, dest=0, tag=11)
 
 
if rank == 0:
 for s in range(size-1):
  print 's', s+1
  l = comm.recv(source=s+1, tag=10)
  print 'l', l
  Tr[:,:,l:l+nl] = comm.recv(source=s+1, tag=11) 
 print Tr
 fd = open('./csv/'+field+'_'+labelo+'_'+str(tt+t0)+'.csv','w')
 print Tr[:,:,:]
 for z in xrange(len(Zlist)):
  print z
  for j in xrange(len(Ylist)):
   for i in xrange(len(Xlist)):
    fd.write(str(Tr[j,i,z])+', ')
   fd.write('\n')
 fd.close()

del mli_pvtu, Tr, coords, data, numPoints, points, DataSliceFile, sliceFilter
gc.collect()

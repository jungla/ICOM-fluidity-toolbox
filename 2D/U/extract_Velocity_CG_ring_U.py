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

label = 'r_1k_B_1F1'
basename = 'ring'

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days
#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

f = open('../../days.list','r')
days = f.readlines()
tt = int(days[int(sys.argv[1])-1])
f.close()

path = '/scratch/jmensa/'+label+'/'

Xlist = np.linspace(-150000,150000,301)
Ylist = np.linspace(-150000,150000,301)
depths = [10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 50, 50, 50, 50, 50, 50, 50, 50, 50]
Zlist = -1*np.cumsum(depths)

[X,Y] = np.meshgrid(Xlist,Ylist)

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
 
nl = len(Zlist)/size
ll = len(Zlist)%size

mli_pvtu = XMLPartitionedUnstructuredGridReader( FileName=[path+basename+'_'+str(tt)+'.pvtu'] )
mli_pvtu.PointArrayStatus = ['Velocity_CG']
 
sliceFilter = Slice(mli_pvtu)
sliceFilter.SliceType.Normal = [0,0,1]
 
if rank == 0:
 U = np.zeros((len(Xlist),len(Ylist),len(Zlist)))
 for n in range(nl+ll):
  layer = n+rank*nl
  print 'layer:', rank, layer 
  sliceFilter.SliceType.Origin = [0,0,-1*layer]
  DataSliceFile = paraview.servermanager.Fetch(sliceFilter)
  points = DataSliceFile.GetPoints()
  numPoints = DataSliceFile.GetNumberOfPoints()
  #
  data=np.zeros((numPoints,3))
  coords=np.zeros((numPoints,3))
  #
  for x in xrange(numPoints):
   data[x,:] = DataSliceFile.GetPointData().GetArray('Velocity_CG').GetTuple(x)
   coords[x] = points.GetPoint(x)
  # 
  U[:,:,layer] = interpolate.griddata((coords[:,0],coords[:,1]),data[:,0],(X,Y),method='linear')
#  print rank, U[:,:,:]

if rank > 0:
 U = np.zeros((len(Xlist),len(Ylist),nl))
 for n in xrange(nl):
  layer = n+rank*nl
  print 'layer:', rank, layer
  sliceFilter.SliceType.Origin = [0,0,-1*layer]
  DataSliceFile = paraview.servermanager.Fetch(sliceFilter)
  points = DataSliceFile.GetPoints()
  numPoints = DataSliceFile.GetNumberOfPoints()
  #
  data=np.zeros((numPoints,3))
  coords=np.zeros((numPoints,3))
  #
  for x in xrange(numPoints):
   data[x,:] = DataSliceFile.GetPointData().GetArray('Velocity_CG').GetTuple(x)
   coords[x] = points.GetPoint(x)
   
  U[:,:,n] = interpolate.griddata((coords[:,0],coords[:,1]),data[:,0],(X,Y),method='linear')
 #  print rank, U[:,:,:]
 
 comm.send(nl*rank+ll, dest=0, tag=10)
 comm.send(U, dest=0, tag=11)
  
if rank == 0:
 for s in range(size-1):
  print 's', s+1
  l = comm.recv(source=s+1, tag=10)
  print 'l', l
  U[:,:,l:l+nl] = comm.recv(source=s+1, tag=11) 
 fd0 = open('./csv/Velocity_CG_0_'+label+'_'+str(tt)+'.csv','a')
 print U[:,:,:]
 for z in xrange(len(Zlist)):
  print z
  for j in xrange(len(Ylist)):
   for i in xrange(len(Xlist)):
    fd0.write(str(U[i,j,z])+', ')
   fd0.write('\n')
 fd0.close()

del mli_pvtu, U, coords, data, numPoints, points, DataSliceFile, sliceFilter
gc.collect()

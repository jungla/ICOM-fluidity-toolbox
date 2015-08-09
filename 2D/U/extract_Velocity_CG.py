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

#label = 'm_10_1'
#labelo = 'm_10_1'

tt = int(sys.argv[1]) 
labelo = sys.argv[2]
label = sys.argv[2]
basename = sys.argv[3]
t0 = int(sys.argv[4])
dimx = int(sys.argv[5])
dimy = int(sys.argv[6])
resx = int(sys.argv[7])
resy = int(sys.argv[8])

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

mli_pvtu = XMLPartitionedUnstructuredGridReader( FileName=[path+basename+'_'+str(tt)+'.pvtu'] )
mli_pvtu.PointArrayStatus = ['Velocity_CG']

print 'opening:', path+basename+'_'+str(tt)+'.pvtu'
 
sliceFilter = Slice(mli_pvtu)
sliceFilter.SliceType.Normal = [0,0,1]
 
if rank == 0:
 U = np.zeros((len(Ylist),len(Xlist),len(Zlist),3))
 for n in range(nl+ll):
  layer = n+rank*nl
  print 'layer:', rank, layer 
  sliceFilter.SliceType.Origin = [0,0,-1*Zlist[layer]]
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
  U[:,:,layer,0] = interpolate.griddata((coords[:,0],coords[:,1]),data[:,0],(X,Y),method='linear')
  U[:,:,layer,1] = interpolate.griddata((coords[:,0],coords[:,1]),data[:,1],(X,Y),method='linear')
  U[:,:,layer,2] = interpolate.griddata((coords[:,0],coords[:,1]),data[:,2],(X,Y),method='linear')
#  print rank, U[:,:,:]

if rank > 0:
 U = np.zeros((len(Ylist),len(Xlist),nl,3))
 for n in xrange(nl):
  layer = n+rank*nl
  print 'layer:', rank, layer
  sliceFilter.SliceType.Origin = [0,0,-1*Zlist[layer]]
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
   
  U[:,:,n,0] = interpolate.griddata((coords[:,0],coords[:,1]),data[:,0],(X,Y),method='linear')
  U[:,:,n,1] = interpolate.griddata((coords[:,0],coords[:,1]),data[:,1],(X,Y),method='linear')
  U[:,:,n,2] = interpolate.griddata((coords[:,0],coords[:,1]),data[:,2],(X,Y),method='linear')
 #  print rank, U[:,:,:]
 
 comm.send(nl*rank+ll, dest=0, tag=10)
 comm.send(U, dest=0, tag=11)
  
if rank == 0:
 for s in range(size-1):
  print 's', s+1
  l = comm.recv(source=s+1, tag=10)
  print 'l', l
  U[:,:,l:l+nl,:] = comm.recv(source=s+1, tag=11) 
 fd0 = open('./csv/Velocity_CG_0_'+labelo+'_'+str(tt+t0)+'.csv','w')
 fd1 = open('./csv/Velocity_CG_1_'+labelo+'_'+str(tt+t0)+'.csv','w')
 fd2 = open('./csv/Velocity_CG_2_'+labelo+'_'+str(tt+t0)+'.csv','w')
 print U[:,:,:]
 for z in xrange(len(Zlist)):
  print z
  for j in xrange(len(Ylist)):
   for i in xrange(len(Xlist)):
    fd0.write(str(U[j,i,z,0])+', ')
    fd1.write(str(U[j,i,z,1])+', ')
    fd2.write(str(U[j,i,z,2])+', ')
   fd0.write('\n')
   fd1.write('\n')
   fd2.write('\n')
 fd0.close()
 fd1.close()
 fd2.close()

del mli_pvtu, U, coords, data, numPoints, points, DataSliceFile, sliceFilter
gc.collect()

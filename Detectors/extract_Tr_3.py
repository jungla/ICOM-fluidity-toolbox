try: paraview.simple
except: from paraview.simple import *

import numpy as np
from mpi4py import MPI
import os
import csv
from scipy import interpolate


comm = MPI.COMM_WORLD

label = 'm_50_7'
label = label+'_3Db_particles'
basename = 'mli_checkpoint'
dayi = 0
dayf = 50
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

Xlist = np.linspace(0,10000,200)
Ylist = np.linspace(0,10000,200)
Zlist = np.linspace(0,-50,51)    

[X,Y] = np.meshgrid(Xlist,Ylist)

depths = [17]

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
 
nl = len(Zlist)/size
ll = len(Zlist)%size

for tt in time:
 print 'Time: ', tt
 for k in range(len(depths)):
  kl = str(k+4)
  print 'Tracer_'+kl+'_CG'
  mli_pvtu = XMLPartitionedUnstructuredGridReader( FileName=[path+'/mli_checkpoint_'+str(tt)+'.pvtu'] )
  mli_pvtu.PointArrayStatus = ['Tracer_'+kl+'_CG']
  
  sliceFilter = Slice(mli_pvtu)
  sliceFilter.SliceType.Normal = [0,0,1]
  
  if rank == 0:
   Tr = np.zeros((len(Xlist),len(Ylist),len(Zlist)))
   for n in range(nl+ll):
    layer = n+rank*nl
#    print 'layer:', rank, layer 
    sliceFilter.SliceType.Origin = [0,0,-1*layer]
    DataSliceFile = paraview.servermanager.Fetch(sliceFilter)
    points = DataSliceFile.GetPoints()
    numPoints = DataSliceFile.GetNumberOfPoints()
    #
    data=np.zeros((numPoints))
    coords=np.zeros((numPoints,3))
    #
    for x in range(numPoints):
     data[x] = DataSliceFile.GetPointData().GetArray('Tracer_'+kl+'_CG').GetValue(x)
     coords[x] = points.GetPoint(x)
    
    Tr[:,:,layer] = interpolate.griddata((coords[:,0],coords[:,1]),data,(X,Y),method='linear')
  #  print rank, Tr[:,:,:]
  
  if rank > 0:
   Tr = np.zeros((len(Xlist),len(Ylist),nl))
   for n in range(nl):
    layer = n+rank*nl
   # print 'layer:', rank, layer
    sliceFilter.SliceType.Origin = [0,0,-1*layer]
    DataSliceFile = paraview.servermanager.Fetch(sliceFilter)
    points = DataSliceFile.GetPoints()
    numPoints = DataSliceFile.GetNumberOfPoints()
    #
    data=np.zeros((numPoints))
    coords=np.zeros((numPoints,3))
    #
    for x in range(numPoints):
     data[x] = DataSliceFile.GetPointData().GetArray('Tracer_'+kl+'_CG').GetValue(x)
     coords[x] = points.GetPoint(x)
    
    Tr[:,:,n] = interpolate.griddata((coords[:,0],coords[:,1]),data,(X,Y),method='linear')
  #  print rank, Tr[:,:,:]
  
   comm.send(nl*rank+ll, dest=0, tag=10)
   comm.send(Tr, dest=0, tag=11)
    
  if rank == 0:
   for s in range(size-1):
   # print 's', s+1
    l = comm.recv(source=s+1, tag=10)
   # print 'l', l
    Tr[:,:,l:l+nl] = comm.recv(source=s+1, tag=11) 
   # print Tr
   fd = open('./csv/Tracer_'+label+'_Tr'+kl+'_'+str(tt)+'.csv','a')
   # print Tr[:,:,:]
   for z in range(len(Zlist)):
   # print z
    for j in xrange(len(Ylist)):
     for i in xrange(len(Xlist)):
      fd.write(str(Tr[i,j,z])+', ')
     fd.write('\n')
   fd.close()

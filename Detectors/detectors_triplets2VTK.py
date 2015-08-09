#!python
# usage: python detectors2VTK.py exp flag tt flag1
# exp: experiment flag
# flag: time flag, split in timesteps if flag = 0
# tt: number of timesteps to plot, starting from 0
# flag1: 0 all particles, 1 only center (for triplets)

import fluidity_tools
import pyvtk
import myfun
import numpy as np
import os
import sys

## split in timesteps if flag = 0

#exp = sys.argv[1]
#flag  = int(sys.argv[2])
#tt = int(sys.argv[3])
#flag1 = int(sys.argv[4])

exp = 'm_50_5_2D_particles'
flag = 1
tt = 180
flag1 = 0

if (flag1 == 0): step = 1
else: step = 4

filename = '/tamay2/mensa/fluidity/'+exp+'/mli_particles.detectors'

try: os.stat('./output/'+exp)
except OSError: os.mkdir('./output/'+exp)

det = fluidity_tools.stat_parser(filename)
keys = det.keys()                                # particles

pt = int(os.popen('grep position '+filename+'| wc -l').read()) # read the number of particles grepping all the positions in the file

print 'particles:',pt
print 'timesteps:',tt
print 'step:',step

# read particles

par = np.zeros((pt,3,tt))

time = det['ElapsedTime']['value']

# read particles

for d in range(pt):
 temp = det['particles_'+myfun.digit(d+1,len(str(pt)))]['position']
 par[d,:,:] = temp[:,0:tt]

# line takes only two points at the time. it has to be defined for each segment

if flag == 0:
# build points coord
#
 for st in range(tt):
  # build points coord
  points = []
  lines = []
  l = 0
#
  for d in range(0,pt-step,step):
   for i in [0,4]:
    for t in range(st-1):
     points.append(par[d+i,:,t])
     lines.append([l,l+1])
     l = l+1
    l = l+1
    points.append(par[d+i,:,st])
#
  vtk = pyvtk.VtkData(pyvtk.UnstructuredGrid(points=points,line=lines),pyvtk.PointData(pyvtk.Scalars(np.zeros(len(points)),'T')),'Unstructured Grid Example')
  label = exp+'_'+myfun.digit(st,3)+'.vtk'
  print label
  vtk.tofile('./output/'+exp+'/'+label,'binary')


if flag == 1:
# build points coord
 points = []
 lines = []
 l = 0

 for d in range(0,pt-step,step):
  for i in [0,4]:
   for t in range(tt-1):
    points.append(par[d+i,:,t])
    lines.append([l,l+1])
    l = l+1
  l = l+1
  points.append(par[d+i,:,tt-1])

 vtk = pyvtk.VtkData(pyvtk.UnstructuredGrid(points=points,line=lines),pyvtk.PointData(pyvtk.Scalars(np.zeros(len(points)),'T')),'Unstructured Grid Example')
 label = exp+'.vtk'
 print label
 vtk.tofile('./output/'+exp+'/'+label,'binary')

#!/share/apps/python/2.7.2/bin/python

import numpy as np
global np


## assign values to a vtk grid from another vtk grid

def replaceScalar(T,coord,coord_o):
 Tn = []
 for id in range(0,len(coord_o)-1):
  id_x  = np.asarray(np.where(np.trunc(coord[:,0]) == np.trunc(coord_o[id,0])))
  id_y  = np.asarray(np.where(np.trunc(coord[:,1]) == np.trunc(coord_o[id,1])))
  id_z  = np.asarray(np.where(np.trunc(coord[:,2]) == np.trunc(-coord_o[id,2])))
  id_T = np.intersect1d(np.intersect1d(id_z,id_y),id_x)
  Tval  = T[id_T]
  coord = np.delete(coord,id_T,0)
  T     = np.delete(T,id_T,0)
  Tn.append(Tval[1])
 return Tn


## digits format

def digit(day,n):
 lday  = str(day)
 while (len(lday) < n):
  lday = '0'+lday
 return lday


# function to sort one matrix: used to write VTK files

def f1(X):
    a,b,c = X.shape
    return [(X[i,j,k]) for k in xrange(c) for j in xrange(b) for i in xrange(a)]


# function to sort three matrices: used to write VTK files

def f3(X,Y,Z):
    a,b,c = X.shape
    return [(X[i,j,k],Y[i,j,k],Z[i,j,k]) for k in xrange(c) for j in xrange(b) for i in xrange(a)]


# remove duplicates from list

def f5(seq, idfun=None):
   # order preserving
   if idfun is None:
       def idfun(x): return x
   seen   = {}
   result = []
   r_id   = []
   id     = 0
   for item in seq:
       marker = idfun(item)
       id = id + 1
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
       r_id.append(id)
   return result, r_id


# define triangles from a rectilinear mesh

def pts2trs(idm,jdm):
 triangles = []
 for j in range(jdm):
  for i in range(idm-1):
   a=i+j*idm
   b=a+1
   c=a+idm
   d=c+1
   triangles.append([a,b,c])
   triangles.append([c,b,d])
   #      print j,i,d,[a,b,c],[c,b,d]
   if d == idm*jdm-1: break
  if d == idm*jdm-1: break
 return triangles


def meshgrid2(*arrs):
    arrs = tuple(reversed(arrs))  #edit
    lens = map(len, arrs)
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz*=s

    ans = []    
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j!=i:
                arr2 = arr2.repeat(sz, axis=j) 
        ans.append(arr2)

    return tuple(ans[::-1])




def smooth2(R,times):
 global np

 ids,jds = R.shape

 Rf = np.zeros([ids,jds]);
 Rf[:,:] = np.nan

 for t in xrange(times):

 # smooth
  for i in xrange(ids):
   for j in xrange(jds):
 
    if ~np.isnan(R[i,j]):
     Rf[i,j] = R[i,j]
     pn = 1
 
     for k in range(4):
 
      if k == 0:
       ip = -1
       jp = 0
      elif k == 1:
       ip = 0
       jp = -1
      elif k == 2:
       ip = 1
       jp = 0
      else:
       ip = 0
       jp = 1
    
      if i+ip>0 and j+jp>0 and i+ip<ids and j+jp<jds:
       if ~np.isnan(R[i+ip,j+jp]):
        Rf[i,j] = Rf[i,j] + R[i+ip,j+jp]
        pn = pn + 1
 
     Rf[i,j] = Rf[i,j]/pn
 
  R = Rf
 return R

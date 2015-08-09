#!python
import numpy as np
from struct import unpack

# returns a(IDM,JDM,KDM), z(IDM,JDM,KDM) for each specific 3D quantity
# label L can be: T,S,U,V,H

def hycomread_3D(filename,L,IDM,JDM,KDM):

 IJDM = IDM*JDM
 npad  = 4096-IJDM%4096
 f = open(filename,'r')
 a = np.zeros((JDM,IDM,KDM))

 if L == 'U':
   print 'reading U'
   id = 1
   Ub = np.zeros((JDM,IDM))
   f.seek(10*4*(npad+IJDM))
   tmp = np.zeros((IJDM))
   for i in xrange(0,IJDM):
    var = unpack('>f',f.read(4))
    var = float(var[0])
    if var > 10**8:
     var = float('NaN')
    tmp[i] = var
   tmp = tmp.reshape(JDM,IDM)
   Ub[:,:] = tmp

 elif L == 'V':
   print 'reading V'
   id = 2
   Vb = np.zeros((JDM,IDM))
   f.seek(10*4*(npad+IJDM))
   tmp = np.zeros((IJDM))
   for i in xrange(0,IJDM):
    var = unpack('>f',f.read(4))
    var = float(var[0])
    if var > 10**8:
     var = float('NaN')
    tmp[i] = var
   tmp = tmp.reshape(JDM,IDM)
   Vb[:,:] = tmp

 elif L == 'H':
   print 'reading H'
   id = 3
 elif L == 'T':
   print 'reading T'
   id = 4
 elif L == 'S':
   print 'reading S'
   id = 5

 # read the archive

 for k in  xrange(0,KDM):
  f.seek((10+k*5+id)*4*(npad+IJDM))
  tmp = np.zeros((IJDM))
  for i in xrange(0,IJDM):
   var = unpack('>f',f.read(4))
   var = float(var[0])
   if var > 10**8:
    var = float('NaN')
   tmp[i] = var
  tmp = tmp.reshape(JDM,IDM)
  if L == 'U': tmp = tmp + Ub
  if L == 'V': tmp = tmp + Vb
  if L == 'H': tmp = tmp/9860
  a[:,:,k] = tmp

 f.close()
 return a

## extract 2D field

def hycomread_2D(filename,IDM,JDM,INDEX):
 IJDM = IDM*JDM
 npad  = 4096-IJDM%4096
 f = open(filename,'r')
 a = np.zeros((JDM,IDM))

 f.seek(INDEX*4*(npad+IJDM))
 tmp = np.zeros((IJDM))

 for i in xrange(IJDM):
  var = unpack('>f',f.read(4))
  var = float(var[0])
  if var > 10**8:
   var = float('NaN')
  tmp[i] = var

 a[:,:] = tmp.reshape(JDM,IDM)

 f.close()
 return a

def binaryread_2D(filename,IDM,JDM,INDEX):
 IJDM = IDM*JDM
 f = open(filename,'r')
 a = np.zeros((JDM,IDM))

 f.seek(INDEX*4*(IJDM))
 tmp = np.zeros((IJDM))

 for i in xrange(IJDM):
  var = unpack('>f',f.read(4))
  var = float(var[0])
  if var > 10**8:
   var = float('NaN')
  tmp[i] = var

 a[:,:] = tmp.reshape(JDM,IDM)

 f.close()
 return a

import re

def getshape(filename):
 f = open(filename,'r')
 first  = f.readline()
 second = f.readline()
 fsplit  = re.split(' ', first.strip())
 ssplit  = re.split(' ', second.strip())
 return int(fsplit[0]),int(ssplit[0])


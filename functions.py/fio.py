import os
import numpy as np
import csv
import scipy.stats as ss
import fluidity_tools
import myfun

global os, np, ss, csv, fluidity_tools, myfun

################# TRACER
def read_particles_csv(filename,xp,yp,zp,tt):
 time = []
 pt = xp*yp*zp
 par = np.zeros((pt,3,tt))
 f = open(filename,'r')
 reader = csv.reader(f)

 j = 0
 k = 0
 for row in reader:
  if k == pt: j = j + 1; k = 0; time.append(float(row[3]))
  i = 0
  for item in row[0:3]: # new line character !!
   par[k,i,j] = float(item)
#   print i
   i = i + 1
  k = k + 1
 f.close()
 time.append(float(row[3]))
# for row in reader:
#  time.append((float(row[3])))
#  par.append((float(row[0]),float(row[1]),float(row[2])))
 return np.asarray(time), np.asarray(par)

def read_Scalar(filepath,xn,yn,zn):
 # pd - list of tracer's name
 # zn,xn,yn - tracer's dims
 # timeTr - tracer's time series
 Tr = np.zeros((yn,xn,zn))
 f = open(filepath,'r')
 reader = csv.reader(f)
 j = 0
 k = 0
 for row in reader: 
  if j == yn: k = k + 1; j = 0
  i = 0
  for item in row[:-1]: # new line character !!
#   print i,j
   Tr[j,i,k] = item
   i = i + 1
  j = j + 1
 #  print np.amax(Tr[z,k,:,:,t])
 #  Tr[:,:,:,t] = Tr[:,:,:,t]/3
 f.close()
 #  TrT = np.reshape(Tr[0,:,:,t],)
 #  TrT = Tr[:,:,:,t]

 To = np.zeros((xn,yn,zn))

 for k in range(zn):
  To[:,:,k] = np.transpose(Tr[:,:,k])
 return To

 return Tr

# read probes

def read_probe(filename,depths,Tr,Xr):
 # space average
 data = []
 with open(filename, 'rb') as f:
  reader = csv.reader(f)
  for row in reader:
   data.append(row)

 data = np.asarray(data)
 data = np.reshape(data,(len(depths),len(Tr),len(Xr)))

 return data

def read_probe(filename,depths,Tr,Xr):
 # space average
 data = []
 with open(filename, 'rb') as f:
  reader = csv.reader(f)
  for row in reader:
   data.append(row)

 data = np.asarray(data)
 data = np.reshape(data,(len(depths),len(Tr),len(Xr)))

 return data

def read_probe_z(filename,Tr,Xr):
 # space average
 data = []
 with open(filename, 'rb') as f:
  reader = csv.reader(f)
  for row in reader:
   data.append(row)

 data = np.asarray(data)
 data = np.reshape(data,(len(Tr),len(Xr)))

 return data



#def read_probe(filename,depths):
# # space average
# data = []
# with open(filename, 'rb') as f:
#  reader = csv.reader(f)
#  for row in reader:
#   data.append(row)
# data = np.asarray(data)
# 
# return data 

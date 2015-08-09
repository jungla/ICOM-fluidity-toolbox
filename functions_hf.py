#!python

import csv
import numpy as np

## from HYCOM 2D -> 1D
# read a 1D csv file and returns a 1D array

def read_h_1d(filename):

 oneD_file = open(filename,'r')
 oneD_csv = csv.reader(oneD_file, delimiter=',')

 oneD = []

 for row in oneD_csv:
  for el in row:
   oneD.append(el)

 oneD = np.asarray(oneD).astype(float)

 oneD_file.close()

 return oneD

## from HYCOM 3D -> 2D
# read a 2D horizontal section from a 3D archive (csv file of the form [rows,cols*depth])
# returns a 1D array

def read_h_2d(filename,dim,k):

 V_file = open(filename,'r')
 V_csv = csv.reader(V_file, delimiter=',')

 V = []

 for row in V_csv:
  V.append(row[(dim*k):dim+(dim*k)])

 V = np.asarray(V).astype(float) 
 V_file.close()

 return V

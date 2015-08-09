#!python

import numpy as np
import csv
from scipy import interpolate

import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt

import matplotlib.tri as tri

data = []

f = open('l01_h_archv_8_009_B.slice.csv', 'rb')
reader = csv.reader(f)
print reader.next()

for rows in reader:
 row = []
 for item in rows:
  row.append(float(item))
 data.append(row)

f.close()

data = np.asarray(data)

Y = max(data[:,7])-data[:,7]
Z = data[:,8]
T = data[:,1]

plt.tricontourf(Y,Z,T)
plt.colorbar()
plt.savefig('contour_slice.eps')
plt.close()

#f = interpolate.Rbf(Y, Z, T, 'linear')

#def val(x,t):
# global f
# Ti = f(x[1],x[2])
# return Ti

#!python

import numpy as np
import csv
from scipy import interpolate

import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt

import matplotlib.tri as tri

data = []

f = open('./data/l01_h_archv_8_009_B_slice.csv', 'rb')
reader = csv.reader(f)
print reader.next()

for rows in reader:
 row = []
 for item in rows:
  row.append(float(item))
 data.append(row)

f.close()

data = np.asarray(data)

Y = data[:,7]-min(data[:,7])
Z = -1*data[:,8]
T = data[:,1]
S = data[:,2]

dZ = [0, 10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 40, 40, 40, 40, 40, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
Zlist = np.cumsum(dZ)
Zlist = np.linspace(0,-900,100)
print Zlist
Ylist = np.linspace(0,100000,250)

[Yi,Zi] = np.meshgrid(Ylist,Zlist)
Yi = np.reshape(Yi,(np.size(Yi),))
Zi = np.reshape(Zi,(np.size(Zi),))

#from scipy.interpolate import Rbf
#rbf = Rbf(Z, Y, T, epsilon=2)
#Ti = rbf(Zi, Yi)

Ti = interpolate.griddata((Y,Z),T,(Yi,Zi),method='linear')
Si = interpolate.griddata((Y,Z),S,(Yi,Zi),method='linear')

f = open('./data/l01_h_archv_8_009_B_slice_i.csv', 'wb')
writer = csv.writer(f)

for l in range(len(Zi)):
 writer.writerow((Yi[l],Zi[l],Ti[l],Si[l]))

Tr = np.reshape(Ti,[len(Zlist),len(Ylist)])
Sr = np.reshape(Si,[len(Zlist),len(Ylist)])

plt.contour(Ylist,Zlist,Tr,40)
plt.colorbar()
plt.savefig('T_ic.eps')
plt.close()

plt.contour(Ylist,Zlist,Sr,40)
plt.colorbar()
plt.savefig('S_ic.eps')
plt.close()
#f = interpolate.Rbf(Y, Z, T, 'linear')

#def val(x,t):
# global f
# Ti = f(x[1],x[2])
# return Ti

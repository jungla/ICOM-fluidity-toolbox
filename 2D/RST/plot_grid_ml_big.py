import os, sys

import fio, myfun
import vtktools
import numpy as np
import matplotlib  as mpl
#mpl.use('ps')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import csv

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

label = 'm_25_1b'
basename = 'mli' 
time = 0

path = '/tamay2/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

x = []
y = []
z = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,]
z = np.cumsum(z)

tlabel = str(time)
while len(tlabel) < 3: tlabel = '0'+tlabel
#
file0 = basename + '_' + str(time) + '.pvtu'
filepath = path+file0
file1 = 'grid_'+label+'_' + tlabel
fileout  = path + file1
#
print 'opening: ', filepath
#
#
nodes=[]
with open(path+'/box.msh', 'rb') as csvfile:
 spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
 for row in spamreader:
  nodes.append(row)

nodes = np.asarray(nodes)

pt = nodes[4]
pt = int(pt[0])
nodes = nodes[5:pt+5]

for node in nodes:
 y.append(node[2])
 x.append(node[1])

x = np.asarray(x).astype(float)
y = np.asarray(y).astype(float)

#x = x[0::8]
#y = y[0::8]
z = -1*np.asarray(z).astype(float)

[X,Z] = np.meshgrid(np.linspace(0,8000,8025/25.),z)
X = np.squeeze(np.reshape(X,(-1,1)))
Z = np.squeeze(np.reshape(Z,(-1,1)))

fig = plt.figure(figsize=(6, 6))
plt.triplot(X/1000,Z)
plt.xlim([0,1])
plt.ylim([-50,0])
plt.xlabel('X [km]',fontsize=16)
plt.ylabel('Z [m]',fontsize=16)
plt.savefig('./plot/'+label+'/'+file1+'_gridV.png')
print       './plot/'+label+'/'+file1+'_gridV.png'
plt.close()

fig = plt.figure(figsize=(6, 6))
plt.triplot(x/1000.,y/1000.)
#plt.axis('equal')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('X [km]',fontsize=16)
plt.ylabel('Y [km]',fontsize=16)
plt.savefig('./plot/'+label+'/'+file1+'_gridH.png')
print       './plot/'+label+'/'+file1+'_gridH.png'
plt.close()

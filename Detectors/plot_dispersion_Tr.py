#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import myfun
import numpy as np
from scipy import optimize
import os
import scipy.stats as sp


def f_exp(t,a,b,c):
    return a + b*np.exp(c*t)

label = 'm_50_7'

# horizontal
depth = 11 #11

pd = range(1,depth,3)
pd = [1, 5, 11, 17, 26]
#pd = [1]

nl = len(pd)

# Tracer second moment

import csv
path = './D2_1200.csv'
timef = 100

val = np.zeros([timef,5])
timeTr = np.zeros([timef])
t = 0

with open(path, 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        #time.append() # change later
        i = 0
        for item in row:
         if i == 0:
          timeTr[t] = float(item)
         else:
          val[t,i-1] = float(item)
#         print float(item),i
         i = i+1
        t = t+1

timeTr = timeTr[timeTr > 0]
D_Tr = np.reshape(val[val > 0],[len(timeTr),5])

# plotting all together

# Ellipses D

pTr, = plt.plot(timeTr/86400,D_Tr[:,0],color=[0,0,0],linewidth=2)

z = 1
pTr5, = plt.plot(timeTr/86400,D_Tr[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
z = 2
pTr11, = plt.plot(timeTr/86400,D_Tr[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
z = 3
pTr17, = plt.plot(timeTr/86400,D_Tr[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
z = 4
pTr26, = plt.plot(timeTr/86400,D_Tr[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)

plt.gca().set_yscale('log')
#plt.gca().set_xscale('log')

plt.xlabel('Time [days]')
plt.ylabel('Dispersion [m^2]')
plt.legend((pTr,pTr5,pTr11,pTr17,pTr26),('Tr 1m','Tr 5m','Tr 11m','Tr 17m','Tr 26m'),loc=4,fontsize=12)

plt.savefig('./plot/'+label+'_23D/D_Tr_'+label+'_23D.eps')
print       './plot/'+label+'_23D/D_Tr_'+label+'_23D.eps' 
plt.close()

import os, sys
import csv
import fio, myfun
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import fio 
import myfun

label = 'm_25_2b_particles'
basename = 'mli' 
dayi  = 324
dayf  = 325
days  = 1

#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

path = './Velocity_CG/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)


# dimensions archives

# Ring exp

Xlist = np.linspace(0,8000,641)
Ylist = np.linspace(0,8000,641)
#Xlist = np.linspace(0,10000,801)
#Ylist = np.linspace(0,4000,321)
Zlist = np.linspace(0,-50,51)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = -1*np.cumsum(dl)

xn = len(Xlist)
yn = len(Ylist)
zn = len(Zlist)

[X,Y,Z] = myfun.meshgrid2(Xlist,Ylist,Zlist)
Y = np.reshape(Y,(np.size(Y),))
X = np.reshape(X,(np.size(X),))
Z = np.reshape(Z,(np.size(Z),))


for time in range(dayi,dayf,days):
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 #

 file0u = path+'Velocity_CG_0_'+label+'_'+str(time)+'.csv'
 file0v = path+'Velocity_CG_1_'+label+'_'+str(time)+'.csv'
 file0w = path+'Velocity_CG_2_'+label+'_'+str(time)+'.csv'
 file1 = 'Velocity_CG_'+label+'_'+str(time)
 print file1
 #

 u = fio.read_Scalar(file0u,xn,yn,zn)
 v = fio.read_Scalar(file0v,xn,yn,zn)
 w = fio.read_Scalar(file0w,xn,yn,zn)
 #
 dt = 1 # s
 mu = np.mean(np.mean(u,0),0)
 mv = np.mean(np.mean(v,0),0)
 mw = np.mean(np.mean(w,0),0)
 dx = mu*dt
 dy = mv*dt
 dz = mw*dt

 fd = open('./UVW_'+file1+'.csv','w')
 fd.write('x, y, z, u, v, w \n')
 for z in range(len(Zlist)):
  fd.write('0, 0, '+str(Zlist[z])+', '+str(mu[z])+', '+str(mv[z])+', '+str(mw[z])+'\n')
 
 fd.close()

import os, sys
import myfun
import numpy as np
import lagrangian_stats
import scipy.interpolate as interpolate
import csv
import matplotlib.pyplot as plt
import advect_functions
import fio 
from intergrid import Intergrid
## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

#label = 'm_25_2_512'
label = 'm_25_1_particles'
dayi  = 960 # 24 hours  
dayf  = 0
days  = -1

#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

path = '../../2D/U/Velocity_CG/'

time = range(dayi,dayf,days)

# dimensions archives

# ML exp

#Xlist = np.linspace(0,10000,801)
#Ylist = np.linspace(0,4000,321)
Xlist = np.linspace(0,2000,161)
Ylist = np.linspace(0,2000,161)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = 1.*np.cumsum(dl)

maps = [Xlist,Ylist,Zlist]

lo = np.array([ 0, 0, 0]) 
hi = np.array([ 2000, 2000, 50])   # highest lat, highest lon
#lo = np.array([ 0, 0, 0]) 
#hi = np.array([ 10000, 4000, 50])   # highest lat, highest lon

[X,Y,Z] = myfun.meshgrid2(Xlist,Ylist,Zlist)
Y = np.reshape(Y,(np.size(Y),))
X = np.reshape(X,(np.size(X),))
Z = np.reshape(Z,(np.size(Z),))

xn = len(Xlist)
yn = len(Ylist)
zn = len(Zlist)

dx = np.gradient(Xlist)
dy = np.gradient(Ylist)
dz = np.gradient(Zlist)

dt = 360
#dt = 1200
time = np.asarray(range(dayi,dayf,days))

# initial particles position

x0 = range(0,2000,25)
y0 = range(0,2000,25)
z0 = range(0,50,10)

dx = 5
dy = 5

xt = []
yt = []
zt = []

for z in z0:
 print z
 for x in x0:
  for y in y0:
   for p in range(3):
    if p == 0:
     xt.append(x)
     yt.append(y)
     zt.append(z)
    if p == 1:
     xt.append(x+dx)
     yt.append(y)
     zt.append(z)
    if p == 2:
     xt.append(x)
     yt.append(y+dy)
     zt.append(z)

x0 = np.asarray(xt)
y0 = np.asarray(yt)
z0 = np.asarray(zt)
print x0.shape

#x0 = range(3000,4000,50)
#y0 = range(2000,3000,50)
#z0 = range(0,30,1)
xp = len(x0)
yp = len(y0)
zp = len(z0)

pt = xp

#[z0,y0,x0] = myfun.meshgrid2(z0,y0,x0)
#x0 = np.reshape(x0, (np.size(x0)))
#y0 = np.reshape(y0, (np.size(y0)))
#z0 = np.reshape(z0, (np.size(z0)))

#levels = np.zeros(x0.shape) + 1.
#levels[np.where(z0 != 2)] = np.nan

#x0 = lo[0] + np.random.uniform( size=(pt) ) * (hi[0] - lo[0])
#y0 = lo[1] + np.random.uniform( size=(pt) ) * (hi[1] - lo[1])
#z0 = lo[2] + np.random.uniform( size=(pt) ) * (hi[2] - lo[2])
#z0 = z0*0-1.

x = np.zeros((pt))
y = np.zeros((pt))
z = np.zeros((pt))

## ADVECT PARTICLES

filename = './traj_'+label+'_'+str(dayi)+'_'+str(dayf)+'.csv'
filename = './traj_'+label+'_'+str(dayi)+'_'+str(dayf)+'_2D_b.csv'
print filename

fd = open(filename,'wb')

for p in range(pt):
 fd.write(str(x0[p])+', '+str(y0[p])+', '+str(-1.*z0[p])+', '+str(time[-1])+'\n')

for t in range(0,len(time)-1):
 print 'time:', time[t],time[t+1],t
 tlabel = str(time[t])
 while len(tlabel) < 3: tlabel = '0'+tlabel

 file0 = path+'Velocity_CG_0_'+label+'_'+str(time[t])+'.csv'
 file1 = path+'Velocity_CG_1_'+label+'_'+str(time[t])+'.csv'
 file2 = path+'Velocity_CG_2_'+label+'_'+str(time[t])+'.csv'

 Ut0 = fio.read_Scalar(file0,xn,yn,zn)
 Vt0 = fio.read_Scalar(file1,xn,yn,zn)
 Wt0 = 0*Ut0 #-1.*fio.read_Scalar(file2,xn,yn,zn)

 file0 = path+'Velocity_CG_0_'+label+'_'+str(time[t+1])+'.csv'
 file1 = path+'Velocity_CG_1_'+label+'_'+str(time[t+1])+'.csv'
 file2 = path+'Velocity_CG_2_'+label+'_'+str(time[t+1])+'.csv'

 Ut1 = fio.read_Scalar(file0,xn,yn,zn)
 Vt1 = fio.read_Scalar(file1,xn,yn,zn)
 Wt1 = 0*Ut1 #-1.*fio.read_Scalar(file2,xn,yn,zn)

 # subcycling
 nt = 20 
 ds = 1.*dt / nt
# for st in range(nt+1):
#  print st
#  Us0 = (Ut1*st + Ut0*(nt-st))/(nt)
#  Us1 = (Ut1*(st+1) + Ut0*(nt-st-1))/(nt) 
#  Vs0 = (Vt1*st + Vt0*(nt-st))/(nt)
#  Vs1 = (Vt1*(st+1) + Vt0*(nt-st-1))/(nt) 
#  Ws0 = (Wt1*st + Wt0*(nt-st))/(nt)
#  Ws1 = (Wt1*(st+1) + Wt0*(nt-st-1))/(nt) 
 # x0,y0,z0 = advect_functions.RK4(x0,y0,z0,Us0,Vs0,Ws0,Us1,Vs1,Ws1,lo,hi,maps,ds)

 x0,y0,z0 = advect_functions.RK4(x0,y0,z0,Ut0,Vt0,Wt0,Ut1,Vt1,Wt1,lo,hi,maps,dt)
 #x0,y0,z0 = advect_functions.EULER(x0,y0,z0,Ut0,Vt0,Wt0,lo,hi,maps,dt)

 x0,y0,z0 = advect_functions.pBC(x0,y0,z0,lo,hi)
# x1,y1,z1 = x0,y0,z0

# write

 for p in range(pt):
  fd.write(str(x0[p])+', '+str(y0[p])+', '+str(-1.*z0[p])+', '+str(time[t])+'\n')

fd.close()
 

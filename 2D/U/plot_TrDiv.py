import os, sys
import myfun
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import lagrangian_stats
import fio

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

label = 'm_25_1_particles'
dayi  = 980
dayf  = 990
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

# ML exp

Xlist = np.linspace(0,2000,161)
Ylist = np.linspace(0,2000,161)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = np.cumsum(dl)

xn = len(Xlist)
yn = len(Ylist)
zn = len(Zlist)

dx, dy = np.meshgrid(np.gradient(Xlist),np.gradient(Ylist))

for time in range(dayi,dayf,days):
 print 'time:', time
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #Temperature_CG_m_50_6e_9.csv
 fileU = path+'Velocity_CG_0_'+label+'_'+str(time+480)+'.csv'
 fileV = path+'Velocity_CG_1_'+label+'_'+str(time+480)+'.csv'
 fileT = '../../Detectors/Tracer_CG/Tracer_1_CG_'+label+'_'+str(time-dayi)+'.csv'
 print fileT
 file1 = 'DivTr_'+label+'_'+str(time)
 #

 U = fio.read_Scalar(fileU,xn,yn,zn)
 V = fio.read_Scalar(fileV,xn,yn,zn)
 T = fio.read_Scalar(fileT,xn,yn,zn)

 for k in range(1):
  dU  = np.asarray(np.gradient(U[:,:,k]))
  dV  = np.asarray(np.gradient(V[:,:,k]))
  Div = dU[0,:,:]/dx + dV[1,:,:]/dy
  fig = plt.figure(figsize=(8,8))
  plt.contourf(Xlist/1000,Ylist/1000,Div,np.linspace(-0.0008,0.0008,30),extend='both',cmap=plt.cm.PiYG)
  plt.colorbar(ticks=np.linspace(-0.0008,0.0008,5))
  plt.contour(Xlist/1000,Ylist/1000,np.mean(T[:,:,:],2),10,colors='k')
  plt.xlabel('X [km]',fontsize=18)
  plt.ylabel('Y [km]',fontsize=18)
  plt.axes().set_aspect('equal')
  plt.xlim(0,2)
  plt.ylim(0,2)
  #plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
  #plt.yticks(fontsize=16)
  plt.savefig('./plot/'+label+'/'+file1+'_'+str(Zlist[k])+'.eps',bbox_inches='tight')
  print       './plot/'+label+'/'+file1+'_'+str(Zlist[k])+'.eps'
  plt.close()
##

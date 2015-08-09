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
dayi  = 481
dayf  = 481+481/4
days  = 2

label = 'm_25_2_512'
dayi  = 0
dayf  = 60
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

#Xlist = np.linspace(0,2000,161)
#Ylist = np.linspace(0,2000,161)
Xlist = np.linspace(0,8000,641)
Ylist = np.linspace(0,4000,321)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = np.cumsum(dl)

xn = len(Xlist)
yn = len(Ylist)
zn = len(Zlist)

Nu = []
Ra = []

alpha = 0.00005
beta = 0.000146056
nu = 0.0005

time = range(dayi,dayf,days)
days = range(4)
day = range(0,240,24)
day = range(0,60,6)

for d in days:
 T = []
 W = []
 DT= []
 for t in day:
  print 'time:', t
  tlabel = str(t)
  while len(tlabel) < 3: tlabel = '0'+tlabel
  #Temperature_CG_m_50_6e_9.csv
  #fileT = '../RST/Temperature_CG/Temperature_CG_'+label+'_'+str(t+d*240+481)+'.csv'
  #fileW = path+'Velocity_CG_2_'+label+'_'+str(t+d*240+481)+'.csv'
  fileT = '../RST/Temperature_CG/Temperature_CG_'+label+'_'+str(t+d*60)+'.csv'
  fileW = path+'Velocity_CG_2_'+label+'_'+str(t+d*60)+'.csv'
  print fileT
  #
  Ttemp = fio.read_Scalar(fileT,xn,yn,zn)+274.5
  T.append(Ttemp)
  W.append(fio.read_Scalar(fileW,xn,yn,zn))
  DT.append(np.mean(np.mean(Ttemp[:,:,-1]-Ttemp[:,:,0],0),0))


 T = np.asarray(T) 
 W = np.asarray(W)
 DT= np.asarray(DT)
 T = np.mean(T,3) 
 W = np.mean(W,3)
 DT= np.mean(DT)
 H = 50
 
 Nu.append(1 + np.mean(np.mean(np.mean(W[:,:,:]*T[:,:,:],0),0),0)/DT*H/alpha)
 Ra.append(beta*DT*9.81*H**3/(alpha*nu))
 print Nu, Ra

fig = plt.figure()
plt.scatter(np.log10(Ra),np.log10(Nu))
plt.savefig('./plot/'+label+'/Nusselt_Rayleigh_'+label+'.eps',bbox_inches='tight')
print       './plot/'+label+'/Nusselt_Rayleigh_'+label+'.eps'
plt.close()
##

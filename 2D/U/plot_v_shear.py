import os, sys
import myfun
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import lagrangian_stats
import fio

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

label_b = 'm_25_1b_particles'
label_w = 'm_25_2b_particles'
label = 'm_25_2b_particles'
dayi  = 60
dayf  = 62
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

Xlist_b = np.linspace(0,8000,641)
Ylist_b = np.linspace(0,8000,641)
Xlist_w = np.linspace(0,8000,641)
Ylist_w = np.linspace(0,8000,641)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = np.cumsum(dl)

xn_w = len(Xlist_w)
yn_w = len(Ylist_w)
xn_b = len(Xlist_b)
yn_b = len(Ylist_b)
zn = len(Zlist)

for time in range(dayi,dayf,days):
 print 'time:', time
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #Temperature_CG_m_50_6e_9.csv
 fileU_w = path+'Velocity_CG_0_'+label_w+'_'+str(time)+'.csv'
 fileV_w = path+'Velocity_CG_1_'+label_w+'_'+str(time)+'.csv'
 fileU_b = path+'Velocity_CG_0_'+label_b+'_'+str(time)+'.csv'
 fileV_b = path+'Velocity_CG_1_'+label_b+'_'+str(time)+'.csv'
 file1 = 'Shear_v_'+label+'_'+str(time)
 #

 U_w = fio.read_Scalar(fileU_w,xn_w,yn_w,zn)
 V_w = fio.read_Scalar(fileV_w,xn_w,yn_w,zn)
 U_b = fio.read_Scalar(fileU_b,xn_b,yn_b,zn)
 V_b = fio.read_Scalar(fileV_b,xn_b,yn_b,zn)

 S_w = []
 S_b = []

 for i in range(xn_w):
  for j in range(yn_w):
   S_w.append((np.gradient(U_w[i,j,:])/np.gradient(Zlist))**2+(np.gradient(V_w[i,j,:])/np.gradient(Zlist))**2)

 for i in range(xn_b):
  for j in range(yn_b):
   S_b.append((np.gradient(U_b[i,j,:])/np.gradient(Zlist))**2+(np.gradient(V_b[i,j,:])/np.gradient(Zlist))**2)


 S_w = np.asarray(S_w)
 S_b = np.asarray(S_b)

 fig = plt.figure(figsize=(7,8))
 pSw, = plt.plot(np.log10(np.mean(S_w,0)),-1*Zlist,'k',linewidth=2)
# pSb, = plt.plot(np.log10(np.mean(S_b,0)),-1*Zlist,'k--',linewidth=2)
# plt.legend([pSw,pSb],['$BW$','$B$'],loc=4,fontsize=20)
 plt.xlabel('$2log(\partial \mathbf{u}/\partial z)$ [$s^{-2}$]',fontsize=22)
 plt.ylabel('Depth [$m$]',fontsize=22)
 plt.xticks(np.linspace(-7,-2,6),np.linspace(-7,-2,6).astype(int),fontsize=18)
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
 plt.yticks(fontsize=18)

 plt.tight_layout()
 plt.savefig('./plot/'+label+'/'+file1+'.eps',bbox_inches='tight')
 print       './plot/'+label+'/'+file1+'.eps'
 plt.close()
##

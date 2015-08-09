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

label_b = 'm_25_1b'
label_w = 'm_25_2b'
label = 'm_25_2b'
dayi  = 36
dayf  = 49
days  = 12

#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

path = './Temperature_CG/'

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
 fileT_w = path+'Temperature_CG_'+label_w+'_'+str(time)+'.csv'
 fileT_b = path+'Temperature_CG_'+label_b+'_'+str(time)+'.csv'
 file1 = 'T_v_'+label+'_'+str(time)
 file2 = 'T_vgrad_'+label+'_'+str(time)
 #

 T_w = fio.read_Scalar(fileT_w,xn_w,yn_w,zn)
 T_b = fio.read_Scalar(fileT_b,xn_b,yn_b,zn)

 Tm_w = []
 Tm_b = []

 Tm_w = np.mean(np.mean(T_w,0),0)
 Tm_b = np.mean(np.mean(T_b,0),0)

 Tg_b = np.gradient(Tm_b)/np.gradient(Zlist)
 Tg_w = np.gradient(Tm_w)/np.gradient(Zlist)


 Tmin = 19.2
 Tmax = 20.2

 fig = plt.figure(figsize=(5,8))
 pTw, = plt.plot(Tm_w,-1*Zlist,'k',linewidth=2)
 pTb, = plt.plot(Tm_b,-1*Zlist,'k--',linewidth=2)
 plt.legend([pTw,pTb],['$BW$','$B$'],loc=3,fontsize=20)
 plt.xlabel('$T$ [$^{\circ}C$]',fontsize=22)
 plt.ylabel('Depth [$m$]',fontsize=22)
 plt.yticks(fontsize=18)
 plt.xticks(np.linspace(Tmin,Tmax,6),fontsize=18)
 plt.xlim(Tmin,Tmax)
 
 plt.tight_layout()
 plt.savefig('./plot/'+label+'/'+file1+'.eps',bbox_inches='tight')
 print       './plot/'+label+'/'+file1+'.eps'
 plt.close()


 fig = plt.figure(figsize=(5,8))
 pTw, = plt.plot(Tg_w,-1*Zlist,'k',linewidth=2)
 pTb, = plt.plot(Tg_b,-1*Zlist,'k--',linewidth=2)
 plt.legend([pTw,pTb],['$BW$','$B$'],loc=3,fontsize=20)
 plt.xlabel('$T$ [$^{\circ}C$]',fontsize=22)
 plt.ylabel('Depth [$m$]',fontsize=22)
 plt.yticks(fontsize=18)
# plt.xticks(np.linspace(Tmin,Tmax,6),fontsize=18)
# plt.xlim(Tmin,Tmax)

 plt.tight_layout()
 plt.savefig('./plot/'+label+'/'+file2+'.eps',bbox_inches='tight')
 print       './plot/'+label+'/'+file2+'.eps'
 plt.close()

##

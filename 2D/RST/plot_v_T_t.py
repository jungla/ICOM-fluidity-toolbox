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

label = 'm_25_1b'
dayi  = 0 
dayf  = 73
days  = 1

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

Xlist = np.linspace(0,8000,641)
Ylist = np.linspace(0,8000,641)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = np.cumsum(dl)

xn = len(Xlist)
yn = len(Ylist)
zn = len(Zlist)

time = np.asarray(range(dayi,dayf,days))

Tm = []

for t in range(dayi,dayf,days):
 print 'time:', t
 tlabel = str(t)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #Temperature_CG_m_50_6e_9.csv
 if t > 48:
  fileT = path+'Temperature_CG_'+label+'_day3_'+str(t-48)+'.csv'
 else:
  fileT = path+'Temperature_CG_'+label+'_'+str(t)+'.csv'
 #

 T = fio.read_Scalar(fileT,xn,yn,zn)


 Tm.append(np.mean(np.mean(T,0),0))

Tm = np.transpose(np.asarray(Tm))

v = np.linspace(19,20,30)
vl = np.linspace(19,20,6)

fig = plt.figure(figsize=(8,5))
pTw = plt.contourf(time,-1*Zlist,Tm,v,extend='both')
plt.xlabel('Time [$hr$]',fontsize=18)
plt.ylabel('Depth [$m$]',fontsize=18)
plt.yticks(fontsize=16)
plt.xticks(np.linspace(dayi,dayf-1,13),np.linspace(dayi,dayf-1,13).astype(int),fontsize=16)
#plt.xlim(Tmin,Tmax)
cb = plt.colorbar(ticks=vl)
cb.ax.tick_params(labelsize=14)

plt.tight_layout()
plt.savefig('./plot/'+label+'/T_v_t_'+label+'.eps',bbox_inches='tight')
print       './plot/'+label+'/T_v_t_'+label+'.eps'
plt.close()


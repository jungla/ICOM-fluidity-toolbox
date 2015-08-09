import os, sys
import csv
import myfun
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import lagrangian_stats
import fio

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

label = 'm_25_1b_particles'

basename = 'drate_'+label

dayi = 60
dayf = 240
days = 1
timeD = np.asarray(range(0,86400*3,1440))

path = './Velocity_CG/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

# dimensions archives

# ML exp

dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = -1*np.cumsum(dl)

drateD = np.zeros((len(timeD),len(Zlist)))

for t in range(len(drateD)):
# print 'read drate', t
 with open('../../2D/U/drate_3+1day/z/'+basename+'_'+str(t+dayi)+'_z.csv', 'rb') as csvfile:
  spamreader = csv.reader(csvfile)
  for row in spamreader:
   drateD[t,:] = row[::-1]

FW_m = -11
FW_M = -8

plt.figure(figsize=(8,4))
plt.contourf(timeD/3600.,Zlist,np.log10(np.rot90(drateD)),np.linspace(FW_m,FW_M,30),extend='both')
#plt.contourf(time,Zlist,np.log10(FW_t25),np.linspace(FW_m,FW_M,30),extend='both')
cb = plt.colorbar(ticks=np.linspace(FW_m,FW_M,7))
cb.ax.tick_params(labelsize=14)

#plt.colorbar()
#plt.plot(time,mld_25,'k')
plt.xlabel('Time [$hr$]',fontsize=18)
plt.ylabel('Depth [$m$]',fontsize=18)
plt.xticks(np.linspace(timeD[0]/3600.,timeD[-1]/3600.,7),np.linspace(72,24*3+72,7).astype(int),fontsize=16)
#plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('./plot/'+label+'/drate_'+label+'.eps',bbox_inches='tight')
print       './plot/'+label+'/drate_'+label+'.eps'
plt.close()


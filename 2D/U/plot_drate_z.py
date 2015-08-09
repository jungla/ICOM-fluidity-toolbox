import os, sys
import myfun
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import lagrangian_stats
import fio

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

#label = 'm_25_1_particles'
#label_25 = 'm_25_1_particles'

label = 'm_25_2_512'
label_25 = 'm_25_2_512'

basename = 'mli' 

#dayi  = 0+481
#days  = 8
#dayf  = 240 + days + dayi

dayi  = 0
days  = 2
dayf  = 60 + days

time = range(dayi,dayf,days)
print time
path = './Velocity_CG/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

# dimensions archives

# ML exp

Ylist_25 = np.linspace(0,4000,321)
Xlist_25 = np.linspace(0,10000,801)

#Ylist_25 = np.linspace(0,2000,161)
#Xlist_25 = np.linspace(0,2000,161)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = -1*np.cumsum(dl)

xn_25 = len(Xlist_25)
yn_25 = len(Ylist_25)
zn = len(Zlist)

dx_25 = np.gradient(Xlist_25) 

dz_25 = np.gradient(Zlist) 


FW_25 = np.zeros((yn_25,xn_25,zn,len(range(dayi,dayf,days))))

#mld_25 = np.zeros(len(range(dayi,dayf,days)))
nu_h = 0.05
nu_v = 0.0005 
for t in range(len(time)):
 print 'time:', time[t]
 tlabel = str(time[t])
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #Velocity_CG_m_50_6e_9.csv
 file0_U = path+'Velocity_CG_0_'+label_25+'_'+str(time[t])+'.csv'
 file0_V = path+'Velocity_CG_1_'+label_25+'_'+str(time[t])+'.csv'
 file0_W = path+'Velocity_CG_2_'+label_25+'_'+str(time[t])+'.csv'
 file1 = 'drate_'+label+'_'+str(time[t])
 file1_25 = 'drate_'+label_25

 U_25 = fio.read_Scalar(file0_U,xn_25,yn_25,zn)
 V_25 = fio.read_Scalar(file0_V,xn_25,yn_25,zn)
 W_25 = fio.read_Scalar(file0_W,xn_25,yn_25,zn)

 for i in range(0,len(Xlist_25),50):
  for j in range(0,len(Ylist_25),50):
   #FW_25[j,i,:,t] = 0.5*nu_h*((np.gradient(U_25[i,j,:]-np.mean(U_25[i,j,:]))/dz_25)**2 + (np.gradient(V_25[i,j,:]-np.mean(V_25[i,j,:]))/dz_25)**2) + 0.5*nu_v*((np.gradient(W_25[i,j,:]-np.mean(W_25[i,j,:]))/dz_25)**2)
   FW_25[j,i,:,t] = 0.5*nu_h*((np.gradient(U_25[i,j,:])/dz_25)**2 + (np.gradient(V_25[i,j,:])/dz_25)**2) + 0.5*nu_v*(np.gradient(W_25[i,j,:])/dz_25)**2

 FW_t25 = np.mean(np.mean(FW_25,0),0)

# plt.figure(figsize=(4,8))
# p25, = plt.semilogx(7.5*0.05*FW_t25[:,t],Zlist,'k--',linewidth=2)

FW_m = -11
FW_M = -7

plt.figure(figsize=(8,4))
plt.contourf(time,Zlist,np.log10(FW_t25),np.linspace(FW_m,FW_M,30),extend='both')
plt.colorbar(ticks=np.linspace(FW_m,FW_M,7))
#plt.colorbar()
#plt.plot(time,mld_25,'k')
plt.xlabel('Time [hr]',fontsize=18)
plt.ylabel('Depth [m]',fontsize=18)
plt.xticks(np.linspace(dayi,dayf-days,13),np.linspace(48,72,13).astype(int))
#plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
#plt.yticks(fontsize=16)
plt.savefig('./plot/'+label+'/'+file1_25+'.eps',bbox_inches='tight')
print       './plot/'+label+'/'+file1_25+'.eps'
plt.close()


import csv
f = open('drate_'+label+'.csv','w')
writer = csv.writer(f)
row = []
row.append('time')

for k in Zlist:
 row.append(k)

writer.writerow(row)
time = np.asarray(time)*1440 + 48*3600

for t in range(len(time)):
 row = []
 #row.append((time[t]-1)*360)
 row.append((time[t]))
 for k in range(len(Zlist)):
  row.append(FW_t25[k,t])
 writer.writerow(row)

f.close()



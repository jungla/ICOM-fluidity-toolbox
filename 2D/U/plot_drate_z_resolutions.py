import os, sys
import myfun
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import lagrangian_stats

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

label = 'm_50_6f'
label_50 = 'm_50_6f'
label_25 = 'm_25_1'
label_10 = 'm_10_1'
basename = 'mli' 
dayi  = 24
dayf  = 49
days  = 1

#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

path = './Velocity_CG/'

# dimensions archives

# ML exp

Xlist_50 = np.linspace(0,2000,41)
Ylist_50 = np.linspace(0,2000,41)
Xlist_25 = np.linspace(0,2000,81)
Ylist_25 = np.linspace(0,2000,81)
Xlist_10 = np.linspace(0,2000,161)
Ylist_10 = np.linspace(0,2000,161)
Zlist = np.linspace(0,-50,51)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = -1*np.cumsum(dl)

xn_50 = len(Xlist_50)
yn_50 = len(Ylist_50)
xn_25 = len(Xlist_25)
yn_25 = len(Ylist_25)
xn_10 = len(Xlist_10)
yn_10 = len(Ylist_10)
zn = len(Zlist)

dx_50 = np.gradient(Xlist_50) 
dx_25 = np.gradient(Xlist_25) 
dx_10 = np.gradient(Xlist_10) 

dz_50 = np.gradient(Zlist) 
dz_25 = np.gradient(Zlist) 
dz_10 = np.gradient(Zlist) 

time = range(dayi,dayf,days)

FW_50 = np.zeros((yn_50,xn_50,zn,len(range(dayi,dayf,days))))
FW_25 = np.zeros((yn_25,xn_25,zn,len(range(dayi,dayf,days))))
FW_10 = np.zeros((yn_10,xn_10,zn,len(range(dayi,dayf,days))))

mld_50 = np.zeros(len(range(dayi,dayf,days)))
mld_25 = np.zeros(len(range(dayi,dayf,days)))
mld_10 = np.zeros(len(range(dayi,dayf,days)))

for t in range(len(time)):
 print 'time:', time[t]
 tlabel = str(time[t])
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #Velocity_CG_m_50_6e_9.csv
 file0_50 = path+'Velocity_CG_0_'+label_50+'_'+str(time[t])+'.csv'
 file0_25 = path+'Velocity_CG_0_'+label_25+'_'+str(time[t])+'.csv'
 file0_10 = path+'Velocity_CG_0_'+label_10+'_'+str(time[t])+'.csv'
 file1 = 'drate_'+label+'_'+str(time[t])
 file1_50 = 'drate_'+label_50
 file1_25 = 'drate_'+label_25
 file1_10 = 'drate_'+label_10

 W_50 = lagrangian_stats.read_Scalar(file0_50,zn,yn_50,xn_50)
 W_25 = lagrangian_stats.read_Scalar(file0_25,zn,yn_25,xn_25)
 W_10 = lagrangian_stats.read_Scalar(file0_10,zn,yn_10,xn_10)

 for i in range(len(Xlist_50)):
  for j in range(len(Ylist_50)):
   FW_50[j,i,:,t] = (np.gradient(W_50[:,j,i])/dz_50)**2
 for i in range(len(Xlist_25)):
  for j in range(len(Ylist_25)):
   FW_25[j,i,:,t] = (np.gradient(W_25[:,j,i])/dz_25)**2
 for i in range(len(Xlist_10)):
  for j in range(len(Ylist_10)):
   FW_10[j,i,:,t] = (np.gradient(W_10[:,j,i])/dz_10)**2

 plt.figure(figsize=(4,8))

 FW_t50 = np.mean(np.mean(FW_50,0),0)
 FW_t25 = np.mean(np.mean(FW_25,0),0)
 FW_t10 = np.mean(np.mean(FW_10,0),0)

 p50, = plt.semilogx(7.5*0.05*FW_t50[:,t],Zlist,'k-',linewidth=2)
 p25, = plt.semilogx(7.5*0.05*FW_t25[:,t],Zlist,'k--',linewidth=2)
 p10, = plt.semilogx(7.5*0.05*FW_t10[:,t],Zlist,'k.-',linewidth=2)

 plt.legend([p50,p25,p10],['50m','25m','10m'],loc=4)
 plt.ylabel(r'depth $[m]$',fontsize=18)
 plt.xlabel(r'$\epsilon [m^2s^{-2}]$',fontsize=18)
 plt.yticks(fontsize=16)
 plt.xticks(fontsize=16)
 plt.savefig('./plot/'+label+'/'+file1+'.eps',bbox_inches='tight')
 print       './plot/'+label+'/'+file1+'.eps'
 plt.close()
  #
 path_T = '../RST/Temperature_CG/'

 file0_50 = path_T+'Temperature_CG_'+label_50+'_'+str(time[t])+'.csv'
 file0_25 = path_T+'Temperature_CG_'+label_25+'_'+str(time[t])+'.csv'
 file0_10 = path_T+'Temperature_CG_'+label_10+'_'+str(time[t])+'.csv'

 T_10 = lagrangian_stats.read_Scalar(file0_10,zn,yn_10,xn_10)
 T_25 = lagrangian_stats.read_Scalar(file0_25,zn,yn_25,xn_25)
 T_50 = lagrangian_stats.read_Scalar(file0_50,zn,yn_50,xn_50)

 mld_t = []
 for x in range(len(Xlist_50)):
  for y in range(len(Ylist_50)):
   ml = T_50[:,x,y]
   mls = np.cumsum(ml)/range(1,len(ml)+1)
   mlst, = np.where(mls>=ml)
   mld_t.append(Zlist[mlst[len(mlst)-1]])
 mld_50[t] = np.mean(mld_t)

 mld_t = []
 for x in range(len(Xlist_25)):
  for y in range(len(Ylist_25)):
   ml = T_25[:,x,y]
   mls = np.cumsum(ml)/range(1,len(ml)+1)
   mlst, = np.where(mls>=ml)
   mld_t.append(Zlist[mlst[len(mlst)-1]])
 mld_25[t] = np.mean(mld_t)

 mld_t = []
 for x in range(len(Xlist_10)):
  for y in range(len(Ylist_10)):
   ml = T_10[:,x,y]
   mls = np.cumsum(ml)/range(1,len(ml)+1)
   mlst, = np.where(mls>=ml)
   mld_t.append(Zlist[mlst[len(mlst)-1]])
 mld_10[t] = np.mean(mld_t)

FW_m = -9
FW_M = -5

plt.figure(figsize=(8,4))
plt.contourf(time,Zlist,np.log10(FW_t50),np.linspace(FW_m,FW_M,30),extend='both')
plt.colorbar()
plt.plot(time,mld_50,'k')
plt.xlabel('Time',fontsize=18)
plt.ylabel('Depth',fontsize=18)
plt.xlim([24,48])
#plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
#plt.yticks(fontsize=16)
plt.savefig('./plot/'+label+'/'+file1_50+'.eps',bbox_inches='tight')
print       './plot/'+label+'/'+file1_50+'.eps'
plt.close() 
###

plt.figure(figsize=(8,4))
plt.contourf(time,Zlist,np.log10(FW_t25),np.linspace(FW_m,FW_M,30),extend='both')
plt.colorbar()
plt.plot(time,mld_25,'k')
plt.xlabel('Time',fontsize=18)
plt.ylabel('Depth',fontsize=18)
plt.xlim([24,48])
#plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
#plt.yticks(fontsize=16)
plt.savefig('./plot/'+label+'/'+file1_25+'.eps',bbox_inches='tight')
print       './plot/'+label+'/'+file1_25+'.eps'
plt.close()

plt.figure(figsize=(8,4))
plt.contourf(time,Zlist,np.log10(FW_t10),np.linspace(FW_m,FW_M,30),extend='both')
plt.colorbar()
plt.plot(time,mld_10,'k')
plt.xlabel('Time',fontsize=18)
plt.ylabel('Depth',fontsize=18)
plt.xlim([24,48])
#plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
#plt.yticks(fontsize=16)
plt.savefig('./plot/'+label+'/'+file1_10+'.eps',bbox_inches='tight')
print       './plot/'+label+'/'+file1_10+'.eps'
plt.close()

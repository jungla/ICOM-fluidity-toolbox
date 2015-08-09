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
label_25 = 'm_25_1b_particles'

#label = 'm_25_2_512'
#label_25 = 'm_25_2_512'

basename = 'mli' 

#dayi  = 0+481
#days  = 8
#dayf  = 240 + days + dayi

dayi  = 0
days  = 1
dayf  = 60 + days

time = range(dayi,dayf,days)

print time

path = './Velocity_CG/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

# dimensions archives

# ML exp

#Ylist_25 = np.linspace(0,4000,321)
#Xlist_25 = np.linspace(0,10000,801)

Ylist_25 = np.linspace(0,8000,641)
Xlist_25 = np.linspace(0,8000,641)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = -1*np.cumsum(dl)

xn_25 = len(Xlist_25)
yn_25 = len(Ylist_25)
zn = len(Zlist)

dx = np.gradient(Xlist_25) 
dy = np.gradient(Ylist_25) 
dz = np.gradient(Zlist) 


FW_25 = np.zeros((yn_25,xn_25,zn,len(time)))

#mld_25 = np.zeros(len(range(dayi,dayf,days)))
nu_h = 0.05
nu_v = 0.0005 

epsilon = np.zeros((xn_25,yn_25,zn,len(time)))

mU = np.zeros((xn_25,yn_25,zn))
mV = np.zeros((xn_25,yn_25,zn))
mW = np.zeros((xn_25,yn_25,zn))

time1 = 1./len(time)

U = np.zeros((xn_25,yn_25,zn,len(time)))
V = np.zeros((xn_25,yn_25,zn,len(time)))
W = np.zeros((xn_25,yn_25,zn,len(time)))

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

 U[:,:,:,t] = fio.read_Scalar(file0_U,xn_25,yn_25,zn)
 V[:,:,:,t] = fio.read_Scalar(file0_V,xn_25,yn_25,zn)
 W[:,:,:,t] = fio.read_Scalar(file0_W,xn_25,yn_25,zn)

# remove mean
 mU = mU + U[:,:,:,t]*time1    
 mV = mV + V[:,:,:,t]*time1    
 mW = mW + W[:,:,:,t]*time1    

for t in xrange(len(time)):
 print 'time:', time[t]

 Ua = U[:,:,:,t] - mU
 Va = V[:,:,:,t] - mV
 Wa = W[:,:,:,t] - mW

 # derivatives

 dUdx = np.zeros((xn_25,yn_25,zn))
# dVdx = np.zeros((xn_25,yn_25,zn))
# dWdx = np.zeros((xn_25,yn_25,zn))

# dUdy = np.zeros((xn_25,yn_25,zn))
 dVdy = np.zeros((xn_25,yn_25,zn))
# dWdy = np.zeros((xn_25,yn_25,zn))

# dUdz = np.zeros((xn_25,yn_25,zn))
# dVdz = np.zeros((xn_25,yn_25,zn))
 dWdz = np.zeros((xn_25,yn_25,zn))

 print 'dUdz, dVdz, dWdz'

 for i in xrange(0,len(Xlist_25),1):
  for j in xrange(0,len(Ylist_25),1):
#   dUdz[i,j,:] = np.gradient(U[i,j,:])/dz 
#   dVdz[i,j,:] = np.gradient(V[i,j,:])/dz 
   dWdz[i,j,:] = np.gradient(Wa[i,j,:])/dz 
 
 print 'dUdy, dVdy, dWdy'
 
 for i in xrange(0,len(Xlist_25),1):
  for k in xrange(len(Zlist)):
#   dUdy[i,:,k] = np.gradient(U[i,:,k])/dy 
   dVdy[i,:,k] = np.gradient(Va[i,:,k])/dy 
#   dWdy[i,:,k] = np.gradient(W[i,:,k])/dy 
 
 print 'dUdx, dVdx, dWdx'
 
 for j in xrange(0,len(Ylist_25),1):
  for k in xrange(len(Zlist)):
   dUdx[:,j,k] = np.gradient(Ua[:,j,k])/dx 
#   dVdx[:,j,k] = np.gradient(V[:,j,k])/dx 
#   dWdx[:,j,k] = np.gradient(W[:,j,k])/dx 

# epsilon[:,:,:,t] = 2*(dUdx)**2 + 2*(dVdy)**2 + 2*(dWdz)**2 + (dUdy)**2 + (dUdz)**2 + (dVdx)**2 + (dVdz)**2 + (dWdx)**2 + (dWdy)**2 + dUdy*dVdx + dUdz*dWdx + dVdx*dUdy + dVdz*dWdy + dWdx*dUdz + dWdy*dVdz
 epsilon[:,:,:,t] = nu_h*2*(dUdx)**2 + nu_h*2*(dVdy)**2 + nu_v*2*(dWdz)**2

   #FW_25[j,i,:,t] = 0.5*nu_h*((np.gradient(U_25[i,j,:]-np.mean(U_25[i,j,:]))/dz_25)**2 + (np.gradient(V_25[i,j,:]-np.mean(V_25[i,j,:]))/dz_25)**2) + 0.5*nu_v*((np.gradient(W_25[i,j,:]-np.mean(W_25[i,j,:]))/dz_25)**2)
#   FW_25[j,i,:,t] = 0.5*nu_h*((np.gradient(U_25[i,j,:])/dz_25)**2 + (np.gradient(V_25[i,j,:])/dz_25)**2) + 0.5*nu_v*(np.gradient(W_25[i,j,:])/dz_25)**2

 FW_t25 = np.mean(np.mean(epsilon,0),0)

 f = open('drate_'+label+'_'+str(time[t])+'_3D.csv','w')
 writer = csv.writer(f)
 
 #row.append((time[t]-1)*360)
 for k in range(zn):
  for j in range(yn_25):
   row = []
   for i in range(xn_25):
    row.append(epsilon[i,j,k,t])
   writer.writerow(row) 
 f.close()


# plt.figure(figsize=(4,8))
# p25, = plt.semilogx(7.5*0.05*FW_t25[:,t],Zlist,'k--',linewidth=2)

time = range(dayi,dayf,days)

FW_m = -11
FW_M = -8

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



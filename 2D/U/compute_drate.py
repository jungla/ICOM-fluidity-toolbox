import os, sys
import csv
import myfun
import numpy as np
import lagrangian_stats
import fio

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

#label = 'm_25_1b_particles'
#label_25 = 'm_25_1b_particles'
label = 'm_25_2_512'
label_25 = 'm_25_2_512'

basename = 'mli' 

#dayi  = 0+481
#days  = 8
#dayf  = 240 + days + dayi

dayi  = 60
days  = 1
dayf  = 120 + days

time = range(dayi,dayf,days)

print time

path = './Velocity_CG/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

# dimensions archives

# ML exp

Ylist_25 = np.linspace(0,4000,321)
Xlist_25 = np.linspace(0,10000,801)

#Ylist_25 = np.linspace(0,8000,641)
#Xlist_25 = np.linspace(0,8000,641)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = -1*np.cumsum(dl)

xn_25 = len(Xlist_25)
yn_25 = len(Ylist_25)
zn = len(Zlist)

dx = np.gradient(Xlist_25) 
dy = np.gradient(Ylist_25) 
dz = np.gradient(Zlist) 

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

for t in range(len(time)):
 print 'time:', time[t]

 Ua = U[:,:,:,t] - mU
 Va = V[:,:,:,t] - mV
 Wa = W[:,:,:,t] - mW

 # derivatives

 dUdx = np.zeros((xn_25,yn_25,zn))
 dVdy = np.zeros((xn_25,yn_25,zn))
 dWdz = np.zeros((xn_25,yn_25,zn))

 print 'dUdz, dVdz, dWdz'

 for i in range(len(Xlist_25)):
  for j in range(len(Ylist_25)):
   dWdz[i,j,:] = np.gradient(Wa[i,j,:])/dz 
 
 print 'dUdy, dVdy, dWdy'
 
 for i in range(len(Xlist_25)):
  for k in range(len(Zlist)):
   dVdy[i,:,k] = np.gradient(Va[i,:,k])/dy 
 
 print 'dUdx, dVdx, dWdx'
 
 for j in range(len(Ylist_25)):
  for k in range(len(Zlist)):
   dUdx[:,j,k] = np.gradient(Ua[:,j,k])/dx 

 epsilon[:,:,:,t] = nu_h*2*(dUdx)**2 + nu_h*2*(dVdy)**2 + nu_v*2*(dWdz)**2

 FW_t25 = np.mean(np.mean(epsilon,0),0)

 f = open('drate_'+label+'_'+str(time[t])+'_3D.csv','w')
 writer = csv.writer(f)
 
 #row.append((time[t]-1)*360)
 for k in range(zn):
  for j in range(xn_25):
   row = []
   for i in range(xn_25):
    row.append(epsilon[i,j,k,t])
   writer.writerow(row) 
 f.close()


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
dayf  = dayi + 6*240
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

Nu = []
Ra = []
Q0 = []
Q1 = []

alpha = 0.00005
beta = 0.000146056/1027.
nu = 0.0005

time = range(dayi,dayf,days)
days = range(4)
day = range(0,240,24)

Q_s = 1000
H_s = 0.58
H_s = 0.67
b0 = 0.35
b1 = 23.0
b0 = 1.0
b1 = 17.0

def Q_surf(t):
 Q_0 = 0
 global np
 if t > 0:
  time = t*360.0/3600.0%24/6
  if time >=0 and time < 2:
   Q_0 = 0
  if time >=2 and time < 3:
   Q_0 = Q_s*(time-2)
  if time >=3 and time < 4:
   Q_0 = Q_s - Q_s*(time-3)
 else:
  Q_0 = 0
 print 'Q_0:', Q_0/(1027.0*4000.0)
 Q = Q_0/(1027.0*4000.0)-245.64/(1027.0*4000.0)
 return Q

for t in time:
 if t*360.0/3600.0%24/6 >= 3:
  print 'time:', t
  tlabel = str(t)
  while len(tlabel) < 3: tlabel = '0'+tlabel
  #Temperature_CG_m_50_6e_9.csv
  fileT = '../RST/Temperature_CG/Temperature_CG_'+label+'_'+str(t)+'.csv'
  fileW = path+'Velocity_CG_2_'+label+'_'+str(t)+'.csv'
  print fileT
  #
  T = fio.read_Scalar(fileT,xn,yn,zn)+274.5
#  W = fio.read_Scalar(fileW,xn,yn,zn)
 
  DT = -np.mean(np.mean(T[:,:,0]-T[:,:,-1],0),0)
#  Q0.append(-np.mean(np.mean(T[:,:,0]))*np.mean(np.mean(W[:,:,0])))
  H = 50.
  Q1.append(-Q_surf(t)) # np.mean(np.mean(W[:,:,0]*T[:,:,0],0),0) # -1*Q_surf(t)
  Nu.append(1 + Q1[-1]/DT*H/alpha)
  Ra.append(beta*DT*9.81*H**3/(alpha*nu))
#  print 'Q0:', Q0[-1], 'Q1:', Q1[-1]
 # for k in range(zn):
 #  DT = np.mean(np.mean(T[:,:,k]-T[:,:,0],0),0)
 #  H = Zlist[k]
 #  Q = -1*Q_surf(t)
 #  print Q
 #  Nu.append(1 + Q/DT*H/alpha)
 #  Ra.append(beta*DT*9.81*H**3/(alpha*nu))
  print Nu[-1], Ra[-1]

Nu = np.asarray(Nu)
Ra = np.asarray(Ra)

#Nup = Nu[106:-5]
#Rap = Ra[106:-5]
Nup = Nu[Nu>0]#[106:-5]
Rap = Ra[Nu>0]#[106:-5]
 
fig = plt.figure(figsize=(6,3))
x = np.log10(Rap)
y = x*0.33+0.5
plt.text(np.min(np.log10(Rap[0]))-0.02,y[0],'1/3')
plt.plot(x,y,'k')
plt.scatter(np.log10(Rap),np.log10(Nup),c='k') #range(len(Nup))) #c=range(len(Nup)))
#plt.scatter(np.log10(Rap),np.log10(0.1*Rap**0.33),c='b') #range(len(Nup))) #c=range(len(Nup)))
plt.xlabel('$log(Ra)$',fontsize=18)
plt.ylabel('$log(Nu)$',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.gray()

#plt.ylim(1,2.5)
#x = np.log10(np.asarray(Rap[:]))

#par = np.polyfit(np.log10(np.asarray(Rap[-4:-1])),np.log10(np.asarray(Nup[-4:-1])), 1)
#print 'slope:', par[0]
#plt.plot(x,y,'k',linewidth=2)

#par = np.polyfit(np.log10(np.asarray(Rap[0:7])),np.log10(np.asarray(Nup[0:7])), 1)
#y = x*par[0]+par[1]
#print 'slope', par[0]
#plt.plot(x,y,'k')

#plt.plot(np.log10(x),np.log10(y))
plt.savefig('./plot/'+label+'/Nusselt_Rayleigh_'+label+'_'+str(t)+'.eps',bbox_inches='tight')
print       './plot/'+label+'/Nusselt_Rayleigh_'+label+'_'+str(t)+'.eps'
plt.close()

#plt.plot(Q0)
#plt.plot(Q1)
#plt.scatter(Q0,Q1)
#plt.xlim(np.min(Q0),np.max(Q0))
#plt.ylim(np.min(Q1),np.max(Q1))
#plt.savefig('./plot/'+label+'/Nusselt_Rayleigh_Q_'+label+'_'+str(t)+'.eps',bbox_inches='tight')
#print       './plot/'+label+'/Nusselt_Rayleigh_Q_'+label+'_'+str(t)+'.eps'
#plt.close()
##

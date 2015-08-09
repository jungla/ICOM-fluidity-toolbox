import os, sys
import fio, myfun
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
from scipy import interpolate
import gc
import myfun
import lagrangian_stats

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

label = 'm_25_3b'
label = 'm_25_2_512'
basename = 'mli' 
dayi  = 0
dayf  = 280
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


t = 0

Xlist = np.linspace(0,4000,161)
Ylist = np.linspace(0,4000,161)
Zlist = np.linspace(0,-50,51)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = np.cumsum(dl)

Xm = np.zeros(len(range(dayi,dayf,days)))
Ym = np.zeros(len(range(dayi,dayf,days)))
X = np.zeros([len(range(dayi,dayf,days)),len(Zlist)])
Y = np.zeros([len(range(dayi,dayf,days)),len(Zlist)])
U = np.zeros(len(range(dayi,dayf,days)))
V = np.zeros(len(range(dayi,dayf,days)))


xn = len(Xlist)
yn = len(Ylist)
zn = len(Zlist)

for time in range(dayi,dayf,days):
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel

 file0u = path+'Velocity_CG_0_'+label+'_'+str(time)+'.csv'
 file0v = path+'Velocity_CG_1_'+label+'_'+str(time)+'.csv'
 file1 = 'Velocity_CG_'+label+'_'+str(time)
 print file1
 #
# xn_50 = 101
# yn_50 = 101
# xn = 101
# yn = 101

 u = lagrangian_stats.read_Scalar(file0u,zn,xn,yn)
 v = lagrangian_stats.read_Scalar(file0v,zn,xn,yn)

 #
# u = np.squeeze(np.reshape(Vel[:,0],[len(Zlist),len(Xlist),len(Ylist)]))
# v = np.squeeze(np.reshape(Vel[:,1],[len(Zlist),len(Xlist),len(Ylist)]))

 for k in range(len(Zlist)):
 # u = np.mean(u,0)
 # v = np.mean(v,0)
  uk = u[k,:,:]
  vk = v[k,:,:]
  #
  dt = 3600  # s

  dx = np.mean(np.mean(uk,0),0)*dt
  dy = np.mean(np.mean(vk,0),0)*dt

  if t == 0:
   X[t,k] = dx
   Y[t,k] = dy
  else:
   X[t,k] = X[t-1,k] + dx
   Y[t,k] = Y[t-1,k] + dy

 um = np.mean(u,0)
 vm = np.mean(v,0)
 dx = np.mean(np.mean(um,0),0)*dt
 dy = np.mean(np.mean(vm,0),0)*dt

 if t == 0:
  Xm[t] = dx
  Ym[t] = dy
 else:
  Xm[t] = Xm[t-1] + dx
  Ym[t] = Ym[t-1] + dy

 U[t] = np.mean(np.mean(um,0),0)
 V[t] = np.mean(np.mean(vm,0),0)

 t = t + 1

time = np.linspace(0,time,time+1)*3600

for k in range(len(Zlist)):
# plt.plot(X[time/86400%1 == 0,k]/1000,Y[time/86400%1 == 0,k]/1000,'ro')
 plt.plot(X[:,k]/1000,Y[:,k]/1000,'k-',linewidth=2)

plt.plot(Xm/1000,Ym/1000,'b-',linewidth=2)

plt.xlabel('X [km]',fontsize=24)
plt.ylabel('Y [km]',fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.savefig('./plot/'+label+'/traj_'+file1+'_snap.eps',bbox_inches='tight')
plt.close()
print 'saved '+'./plot/'+label+'/traj_'+file1+'_snap.eps\n'


plt.plot(Xm/1000,Ym/1000,'b-',linewidth=2)

plt.xlabel('X [km]',fontsize=24)
plt.ylabel('Y [km]',fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.savefig('./plot/'+label+'/traj_M_'+file1+'_snap.eps',bbox_inches='tight')
plt.close()
print 'saved '+'./plot/'+label+'/traj_M_'+file1+'_snap.eps\n'

#U - np.mean(U)
#V - np.mean(V)

plt.plot(time[0:len(U)],U-np.mean(U),'k',linewidth=2)
plt.plot(time[0:len(V)],V-np.mean(V),'k--',linewidth=2)

plt.xlabel('time [$s$]')
plt.ylabel('U\',V\' [$m\,s^{-1}$]')
plt.savefig('./plot/'+label+'/U_'+file1+'_t.eps',bbox_inches='tight')
plt.close()
print 'saved '+'./plot/'+label+'/U_'+file1+'_t.eps\n'

import scipy.fftpack

N = len(time[1:])
dt = time[1]-time[0]

# calculate the fast fourier transform
# y2 is the solution to the under-damped oscillator from the previous section
FU = scipy.fftpack.fft(U[:]) 
FV = scipy.fftpack.fft(V[:]) 

# calculate the frequencies for the components in F
w = scipy.fftpack.fftfreq(N, dt)

# I Freq

iF = 1e-4/(2*np.pi)

pU, = plt.plot(w[w>0], abs(FU[w>0]),'k',linewidth=2)
pV, = plt.plot(w[w>0], abs(FV[w>0]),'k--',linewidth=2)
plt.legend([pU,pV],['U','V'])
plt.vlines(iF,0,0.3,'r',linewidth=2)
plt.xlabel(r'frequency $[hr^{-1}]$',fontsize=24)
plt.ylabel(r'Amplitude $[m\,s^{-1}]$',fontsize=24)
plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=22)
plt.xlim((3.56125356e-06,6.95e-05))
plt.yticks(fontsize=22)
plt.savefig('./plot/'+label+'/UV_'+file1+'_spec.eps',bbox_inches='tight')
plt.close()
print 'saved '+'./plot/'+label+'/UV_'+file1+'_spec.eps\n'

import os, sys
import fio, myfun
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
from scipy import interpolate
import csv
import myfun
import lagrangian_stats

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

label = 'm_25_2b_particles'
basename = 'mli' 
dayi  = 60
dayf  = 290
days  = 1

#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

path = './Velocity_CG/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)


dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = np.cumsum(dl)

Xm = np.zeros(len(range(dayi,dayf,days)))
Ym = np.zeros(len(range(dayi,dayf,days)))
X = np.zeros([len(range(dayi,dayf,days)),len(Zlist)])
Y = np.zeros([len(range(dayi,dayf,days)),len(Zlist)])
U = np.zeros(len(range(dayi,dayf,days)))
V = np.zeros(len(range(dayi,dayf,days)))

for t in range(len(range(dayi,dayf,days))):
 print t
 file1 = 'Velocity_CG_'+label+'_'+str(t)
 file = './Ekman/UVW_Velocity_CG_'+label+'_'+str(t)+'.csv'
 f = open(file,'r')
 reader = csv.reader(f)
 reader.next()
 depth = []

 u = []
 v = []
 w = []

 for row in reader:
  depth.append(float(row[2]))
  u.append(float(row[3]))
  v.append(float(row[4]))
  #w.append(float(row[5]))
 f.close()

 for k in range(len(Zlist)):
 # u = np.mean(u,0)
 # v = np.mean(v,0)
  uk = u[k]
  vk = v[k]
  #
  dt = 1440  # s

  dx = uk*dt
  dy = vk*dt

  if t == 0:
   X[t,k] = dx
   Y[t,k] = dy
  else:
   X[t,k] = X[t-1,k] + dx
   Y[t,k] = Y[t-1,k] + dy

 um = np.mean(u,0)
 vm = np.mean(v,0)
 dx = um*dt
 dy = vm*dt

 if t == 0:
  Xm[t] = dx
  Ym[t] = dy
 else:
  Xm[t] = Xm[t-1] + dx
  Ym[t] = Ym[t-1] + dy

 U[t] = um
 V[t] = vm

 t = t + 1



for k in range(len(Zlist)):
# plt.plot(X[time/86400%1 == 0,k]/1000,Y[time/86400%1 == 0,k]/1000,'ro')
 plt.plot(X[:,k]/1000,Y[:,k]/1000,'k-',linewidth=2)
plt.plot(Xm/1000,Ym/1000,'b-',linewidth=2)
k=0
plt.text(X[-1,k]/1000-2,Y[-1,k]/1000+2,str(Zlist[k])+'m',fontsize=22)
k=2
plt.text(X[-1,k]/1000+0.5,Y[-1,k]/1000-2.5,str(Zlist[k])+'m',fontsize=22)
k=5
plt.text(X[-1,k]/1000-2.2,Y[-1,k]/1000-2,str(Zlist[k])+'m',fontsize=22)
k=20
plt.text(X[-1,k]/1000,Y[-1,k]/1000+1,str(Zlist[k])+'m',fontsize=22)

plt.xlabel('X [km]',fontsize=24)
plt.ylabel('Y [km]',fontsize=24)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlim(-15,60)
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

time = np.asarray(range(dayi,dayf,days))*dt

plt.plot(time[0:len(U)]/3600.,U-np.mean(U),'k',linewidth=2)
plt.plot(time[0:len(V)]/3600.,V-np.mean(V),'k--',linewidth=2)

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

fmin = 3.56125356e-06
fmax = 6.95e-05

pU, = plt.plot(w[w>0], abs(FU[w>0]),'k',linewidth=2)
pV, = plt.plot(w[w>0], abs(FV[w>0]),'k--',linewidth=2)
plt.legend([pU,pV],['U','V'])
plt.vlines(iF,0,0.6,'r',linewidth=2)
plt.xlabel(r'frequency $[hr^{-1}]$',fontsize=24)
plt.ylabel(r'Amplitude $[m\,s^{-1}]$',fontsize=24)
#plt.xticks(np.linspace(fmin,fmax,7),(np.round(np.linspace(fmin,fmax,7))*360000)/100,fontsize=22)
plt.xticks([fmin,fmax,iF],[np.round(fmin*360000)/100,np.round(fmax*360000)/100,np.round(iF*360000)/100],fontsize=22)

plt.xlim((fmin,fmax))
plt.ylim((0,0.4))
plt.yticks(fontsize=22)
plt.savefig('./plot/'+label+'/UV_'+file1+'_spec.eps',bbox_inches='tight')
plt.close()
print 'saved '+'./plot/'+label+'/UV_'+file1+'_spec.eps\n'

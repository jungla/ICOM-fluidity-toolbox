import os, sys
import myfun
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import scipy.fftpack
import fio

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

label = 'm_25_1'
basename = 'mli' 
dayi  = 48
dayf  = 49
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
Zlist = np.linspace(0,-50,51)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = np.cumsum(dl)

xn = len(Xlist)
yn = len(Ylist)
zn = len(Zlist)

dx = np.diff(Xlist) 

for time in range(dayi,dayf,days):
 print 'time:', time
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #Velocity_CG_m_50_6e_9.csv
 fileU = path+'Velocity_CG_0_'+label+'_'+str(time)+'.csv'
 fileV = path+'Velocity_CG_1_'+label+'_'+str(time)+'.csv'
 file1 = 'Velocity_CG_'+label+'_'+str(time)
 #
 U = fio.read_Scalar(fileU,xn,yn,zn)
 V = fio.read_Scalar(fileV,xn,yn,zn)

 W = 0.5*(U**2 + V**2)

 FW = np.zeros((xn/1,yn))

 #
 for k in range(0,len(Zlist),5):

  for j in range(len(Ylist)):
   tempfft = scipy.fftpack.fft(W[:,j,k],xn) 
   FW[:,j] = abs(tempfft)**2
  w = scipy.fftpack.fftfreq(xn, dx[1])
#  w = scipy.fftpack.fftshift(w)
  FWp = np.mean(FW,1)*2/xn

  p25, = plt.loglog(w[w>0], FWp[w>0],'k',linewidth=2)
  x = np.linspace(10**-2,10**-3,10)
  p53, = plt.plot(x,np.power(x,-5/3.)/10**14,'r',linewidth=2)
  p2, = plt.plot(x,np.power(x,-2.)/10**14,'g',linewidth=2)
  p3, = plt.plot(x,np.power(x,-3.)/10**14,'b',linewidth=2)
  plt.xlabel(r'k $[m^{-1}]$',fontsize=18)
  plt.ylabel(r'$KE^2$ $[m^2s^{-2}]$',fontsize=18)
  plt.legend([p53,p2,p3],['-5/3','-2','-3'])
#  plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(1/np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)),fontsize=16)
  #plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
  plt.yticks(fontsize=16)
  plt.xticks(fontsize=16)
  plt.savefig('./plot/'+label+'/'+file1+'_'+str(Zlist[k])+'_spec.eps',bbox_inches='tight')
  print       './plot/'+label+'/'+file1+'_'+str(Zlist[k])+'_spec.eps'
  plt.close()
  #

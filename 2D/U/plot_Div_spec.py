import os, sys
import myfun
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import lagrangian_stats
import scipy.fftpack

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days


#label = 'm_25_2_512'
label = 'm_25_1_particles'
dayi  = 481 #10*24*2
dayf  = 581 #10*24*4
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

z = 1

for time in range(dayi,dayf,days):
 print 'time:', time
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #Velocity_CG_m_50_6e_9.csv
 fileU = path+'Velocity_CG_0_'+label+'_'+str(time)+'.csv'
 fileV = path+'Velocity_CG_1_'+label+'_'+str(time)+'.csv'
 fileT = '../RST/Temperature_CG/Temperature_CG_'+label+'_'+str(time)+'.csv'
 file1 = 'Divergence_'+label+'_'+str(time)
 #

 U = lagrangian_stats.read_Scalar(fileU,xn,yn,zn)
 V = lagrangian_stats.read_Scalar(fileV,xn,yn,zn)
 T = lagrangian_stats.read_Scalar(fileT,xn,yn,zn)

 for k in range(0,len(Zlist),5):
  dU  = np.asarray(np.gradient(U[:,:,k]))
  dV  = np.asarray(np.gradient(V[:,:,k]))
  Div = dU[0,:,:]/dx + dV[1,:,:]/dy

  #
  FT = np.zeros((xn/1,yn))
  #
  for j in range(len(Ylist)):
   tempfft = scipy.fftpack.fft(Div[:,j]**2,xn) 
   FT[:,j] = abs(tempfft)**2
  w = scipy.fftpack.fftfreq(xn, dx[1])
 #   w = scipy.fftpack.fftshift(w)
  FTp = np.mean(FT,1)/xn

  fig = plt.figure(figsize=(10,8))
  p25, = plt.loglog(w[w>0], FTp[w>0],'r',linewidth=2)
  plt.plot([5*10**-3, 5*10**-2],[5*10**3 , 5*10**-( -3+5/3.)],'k',linewidth=1.5)
  plt.plot([5*10**-3, 5*10**-2],[5*10**3 , 5*10**-( -3+3.)],'k',linewidth=1.5)
  plt.plot([5*10**-3, 5*10**-2],[5*10**3 , 5*10**-( -3+1.)],'k',linewidth=1.5)
  plt.text(6*10**-2, 5*10**-( -3+5/3.), '-5/3',fontsize=18)
  plt.text(6*10**-2, 5*10**-( -3+3.), '-3',fontsize=18)
  plt.text(6*10**-2, 5*10**-( -3+1.), '-1',fontsize=18)
  plt.text(10**-3, 10**2,str(time*360./3600)+'hr',fontsize=18)
 
  plt.xlabel(r'k $[m^{-1}]$',fontsize=20)
  plt.ylabel(r'PSD',fontsize=20)
  plt.yticks(fontsize=18)
  plt.xticks(fontsize=18)
  plt.xlim([1/2000.,1/10.])
  plt.savefig('./plot/'+label+'/Divergence_'+str(z)+'_CG_'+label+'_'+tlabel+'_spec.eps',bbox_inches='tight')
  print       './plot/'+label+'/Divergence_'+str(z)+'_CG_'+label+'_'+tlabel+'_spec.eps'
  plt.close()
  #
  v = np.linspace(0, 10, 10, endpoint=True)
  vl = np.linspace(0, 10, 5, endpoint=True)
 
  fig = plt.figure(figsize=(6,6))
  fig.add_subplot(111,aspect='equal')
  plt.contourf(Xlist/1000,Ylist/1000,T,v,extend='both',cmap=plt.cm.PiYG)
  plt.colorbar(ticks=vl)
  plt.title(str(np.round(10*(time*360./3600))/10.0)+'h')
  plt.ylabel('Y [km]',fontsize=16)
  plt.xlabel('X [km]',fontsize=16)
  plt.savefig('./plot/'+label+'/Divergence_'+str(z)+'_CG_'+label+'_'+str(time)+'.eps',bbox_inches='tight')
  print      './plot/'+label+'/Divergence_'+str(z)+'_CG_'+label+'_'+str(time)+'.eps'
  plt.close()


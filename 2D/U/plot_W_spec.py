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

label  = 'm_25_2b_tracer'
label_BW = 'm_25_2b_tracer'
label_B = 'm_25_1b_tracer'
basename = 'mli' 
dayi  = 12
dayf  = 13
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

Xlist = np.linspace(0,8000,641)
Ylist = np.linspace(0,8000,641)
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
 #Velocity_CG_m_6e_9.csv
 file0_B = path+'Velocity_CG_2_'+label_B+'_'+str(time)+'.csv'
 file0_BW = path+'Velocity_CG_2_'+label_BW+'_'+str(time)+'.csv'
 file1 = 'Velocity_CG_2_'+label+'_'+str(time)
 file1 = 'Velocity_CG_2_'+label+'_'+str(time)
 #

 T_B = lagrangian_stats.read_Scalar(file0_B,zn,xn,yn)
 T_BW = lagrangian_stats.read_Scalar(file0_BW,zn,xn,yn)

 FT_B = np.zeros((xn/1,yn))
 FT_BW = np.zeros((xn/1,yn))

 #

 for k in range(1):
  for j in range(0,len(Ylist),100):
   print j
   plt.plot(T_B[k,j,:]-np.mean(T_B[k,j,:]))
   plt.savefig('./plot/'+label+'/'+file1+'_'+str(Zlist[k])+'_'+str(j)+'_sec.eps',bbox_inches='tight')
   plt.close()
   print       './plot/'+label+'/'+file1+'_'+str(Zlist[k])+'_'+str(j)+'_sec.eps' 



  for j in range(len(Ylist)):
   tempfft = scipy.fftpack.fft(T_B[k,j,:]**2,xn) 
   FT_B[:,j] = abs(tempfft)**2
   tempfft = scipy.fftpack.fft(T_BW[k,j,:]**2,xn) 
   FT_BW[:,j] = abs(tempfft)**2
#   FT_B[:,j] = scipy.fftpack.fft(T_B[k,j,:],xn) 
#   FT_B[:,j] = abs(tempfft)**2
#   FT_BW[:,j] = scipy.fftpack.fft(T_BW[k,j,:],xn) 
#   FT_BW[:,j] = abs(tempfft)**2

  w = scipy.fftpack.fftfreq(xn, dx[1])
#  w = scipy.fftpack.fftshift(w)
  FTp_B = np.mean(FT_B,1)/xn
  FTp_BW = np.mean(FT_BW,1)/xn

  fig = plt.figure(figsize=(10,8))
  pB, = plt.plot(w[w>0], FTp_B[w>0],'r',linewidth=2)
  pBW, = plt.plot(w[w>0], FTp_BW[w>0],'b',linewidth=2)
  plt.legend((pB,pBW),('B','BW'),fontsize=24,loc=1)
#  pU, = plt.plot(w, FTp,'b',linewidth=2)
#  pU, = plt.plot(w_25, FTp_25,'r',linewidth=2)
#  plt.ylim(0,1)

#  plt.plot([4*10**-3, 4*10**-2],[4*10**-1, 4*10**-(1+5/3.)],'k',linewidth=1.5)
#  plt.plot([4*10**-3, 4*10**-2],[4*10**-1, 4*10**-(1+3.)],'k',linewidth=1.5)
#  plt.text(5*10**-2, 4*10**-(1+5/3.), '-5/3',fontsize=24)
#  plt.text(5*10**-2, 4*10**-(1+3.), '-3',fontsize=24)
#  plt.xscale('log')
#  pU, = plt.loglog(w_10[w_10>0], FTp_10[w_10>0],'k.',linewidth=2)
  plt.xlabel(r'$\lambda$ $[m]$',fontsize=26)
  plt.ylabel('Temperature PSD',fontsize=24)
#  plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(1/np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)),fontsize=16)
  #plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
  plt.yticks(fontsize=24)
  plt.xticks(fontsize=24)
#  plt.xticks([0.1,0.01,0.001],[10**1,10**2,10**3],fontsize=24)
#  plt.xlim([1/2000.,1/10.])
#  plt.xlim([1/8000.,0.005])
#  plt.xlim(8000,0)
#  plt.ylim([0.,0.8 ])
  plt.savefig('./plot/'+label+'/'+file1+'_'+str(Zlist[k])+'_spec.eps',bbox_inches='tight')
  print       './plot/'+label+'/'+file1+'_'+str(Zlist[k])+'_spec.eps'
  plt.close()
  #

#  # PDF
#
#  vals50,bins50 = np.histogram(T[k,:,:],50,(18.6,20.1),normed=True)  
#
#  bins = np.linspace(18.6,19.8,50)
#
#  fig = plt.figure(figsize=(8,8))
#  ph50, = plt.plot(bins,vals50,'k--')
#  plt.ylabel(r'PDF',fontsize=22)
#  plt.xlabel('Temperature $[^\circ C]$',fontsize=22)
##  plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(1/np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)),fontsize=16)
#  #plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
#  plt.yticks(fontsize=20)
#  plt.xticks(np.linspace(18.6,20.1,7),np.linspace(18.6,20.1,7),fontsize=20)
#  plt.tight_layout()
#
#  plt.savefig('./plot/'+label+'/'+file1+'_'+str(Zlist[k])+'_hist.eps')
#  print       './plot/'+label+'/'+file1+'_'+str(Zlist[k])+'_hist.eps'
#  plt.close()
#
#
#  Tm = 18.6 #min(np.min(T_10[k,:,:]),np.min(T_25[k,:,:]),np.min(T[k,:,:]))
#  TM = 19.8 #max(np.max(T_10[k,:,:]),np.max(T_25[k,:,:]),np.max(T[k,:,:]))
##  print Tm,TM
#
#  plt.contourf(Xlist/1000,Ylist/1000,T[k,:,:],np.linspace(Tm,TM,30),extend='both')
#  cb = plt.colorbar(ticks=np.linspace(Tm,TM,5))
#  cb.ax.tick_params(labelsize=22) 
#  plt.xlabel('X [km]',fontsize=24)
#  plt.ylabel('Y [km]',fontsize=24)
#  plt.xticks(fontsize=22)
#  plt.yticks(fontsize=22)
#  plt.axes().set_aspect('equal')
#  plt.xlim(0,2)
#  plt.ylim(0,2)
#  #plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
#  #plt.yticks(fontsize=16)
#  plt.savefig('./plot/'+label+'/'+file1+'_'+str(Zlist[k])+'.eps',bbox_inches='tight')
#  print       './plot/'+label+'/'+file1+'_'+str(Zlist[k])+'.eps'
#  plt.close() 

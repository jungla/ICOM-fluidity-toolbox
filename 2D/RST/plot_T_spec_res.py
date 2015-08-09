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

label = 'm_50_6f'
label_50 = 'm_50_6f'
label_25 = 'm_25_1'
label_10 = 'm_10_1'
basename = 'mli' 
dayi  = 36
dayf  = 49
days  = 1

#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

path = './Temperature_CG/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

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
Zlist = np.cumsum(dl)

xn_50 = len(Xlist_50)
yn_50 = len(Ylist_50)
xn_25 = len(Xlist_25)
yn_25 = len(Ylist_25)
xn_10 = len(Xlist_10)
yn_10 = len(Ylist_10)
zn = len(Zlist)

dx_50 = np.diff(Xlist_50) 
dx_25 = np.diff(Xlist_25) 
dx_10 = np.diff(Xlist_10) 

for time in range(dayi,dayf,days):
 print 'time:', time
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #Temperature_CG_m_50_6e_9.csv
 file0_50 = path+'Temperature_CG_'+label_50+'_'+str(time)+'.csv'
 file0_25 = path+'Temperature_CG_'+label_25+'_'+str(time)+'.csv'
 file0_10 = path+'Temperature_CG_'+label_10+'_'+str(time)+'.csv'
 file1 = 'Temperature_CG_'+label+'_'+str(time)
 file1_50 = 'Temperature_CG_'+label_50+'_'+str(time)
 file1_25 = 'Temperature_CG_'+label_25+'_'+str(time)
 file1_10 = 'Temperature_CG_'+label_10+'_'+str(time)
 #
# xn_50 = 101
# yn_50 = 101
# xn_25 = 101
# yn_25 = 101

 T_50 = lagrangian_stats.read_Scalar(file0_50,zn,xn_50,yn_50)
 T_25 = lagrangian_stats.read_Scalar(file0_25,zn,xn_25,yn_25)
 T_10 = lagrangian_stats.read_Scalar(file0_10,zn,xn_10,yn_10)

# xn_50 = 41
# yn_50 = 41
# xn_25 = 81
# yn_25 = 81

# T_50 = T_50[:,0:xn_50,0:yn_50]
# T_25 = T_25[:,0:xn_25,0:yn_25]

# Xlist_50 = np.linspace(0,2000,xn_50)
# Ylist_50 = np.linspace(0,2000,yn_50)
# Xlist_25 = np.linspace(0,2000,xn_25)
# Ylist_25 = np.linspace(0,2000,yn_25)

 FT_50 = np.zeros((xn_50/1,yn_50))
 FT_25 = np.zeros((xn_25/1,yn_25))
 FT_10 = np.zeros((xn_10/1,yn_10))

 #
 for k in range(1):

  for j in range(len(Ylist_50)):
   tempfft = scipy.fftpack.fft(T_50[k,j,:],xn_50) 
   FT_50[:,j] = abs(tempfft)**2
  w_50 = scipy.fftpack.fftfreq(xn_50, dx_50[1])
#  w_50 = scipy.fftpack.fftshift(w_50)
  FTp_50 = np.mean(FT_50,1)/xn_50

  for j in range(len(Ylist_25)):
   tempfft = scipy.fftpack.fft(T_25[k,j,:],xn_25) 
   FT_25[:,j] = abs(tempfft)**2
  w_25 = scipy.fftpack.fftfreq(xn_25, dx_25[1])
#  w_25 = scipy.fftpack.fftshift(w_25)
  FTp_25 = np.mean(FT_25,1)/xn_25

  for j in range(len(Ylist_10)):
   tempfft = scipy.fftpack.fft(T_10[k,j,:],xn_10) 
   FT_10[:,j] = abs(tempfft)**2
  w_10 = scipy.fftpack.fftfreq(xn_10, dx_10[1])
#  w_10 = scipy.fftpack.fftshift(w_10)
  FTp_10 = np.mean(FT_10,1)/xn_10

  fig = plt.figure(figsize=(10,8))
  p50, = plt.loglog(w_50[w_50>0], FTp_50[w_50>0],'b',linewidth=2)
  p25, = plt.loglog(w_25[w_25>0], FTp_25[w_25>0],'r',linewidth=2)
  p10, = plt.loglog(w_10[w_10>0], FTp_10[w_10>0],'k',linewidth=2)
  plt.legend([p50,p25,p10],['$B50_m$','$B25_m$','$B10_m$'],fontsize=24,loc=3)
#  pU, = plt.plot(w_50, FTp_50,'b',linewidth=2)
#  pU, = plt.plot(w_25, FTp_25,'r',linewidth=2)
#  plt.ylim(0,1)

#  plt.plot([0.5*10**-3, 4*10**-3],[4*10**-3, 0.5*10**-3],'k',linewidth=1.5)
#  plt.plot([0.5*10**-3, 4*10**-3],[3.*4*10**-3, 0.5*10**-3],'k',linewidth=1.5)
  plt.plot([4*10**-3, 4*10**-2],[4*10**-1, 4*10**-(1+5/3.)],'k',linewidth=1.5)
  plt.plot([4*10**-3, 4*10**-2],[4*10**-1, 4*10**-(1+3.)],'k',linewidth=1.5)
#  plt.plot([4*10**-3, 4*10**-2],[4*10**-1, 4*10**-(1+1.)],'k',linewidth=1.5)
  plt.text(5*10**-2, 4*10**-(1+5/3.), '-5/3',fontsize=24)
  plt.text(5*10**-2, 4*10**-(1+3.), '-3',fontsize=24)
#  plt.text(5*10**-2, 4*10**-(1+1.), '-1',fontsize=24)
#  plt.text(0.3*10**-3, 3.*4*10**-3, '-3')
#  plt.text(0.3*10**-3, 5./3.*4*10**-3, '-5/3')
  plt.xscale('log')
#  pU, = plt.loglog(w_10[w_10>0], FTp_10[w_10>0],'k.',linewidth=2)
  plt.xlabel(r'k $[m^{-1}]$',fontsize=26)
  plt.ylabel('Temperature PSD',fontsize=24)
#  plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(1/np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)),fontsize=16)
  #plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
  plt.yticks(fontsize=24)
#  plt.xticks(fontsize=24)
  plt.xticks([0.1,0.01,0.001,1/500.],[10**-1,10**-2,10**-3,1/500.],fontsize=24)
  plt.xlim([1/2000.,1/10.])
  plt.savefig('./plot/'+label+'/'+file1+'_'+str(Zlist[k])+'_spec.eps',bbox_inches='tight')
  print       './plot/'+label+'/'+file1+'_'+str(Zlist[k])+'_spec.eps'
  plt.close()
  #

  # PDF

  vals50,bins50 = np.histogram(T_50[k,:,:],50,(18.6,20.1),normed=True)  
  vals25,bins25 = np.histogram(T_25[k,:,:],50,(18.6,20.1),normed=True)  
  vals10,bins10 = np.histogram(T_10[k,:,:],50,(18.6,20.1),normed=True)  

  bins = np.linspace(18.6,19.8,50)

  fig = plt.figure(figsize=(8,8))
  ph50, = plt.plot(bins,vals50,'k--')
  ph25, = plt.plot(bins,vals25,'k.-')
  ph10, = plt.plot(bins,vals10,'k',linewidth=2)
  plt.ylabel(r'PDF',fontsize=22)
  plt.xlabel('Temperature $[^\circ C]$',fontsize=22)
#  plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(1/np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)),fontsize=16)
  #plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
  plt.yticks(fontsize=20)
  plt.xticks(np.linspace(18.6,20.1,7),np.linspace(18.6,20.1,7),fontsize=20)
  plt.tight_layout()

  plt.legend([ph50,ph25,ph10],['$B50_m$','$B25_m$','$B10_m$'],loc=2,fontsize=20)
  plt.savefig('./plot/'+label+'/'+file1+'_'+str(Zlist[k])+'_hist.eps')
  print       './plot/'+label+'/'+file1+'_'+str(Zlist[k])+'_hist.eps'
  plt.close()


  Tm = 18.6 #min(np.min(T_10[k,:,:]),np.min(T_25[k,:,:]),np.min(T_50[k,:,:]))
  TM = 19.8 #max(np.max(T_10[k,:,:]),np.max(T_25[k,:,:]),np.max(T_50[k,:,:]))
#  print Tm,TM

  plt.contourf(Xlist_50/1000,Ylist_50/1000,T_50[k,:,:],np.linspace(Tm,TM,30),extend='both')
  cb = plt.colorbar(ticks=np.linspace(Tm,TM,5))
  cb.ax.tick_params(labelsize=22) 
  plt.xlabel('X [km]',fontsize=24)
  plt.ylabel('Y [km]',fontsize=24)
  plt.xticks(fontsize=22)
  plt.yticks(fontsize=22)
  plt.axes().set_aspect('equal')
  plt.xlim(0,2)
  plt.ylim(0,2)
  #plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
  #plt.yticks(fontsize=16)
  plt.savefig('./plot/'+label+'/'+file1_50+'_'+str(Zlist[k])+'.eps',bbox_inches='tight')
  print       './plot/'+label+'/'+file1_50+'_'+str(Zlist[k])+'.eps'
  plt.close() 
###
  plt.contourf(Xlist_25/1000,Ylist_25/1000,T_25[k,:,:],np.linspace(Tm,TM,30),extend='both')
  cb = plt.colorbar(ticks=np.linspace(Tm,TM,5))
  cb.ax.tick_params(labelsize=22) 
  plt.xlabel('X [km]',fontsize=24)
  plt.ylabel('Y [km]',fontsize=24)
  plt.xticks(fontsize=22)
  plt.yticks(fontsize=22)
  plt.axes().set_aspect('equal')
  plt.xlim(0,2)
  plt.ylim(0,2)
  #plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
  #plt.yticks(fontsize=16)
  plt.savefig('./plot/'+label+'/'+file1_25+'_'+str(Zlist[k])+'.eps',bbox_inches='tight')
  print       './plot/'+label+'/'+file1_25+'_'+str(Zlist[k])+'.eps'
  plt.close()
##
  plt.contourf(Xlist_10/1000,Ylist_10/1000,T_10[k,:,:],np.linspace(Tm,TM,30),extend='both')
  cb = plt.colorbar(ticks=np.linspace(Tm,TM,5))
  cb.ax.tick_params(labelsize=22) 
  plt.xlabel('X [km]',fontsize=24)
  plt.ylabel('Y [km]',fontsize=24)
  plt.xticks(fontsize=22)
  plt.yticks(fontsize=22)
  plt.axes().set_aspect('equal')
  plt.xlim(0,2)
  plt.ylim(0,2)
  #plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
  #plt.yticks(fontsize=16)
  plt.savefig('./plot/'+label+'/'+file1_10+'_'+str(Zlist[k])+'.eps',bbox_inches='tight')
  print       './plot/'+label+'/'+file1_10+'_'+str(Zlist[k])+'.eps'
  plt.close()
###
##

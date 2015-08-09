#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import myfun
import numpy as np
from scipy import optimize
import os
import scipy.stats as sp
import scipy
import lagrangian_stats
import advect_functions
import csv

Q_s = 1000
H_s = 0.58
H_s = 0.67
b0 = 0.35
b1 = 23.0
b0 = 1.0
b1 = 17.0

def ic(X,t):
 Q_0 = 0
 global np
 if t > 0:
  time = t/3600.0%24/6
  if time >=0 and time < 2:
   Q_0 = 0
  if time >=2 and time < 3:
   Q_0 = Q_s*(time-2)
  if time >=3 and time < 4:
   Q_0 = Q_s - Q_s*(time-3)
 else:
  Q_0 = 0
# print Q_0
 Q = Q_0*(H_s*np.exp(X[2]/b0)+(1-H_s)*np.exp(X[2]/b1))
 return Q

flux_t = np.asarray(range(60,240,1))*1440. #np.linspace(0,24*3600.0,49*10)
TS = np.zeros((len(flux_t)))

for t in range(len(flux_t)):
 TS[t] = ic((1,1,0),flux_t[t])

# read offline
print 'reading offline'

#exp = 'm_25_1_tracer'
#label = 'm_25_1_tracer'
#filename = 'traj_m_25_1_tracer_0_640_.csv'
#filename3D = 'traj_m_25_1_tracer_0_640_3D.csv'
#tt = 640 # IC + 24-48 included

label = 'm_25_2b_particles'
label_BW = 'm_25_2b_particles'
label_B = 'm_25_1b_particles'


time = np.asarray(range(60,240,2))*1440./3600.

meanDiffB = []
meanDiffBW = []

for di in range(0,100,10):
 Diff_B = [] #np.zeros((tt_B,nl))
 Diff_BW = [] #np.zeros((tt_B,nl))

 df = di + 5 
 for tday in range(60,240,2): #240
  dayi  = tday #10*24*1  
  dayf  = tday+100 #10*24*4
 
  filename_B = './csv/RD_v_'+label_B+'_'+str(dayi)+'_'+str(dayf)+'.csv'
  filename_BW = './csv/RD_v_'+label_BW+'_'+str(dayi)+'_'+str(dayf)+'.csv'
 
 # print filename_B
 # print filename_BW
 
  time_B, RD_B = lagrangian_stats.read_dispersion(filename_B)
  time_BW, RD_BW = lagrangian_stats.read_dispersion(filename_BW)
 
  gradBW,poop = np.gradient(RD_BW)
  Diff_BW.append(0.5*np.mean(gradBW[di:df,:]/1440.,0))
 
  gradB,poop = np.gradient(RD_B)
  Diff_B.append(0.5*np.mean(gradB[di:df,:]/1440.,0))
  
  # relative D
  
  # RD 
 
  if tday%5==0:  
   p_5_B, = plt.plot(time_B/3600.,RD_B[:,0],color=[0,0,1],linewidth=3)
 #  z = 1
 #  p_10_B, = plt.plot(time_B/3600.,RD_B[:,z],'--',color=[0,0,1],linewidth=3)
 #  z = 2
 #  p_15_B, = plt.plot(time_B/3600.,RD_B[:,z],'-.',color=[0,0,1],linewidth=3)
   
   p_5_BW, = plt.plot(time_BW/3600.,RD_BW[:,0],color=[1,0,0],linewidth=3)
 #  z = 1
 #  p_10_BW, = plt.plot(time_BW/3600.,RD_BW[:,z],'--',color=[1,0,0],linewidth=3)
 #  z = 2
 #  p_15_BW, = plt.plot(time_BW/3600.,RD_BW[:,z],'-.',color=[1,0,0],linewidth=3)
  
 #plt.gca().set_yscale('log',basey=10)
 #plt.gca().set_xscale('log',basex=10)
 
 plt.xlabel(r'Time $[hr]$',fontsize=24)
 plt.ylabel(r'$\sigma^2_{C_z}$ $[m^2]$',fontsize=24)
 #plt.legend((p_5_B,p_10_B,p_15_B,p_5_BW,p_10_BW,p_15_BW),('B 5m','B 10m','B 15m','BW 5m','BW 10m','BW 15m'),loc=1,fontsize=20,ncol=2)
 plt.legend((p_5_B,p_5_BW),('B 5m','BW 5m'),loc=1,fontsize=20,ncol=2)
 #plt.xlim((np.log10(1440),np.log10(72*3600)))
 #plt.ylim((0,30))
 #plt.xlim((24,72))
 
 v = np.linspace(0,72,7)+24
 vl = (np.linspace(0,72,7)+72).astype(int)
 
 #plt.xticks(vind,['48.4','72','96','120'],fontsize=16)
 plt.xticks(v,vl,fontsize=20)
 plt.yticks(fontsize=20)
 
 plt.tight_layout()
 
 plt.savefig('./plot/'+label+'/RDv_time_'+label+'.eps')
 print       './plot/'+label+'/RDv_time_'+label+'.eps'
 plt.close()
 
 
 ####
 ax0 = plt.subplot2grid((3,1), (0, 0), rowspan=2)
 ax1 = plt.subplot2grid((3,1), (2, 0), rowspan=1)
 
 #fig, ax = plt.subplots(2,sharex=True,figsize=(8,8))
 
 Diff_B = np.asarray(Diff_B)
 Diff_BW = np.asarray(Diff_BW)
 
 p_5_B, = ax0.plot(time,Diff_B[:,0],'ro-',linewidth=3)
 #p_10_B, = ax1.plot(time,Diff_B[:,1],'v-',color=[1,0,0],linewidth=3)
 #p_15_B, = ax1.plot(time,Diff_B[:,2],'s-',color=[1,0,0],linewidth=3)
 p_5_BW, = ax0.plot(time,Diff_BW[:,0],'o-',color=[0,0,1],linewidth=3)
 #p_10_BW, = ax1.plot(time,Diff_BW[:,1],'v-',color=[0,0,1],linewidth=3)
 #p_15_BW, = ax1.plot(time,Diff_BW[:,2],'s-',color=[0,0,1],linewidth=3)
 
# print 'Kz BW max', np.max(Diff_BW[:,0])
# print 'Kz BW min', np.min(Diff_BW[:,0])
 print 'Kz BW mean', np.mean(Diff_BW[:,0])
# print 'Kz B max', np.max(Diff_B[:,0])
# print 'Kz B min', np.min(Diff_B[:,0])
 print 'Kz B mean', np.mean(Diff_B[:,0])

 meanDiffB.append(np.mean(Diff_B[:,0]))
 meanDiffBW.append(np.mean(Diff_BW[:,0]))
 
 ax0.set_ylabel(r'$K_z$ $[m^2s^{-1}]$',fontsize=20)
 #plt.legend((p_5_B,p_5_BW,p_10_B,p_10_BW,p_15_B,p_15_BW),('B 5m','BW 5m','B 10m','BW 10m','B 15m','BW 15m'),loc=1,fontsize=16,ncol=3)
 ax0.legend((p_5_B,p_5_BW),('B 5m','BW 5m'),loc=1,fontsize=20,ncol=2)
 #plt.xlim((24,96))
 ax0.set_ylim((0.0001,0.0014))
 ax0.set_xticks(())
 ax0.tick_params(axis='y',labelsize=16)
 
 ax1.plot(flux_t/3600,TS-245.36,'r',linewidth=3)
 plt.ylabel('$Q_0-Q_L$ $[Wm^{-2}]$',fontsize=16)
 plt.xlabel(r'Time $[hr]$',fontsize=18)
 plt.xticks(v,vl,fontsize=16)
 plt.yticks(fontsize=14)
 
 plt.tight_layout()
 plt.savefig('./plot/'+label+'/Diff_RD_'+label+'_'+str(di)+'_'+str(df)+'.eps')
 print       './plot/'+label+'/Diff_RD_'+label+'_'+str(di)+'_'+str(df)+'.eps'
 plt.close()


plt.plot(range(0,100,10),meanDiffB,'b',linewidth=2)
plt.plot(range(0,100,10),meanDiffBW,'r',linewidth=2)
plt.savefig('./plot/'+label+'/Diff_RD_time.eps')
print       './plot/'+label+'/Diff_RD_time.eps'
plt.close()

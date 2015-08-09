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
dayi  = 960
dayf  = dayi+240
days  = 1

#label = 'm_25_2_512_tracer'
#dayi  = 0
3days  = 10
#dayf  = dayi+240+days

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
#Xlist = np.linspace(0,10000,801)
#Ylist = np.linspace(0,4000,321)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = np.cumsum(dl)

xn = len(Xlist)
yn = len(Ylist)
zn = len(Zlist)

dx, dy = np.meshgrid(np.gradient(Ylist),np.gradient(Xlist))

corrT = []
timeT = np.asarray(range(dayi,dayf,days))
timeL = timeT*360/3600.+48
for time in timeT:
 print 'time:', time
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #Temperature_CG_m_50_6e_9.csv
 fileU = path+'Velocity_CG_0_'+label+'_'+str(time)+'.csv' #+480
 fileV = path+'Velocity_CG_1_'+label+'_'+str(time)+'.csv' #+480
 fileT = '../RST/Tracer_CG/Tracer_1_CG_'+label+'_'+str(time)+'.csv' #-960
 print fileT
 file1 = 'DivTr_'+label+'_'+str(time)
 #

 U = fio.read_Scalar(fileU,xn,yn,zn)
 V = fio.read_Scalar(fileV,xn,yn,zn)
 T = fio.read_Scalar(fileT,xn,yn,zn)

 for k in range(1):
  dU  = np.asarray(np.gradient(U[:,:,k]))
  dV  = np.asarray(np.gradient(V[:,:,k]))
  Div = dU[0,:,:]/dx + dV[1,:,:]/dy
  T = np.sum(T[:,:,:],2)/50.

 # Div = Div-np.mean(Div)
 # norm=np.linalg.norm(Div)
 # Div = -Div/norm
  
 # T = T-np.mean(T) 
 # norm=np.linalg.norm(T)
 # T = T/norm

  X=np.vstack(( np.reshape(Div,(1,xn*yn)), np.reshape(T,(1,xn*yn))))
  corr = np.corrcoef(X)
  print corr
  corrT.append(corr[0,1])


  if time - 960 == 33:
   plt.subplots(figsize=(7,6))
   plt.contourf(Xlist/1000,Ylist/1000,Div,np.linspace(-0.001,0.001,30),extend='both',cmap=plt.cm.PiYG)
   cb = plt.colorbar(ticks=np.linspace(-0.001,0.001,5))
   cb.ax.tick_params(labelsize=16)   
   plt.xlabel('X [km]',fontsize=18)
   plt.ylabel('Y [km]',fontsize=18)
   plt.axes().set_aspect('equal')
   plt.title(r'$\nabla\cdot u$',fontsize=18)
   plt.xlim(0,2)
   plt.ylim(0,2)
   #plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
   plt.xticks(fontsize=16)
   plt.yticks(fontsize=16)
   plt.savefig('./plot/'+label+'/'+file1+'_Div_'+str(Zlist[k])+'.eps')
   print       './plot/'+label+'/'+file1+'_Div_'+str(Zlist[k])+'.eps'
   plt.close()
# 
   plt.subplots(figsize=(7,6))
   plt.contourf(Xlist/1000,Ylist/1000,T,np.linspace(0,0.4,30),extend='both',cmap=plt.cm.PiYG)
   cb = plt.colorbar(ticks=np.linspace(0,0.4,5))
#   cb = plt.colorbar()
   cb.ax.tick_params(labelsize=16)   
   plt.xlabel('X [km]',fontsize=18)
   plt.ylabel('Y [km]',fontsize=18)
   plt.axes().set_aspect('equal')
   plt.title(r'C',fontsize=18)
   plt.xlim(0,2)
   plt.ylim(0,2)
   plt.xticks(fontsize=16)
   plt.yticks(fontsize=16)
   #plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
   #plt.yticks(fontsize=16)
   plt.savefig('./plot/'+label+'/'+file1+'_T_'+str(Zlist[k])+'.eps') #,bbox_inches='tight')
   print       './plot/'+label+'/'+file1+'_T_'+str(Zlist[k])+'.eps'
   plt.close()
#

  ## corr heat map
#  heatmap, xedges, yedges = np.histogram2d(np.reshape(Div,(xn*yn,)), np.reshape(np.mean(T[:,:,:],2),(xn*yn,)), bins=50)
#  extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#  plt.contourf(heatmap,30,extent=extent)
#  plt.savefig('./plot/'+label+'/'+file1+'_heat_'+str(Zlist[k])+'.eps',bbox_inches='tight')
#  print './plot/'+label+'/'+file1+'_heat_'+str(Zlist[k])+'.eps'
#  plt.close()

fig = plt.figure(figsize=(8,4))
plt.plot(timeL,corrT,'k',linewidth=2)
plt.xlabel('Time [$hr$]',fontsize=18)
plt.ylabel(r'$\rho(\nabla \cdot u, C)$',fontsize=18)
plt.xticks(np.linspace(48,72,5),np.linspace(48,72,5).astype(int),fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(-1,1)
plt.xlim(48,72)
plt.tight_layout()
#plt.ylabel(r'corr(T,$nabla\cdot U$)',fontsize=18)
#plt.axes().set_aspect('equal')
#plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
#plt.yticks(fontsize=16)
plt.savefig('./plot/'+label+'/'+file1+'_t_'+str(Zlist[k])+'.eps') #,bbox_inches='tight')
print       './plot/'+label+'/'+file1+'_t_'+str(Zlist[k])+'.eps'
plt.close()
##

import os, sys
import myfun
import numpy as np
#import matplotlib  as mpl
#mpl.use('ps')
import matplotlib.pyplot as plt
from scipy import interpolate
import lagrangian_stats
import fio
import scipy

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

label_B = 'm_25_1b_tracer'
label_BV = 'm_25_1b_tracer'
label_BW = 'm_25_2b_tracer'
label_BWV = 'm_25_2b_tracer'
#label_BW = 'm_25_2_512_tracer'
#label_BWV = 'm_25_2_512'

dayi  = 0 
days  = 1
dayf  = 91

timeT = np.asarray(range(dayi,dayf,days))
timeL = timeT*1440
print timeL

#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

path = './Velocity_CG/'

try: os.stat('./plot/'+label_B)
except OSError: os.mkdir('./plot/'+label_B)

# dimensions archives

# ML exp

Xlist_B = np.linspace(0,8000,641)
Ylist_B = np.linspace(0,8000,641)
Xlist_BW = np.linspace(0,8000,641)
Ylist_BW = np.linspace(0,8000,641)
#Xlist_BW = np.linspace(0,10000,801)
#Ylist_BW = np.linspace(0,4000,321)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = np.cumsum(dl)

xn_B = len(Xlist_B)
yn_B = len(Ylist_B)
xn_BW = len(Xlist_BW)
yn_BW = len(Ylist_BW)
zn = len(Zlist)

dx_BW, dy_BW = np.meshgrid(np.gradient(Ylist_BW),np.gradient(Xlist_BW))
dx_B, dy_B = np.meshgrid(np.gradient(Ylist_B),np.gradient(Xlist_B))

covT = np.zeros((2,len(timeL)))
#covT = covT+1.
covTn = np.zeros((2,len(timeL)))
meanT = np.zeros((2,len(timeL)))
meanTDiv = np.zeros((2,len(timeL)))
meanDiv = np.zeros((2,len(timeL)))
corrT = np.zeros((2,len(timeL)))

for time in range(len(timeT)):
 print 'time:', time
 tlabel = str(timeT[time])
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #Temperature_CG_m_50_6e_9.csv
 fileU_B = path+'Velocity_CG_0_'+label_BV+'_'+str(timeT[time])+'.csv' #+480
 fileV_B = path+'Velocity_CG_1_'+label_BV+'_'+str(timeT[time])+'.csv' #+480
 fileT_B = '../RST/Tracer_CG/Tracer_1_CG_'+label_B+'_'+str(timeT[time])+'.csv' #-960
 fileU_BW = path+'Velocity_CG_0_'+label_BWV+'_'+str(timeT[time])+'.csv' #+480
 fileV_BW = path+'Velocity_CG_1_'+label_BWV+'_'+str(timeT[time])+'.csv' #+480
 fileT_BW = '../RST/Tracer_CG/Tracer_1_CG_'+label_BW+'_'+str(timeT[time])+'.csv' #-960
 print fileT_BW, fileT_B
 file1_B = 'DivTr_'+label_B+'_'+str(timeT[time])
 file1_BW = 'DivTr_'+label_BW+'_'+str(timeT[time])
 #

 U_B = fio.read_Scalar(fileU_B,xn_B,yn_B,zn)
 V_B = fio.read_Scalar(fileV_B,xn_B,yn_B,zn)
 T_B = fio.read_Scalar(fileT_B,xn_B,yn_B,zn)
 U_BW = fio.read_Scalar(fileU_BW,xn_BW,yn_BW,zn)
 V_BW = fio.read_Scalar(fileV_BW,xn_BW,yn_BW,zn)
 T_BW = fio.read_Scalar(fileT_BW,xn_BW,yn_BW,zn)
# U_BW = U_B
# V_BW = V_B
# T_BW = T_B

# for k in [0]:
#  print 'depth',Zlist[k]
#  dU_B  = np.asarray(np.gradient(np.mean(U_B[:,:,2:3],2)))
#  dV_B  = np.asarray(np.gradient(np.mean(V_B[:,:,2:3],2)))
#  Div_B = dU_B[0,:,:]/dx_B + dV_B[1,:,:]/dy_B
#  T_B = np.mean(T_B[:,:,2:3],2)
#  dU_BW  = np.asarray(np.gradient(np.mean(U_BW[:,:,2:3],2)))
#  dV_BW  = np.asarray(np.gradient(np.mean(V_BW[:,:,2:3],2)))
#  Div_BW = dU_BW[0,:,:]/dx_BW + dV_BW[1,:,:]/dy_BW
#  T_BW = np.mean(T_BW[:,:,2:3],2)


 DivK_B = np.zeros((3,xn_B,yn_B))
 DivK_BW = np.zeros((3,xn_B,yn_B))
 
 for k in range(1,4):
  print 'depth',Zlist[k]
  dU_B  = np.asarray(np.gradient(U_B[:,:,k]))
  dV_B  = np.asarray(np.gradient(V_B[:,:,k]))
  DivK_B[k-1,:,:] = dU_B[0,:,:]/dx_B + dV_B[1,:,:]/dy_B
  dU_BW  = np.asarray(np.gradient(U_BW[:,:,k]))
  dV_BW  = np.asarray(np.gradient(V_BW[:,:,k]))
  DivK_BW[k-1,:,:] = dU_BW[0,:,:]/dx_BW + dV_BW[1,:,:]/dy_BW

 T_B = np.mean(T_B[:,:,1:4],2)
 T_BW = np.mean(T_BW[:,:,1:4],2)
 Div_B = np.mean(DivK_B[:,:,:],0)
 Div_BW = np.mean(DivK_BW[:,:,:],0)

 X = np.vstack(( -1*np.reshape(Div_B,(1,xn_B*yn_B)), np.reshape(T_B,(1,xn_B*yn_B))))
 covT[0,time] = np.mean((X[0,:]-np.mean(X[0,:]))*(X[1,:]-np.mean(X[1,:])))
 meanT[0,time] = np.mean(T_B)
 meanDiv[0,time] = np.mean(Div_B)
 meanTDiv[0,time] = np.mean(T_B*Div_B)
 corrT[0,time] = 1-scipy.spatial.distance.correlation(-1*np.reshape(Div_B,(1,xn_B*yn_B)), np.reshape(T_B,(1,xn_B*yn_B)))
 print 'B',corrT[0,time]

 X = np.vstack(( -1*np.reshape(np.transpose(Div_BW),(1,xn_BW*yn_BW)), np.reshape(np.transpose(T_BW),(1,xn_BW*yn_BW))))
 covT[1,time] = np.mean((X[0,:]-np.mean(X[0,:]))*(X[1,:]-np.mean(X[1,:])))
 meanT[1,time] = np.mean(T_BW)
 meanDiv[1,time] = np.mean(Div_BW)
 meanTDiv[1,time] = np.mean(T_BW*Div_BW)
 corrT[1,time] = 1-scipy.spatial.distance.correlation(-1*np.reshape(Div_BW,(1,xn_BW*yn_BW)), np.reshape(T_BW,(1,xn_BW*yn_BW)))
 print 'BW',corrT[1,time]

 print 'time',timeL[time] 
 if timeL[time] > 0: #== 3.2*3600 or timeL[time] == 5.2*3600: # == 3600*4.:
  mdiv = -0.0003
  Mdiv = 0.00015
  plt.subplots(figsize=(7,6))
  plt.contourf(Xlist_B/1000,Ylist_B/1000,Div_B,np.linspace(mdiv,Mdiv,30),extend='both',cmap=plt.cm.PiYG)
  cb = plt.colorbar(ticks=np.linspace(mdiv,Mdiv,5))
  cb.ax.tick_params(labelsize=16)   
  plt.xlabel('X [$km$]',fontsize=18)
  plt.ylabel('Y [$km$]',fontsize=18)
  plt.axes().set_aspect('equal')
  plt.title(r'$\nabla\cdot u$',fontsize=18)
  plt.xlim(0,8)
  plt.ylim(0,8)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig('./plot/'+label_B+'/'+file1_B+'_Div_'+str(Zlist[k])+'.png')
  print       './plot/'+label_B+'/'+file1_B+'_Div_'+str(Zlist[k])+'.png'
  plt.close()
  plt.subplots(figsize=(7,6))
  plt.contourf(Xlist_B/1000,Ylist_B/1000,T_B,np.linspace(0,1,30),extend='both',cmap=plt.cm.PiYG)
  cb = plt.colorbar(ticks=np.linspace(0,1,5))
  cb.ax.tick_params(labelsize=16)   
  plt.xlabel('X [$km$]',fontsize=18)
  plt.ylabel('Y [$km$]',fontsize=18)
  plt.axes().set_aspect('equal')
  plt.title(r'C',fontsize=18)
  plt.xlim(0,8)
  plt.ylim(0,8)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig('./plot/'+label_B+'/'+file1_B+'_T_'+str(Zlist[k])+'.png') #,bbox_inches='tight')
  print       './plot/'+label_B+'/'+file1_B+'_T_'+str(Zlist[k])+'.png'
  plt.close()

  plt.subplots(figsize=(7,6))
  plt.contourf(Xlist_BW/1000,Ylist_BW/1000,np.transpose(Div_BW),np.linspace(mdiv,Mdiv,30),extend='both',cmap=plt.cm.PiYG)
  cb = plt.colorbar(ticks=np.linspace(mdiv,Mdiv,5))
  cb.ax.tick_params(labelsize=16)
  plt.xlabel('X [$km$]',fontsize=18)
  plt.ylabel('Y [$km$]',fontsize=18)
  plt.title(r'$\nabla\cdot u$',fontsize=18)
  plt.axes().set_aspect('equal')
  plt.xlim(0,8)
  plt.ylim(0,8)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig('./plot/'+label_BW+'/'+file1_BW+'_Div_'+str(Zlist[k])+'.png')
  print       './plot/'+label_BW+'/'+file1_BW+'_Div_'+str(Zlist[k])+'.png'
  plt.close()

  plt.subplots(figsize=(7,6))
  plt.contourf(Xlist_BW/1000,Ylist_BW/1000,np.transpose(T_BW),np.linspace(0,1,30),extend='both',cmap=plt.cm.PiYG)
  cb = plt.colorbar(ticks=np.linspace(0,1,5))
  cb.ax.tick_params(labelsize=16)
  plt.xlabel('X [$km$]',fontsize=18)
  plt.ylabel('Y [$km$]',fontsize=18)
  plt.title(r'C',fontsize=18)
  plt.axes().set_aspect('equal')
  plt.xlim(0,8)
  plt.ylim(0,8)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.savefig('./plot/'+label_BW+'/'+file1_BW+'_T_'+str(Zlist[k])+'.png') #,bbox_inches='tight')
  print       './plot/'+label_BW+'/'+file1_BW+'_T_'+str(Zlist[k])+'.png'
  plt.close()

  ## cov heat map
#  heatmap, xedges, yedges = np.histogram2d(np.reshape(Div,(xn*yn,)), np.reshape(np.mean(T[:,:,:],2),(xn*yn,)), bins=50)
#  extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#  plt.contourf(heatmap,30,extent=extent)
#  plt.savefig('./plot/'+label+'/'+file1+'_heat_'+str(Zlist[k])+'.png',bbox_inches='tight')
#  print './plot/'+label+'/'+file1+'_heat_'+str(Zlist[k])+'.png'
#  plt.close()

timeP = timeL[:]/3600. #+86400

fig = plt.figure(figsize=(8,4))
#plt.xscale('log')

covTn[0,:] = (covT[0,:])/np.max(covT[0,:])
covTn[1,:] = (covT[1,:])/np.max(covT[1,:])

timePlog = timeP

pl_B, = plt.plot(timePlog,covT[0,:],'k',linewidth=2)
pl_BW, = plt.plot(timePlog,covT[1,:],'k--',linewidth=2)
plt.legend([pl_B,pl_BW],['$B$','$BW$'],loc=1)
plt.xlabel('Time [$hr$]',fontsize=18)
plt.ylabel(r'$cov(\nabla \cdot u, C)$',fontsize=18)
#labels = np.log10(np.linspace(86400,86400+36*3600,19))
#labels[0] = np.log10(1440)
#plt.xticks(np.log10(labels),labels[1:].astype(int),fontsize=16)

#plt.xticks(labels,['0','2','4','6','8','10','12','','','18','','','24','','','','','','36'],fontsize=16)
plt.xticks(np.linspace(0,36,13),(np.linspace(0,36,13)+72).astype(int),fontsize=16)
plt.yticks(fontsize=16)

plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

#plt.ylim(-.1,1.1)
#plt.xlim(labels[0],labels[-1])
plt.xlim(0,36) #labels[0],labels[-1])
plt.tight_layout()
#plt.ylabel(r'cov(T,$nabla\cdot U$)',fontsize=18)
#plt.axes().set_aspect('equal')
#plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
plt.savefig('./plot/'+label_BW+'/'+file1_BW+'_t_'+str(Zlist[k])+'.png') #,bbox_inches='tight')
print       './plot/'+label_BW+'/'+file1_BW+'_t_'+str(Zlist[k])+'.png'
plt.close()



pl_B, = plt.plot(timePlog,meanT[0,:],'k',linewidth=2)
pl_BW, = plt.plot(timePlog,meanT[1,:],'k--',linewidth=2)
plt.legend([pl_B,pl_BW],['$B$','$BW$'],loc=1)
plt.xlabel('Time [$hr$]',fontsize=18)
plt.ylabel(r'$<C>$',fontsize=18)
#labels = np.log10(np.linspace(86400,86400+36*3600,19))
#labels[0] = np.log10(1440)
#plt.xticks(np.log10(labels),labels[1:].astype(int),fontsize=16)

#plt.xticks(labels,['0','2','4','6','8','10','12','','','18','','','24','','','','','','36'],fontsize=16)
plt.xticks(np.linspace(0,36,13),(np.linspace(0,36,13)+72).astype(int),fontsize=16)
plt.yticks(fontsize=16)

plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

#plt.ylim(-.1,1.1)
#plt.xlim(labels[0],labels[-1])
plt.xlim(0,36) #labels[0],labels[-1])
plt.tight_layout()
#plt.ylabel(r'cov(T,$nabla\cdot U$)',fontsize=18)
#plt.axes().set_aspect('equal')
#plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
plt.savefig('./plot/'+label_BW+'/'+file1_BW+'_meanT_t_'+str(Zlist[k])+'.png') #,bbox_inches='tight')
print       './plot/'+label_BW+'/'+file1_BW+'_meanT_t_'+str(Zlist[k])+'.png'
plt.close()
#


pl_B, = plt.plot(timePlog,meanTDiv[0,:],'k',linewidth=2)
pl_BW, = plt.plot(timePlog,meanTDiv[1,:],'k--',linewidth=2)
plt.legend([pl_B,pl_BW],['$B$','$BW$'],loc=1)
plt.xlabel('Time [$hr$]',fontsize=18)
plt.ylabel(r'$<C\cdot \nabla\cdot u>$',fontsize=18)
#labels = np.log10(np.linspace(86400,86400+36*3600,19))
#labels[0] = np.log10(1440)
#plt.xticks(np.log10(labels),labels[1:].astype(int),fontsize=16)

#plt.xticks(labels,['0','2','4','6','8','10','12','','','18','','','24','','','','','','36'],fontsize=16)
plt.xticks(np.linspace(0,36,13),(np.linspace(0,36,13)+72).astype(int),fontsize=16)
plt.yticks(fontsize=16)

plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

#plt.ylim(-.1,1.1)
#plt.xlim(labels[0],labels[-1])
plt.xlim(0,36) #labels[0],labels[-1])
plt.tight_layout()
#plt.ylabel(r'cov(T,$nabla\cdot U$)',fontsize=18)
#plt.axes().set_aspect('equal')
#plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
plt.savefig('./plot/'+label_BW+'/'+file1_BW+'_meanTDiv_t_'+str(Zlist[k])+'.png') #,bbox_inches='tight')
print       './plot/'+label_BW+'/'+file1_BW+'_meanTDiv_t_'+str(Zlist[k])+'.png'
plt.close()
#


pl_B, = plt.plot(timePlog,meanDiv[0,:],'k',linewidth=2)
pl_BW, = plt.plot(timePlog,meanDiv[1,:],'k--',linewidth=2)
plt.legend([pl_B,pl_BW],['$B$','$BW$'],loc=1)
plt.xlabel('Time [$hr$]',fontsize=18)
plt.ylabel(r'$<\nabla\cdot u>$',fontsize=18)
#labels = np.log10(np.linspace(86400,86400+36*3600,19))
#labels[0] = np.log10(1440)
#plt.xticks(np.log10(labels),labels[1:].astype(int),fontsize=16)

#plt.xticks(labels,['0','2','4','6','8','10','12','','','18','','','24','','','','','','36'],fontsize=16)
plt.xticks(np.linspace(0,36,13),(np.linspace(0,36,13)+72).astype(int),fontsize=16)
plt.yticks(fontsize=16)

plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

#plt.ylim(-.1,1.1)
#plt.xlim(labels[0],labels[-1])
plt.xlim(0,36) #labels[0],labels[-1])
plt.tight_layout()
#plt.ylabel(r'cov(T,$nabla\cdot U$)',fontsize=18)
#plt.axes().set_aspect('equal')
#plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
plt.savefig('./plot/'+label_BW+'/'+file1_BW+'_meanDiv_t_'+str(Zlist[k])+'.png') #,bbox_inches='tight')
print       './plot/'+label_BW+'/'+file1_BW+'_meanDiv_t_'+str(Zlist[k])+'.png'
plt.close()



pl_B, = plt.plot(timePlog,corrT[0,:],'k',linewidth=2)
pl_BW, = plt.plot(timePlog,corrT[1,:],'k--',linewidth=2)
plt.legend([pl_B,pl_BW],['$B$','$BW$'],loc=1)
plt.xlabel('Time [$hr$]',fontsize=18)
plt.ylabel(r'$corr(\nabla \cdot u, C)$',fontsize=18)
#labels = np.log10(np.linspace(86400,86400+36*3600,19))
#labels[0] = np.log10(1440)
#plt.xticks(np.log10(labels),labels[1:].astype(int),fontsize=16)

#plt.xticks(labels,['0','2','4','6','8','10','12','','','18','','','24','','','','','','36'],fontsize=16)
plt.xticks(np.linspace(0,36,13),(np.linspace(0,36,13)+72).astype(int),fontsize=16)
plt.yticks(fontsize=16)

plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

plt.ylim(0,1)
#plt.xlim(labels[0],labels[-1])
plt.xlim(0,36) #labels[0],labels[-1])
plt.tight_layout()
#plt.ylabel(r'cov(T,$nabla\cdot U$)',fontsize=18)
#plt.axes().set_aspect('equal')
#plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
plt.savefig('./plot/'+label_BW+'/'+file1_BW+'_corr_t_'+str(Zlist[k])+'.png') #,bbox_inches='tight')
print       './plot/'+label_BW+'/'+file1_BW+'_corr_t_'+str(Zlist[k])+'.png'
plt.close()

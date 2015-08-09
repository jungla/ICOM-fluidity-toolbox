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

label_B = 'm_25_1b_tracer'
#label_BW = 'm_25_1_tracer'
label_BW = 'm_25_2b_tracer'

dayi  = 0
days  = 1 
dayf  = 91

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
#Xlist_BW = np.linspace(0,10000,801)
#Ylist_BW = np.linspace(0,4000,321)
Xlist_BW = np.linspace(0,8000,641)
Ylist_BW = np.linspace(0,8000,641)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = np.cumsum(dl)

xn_B = len(Xlist_B)
yn_B = len(Ylist_B)
xn_BW = len(Xlist_BW)
yn_BW = len(Ylist_BW)
zn = len(Zlist)

dx_BW, dy_BW = np.meshgrid(np.gradient(Ylist_BW),np.gradient(Xlist_BW))
dx_B, dy_B = np.meshgrid(np.gradient(Ylist_B),np.gradient(Xlist_B))

timeT = np.asarray(range(dayi,dayf,days))
timeL = timeT*1440

varT = np.zeros((2,3,len(timeL)))
varTn = np.zeros((2,3,len(timeL)))
Tlist = [1,2,4]

for time in range(len(timeT)):
 print 'time:', time
 tlabel = str(timeT[time])
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #Temperature_CG_m_50_6e_9.csv
 fileT_B_1 = '../RST/Tracer_CG/Tracer_1_CG_'+label_B+'_'+str(timeT[time])+'.csv' #-960
 fileT_BW_1 = '../RST/Tracer_CG/Tracer_1_CG_'+label_BW+'_'+str(timeT[time])+'.csv' #-960
 fileT_B_2 = '../RST/Tracer_CG/Tracer_2_CG_'+label_B+'_'+str(timeT[time])+'.csv' #-960
 fileT_BW_2 = '../RST/Tracer_CG/Tracer_2_CG_'+label_BW+'_'+str(timeT[time])+'.csv' #-960
 fileT_B_4 = '../RST/Tracer_CG/Tracer_4_CG_'+label_B+'_'+str(timeT[time])+'.csv' #-960
 fileT_BW_4 = '../RST/Tracer_CG/Tracer_4_CG_'+label_BW+'_'+str(timeT[time])+'.csv' #-960
 print fileT_BW_1, fileT_B_1
 print fileT_BW_2, fileT_B_2
 print fileT_BW_4, fileT_B_4
 file1_B = 'Tr_var_'+label_B+'_'+str(timeT[time])
 file1_BW = 'Tr_var_'+label_BW+'_'+str(timeT[time])
 #

 T_B_1 = fio.read_Scalar(fileT_B_1,xn_B,yn_B,zn)
 T_BW_1 = fio.read_Scalar(fileT_BW_1,xn_BW,yn_BW,zn)

# T_B_2 = fio.read_Scalar(fileT_B_2,xn_B,yn_B,zn)
# T_BW_2 = fio.read_Scalar(fileT_BW_2,xn_BW,yn_BW,zn)
# T_B_4 = fio.read_Scalar(fileT_B_4,xn_B,yn_B,zn)
# T_BW_4 = fio.read_Scalar(fileT_BW_4,xn_BW,yn_BW,zn)

 T_B = np.sum(T_B_1[:,:,1:4],2)/3.
 T_BW = np.sum(T_BW_1[:,:,1:4],2)/3. 
 M_T_B = np.mean(T_B)
 M_T_BW = np.mean(T_BW)
 varT[0,0,time] = np.mean((T_B-M_T_B)**2)
 varT[1,0,time] = np.mean((T_BW-M_T_BW)**2)

# T_B = np.sum(T_B_2[:,:,4:7],2)/3.
# T_BW = np.sum(T_BW_2[:,:,4:7],2)/3. 
# M_T_B = np.mean(T_B)
# M_T_BW = np.mean(T_BW)
# varT[0,1,time] = np.mean((T_B-M_T_B)**2)
# varT[1,1,time] = np.mean((T_BW-M_T_BW)**2)
#
# T_B = np.sum(T_B_4[:,:,16:19],2)/3.
# T_BW = np.sum(T_BW_4[:,:,16:19],2)/3. 
# M_T_B = np.mean(T_B)
# M_T_BW = np.mean(T_BW)
# varT[0,2,time] = np.mean((T_B-M_T_B)**2)
# varT[1,2,time] = np.mean((T_BW-M_T_BW)**2)

timeP = timeL[1:]
#
#fig = plt.figure(figsize=(9,6))
#
#pl_B_1, = plt.plot(np.log10(timeP),varT[0,0,1:]/np.max(varT[0,0,1:]),'r',linewidth=2)
#pl_BW_1, = plt.plot(np.log10(timeP),varT[1,0,1:]/np.max(varT[1,0,1:]),'b',linewidth=2)
#pl_B_2, = plt.plot(np.log10(timeP),varT[0,1,1:]/np.max(varT[0,1,1:]),'r--',linewidth=2)
#pl_BW_2, = plt.plot(np.log10(timeP),varT[1,1,1:]/np.max(varT[1,1,1:]),'b--',linewidth=2)
#pl_B_4, = plt.plot(np.log10(timeP),varT[0,2,1:]/np.max(varT[0,2,1:]),'r-.',linewidth=2)
#pl_BW_4, = plt.plot(np.log10(timeP),varT[1,2,1:]/np.max(varT[1,2,1:]),'b-.',linewidth=2)
#plt.legend([pl_B_1,pl_B_2,pl_B_4,pl_BW_1,pl_BW_2,pl_BW_4],['$B$ $1m$','$B$ $5m$','$B$ $17m$','$BW$ $1m$','$BW$ $5m$','$BW$ $17m$'],loc=2)
#plt.xlabel('Time [$hr$]',fontsize=18)
#plt.ylabel(r'$<(C - <C>)^2>$',fontsize=18)
#
#labels = np.log10(np.linspace(0,36*3600,19))
#labels[0] = np.log10(1440)
#
#plt.xticks(labels,['0.4','2','4','6','8','10','12','','','18','','','24','','','','','','36'],fontsize=16)
#plt.yticks(fontsize=16)
#
##plt.ylim(0,0.005)
#plt.xlim(labels[0],36*3600)
#plt.tight_layout()
#plt.savefig('./plot/'+label_BW+'/'+file1_BW+'.eps') #,bbox_inches='tight')
#print       './plot/'+label_BW+'/'+file1_BW+'.eps'
#plt.close()

timeP = timeL/3600.#+86400


for k in [0]:
 varTn[0,k,:] = (varT[0,k,:])/np.max(varT[0,k,:])
 varTn[1,k,:] = (varT[1,k,:])/np.max(varT[1,k,:])
 fig = plt.figure(figsize=(8,4))
 #
 pl_B_1, = plt.plot(timeP,varT[0,k,:],'k',linewidth=2)
 pl_BW_1, = plt.plot(timeP,varT[1,k,:],'k--',linewidth=2)
 plt.legend([pl_B_1,pl_BW_1],['$B$','$BW$'],loc=1)
 plt.xlabel('Time [$hr$]',fontsize=18)
 plt.ylabel(r'$<(C - <C>)^2>$',fontsize=18)
 #
# labels = np.log10(np.linspace(86400,86400+36*3600,19))
# labels[0] = 0 #np.log10(1440)
 #
# plt.xticks(labels,['0','2','4','6','8','10','12','','','18','','','24','','','','','','36'],fontsize=16)
 plt.xticks(np.linspace(0,36,13),(np.linspace(0,36,13)+72).astype(int),fontsize=16)
 plt.yticks(fontsize=16)
 plt.ylim(-.01,0.15) 
#plt.ylim(-0.01,0.14)
# plt.xlim(np.log10(86400),np.log10(timeP[-1]))
 plt.xlim(0,36) #labels[0],labels[-1]) #np.log10(86400),np.log10(timeP[-1]))
 plt.tight_layout()
 plt.savefig('./plot/'+label_BW+'/'+file1_BW+'_t_'+str(Tlist[k])+'.eps') #,bbox_inches='tight')
 print       './plot/'+label_BW+'/'+file1_BW+'_t_'+str(Tlist[k])+'.eps'
 plt.close()

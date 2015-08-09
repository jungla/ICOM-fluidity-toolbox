import numpy as np
import csv
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
from scipy.integrate import simps, trapz

def tracer_m0_z(Zlist,Tr,deltaz):
 H = 3 #float(max(Zlist)-min(Zlist))
 Zlist = Zlist - deltaz # center of the distribution
 # strip nans
 S00 = simps(Tr,Zlist)
# S00 = S00/float(N); S01 = S01/float(N); S02 = S02/float(N)
 return S00


def tracer_m1_z(Zlist,Tr,deltaz):
 H = 3 #float(max(Zlist)-min(Zlist))
 Zlist = Zlist - deltaz # center of the distribution
 # strip nans
 S01 = np.trapz(Tr*(Zlist), Zlist)/H
 return S01

def tracer_m2_z(Zlist,Tr,deltaz):
 H = 3 #float(max(Zlist)-min(Zlist))
 Zlist = Zlist - deltaz # center of the distribution
 # strip nans
 S00 = np.trapz(Tr, Zlist)/H
 S01 = np.trapz(Tr*(Zlist), Zlist)/H
 S02 = np.trapz(Tr*(Zlist)**2, Zlist)/H
# S00 = S00/float(N); S01 = S01/float(N); S02 = S02/float(N)
 return (S02-S01**2)/S00




 return S01

label_BW = 'm_25_2b_tracer'
label_B = 'm_25_2b_tracer'
label = 'm_25_1b_tracer'
basename_BW = 'Tracer_CG_1_'+label_BW
basename_B = 'Tracer_CG_1_'+label_B

dayi = 0
dayf = 90
days = 1

dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = np.cumsum(dl) + 1

time = range(dayi,dayf,days)

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days
#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

path = './Tracer_CG/z/'

# read csv

Tracer_B = []
Tracer_BW = []

for t in time:
# print 'read drate', t
 with open('./Tracer_CG/z/'+basename_B+'_'+str(t)+'_z_clip.csv', 'rb') as csvfile:
  spamreader = csv.reader(csvfile)
  for row in spamreader:
   Tracer_B.append(row[:])

for t in time:
 with open('./Tracer_CG/z/'+basename_BW+'_'+str(t)+'_z_clip.csv', 'rb') as csvfile:
  spamreader = csv.reader(csvfile)
  for row in spamreader:
   Tracer_BW.append(row[:])

Tracer_B = np.asarray(Tracer_B).astype(float)
Tracer_BW = np.asarray(Tracer_BW).astype(float)

plt.contourf(np.asarray(time)*1440/3600.,-1*Zlist,np.transpose(Tracer_B),60)
plt.xlabel('Time [hr]',fontsize=18)
plt.ylabel('Depth [m]',fontsize=18)
plt.colorbar()
plt.title('Tracer mean concentration $B25_m$')
plt.savefig('./plot/'+label+'/Tracer_'+label_B+'.eps',bbox_inches='tight')
print       './plot/'+label+'/Tracer_'+label_B+'.eps'
plt.close()

plt.contourf(np.asarray(time)*1440/3600.,-1*Zlist,np.transpose(Tracer_BW),60)
plt.xlabel('Time [hr]',fontsize=18)
plt.ylabel('Depth [m]',fontsize=18)
plt.colorbar()
plt.title('Tracer mean concentration $BW25_m$')
plt.savefig('./plot/'+label+'/Tracer_'+label_BW+'.eps',bbox_inches='tight')
print       './plot/'+label+'/Tracer_'+label_BW+'.eps'
plt.close()

# plot trace concentration at various times

pB, = plt.plot(Zlist,Tracer_B[0,:],'k',linewidth=2)
plt.plot(Zlist,Tracer_B[5,:],'k',linewidth=2)
plt.plot(Zlist,Tracer_B[10,:],'k',linewidth=2)
plt.plot(Zlist,Tracer_B[15,:],'k',linewidth=2)
pBW, = plt.plot(Zlist,Tracer_BW[0,:],'k--',linewidth=2)
plt.plot(Zlist,Tracer_BW[5,:],'k--',linewidth=2)
plt.plot(Zlist,Tracer_BW[10,:],'k--',linewidth=2)
plt.plot(Zlist,Tracer_BW[15,:],'k--',linewidth=2)
plt.legend((pB,pBW),('$B25_m$','$BW25_m$'),loc=4)
plt.xlim(0,25)
plt.savefig('./plot/'+label+'/Tracer_time.eps',bbox_inches='tight')
print       './plot/'+label+'/Tracer_time.eps'
plt.close()

deltaz = 0

#0th mom

Tr_disp_B = []
Tr_disp_BW = []

for t in time:
 Tr_disp_B.append(tracer_m0_z(Zlist,Tracer_B[t,:],deltaz))
 Tr_disp_BW.append(tracer_m0_z(Zlist,Tracer_BW[t,:],deltaz))

pB, = plt.plot(np.asarray(time)*1440/3600.,Tr_disp_B,'k',linewidth=2)
pBW, = plt.plot(np.asarray(time)*1440/3600.,Tr_disp_BW,'k--',linewidth=2)
plt.legend((pB,pBW),('$B25_m$','$BW25_m$'),loc=4)
plt.xlabel('Time [hr]',fontsize=18)
plt.ylabel('Tracer 0th moment',fontsize=18)
#plt.xticks(np.linspace(timeD[0]/3600.,timeD[-1]/3600.,7),np.linspace(48,24*3+48,7).astype(int))
#plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
#plt.yticks(fontsize=16)
plt.savefig('./plot/'+label+'/Tr_0mom_'+label+'.eps',bbox_inches='tight')
print       './plot/'+label+'/Tr_0mom_'+label+'.eps'
plt.close()


#1st mom

Tr_disp_B = []
Tr_disp_BW = []

for t in time:
 Tr_disp_B.append(tracer_m1_z(Zlist,Tracer_B[t,:],deltaz))
 Tr_disp_BW.append(tracer_m1_z(Zlist,Tracer_BW[t,:],deltaz))

pB, = plt.plot(np.asarray(time)*1440/3600.,Tr_disp_B,'k',linewidth=2)
pBW, = plt.plot(np.asarray(time)*1440/3600.,Tr_disp_BW,'k--',linewidth=2)
plt.legend((pB,pBW),('$B25_m$','$BW25_m$'),loc=4)
plt.xlabel('Time [hr]',fontsize=18)
plt.ylabel('Tracer 1st moment',fontsize=18)
#plt.xticks(np.linspace(timeD[0]/3600.,timeD[-1]/3600.,7),np.linspace(48,24*3+48,7).astype(int))
#plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
#plt.yticks(fontsize=16)
plt.savefig('./plot/'+label+'/Tr_1mom_'+label+'.eps',bbox_inches='tight')
print       './plot/'+label+'/Tr_1mom_'+label+'.eps'
plt.close()

#2nd mom

Tr_disp_B = []
Tr_disp_BW = []

for t in time:
 Tr_disp_B.append(tracer_m2_z(Zlist,Tracer_B[t,:],deltaz))
 Tr_disp_BW.append(tracer_m2_z(Zlist,Tracer_BW[t,:],deltaz))

pB, = plt.plot(np.asarray(time)*1440/3600.,Tr_disp_B,'k',linewidth=2)
pBW, = plt.plot(np.asarray(time)*1440/3600.,Tr_disp_BW,'k--',linewidth=2)
plt.legend((pB,pBW),('$B25_m$','$BW25_m$'),loc=4)
plt.xlabel('Time [hr]',fontsize=18)
plt.ylabel('Tracer 2nd moment',fontsize=18)
#plt.xticks(np.linspace(timeD[0]/3600.,timeD[-1]/3600.,7),np.linspace(48,24*3+48,7).astype(int))
#plt.xticks(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7),np.round(np.linspace(np.min(w[w>0]),np.max(w[w>0]),7)*360000)/100,fontsize=16)
#plt.yticks(fontsize=16)
plt.savefig('./plot/'+label+'/Tr_2mom_'+label+'.eps',bbox_inches='tight')
print       './plot/'+label+'/Tr_2mom_'+label+'.eps'
plt.close()





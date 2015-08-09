#!~/python
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import myfun
import numpy as np
import lagrangian_stats
import csv
import advect_functions

# read offline
print 'reading particles'

exp = 'm_25_2b'
label = 'm_25_2b'
filename2D_BW = './csv/RD_2D_m_25_2b_particles.csv'
tt_BW = 439 # IC + 24-48 included

dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = 1.*np.cumsum(dl)

depths = [5, 10, 15] 
depthid = [1, 2, 3] 

nl = len(depths)

RD_2D_BW = [] #np.zeros((tt_BW,nl))
time2D_BW = []


with open(filename2D_BW, 'r') as csvfile:
 spamreader = csv.reader(csvfile)
 spamreader.next()
 for row in spamreader:
  time2D_BW.append(row[0])
  RD_2D_BW.append(row[1:])

time2D_BW = np.asarray(time2D_BW).astype(float) 
RD_2D_BW = np.asarray(RD_2D_BW).astype(float) 

time = time2D_BW[:]

# cut particles to time of interest

timeD = np.asarray(range(0,3*86400,1440))/3600.
vtime = time - time[0]


# read 3D eps and get eps at particle's location

drateD_BW = np.zeros((len(timeD),len(Zlist)))

for t in range(len(timeD)):
# print 'read drate', t
 with open('../../2D/U/drate_3+1day/z/drate_m_25_2b_particles_'+str(t+60)+'_z.csv', 'rb') as csvfile:
  spamreader = csv.reader(csvfile)
  for row in spamreader:
   drateD_BW[t,:] = row[:]  

# test drate
#plt.contourf(timeD/86400.,Zlist,np.log10(np.rot90(drateD_B)),30)
#plt.colorbar()
#plt.savefig('./plot/'+label+'/drate_B_'+label+'.eps')
#plt.close()
#
#plt.contourf((timeD)/3600.+48,Zlist,np.log10(np.rot90(drateD_BW)),30)
#plt.colorbar()
#plt.savefig('./plot/'+label+'/drate_BW_'+label+'.eps')
#print       './plot/'+label+'/drate_BW_'+label+'.eps'
#plt.close()


# normalized RD
fig = plt.figure(figsize=(8, 5))
time2D_BW = time2D_BW - time2D_BW[0]
R2D5_BW, = plt.plot(np.log10(time2D_BW[:]/3600.),np.log10(RD_2D_BW[:,0]/time2D_BW[:]**3),'k',linewidth=1)
R2D10_BW, = plt.plot(np.log10(time2D_BW[:]/3600.),np.log10(RD_2D_BW[:,1]/time2D_BW[:]**3),'k--',linewidth=1)
R2D15_BW, = plt.plot(np.log10(time2D_BW[:]/3600.),np.log10(RD_2D_BW[:,2]/time2D_BW[:]**3),'k-.',linewidth=1)

intm = 0.3*86400; intM = 2.5*86400; interval = (vtime > intm) * (vtime < intM)
R2D5_BW, = plt.plot(np.log10(time2D_BW[interval]/3600.),np.log10(RD_2D_BW[interval,0]/time2D_BW[interval]**3),'k',linewidth=3.5)

intm = 0.4*86400; intM = 3*86400; interval = (vtime > intm) * (vtime < intM)
R2D10_BW, = plt.plot(np.log10(time2D_BW[interval]/3600.),np.log10(RD_2D_BW[interval,1]/time2D_BW[interval]**3),'k--',linewidth=3.5)

intm = 0.6*86400; intM = 3*86400; interval = (vtime > intm) * (vtime < intM)
R2D15_BW, = plt.plot(np.log10(time2D_BW[interval]/3600.),np.log10(RD_2D_BW[interval,2]/time2D_BW[interval]**3),'k-.',linewidth=3.5)

#plt.legend((R2D5_BW,R2D10_BW,R2D15_BW,R2D5_B,R2D10_B,R2D15_B),('$BW25_m$ 5m','$BW25_m$ 10m','$BW25_m$ 15m','$B25_m$ 5m','$B25_m$ 10m','$B25_m$ 15m'),    loc=1,fontsize=16,ncol=2)
plt.legend((R2D5_BW,R2D10_BW,R2D15_BW),('5m','10m','15m'),    loc=1,fontsize=16,ncol=3)
plt.xlabel('Time $[hr]$',fontsize=20)
plt.ylabel('$log(\sigma^2_D t^{-3})$ ',fontsize=20)
plt.yticks(fontsize=16)
plt.ylim()
#ind = [0.,12.,24.,36.,48.,60.,72.,84.,96.,108.,120.,132.,144.,156.,168.,180.,192.]
ind = np.asarray([0.,12.,24.,48.,96.,192.])
#ind = np.linspace(0,24*8,7)
ind[0] = 1440/3600.
vind = np.log10(ind);# vind[0]=np.log10(1440/3600.)
plt.xticks(vind,['72.4','84','96','144','168','264'],fontsize=16)
plt.tight_layout()
plt.savefig('./plot/'+label+'/RDt3_2_BW_'+label+'.eps')
print       './plot/'+label+'/RDt3_2_BW_'+label+'.eps'
plt.close()

# Rich 2D-3D

#fig = plt.figure(figsize=(8, 5))

fig, ax1 = plt.subplots(figsize=(8, 5))
#
# BW
intm = 0.3*86400; intM = 2.5*86400; interval = (vtime > intm) * (vtime < intM)
Rich = RD_2D_BW[interval,0]/time2D_BW[interval]**3/(drateD_BW[interval,depths[0]])
print 'Rich  2D 5m: mean', np.mean(Rich), 'std', np.std(Rich)
print 'Drate 2D 5m: mean', np.mean(drateD_BW[interval,depths[0]]), 'std', np.std(drateD_BW[interval,depths[0]])
R2D5_BW, = ax1.plot(time2D_BW[interval]/3600.,Rich,'r',linewidth=2)

intm = 0.4*86400; intM = 3*86400; interval = (vtime > intm) * (vtime < intM)
Rich = RD_2D_BW[interval,1]/time2D_BW[interval]**3/(drateD_BW[interval,depths[1]])
print 'Rich  2D 10m: mean', np.mean(Rich), 'std', np.std(Rich)
print 'Drate 2D 10m: mean', np.mean(drateD_BW[interval,depths[1]]), 'std', np.std(drateD_BW[interval,depths[1]])
R2D10_BW, = ax1.plot(time2D_BW[interval]/3600.,Rich,'r--',linewidth=2)

intm = 0.6*86400; intM = 3*86400; interval = (vtime > intm) * (vtime < intM)
Rich = RD_2D_BW[interval,2]/time2D_BW[interval]**3/(drateD_BW[interval,depths[2]])
print 'Rich  2D 15m: mean', np.mean(Rich), 'std', np.std(Rich)
print 'Drate 2D 15m: mean', np.mean(drateD_BW[interval,depths[2]]), 'std', np.std(drateD_BW[interval,depths[2]])
R2D15_BW, = ax1.plot(time2D_BW[interval]/3600.,Rich,'r-.',linewidth=2)

#for tic in plt.xaxis.get_minor_ticks():
#    tic.tick1On = tic.tick2On = False

#plt.legend((R2D1,R3D1,R2D5,R3D5,R2D17,R3D17),('2D 5m','3D 5m','2D 10m','3D 10m','2D 15m','3D 15m'),loc=3,fontsize=16,ncol=3)
#plt.legend((R2D5_BW,R2D10_BW,R2D15_BW,R2D5_B,R2D10_B,R2D15_B),('$BW25_m$ 5m','$BW25_m$ 10m','$BW25_m$ 15m','$B25_m$ 5m','$B25_m$ 10m','$B25_m$ 15m'),loc=2,fontsize=16,ncol=2)


dummy5, = ax1.plot([],[],'k',linewidth=2)
dummy10, = ax1.plot([],[],'k--',linewidth=2)
dummy15, = ax1.plot([],[],'k-.',linewidth=2)
ax1.legend((dummy5,dummy10,dummy15),('5m','10m','15m'),loc=1,fontsize=14,ncol=3)

#import matplotlib.lines as mlines
#l5 = mlines.Line2D([], [],'-',color='black', label='5m')
#l10 = mlines.Line2D([], [],'--',color='black', label='10m')
#l15 = mlines.Line2D([], [],'-.',color='black', label='15m')
#ax1.legend(handles=[l5,l10,l15],loc=1,fontsize=16,ncol=3)

ax1.set_xlabel('Time $[hr]$',fontsize=20)
ax1.set_ylabel('$\sigma^2_D t^{-3} \epsilon^{-1}$ ',fontsize=20,color='r')
ax1.set_ylim(0.02,0.18)

for tl in ax1.get_yticklabels():
    tl.set_color('r')
    tl.set_fontsize(16)

#plt.ylim(0.02,0.18)
#plt.yticks(fontsize=16)
ind = np.linspace(72,24*3+72,13)
ind[0] = 52

ax2 = ax1.twinx()
DR5, = ax2.plot(timeD,drateD_BW[:,depths[0]],'b',linewidth=2)
DR10, = ax2.plot(timeD,drateD_BW[:,depths[1]],'b--',linewidth=2)
DR15, = ax2.plot(timeD,drateD_BW[:,depths[2]],'b-.',linewidth=2)
#ax2.legend((DR5,DR10,DR15),('5m','10m','15m'),loc=1,fontsize=14,ncol=3)

ax2.set_ylim(0.,1.2e-8)
ax2.set_ylabel('$\epsilon$ ',fontsize=20,color='b')
for tl in ax2.get_yticklabels():
    tl.set_color('b')
    tl.set_fontsize(16)

plt.xlim(0,72)
#plt.xticks(ind,['','54','60','66','72','78','84','90','96','102','108','114','120'],fontsize=16)
plt.xticks(np.linspace(0,72,13),np.linspace(0,72,13).astype(int),fontsize=16)
plt.tight_layout()
plt.savefig('./plot/'+label+'/Rich_2_BW_'+label+'.eps')
print       './plot/'+label+'/Rich_2_BW_'+label+'.eps'
plt.close()


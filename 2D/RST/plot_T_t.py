import os, sys
import vtktools
import fluidity_tools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt

#label = sys.argv[1]
#basename = sys.argv[2]


path0 = path1 = path2 = '../RST/stat_files/'
path2b = '/scratch/jmensa/m_10_1/'

#
file0 = 'm_50_6f.stat'
filepath0 = path0+file0
stat0 = fluidity_tools.stat_parser(filepath0)

file1 = 'm_25_1.stat'
filepath1 = path1+file1
stat1 = fluidity_tools.stat_parser(filepath1)

file2 = 'm_10_1.stat'
filepath2 = path2+file2
stat2 = fluidity_tools.stat_parser(filepath2)

file2b = 'mli_checkpoint.stat'
filepath2b = path2b+file2b
stat2b = fluidity_tools.stat_parser(filepath2b)

#file1 = 'ring.stat'
#filepath1 = path1+file1
#stat1 = fluidity_tools.stat_parser(filepath1)

time0 = stat0["ElapsedTime"]["value"]/86400.0
time1 = stat1["ElapsedTime"]["value"]/86400.0
time2 = stat2["ElapsedTime"]["value"]/86400.0
time2b = stat2b["ElapsedTime"]["value"]/86400.0
#time1 = stat1["ElapsedTime"]["value"]/86400.0

Temp0 = stat0["BoussinesqFluid"]["Temperature_CG"]["l2norm"]
Temp1 = stat1["BoussinesqFluid"]["Temperature_CG"]["l2norm"]
Temp2 = stat2["BoussinesqFluid"]["Temperature_CG"]["l2norm"]
Temp2b = stat2b["BoussinesqFluid"]["Temperature_CG"]["l2norm"]
#KE1 = 0.5*np.sqrt(stat1["BoussinesqFluid"]["Temperature_CG"]["l2norm"])

Temp2 = Temp2[np.where(time2<time2b[0])]
time2 = time2[np.where(time2<time2b[0])]
Temp2 = np.hstack((Temp2,Temp2b))
time2 = np.hstack((time2,time2b))

dayf = np.min((len(time0),len(time1),len(time2)))

Temp0a = 0
Temp1a = Temp1[:dayf]-Temp0[:dayf]
Temp2a = Temp2[:dayf]-Temp0[:dayf]

T0 = len(time0)
T1 = len(time1)
T2 = len(time2)

# volume
V = 1 #2000.0*2000.0*50

# plot KE
fig = plt.figure(figsize=(6,3))
#T10, = plt.plot(time0[np.where(time0<=4)], Temp0a[np.where(time0<=4)]/V, 'r-',linewidth=1.5)
T50, = plt.plot(time1[np.where(time1<=5)], Temp1a[np.where(time1<=5)]/V, 'r',linewidth=1.5)
T25, = plt.plot(time2[np.where(time2<=5)], Temp2a[np.where(time2<=5)]/V, 'b',linewidth=1.5)
#plt.ylim([0.0014, 0.00142])
plt.xlim([0, 7])
plt.xticks(np.linspace(0,5,6),np.linspace(0,5,6).astype(int))
#plt.xlim([0.920, 0.980])
plt.legend([T50,T25],['$B50_m$ - $B10_m$','$B25_m$ - $B10_m$'])
#plt.plot(time1, KE1/V, 'k',linewidth=1.5)
plt.xlabel("Time $[days]$",fontsize=18)
plt.ylabel("$l^2-norm$ T' $[^\circ C]$",fontsize=18)
plt.tight_layout()
plt.savefig('./plot/T_t_50_25_10.eps')
plt.close()
print 'saved '+'./plot/T_t_50_25_10.eps\n'


# plot KE
#Temp1 = Temp1-np.mean(Temp1)

fig = plt.figure(figsize=(6,3))
#T50, = plt.plot(time0[np.where(time0<=4)], Temp0[np.where(time0<=4)]/V, 'r-',linewidth=1.5)
T10, = plt.plot(time0[np.where(time0<=5)], Temp0[np.where(time0<=5)]/V, 'g',linewidth=1.5)
#T10, = plt.plot(time2[np.where(time2<=4)], Temp2[np.where(time2<=4)]/V, 'b-',linewidth=1.5)
#T10, = plt.plot(time2b[np.where(time2b<=4)], Temp2b[np.where(time2b<=4)]/V, 'b-',linewidth=1.5)
#plt.ylim([0.0014, 0.00142])
plt.xlim([0, 7])
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xticks(np.linspace(0,5,6),np.linspace(0,5,6).astype(int))
#plt.xlim([0.920, 0.980])
plt.legend([T10],['$B10_m$'])
#plt.plot(time1, KE1/V, 'k',linewidth=1.5)
plt.xlabel("Time $[days]$",fontsize=18)
plt.ylabel("$l^2-norm$ T $[^\circ C]$",fontsize=18)
plt.tight_layout()

#plt.ylim([1.48, 1.52])

plt.savefig('./plot/T_t_25.eps')
plt.close()
print 'saved '+'./plot/T_t_25.eps\n'

#

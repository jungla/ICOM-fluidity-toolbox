import os, sys
import vtktools
import fluidity_tools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt

#label = sys.argv[1]
#basename = sys.argv[2]


path = '../RST/stat_files/'

file1 = 'm_25_1b.stat'
filepath1 = path+file1
stat1 = fluidity_tools.stat_parser(filepath1)

file1b = 'm_25_1b_checkpoint.stat'
filepath1b = path+file1b
stat1b = fluidity_tools.stat_parser(filepath1b)

#file1c = 'm_25_1b_checkpoint_2.stat'
#filepath1c = path+file1c
#stat1c = fluidity_tools.stat_parser(filepath1c)

file2 = 'm_25_2b.stat'
filepath2 = path+file2
stat2 = fluidity_tools.stat_parser(filepath2)

file2b = 'm_25_2b_checkpoint.stat'
filepath2b = path+file2b
stat2b = fluidity_tools.stat_parser(filepath2b)

file2c = 'm_25_2b_checkpoint_2.stat'
filepath2c = path+file2c
stat2c = fluidity_tools.stat_parser(filepath2c)

#file2d = 'm_25_2_512_checkpoint_2.stat'
#filepath2d = path+file2d
#stat2d = fluidity_tools.stat_parser(filepath2d)

time1 = stat1["ElapsedTime"]["value"]/86400.0
time1b = stat1b["ElapsedTime"]["value"]/86400.0
#time1c = stat1c["ElapsedTime"]["value"]/86400.0
time2 = stat2["ElapsedTime"]["value"]/86400.0
time2b = stat2b["ElapsedTime"]["value"]/86400.0
time2c = stat2c["ElapsedTime"]["value"]/86400.0
#time2d = stat2d["ElapsedTime"]["value"]/86400.0

Vel1 = stat1["BoussinesqFluid"]["Velocity_CG%3"]["l2norm"]
Vel1b = stat1b["BoussinesqFluid"]["Velocity_CG%3"]["l2norm"]
#Vel1c = stat1c["BoussinesqFluid"]["Velocity_CG%3"]["l2norm"]
Vel2 = stat2["BoussinesqFluid"]["Velocity_CG%3"]["l2norm"]
Vel2b = stat2b["BoussinesqFluid"]["Velocity_CG%3"]["l2norm"]
Vel2c = stat2c["BoussinesqFluid"]["Velocity_CG%3"]["l2norm"]
#Vel2d = stat2d["BoussinesqFluid"]["Velocity_CG%3"]["l2norm"]

Vel1 = Vel1[np.where(time1<time1b[0])]
time1 = time1[np.where(time1<time1b[0])]
Vel1 = np.hstack((Vel1,Vel1b))
time1 = np.hstack((time1,time1b))

#Vel1 = Vel1[np.where(time1<time1c[0])]
#time1 = time1[np.where(time1<time1c[0])]
#Vel1 = np.hstack((Vel1,Vel1c))
#time1 = np.hstack((time1,time1c))

Vel2 = Vel2[np.where(time2<time2b[0])]
time2 = time2[np.where(time2<time2b[0])]
Vel2 = np.hstack((Vel2,Vel2b))
time2 = np.hstack((time2,time2b))

Vel2 = Vel2[np.where(time2<time2c[0])]
time2 = time2[np.where(time2<time2c[0])]
Vel2 = np.hstack((Vel2,Vel2c))
time2 = np.hstack((time2,time2c))

#Vel2 = Vel2[np.where(time2<time2d[0])]
#time2 = time2[np.where(time2<time2d[0])]
#Vel2 = np.hstack((Vel2,Vel2d))
#time2 = np.hstack((time2,time2d))

dayf = np.min((len(time1),len(time2)))

Vel1a = Vel1#[:dayf]
Vel2a = Vel2#[:dayf]

T1 = len(time1)
T2 = len(time2)

# volume
V1 = 4883616 #8000.0*8000.0*50 #270030 #135015 #1120*244 #*10000.0*4000.0*50
V2 = 4883616 #8000.0*8000.0*50 #168660 #84330 #1280*240  #*8000.0*8000.0*50

# plot KE
fig = plt.figure(figsize=(6,3))
#T10, = plt.plot(time0[np.where(time0<=4)], Vel0a[np.where(time0<=4)]/V, 'r-',linewidth=1.5)
TB25, = plt.plot(time1[np.where(time1<=9)], Vel1a[np.where(time1<=9)]/V1, 'r',linewidth=1.5)
TBW25, = plt.plot(time2[np.where(time2<=9)], Vel2a[np.where(time2<=9)]/V2, 'b',linewidth=1.5)
#plt.ylim([0.0014, 0.00142])
plt.xlim([0, 13])
plt.xticks(np.linspace(0,9,10),np.linspace(0,9,10).astype(int))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.plot([3,3],[0,6e-9],'k--')
#plt.xlim([0.920, 0.980])
plt.legend([TB25,TBW25],['$B$','$BW$'])
#plt.legend([TB25],['$B25_m$'])
#plt.plot(time1, KE1/V, 'k',linewidth=1.5)
plt.xlabel("Time $[days]$",fontsize=18)
plt.ylabel("mean |W| $[ms^{-1}]$",fontsize=18)
plt.tight_layout()
plt.savefig('./plot/W_t.eps')
plt.close()
print 'saved '+'./plot/W_t.eps'

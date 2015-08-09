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

file1c = 'm_25_1b_checkpoint_2.stat'
filepath1c = path+file1c
stat1c = fluidity_tools.stat_parser(filepath1c)

file2 = 'm_25_2.stat'
filepath2 = path+file2
stat2 = fluidity_tools.stat_parser(filepath2)

file2b = 'm_25_2_checkpoint.stat'
filepath2b = path+file2b
stat2b = fluidity_tools.stat_parser(filepath2b)

file2c = 'm_25_2_512_checkpoint.stat'
filepath2c = path+file2c
stat2c = fluidity_tools.stat_parser(filepath2c)

file2d = 'm_25_2_512_checkpoint_2.stat'
filepath2d = path+file2d
stat2d = fluidity_tools.stat_parser(filepath2d)


#file1 = 'ring.stat'
#filepath1 = path1+file1
#stat1 = fluidity_tools.stat_parser(filepath1)

time1 = stat1["ElapsedTime"]["value"]/86400.0
time1b = stat1b["ElapsedTime"]["value"]/86400.0
time1c = stat1c["ElapsedTime"]["value"]/86400.0
time2 = stat2["ElapsedTime"]["value"]/86400.0
time2b = stat2b["ElapsedTime"]["value"]/86400.0
time2c = stat2c["ElapsedTime"]["value"]/86400.0
time2d = stat2d["ElapsedTime"]["value"]/86400.0

Temp1 = stat1["BoussinesqFluid"]["Temperature_CG"]["integral"]
Temp1b = stat1b["BoussinesqFluid"]["Temperature_CG"]["integral"]
Temp1c = stat1c["BoussinesqFluid"]["Temperature_CG"]["integral"]
Temp2 = stat2["BoussinesqFluid"]["Temperature_CG"]["integral"]
Temp2b = stat2b["BoussinesqFluid"]["Temperature_CG"]["integral"]
Temp2c = stat2c["BoussinesqFluid"]["Temperature_CG"]["integral"]
Temp2d = stat2d["BoussinesqFluid"]["Temperature_CG"]["integral"]

Temp1 = Temp1[np.where(time1<time1b[0])]
time1 = time1[np.where(time1<time1b[0])]
Temp1 = np.hstack((Temp1,Temp1b))
time1 = np.hstack((time1,time1b))

Temp1 = Temp1[np.where(time1<time1c[0])]
time1 = time1[np.where(time1<time1c[0])]
Temp1 = np.hstack((Temp1,Temp1c))
time1 = np.hstack((time1,time1c))


Temp2 = Temp2[np.where(time2<time2b[0])]
time2 = time2[np.where(time2<time2b[0])]
Temp2 = np.hstack((Temp2,Temp2b))
time2 = np.hstack((time2,time2b))

Temp2 = Temp2[np.where(time2<time2c[0])]
time2 = time2[np.where(time2<time2c[0])]
Temp2 = np.hstack((Temp2,Temp2c))
time2 = np.hstack((time2,time2c))

Temp2 = Temp2[np.where(time2<time2d[0])]
time2 = time2[np.where(time2<time2d[0])]
Temp2 = np.hstack((Temp2,Temp2d))
time2 = np.hstack((time2,time2d))

dayf = np.min((len(time1),len(time2)))

Temp1a = Temp1#[:dayf]
Temp2a = Temp2[0:-1:2]#[:dayf]

time1a = time1
time2a = time2[0:-1:2]

# volume
V1 = 8000.0*8000.0*50 #270030 #135015 #1120*244 #*10000.0*4000.0*50
V2 = 10000.0*4000.0*50 #168660 #84330 #1280*240  #*8000.0*8000.0*50

# plot KE
fig = plt.figure(figsize=(6,3))
#T10, = plt.plot(time0[np.where(time0<=4)], Temp0a[np.where(time0<=4)]/V, 'r-',linewidth=1.5)
TB25, = plt.plot(time1a[np.where(time1a<=5)], Temp1a[np.where(time1a<=5)]/V1, 'r',linewidth=1.5)
TBW25, = plt.plot(time2a[np.where(time2a<=5)], Temp2a[np.where(time2a<=5)]/V2, 'b',linewidth=1.5)
#plt.ylim([0.0014, 0.00142])
plt.xlim([0, 8])
plt.xticks(np.linspace(0,5,6),np.linspace(0,5,6).astype(int))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.xlim([0.920, 0.980])
plt.legend([TB25,TBW25],['$B25_m$','$BW25_m$'])
#plt.legend([TB25],['$B25_m$'])
#plt.plot(time1, KE1/V, 'k',linewidth=1.5)
plt.xlabel("Time $[days]$",fontsize=18)
plt.ylabel("$l^2-norm$ T' $[^\circ C]$",fontsize=18)
plt.tight_layout()
plt.savefig('./plot/T_t_25.eps')
plt.close()
print 'saved '+'./plot/T_t_25.eps\n'


tt = (Temp2a[np.where(time2a[1:]<=5)]/V2 - Temp1a[np.where(time1a[1:]<=5)]/V1)/time1a[np.where(time1a[1:]<=5)]
plt.plot(tt[1:],linewidth=1.5)
plt.savefig('./plot/T_day_25.eps')
print       './plot/T_day_25.eps'
plt.close()#


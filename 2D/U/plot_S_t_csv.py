import os, sys
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import csv

#label = sys.argv[1]
#basename = sys.argv[2]


path = '../RST/stat_files_csv/'
path = './'

file1a = 'Savg_m_25_1b.csv'
filepath1a = path+file1a

file1 = 'Savg_m_25_1b_particles.csv'
filepath1 = path+file1

file2a = 'Savg_m_25_2b.csv'
filepath2a = path+file2a

file2 = 'Savg_m_25_2b_particles.csv'
filepath2 = path+file2

time1 = []
time2 = []
Vel1 = []
Vel2 = []
time1a = []
time2a = []
Vel1a = []
Vel2a = []

with open(filepath1a, 'r') as csvfile:
 spamreader = csv.reader(csvfile)
 for row in spamreader:
  time1a.append(row[0])
  Vel1a.append(row[1])

with open(filepath1, 'r') as csvfile:
 spamreader = csv.reader(csvfile)
 for row in spamreader:
  time1.append(row[0])
  Vel1.append(row[1])

with open(filepath2a, 'r') as csvfile:
 spamreader = csv.reader(csvfile)
 for row in spamreader:
  time2a.append(row[0])
  Vel2a.append(row[1])

with open(filepath2, 'r') as csvfile:
 spamreader = csv.reader(csvfile)
 for row in spamreader:
  time2.append(row[0])
  Vel2.append(row[1])


time1 = np.asarray(time1).astype(float)/86400. + 2
time2 = np.asarray(time2).astype(float)/86400. + 2
Vel1 = np.asarray(Vel1).astype(float)
Vel2 = np.asarray(Vel2).astype(float)

time1a = np.asarray(time1a).astype(float)*3600./1440./86400.
time2a = np.asarray(time2a).astype(float)*3600./1440./86400.
Vel1a = np.asarray(Vel1a).astype(float)
Vel2a = np.asarray(Vel2a).astype(float)

V1 = 1 
V2 = 1 

# plot KE
fig = plt.figure(figsize=(6,3))
#T10, = plt.plot(time0[np.where(time0<=4)], Vel0a[np.where(time0<=4)]/V, 'r-',linewidth=1.5)
TB25, = plt.plot(time1a[np.where(time1a<=9)], Vel1a[np.where(time1a<=9)]/V1, 'r',linewidth=1.5)
TBW25, = plt.plot(time2a[np.where(time2a<=9)], Vel2a[np.where(time2a<=9)]/V2, 'b',linewidth=1.5)
TB25, = plt.plot(time1[np.where(time1<=9)], Vel1[np.where(time1<=9)]/V1, 'r',linewidth=1.5)
TBW25, = plt.plot(time2[np.where(time2<=9)], Vel2[np.where(time2<=9)]/V2, 'b',linewidth=1.5)

plt.plot([3,3],[0,3.5e-2],'k--')

plt.xlim([0, 8])
plt.ylim(0.,3.5e-2)
plt.xticks(np.linspace(0,8,9),np.linspace(0,8,9).astype(int))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#plt.xlim([0.920, 0.980])
plt.legend([TB25,TBW25],['$B$','$BW$'],ncol=2)
#plt.legend([TB25],['$B25_m$'])
#plt.plot(time1, KE1/V, 'k',linewidth=1.5)
plt.xlabel("Time $[days]$",fontsize=18)
plt.ylabel("$<|\mathbf{u}|>$ $[ms^{-1}]$",fontsize=18)
plt.tight_layout()
plt.savefig('./plot/S_t_25.eps')
plt.close()
print 'saved '+'./plot/S_t_25.eps'

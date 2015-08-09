import os, sys
import vtktools
import fluidity_tools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt

#label = sys.argv[1]
#basename = sys.argv[2]


path0 = '/tamay2/mensa/fluidity/m_50_6f/'
path1 = '/tamay2/mensa/fluidity/m_25_1/'
path2 = '/tamay2/mensa/fluidity/m_10_1/'
#path1 = '/tamay2/mensa/fluidity/r_1k_B_1F0/'

#
file0 = 'mli.stat'
filepath0 = path0+file0
stat0 = fluidity_tools.stat_parser(filepath0)

file1 = 'mli.stat'
filepath1 = path1+file1
stat1 = fluidity_tools.stat_parser(filepath1)

file2 = 'mli.stat'
filepath2 = path2+file2
stat2 = fluidity_tools.stat_parser(filepath2)

#file1 = 'ring.stat'
#filepath1 = path1+file1
#stat1 = fluidity_tools.stat_parser(filepath1)

time0 = stat0["ElapsedTime"]["value"]/86400.0
time1 = stat1["ElapsedTime"]["value"]/86400.0
time2 = stat2["ElapsedTime"]["value"]/86400.0
#time1 = stat1["ElapsedTime"]["value"]/86400.0

Temp0 = stat0["BoussinesqFluid"]["Velocity_CG%3"]["l2norm"]
Temp1 = stat1["BoussinesqFluid"]["Velocity_CG%3"]["l2norm"]
Temp2 = stat2["BoussinesqFluid"]["Velocity_CG%3"]["l2norm"]
#KE1 = 0.5*np.sqrt(stat1["BoussinesqFluid"]["Velocity_CG%3"]["l2norm"])

T0 = len(time0)
T1 = len(time1)
T2 = len(time2)

# volume
V = 2000.0*2000.0*50
t1 = 0
# plot KE
fig = plt.figure(figsize=(6,2))
T50, = plt.plot(time0[t1:T0], Temp0[t1:T0]/V, 'r-',linewidth=1.5)
T25, = plt.plot(time1[t1:T1], Temp1[t1:T1]/V, 'g-',linewidth=1.5)
T10, = plt.plot(time2[t1:T2], Temp2[t1:T2]/V, 'b-',linewidth=1.5)
plt.ylim([1e-8, 2.2e-8])
#plt.xlim([0, 20])
plt.xlim([0, 9])
plt.legend([T50,T25,T10],['50m','25m','10m'])
#plt.plot(time1, KE1/V, 'k',linewidth=1.5)
plt.xlabel("Time $[days]$")
plt.ylabel("$l^2$-norm V $[ms^{-1}]$")

#plt.ylim([1.48, 1.52])

plt.savefig('./plot/V_t_50_25_10.eps',bbox_inches='tight')
plt.close()
print 'saved '+'./plot/V_t_50_25_10.eps\n'
#

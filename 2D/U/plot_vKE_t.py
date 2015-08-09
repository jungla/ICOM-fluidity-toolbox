import os, sys
import vtktools
import fluidity_tools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt

#label = sys.argv[1]
#basename = sys.argv[2]


path0 = '/tamay2/mensa/fluidity/r_1k_B_3/'
#path0 = '/tamay2/mensa/fluidity/r_5k_B_0_particles/'
path1 = '/tamay2/mensa/fluidity/r_1k_B_1F1/'

#
file0 = 'ring.stat'
filepath0 = path0+file0
stat0 = fluidity_tools.stat_parser(filepath0)

file1 = 'ring.stat'
filepath1 = path1+file1
stat1 = fluidity_tools.stat_parser(filepath1)

time0 = stat0["ElapsedTime"]["value"]/86400.0
time1 = stat1["ElapsedTime"]["value"]/86400.0

KE0 = 0.5*np.sqrt(stat0["BoussinesqFluid"]["Velocity_CG%3"]["l2norm"])
KE1 = 0.5*np.sqrt(stat1["BoussinesqFluid"]["Velocity_CG%3"]["l2norm"])

# volume
V = (180000.0/2)**2*np.pi*900

# plot KE
fig = plt.figure(figsize=(6,3))
plt.plot(time0, KE0/V, 'k--',linewidth=1.5)
plt.plot(time1, KE1/V, 'k',linewidth=1.5)
plt.xlabel("Time $[days]$")
plt.ylabel("Vertical KE Density $[s^{-2}m^{-1}]$")

#plt.ylim([1.48, 1.52])

plt.savefig('./plot/KEv_t_r_1k.eps',bbox_inches='tight')
plt.close()
print 'saved '+'./plot/KEv_t_r_1k.eps\n'
#

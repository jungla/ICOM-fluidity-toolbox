import os, sys
import vtktools
import fluidity_tools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt

label = sys.argv[1]
basename = sys.argv[2]

path = '/tamay2/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

#
file0 = basename+'.stat'
filepath0 = path+file0
stat0 = fluidity_tools.stat_parser(filepath0)

#file1 = 'mli_checkpoint.stat'
#filepath1 = path+file1
#stat1 = fluidity_tools.stat_parser(filepath1)

time0 = stat0["ElapsedTime"]["value"]/86400.0
#time1 = stat1["ElapsedTime"]["value"]/86400.0

KE0 = 0.5*np.sqrt(stat0["BoussinesqFluid"]["Velocity%magnitude"]["l2norm"])
#KE1 = 0.5*np.sqrt(stat1["BoussinesqFluid"]["Velocity%magnitude"]["l2norm"])

# plot KE
fig = plt.figure()
plt.plot(time0[50:], KE0[50:], color='k',linewidth=1.5)
#plt.plot(time1, KE1, color='k',linewidth=1.5)
plt.xlabel("Time $[days]$")
plt.ylabel("Kinetic Energy $[m^2/s^2]$")

#plt.xlim([-0.000017, 0.00004])
#plt.xticks([-0.00001, 0.00002])
plt.savefig('./plot/'+label+'/KE_t_'+label+'.eps',bbox_inches='tight')
plt.close()
print 'saved '+'./plot/'+label+'/KE_t_'+label+'.eps\n'
#

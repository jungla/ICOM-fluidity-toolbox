import os, sys
import vtktools
import fluidity_tools
import numpy as np
import matplotlib  as mpl
#mpl.use('ps')
import matplotlib.pyplot as plt

label0 = 'm_25_3b'
label1 = 'm_25_3b'
basename1 = 'mli_checkpoint'
basename0 = 'mli'

path0 = '/tamay2/mensa/fluidity/'+label0+'/'
path1 = '/tamay2/mensa/fluidity/'+label1+'/'

try: os.stat('./plot/'+label1)
except OSError: os.mkdir('./plot/'+label1)

#
file0 = basename0+'.stat'
filepath0 = path0+file0
stat0 = fluidity_tools.stat_parser(filepath0)

file1 = basename1+'.stat'
filepath1 = path1+file1
stat1 = fluidity_tools.stat_parser(filepath1)

time0 = stat0["ElapsedTime"]["value"]/86400.0
time1 = stat1["ElapsedTime"]["value"]/86400.0

KE0 = stat0["BoussinesqFluid"]["Temperature_CG"]["l2norm"]/np.sqrt((10000*10000*50))
KE1 = stat1["BoussinesqFluid"]["Temperature_CG"]["l2norm"]/np.sqrt((10000*10000*50))

#time1[time1 > 1] = np.nan
#KE1[time1 > 1] = np.nan

# plot KE
fig = plt.figure()
plt.plot(time0[:], KE0[:], '--k',linewidth=3)
plt.plot(time1[:], KE1[:], '-k',linewidth=3)
plt.xlabel("Time $[days]$", fontsize=22)
plt.ylabel("rms Temperature $[C]$", fontsize=22)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

#plt.xlim([-0.000017, 0.00004])
#plt.xticks([-0.00001, 0.00002])
plt.tight_layout()
plt.savefig('./plot/'+label1+'/T_t_'+label1+'.png',bbox_inches='tight')
plt.close()
print 'saved '+'./plot/'+label1+'/T_t_'+label1+'.png\n'
#

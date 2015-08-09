import os, sys
import vtktools
import fluidity_tools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt

#label = sys.argv[1]
#basename = sys.argv[2]


path0 = './stat_files_bkup/'
label = 'm_25_2b'

file0 = label+'_tracer.stat'
filepath0 = path0+file0
stat0 = fluidity_tools.stat_parser(filepath0)

time0 = stat0["ElapsedTime"]["value"]/86400.0

Temp1 = stat0["BoussinesqFluid"]["Tracer_1_CG"]["integral"]
Temp2 = stat0["BoussinesqFluid"]["Tracer_2_CG"]["integral"]
Temp4 = stat0["BoussinesqFluid"]["Tracer_4_CG"]["integral"]

#Temp1 = stat0["BoussinesqFluid"]["Temperature_CG"]["min"]
#Temp2 = stat0["BoussinesqFluid"]["Temperature_CG"]["min"]
#Temp4 = stat0["BoussinesqFluid"]["Temperature_CG"]["min"]
#Temp2 = stat0["BoussinesqFluid"]["Tracer_2_CG"]["min"]
#Temp4 = stat0["BoussinesqFluid"]["Tracer_4_CG"]["min"]

# plot KE
V = 1
fig = plt.figure(figsize=(6,3))
T1, = plt.plot(time0[np.where(time0<=4)], Temp1[np.where(time0<=4)]/V, 'r-',linewidth=1.5)
T2, = plt.plot(time0[np.where(time0<=4)], Temp2[np.where(time0<=4)]/V, 'g-',linewidth=1.5)
T4, = plt.plot(time0[np.where(time0<=4)], Temp4[np.where(time0<=4)]/V, 'b-',linewidth=1.5)

plt.legend((T1,T2,T4),('2m','5m','17m'))

plt.xlim([2, 4.5])
#plt.xticks(np.linspace(0,5,6),np.linspace(0,5,6).astype(int))
plt.xlabel("Time $[days]$",fontsize=18)
plt.ylabel("integral C",fontsize=18)
plt.tight_layout()
plt.savefig('./plot/Tracer_'+label+'.eps')
plt.close()
print      './plot/Tracer_'+label+'.eps'


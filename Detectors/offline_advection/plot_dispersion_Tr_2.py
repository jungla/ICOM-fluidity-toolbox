#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import myfun
import numpy as np
import os
import fio
import csv
import lagrangian_stats

label = 'm_25_2_512'
#label = 'm_25_1_particles'
dayi  = 0 #10*24*2
dayf  = 269 #10*24*4
days  = 1

#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

time = np.asarray(range(dayi,dayf,days))

timeTr = (time)*1200 + 48*3600 - 1200

depths = [1,5,17]
Trid = [1,2,4]

D_Tr = np.zeros([len(time),len(depths)])

# Tracer second moment

for z in range(len(depths)):
 print z
 f0 = open('D_Tracer_'+str(Trid[z])+'_CG_'+label+'.csv','r')
 r0 = csv.reader(f0)
 vals = []
 for row in r0:
  time,val = row[0].split(', ') 
  vals.append(float(val))
 D_Tr[:,z] = np.asarray(vals[dayi:dayf:days])
 f0.close()


# plot dispersion Tr

nl = len(depths)

pTr, = plt.plot(timeTr/3600.,D_Tr[:,0],color=[0,0,0],linewidth=2)
z = 1
pTr5, = plt.plot(timeTr/3600.,D_Tr[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
z = 2
pTr17, = plt.plot(timeTr/3600.,D_Tr[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)

plt.gca().set_yscale('log')
#plt.gca().set_xscale('log')

plt.xlabel('Time [days]')
plt.xlim([48, 140])
plt.ylabel('Dispersion [m^2]')
plt.legend((pTr,pTr5,pTr17),('Tr 1m','Tr 5m','Tr 17m'),loc=4,fontsize=12)

plt.savefig('./plot/'+label+'/D_Tr_'+label+'.eps')
print       './plot/'+label+'/D_Tr_'+label+'.eps' 
plt.close()

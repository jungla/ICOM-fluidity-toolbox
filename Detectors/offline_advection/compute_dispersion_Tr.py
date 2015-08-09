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
dayf  = 230 #10*24*4
days  = 1

#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

path = '../Tracer_CG/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

time = np.asarray(range(dayi,dayf,days))

timeTr = (time)*1200 + 48*3600 - 1200

depths = [1,5,17]
Trid = [1,2,4]

D_Tr = np.zeros([len(range(dayi,dayf,days)),len(depths)])

# Tracer second moment

Xlist = np.linspace(0,10000,801)
Ylist = np.linspace(0,4000,321)
#Xlist = np.linspace(0,2000,161)
#Ylist = np.linspace(0,2000,161)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = 1.*np.cumsum(dl)

xn = len(Xlist)
yn = len(Ylist)
zn = len(Zlist)

f0 = open('D_Tracer_'+str(Trid[0])+'_CG_'+label+'.csv','w')
w0 = csv.writer(f0)
f1 = open('D_Tracer_'+str(Trid[1])+'_CG_'+label+'.csv','w')
w1 = csv.writer(f1)
f2 = open('D_Tracer_'+str(Trid[2])+'_CG_'+label+'.csv','w')
w2 = csv.writer(f2)

for t in range(len(time)):
 tlabel = str(t)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 for z in range(len(depths)):
  file0 = path+'Tracer_'+str(Trid[z])+'_CG_'+label+'_'+str(time[t])+'.csv'
  #print file0
  Tr = fio.read_Scalar(file0,xn,yn,zn)

  D_Tr[t,z] = lagrangian_stats.tracer_d2_bis(Xlist,Ylist,3500.0,np.sum(Tr,2)/3.0)

 print t

 w0.writerow([str(t)+', '+str(D_Tr[t,0])])
 w1.writerow([str(t)+', '+str(D_Tr[t,1])])
 w2.writerow([str(t)+', '+str(D_Tr[t,2])])
# plot dispersion Tr

f0.close()
f1.close()
f2.close()

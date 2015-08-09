import os, sys
import fio
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import myfun

#label = 'm_25_2_512'
label = 'm_25_1_particles'
dayi  = 0 #10*24*2
dayf  = 600 #10*24*4
days  = 1

#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

path = './Tracer_CG/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

time = range(dayi,dayf,days)

# dimensions archives

# ML exp

#Xlist = np.linspace(0,10000,801)
#Ylist = np.linspace(0,4000,321)
Xlist = np.linspace(0,2000,161)
Ylist = np.linspace(0,2000,161)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = 1.*np.cumsum(dl)

xn = len(Xlist)
yn = len(Ylist)
zn = len(Zlist)

for t in range(0,len(time),10):
 tlabel = str(t)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 for z in [1]:
  file0 = path+'Tracer_'+str(z)+'_CG_'+label+'_'+str(time[t])+'.csv'
  #print file0
  T = fio.read_Scalar(file0,xn,yn,zn)
  #
  v = np.linspace(0, 1, 30, endpoint=True)
  vl = np.linspace(0, 1, 5, endpoint=True)
  #
  fig = plt.figure(figsize=(6,5))
  plt.contourf(Ylist/1000,-1*Zlist,np.transpose(np.mean(T,0)),30,extend='both',cmap=plt.cm.PiYG)
  plt.colorbar()
#  plt.colorbar(ticks=vl)
  plt.title(str(np.round(10*(time[t]*1200/3600.0+24*6))/10.0)+'h')
  plt.ylabel('Z [m]',fontsize=16)
  plt.xlabel('X [km]',fontsize=16)
  plt.savefig('./plot/'+label+'/Tracer_'+str(z)+'_CG_'+label+'_'+str(time[t])+'_v.eps')
  plt.close()
  print       './plot/'+label+'/Tracer_'+str(z)+'_CG_'+label+'_'+str(time[t])+'_v.eps'

import os, sys

import fio, myfun
import vtktools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import lagrangian_stats 
import scipy.interpolate

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

label = 'r_1k_B_1F1'
basename = 'ring' 

#files = '../../days.list'

#f = open(files,'r')
#files = f.readlines()
#f.close()

dayi  = 120
dayf  = 241
days  = 1

files = range(dayi,dayf,days)

#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

path = '/tamay2/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

# dimensions archives
xn = 301
yn = 301
Xlist = np.linspace(-150000,150000,xn)
Ylist = np.linspace(-150000,150000,yn)

depths = [0, 10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 50, 50, 50, 50, 50, 50, 50, 50, 49]
zn = len(depths)
Zlist = -1*np.cumsum(depths)

v = np.zeros((len(Zlist),2))

#for time in range(dayi,dayf,days):
for file in files:
 time = int(file)
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file0 = basename + '_' + str(time) + '.pvtu'
 filepath = path+file0
 file1 = label+'_' + tlabel
 fileout  = path + file1
 #
 print 'opening: ', filepath
 #
 Rho = lagrangian_stats.read_Scalar('/nethome/jmensa/scripts_fluidity/2D/RST/Density_CG/Density_CG_'+label+'_'+str(time),zn,xn,yn,[time])
 #
 #
 for d in range(9,len(Zlist),100):
  if file == files[0]:
   v[d,:] = np.nanmin(Rho[d,:,:,0]), np.nanmax(Rho[d,:,:,0])
  fig = plt.figure(figsize=(7, 6))
  plt.contourf(Xlist/1000.0,Ylist/1000.0,Rho[d,:,:,0],np.linspace(v[d,0],v[d,1],50,endpoint=True),extend='both',cmap=plt.cm.PiYG)
  plt.colorbar(ticks=np.linspace(v[d,0],v[d,1],5,endpoint=True))
  plt.axis('equal')
  plt.xlabel('X (Km)',fontsize=16)
  plt.ylabel('Y (Km)',fontsize=16)
  plt.title('day '+str(time/4.0),fontsize=16)
  plt.savefig('./plot/'+label+'/R_'+file1+'_'+str(Zlist[d])+'_h_snap.eps')
  print       './plot/'+label+'/R_'+file1+'_'+str(Zlist[d])+'_h_snap.eps'
  plt.close()

import scipy.stats as ss
import os, sys
import scipy
from scipy import interpolate
import lagrangian_stats
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import myfun


label = 'r_1k_B_1'
basename = 'ring'
dayi = 0
dayf = 10
days = 1

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days
#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

path = '/tamay2/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

xn = 301
yn = 301
zn = 51

depths = [0, 10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 30, 50, 50, 50, 50, 50, 50, 50, 50, 49]
Zlist = -1*np.cumsum(depths)

Xlist = np.linspace(-150000,150000,301)
Ylist = np.linspace(-150000,150000,301)

[X,Y] = np.meshgrid(Xlist,Ylist)

time = range(dayi,dayf,days)

t = 0

for tt in time:
 tlabel = str(tt)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file1 = label+'_' + tlabel
 #
 Rho = lagrangian_stats.read_Scalar('./Density_CG/Density_CG_'+label+'_'+str(tt),zn,xn,yn,[tt])
 #
 for z in range(5,zn,25):
  # determine speed direction
   #
  plt.figure()
  plt.contourf(Xlist,Ylist,Rho[z,:,:,0],50,extend='both',cmap=plt.cm.PiYG)
  plt.colorbar()
  plt.ylabel('Y [Km]',fontsize=18)
  plt.xlabel('X [Km]',fontsize=18)
  plt.savefig('./plot/'+label+'/T_'+str(z)+'_'+file1+'.eps')
  plt.close()
  print       './plot/'+label+'/T_'+str(z)+'_'+file1+'.eps'

import scipy.stats as ss
import os, sys
import scipy
from scipy import interpolate
import fio 
import numpy as np
import matplotlib  as mpl
#mpl.use('ps')
import matplotlib.pyplot as plt
import myfun


label = 'm_25_2b_particles'
basename = 'mli'
dayi = 0 
dayf = 49
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

xn = 641
yn = 641
zn = 51

Xlist = np.linspace(0,8000,xn)# x co-ordinates of the desired array shape
Zlist = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] 
Zlist = np.linspace(0,-50,zn)
Ylist = np.linspace(0,8000,yn)# y co-ordinates of the desired array shape
[X,Y] = np.meshgrid(Xlist,Ylist)

time = range(dayi,dayf,days)

t = 0

for tt in time:
 tlabel = str(tt)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file1 = label+'_' + tlabel
 #
 Temp = fio.read_Scalar('./Temperature_CG/Temperature_CG_'+label+'_'+str(tt)+'.csv',xn,yn,zn)
 #
 for z in [0]:
  # determine speed direction
   #
  plt.figure()
  plt.contourf(Xlist,Ylist,Temp[:,:,0],50,extend='both',cmap=plt.cm.PiYG)
  plt.colorbar()
  plt.ylabel('Y [Km]',fontsize=18)
  plt.xlabel('X [Km]',fontsize=18)
  plt.savefig('./plot/'+label+'/T_'+str(z)+'_'+file1+'.png')
  plt.close()
  print       './plot/'+label+'/T_'+str(z)+'_'+file1+'.png'

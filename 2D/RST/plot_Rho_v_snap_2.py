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

dayi  = 1
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

[X,Y] = np.meshgrid(Xlist,Ylist)
X = np.reshape(X,(np.size(X),))
Y = np.reshape(Y,(np.size(Y),))

latitude = 0

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
 #
 #
 Rho = lagrangian_stats.read_Scalar('/nethome/jmensa/scripts_fluidity/2D/RST/Density_CG/Density_CG_'+label+'_'+str(time),zn,xn,yn,[time])
 #

 mld = np.zeros(len(Xlist))
 depthi = np.linspace(0,np.max(abs(Zlist)),200)

 for x in range(len(Xlist)):
  ml = Rho[:,x,150,0]
 # print ml
  ml[1:] = Rho[:-1,x,150,0]
 # ml[0] = ml[1] # temp hack to fix extraction
  #
  f = scipy.interpolate.interp1d(abs(Zlist),ml,kind='linear')
  mli = f(depthi)
  mls = np.cumsum(mli)/(range(1,len(depthi)+1))
  mls = np.round(mls*100)
  mli = np.round(mli*100)
  mlst, = np.where(mls>=mli)
  mld[x] = depthi[mlst[-1]]

#plt.plot(np.mean(mld,1))
#plt.savefig('./plot/'+label+'/'+file1+'_MLD.eps',bbox_inches='tight')
#plt.close()

 # Density

 fig = plt.figure(figsize=(6, 7))
 
# for d in depths:
#  plt.axhline(y=d, xmin=-150000, xmax=150000,color='k',linestyle='--')
 if file == files[0]:
  rmin = np.min(Rho[:,:,150,0])
  rmax = np.max(Rho[:,:,150,0])
 v = np.linspace(rmin, rmax, 50, endpoint=True)
# vl = np.linspace(-1e-6, 1e-6, 5, endpoint=True)
 # 
 plt.contourf(Xlist/1000,Zlist/1000.0,Rho[:,:,150,0],v,extend='both',cmap=plt.cm.PiYG)
 plt.colorbar()
 plt.plot(Xlist/1000,-mld/1000.0,'k',linewidth=1)
 plt.title('day '+str(time/4.0))
 plt.xlabel('X (Km)')
 plt.ylabel('Z (Km)')
 plt.savefig('./plot/'+label+'/R_'+file1+'_v_snap.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/R_'+file1+'_v_snap.eps\n'

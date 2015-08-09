import os, sys
import gc
import vtktools
import numpy as np
import matplotlib  as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import myfun
from scipy import interpolate

gc.enable()

label = 'm_50_6b_3D_particles'
basename = 'mli_checkpoint'
dayi = 0
dayf = 1
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

xn = 50
yn = 50
zn = 30


Xlist = np.linspace(0,5000,xn)# x co-ordinates of the desired array shape
Zlist = np.linspace(0,-30,zn)# x co-ordinates of the desired array shape
Ylist = np.linspace(0,5000,yn)# y co-ordinates of the desired array shape


[X,Y,Z] = myfun.meshgrid2(Xlist,Ylist,Zlist)
Y = np.reshape(Y,(np.size(Y),))
X = np.reshape(X,(np.size(X),))
Z = np.reshape(Z,(np.size(Z),))


pts = zip(X,Y,Z)
pts = vtktools.arr(pts)

gc.collect()

for time in range(dayi,dayf,days):
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file0 = basename+'_' + str(time) + '.pvtu'
 filepath = path+file0
 file1 = label+'_' + tlabel
 fileout  = path + file1
 #
 print 'opening ', filepath
 #
 data = vtktools.vtu(filepath)
 print 'extracting points'
 T = data.ProbeData(pts,'Tracer_1_CG')
# del data
 gc.collect()
 #
 Tr = np.squeeze(np.reshape(T,[len(Zlist),len(Xlist),len(Ylist)]))
 Ts = np.sum(Tr,0)
 Ts[Ts < 1e-12] = np.nan
 #
 print 'plotting'
 v = np.linspace(-1e-6, 1e-6, 30, endpoint=True)
 vl = np.linspace(-1e-6, 1e-6, 5, endpoint=True)
 #
 plt.figure()
 plt.contourf(Xlist/1000,Ylist/1000,Ts,50,extend='both',cmap=plt.cm.PiYG)
 # plt.colorbar()
 # plt.colorbar(ticks=vl)
 plt.colorbar()
 plt.ylabel('Y [Km]')
 plt.xlabel('X [Km]')
 plt.title('Density')
 plt.savefig('./plot/'+label+'/Tracer_1_'+file1+'_h_snap.eps')
 plt.close()
 print 'saved '+'./plot/'+label+'/Tracer_1_'+file1+'_h_snap.eps\n'

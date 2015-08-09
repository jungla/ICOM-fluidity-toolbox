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

label = 'm_3k_B_1'
basename = 'ring'
dayi = 50
dayf = 51
days = 1

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days
label = sys.argv[1]
basename = sys.argv[2]
dayi  = int(sys.argv[3])
dayf  = int(sys.argv[4])
days  = int(sys.argv[5])

path = '/tamay2/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)


xn = 300
yn = 300

Xlist = np.linspace(-150000,150000,xn)# x co-ordinates of the desired array shape
Ylist = np.linspace(-150000,150000,yn)# x co-ordinates of the desired array shape

[X,Y] = np.meshgrid(Xlist,Ylist) 
X = np.reshape(X,(np.size(X),))
Y = np.reshape(Y,(np.size(Y),))

gc.collect()

depth = -50.0

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
 data.Crop(np.min(Xlist),np.max(Xlist),np.min(Ylist),np.max(Ylist),depth,depth+100)
 coords = data.GetLocations()
 V = data.GetVectorField('Velocity_CG')
 del data
 gc.collect()
 #
 W = V[np.around(coords[:,2])==depth,2]
 del V
 Cw = coords[np.around(coords[:,2])==depth,:]
 Wr = interpolate.griddata((Cw[:,0],Cw[:,1]),W,(X,Y),method='linear')
 Wr = np.reshape(Wr,[len(Xlist),len(Ylist)])
 #
 #Vt[X**2+Y**2>np.max(coords[:,0])**2]=np.nan
 #
 print 'plotting'
 v = np.linspace(-1e-5, 1e-5, 30, endpoint=True)
 vl = np.linspace(-1e-5, 1e-5, 5, endpoint=True)
 #
 plt.figure()
 plt.contourf(Xlist/1000,Ylist/1000,Wr,v,extend='both',cmap=plt.cm.PiYG)
 # plt.colorbar()
 plt.colorbar(ticks=vl)
 plt.ylabel('Y [Km]')
 plt.xlabel('X [Km]')
 plt.title('W')
 plt.savefig('./plot/'+label+'/W_'+file1+'_'+str(abs(depth))+'_h_snap.pdf')
 plt.close()
 print 'saved '+'./plot/'+label+'/W_'+file1+'_'+str(abs(depth))+'_h_snap.pdf\n'
  # density
#  plt.figure()
#  plt.contourf(Xlist,Ylist,R[d,:,:],50,extend='both',cmap=plt.cm.PiYG)
#  plt.colorbar()
#  plt.ylabel('Longitude')
#  plt.xlabel('Latitude')
#  plt.title('Density')
#  plt.savefig('./plot/'+label+'/R_'+file1+'_'+str(abs(depths[d]))+'.eps',bbox_inches='tight')
#  plt.close()
#  print 'saving'+'./plot/'+label+'/R_'+file1+'_'+str(abs(depths[d]))+'.eps'

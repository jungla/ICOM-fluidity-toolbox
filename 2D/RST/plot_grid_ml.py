import os, sys

import fio, myfun
import vtktools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import gc

gc.enable()

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

label = 'm_50_6f'
basename = 'mli' 
dayi  = 0
dayf  = 1
days  = 1

#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

path = '/tamay2/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)
#

# dimensions archives

for time in range(dayi,dayf,days):
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file0 = basename + '_' + str(time) + '.pvtu'
 filepath = path+file0
 file1 = 'grid_'+label+'_' + tlabel
 fileout  = path + file1
 #
 print 'opening: ', filepath
 #
 #
 data = vtktools.vtu(filepath)
 print 'fields: ', data.GetFieldNames()
 coords = data.GetLocations()
 fig = plt.figure(figsize=(6, 6))
 z = coords[np.around(coords[:,1])==0.0,2] 
 x = coords[np.around(coords[:,1])==0.0,0] 
 plt.triplot(x/1000,z)
 plt.xlim([0,2])
 plt.ylim([-50,0])
 plt.xlabel('X [km]',fontsize=16)
 plt.ylabel('Z [m]',fontsize=16)
 plt.savefig('./plot/'+label+'/'+file1+'_gridV.eps')
 print       './plot/'+label+'/'+file1+'_gridV.eps'
 plt.close()
 #
 fig = plt.figure(figsize=(6, 6))
 x = coords[np.around(coords[:,2])==0.0,0] 
 y = coords[np.around(coords[:,2])==0.0,1] 
 plt.triplot(x/1000,y/1000)
# plt.axis('equal')
 plt.xlim([0,2])
 plt.ylim([0,2])
 plt.xlabel('X [km]',fontsize=16)
 plt.ylabel('Y [km]',fontsize=16)
 plt.savefig('./plot/'+label+'/'+file1+'_gridH.eps')
 print       './plot/'+label+'/'+file1+'_gridH.eps'
 plt.close()

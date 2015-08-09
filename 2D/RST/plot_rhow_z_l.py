import os, sys

import fio, myfun
import vtktools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days

label = sys.argv[1]
dayi  = int(sys.argv[2])
dayf  = int(sys.argv[3])
days  = int(sys.argv[4])

path = '/tamay/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

file0 = 'mli_' + str(1) + '.pvtu'
filepath = path+file0
#
data = vtktools.vtu(filepath)
coords = data.GetLocations()
depths = sorted(list(set(coords[:,2])))

fd = open('./plot/'+label+'/'+label+'_rhow_z_temp.csv','a')

for elem in depths:
 fd.write(', '+ str(elem))
fd.write('\n')

for time in range(dayi,dayf,days):
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file0 = 'mli_' + str(time) + '.pvtu'
 filepath = path+file0
 file1 = 'rhow_'+label+'_' + tlabel
 fileout  = path + file1
 #
 print 'opening: ', filepath
 #
 #
 data = vtktools.vtu(filepath)
 print 'fields: ', data.GetFieldNames()
 print 'extract V, R'
 V = data.GetVectorField('Velocity_CG')
 R = data.GetScalarField('Density_CG')
 rho = []
 w = []
 for d in range(len(depths)):
  rho.append(np.hist(R[coords[:,2]==depths[d]]))
  w.append(np.mean(V[coords[:,2]==depths[d],2]))
 rho = np.asarray(rho)
 w = np.asarray(w)
 #del data
 #
 print 'max: ', (w*rho).max(), 'min: ', (w*rho).min()
 #

 # RHO*W
 fig = plt.figure(figsize=(2,5))
 ax = fig.add_subplot(111)
 rhow_z = rho*w
 plt.plot([0, 0], [min(depths), max(depths)], color='k', linestyle='--', linewidth=1)
 plt.plot(rhow_z,depths,color='0.75')
 plt.xlabel('$<w*rho>$')
 plt.ylabel('Z (m)')
 plt.xlim([-0.002, 0.002])
 plt.xticks([-0.002, 0.002])
 plt.savefig('./plot/'+label+'/'+file1+'_z.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/'+file1+'_z.eps\n'

 # save to csv
 fd.write(str(time))
 for elem in rhow_z:
  fd.write(', '+ str(elem)) 
 fd.write('\n')

 #dRHO*W/dZ 
 fig = plt.figure(figsize=(2,5))
 ax = fig.add_subplot(111)
 rhow_z = np.diff(rho*w)/np.diff(depths)
 plt.plot([0, 0], [min(depths[1:]), max(depths[1:])], color='k', linestyle='--', linewidth=1)
 plt.plot(rhow_z,depths[1:],color='0.75')
 plt.xlabel('$<d(w*rho)/dz>$')
 plt.ylabel('Z (m)')
 plt.xlim([-0.0015, 0.0005])
 plt.xticks([-0.0015, 0.0005])
 plt.savefig('./plot/'+label+'/'+file1+'_z_dz.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/'+file1+'_z_dz.eps\n'
 
 os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'_z.jpg ./plot/'+label+'/'+file1+'_z.eps')
 os.system('mogrify ./plot/'+label+'/'+file1+'_z.jpg -trim ./plot/'+label+'/'+file1+'_z.jpg')
 os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'_z_dz.jpg ./plot/'+label+'/'+file1+'_z_dz.eps')
 os.system('mogrify ./plot/'+label+'/'+file1+'_z_dz.jpg -trim ./plot/'+label+'/'+file1+'_z_dz.jpg')

fd.close()

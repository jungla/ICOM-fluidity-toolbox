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
basename = sys.argv[2]
dayi  = int(sys.argv[3])
dayf  = int(sys.argv[4])
days  = int(sys.argv[5])

path = '/tamay2/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

file0 = basename+'_' + str(1) + '.pvtu'
filepath = path+file0
#
data = vtktools.vtu(filepath)
coords = data.GetLocations()
depths = sorted(list(set(coords[:,2])))

for time in range(dayi,dayf,days):
 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file0 = basename+'_' + str(time) + '.pvtu'
 filepath = path+file0
 file1 = 'w_'+label+'_' + tlabel
 fileout  = path + file1
 #
 print 'opening: ', filepath
 #
 #
 data = vtktools.vtu(filepath)
 print 'fields: ', data.GetFieldNames()
 print 'extract V, R'
 V = data.GetVectorField('Velocity_CG')
 wM = []
 wS = []
 for d in range(len(depths)):
  wM.append(np.mean(abs(V[coords[:,2]==depths[d],2])))
  wS.append(np.std(abs(V[coords[:,2]==depths[d],2])))
 wM = np.asarray(wM)
 wS = np.asarray(wS)
 #del data
 #
 print 'max: ', (wM).max(), 'min: ', (wM).min()
 #
 #
 # W
 fig = plt.figure(figsize=(3,6))
 ax = fig.add_subplot(111)
 plt.plot([0, 0], [min(depths), max(depths)], color='k', linestyle='--', linewidth=1)
 plt.plot(wM,depths,color='0.2')
# plt.plot(wM+wS,depths,color='0.75')
# plt.plot(wM-wS,depths,color='0.75')
 plt.xlabel('$<w>$')
 plt.ylabel('Z (m)')
# plt.xlim([min(wM-wS)*1.1, max(wM+wS)*1.1])
# plt.xticks([min(wM-wS), max(wM+wS)])
# plt.xlim([-0.0000017, 0.000004])
 plt.xlim([-1.5e-6, 1.5e-6])
 plt.xticks([-1e-6, 1e-6])
 plt.savefig('./plot/'+label+'/'+file1+'_z.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/'+file1+'_z.eps\n'
 #
 os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'_z.jpg ./plot/'+label+'/'+file1+'_z.eps')
 os.system('mogrify ./plot/'+label+'/'+file1+'_z.jpg -trim ./plot/'+label+'/'+file1+'_z.jpg')

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
depths = sorted(list(set(coords[:,2])),reverse=True)

XM = max(coords[0,:])
Xm = min(coords[0,:])
YM = max(coords[1,:])
Ym = min(coords[1,:])

X  = np.linspace(Xm,XM,10000/50)
Xl = np.diff(X)/2+X[:-1]
#Y = np.linspace(Ym,YM,10000/50)
#Yl = np.diff(Y)/2+Y[:-1]

flux = []

flux_t = np.linspace(0,24,50)

for t in flux_t:
 tt = t%24/6.0
 if tt >=0 and tt < 1:
  flux.append(-250 + 1000.0*tt)
 elif tt >= 1 and tt < 2:
  flux.append(1750 - 1000.0*tt)
 else:
  flux.append(-250.0)

def fflux(t):
 tt = t%24/6.0
 if tt >=0 and tt < 1:
  return -250 + 1000.0*tt
 elif tt >= 1 and tt < 2:
  return 1750 - 1000.0*tt
 else:
  return -250.0


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
 #
 V = data.GetVectorField('Velocity_CG')
 R = data.GetScalarField('Density_CG')
 #
 rho = np.zeros([len(depths),len(Xl)])
 w = np.zeros([len(depths),len(Xl)])
 for d in range(len(depths)):
  rho_t = R[np.where(coords[:,2]==depths[d])]
  w_t, = V[np.where(coords[:,2]==depths[d]),2]
  x_t, = coords[np.where(coords[:,2]==depths[d]),0]
  for x in range(len(Xl)):
   rho[d,x] = np.mean(rho_t[np.where(x_t[:] > X[x]) and np.where(x_t[:] <= X[x+1])])
   w[d,x] = np.mean(w_t[np.where(x_t[:] > X[x]) and np.where(x_t[:] <= X[x+1])])

 # create arrays of velocity and temperature values at the desired points
 #
 # Msh-3.2$ vi plot_rhow_z.py

 mld = []

 for row in range(len(Xl)):
  ml = rho[:,row]
  mls = np.cumsum(ml)/range(1,len(ml)+1)
  mlst, = np.where(mls<=ml)
  mld.append((depths[mlst[0]]+depths[mlst[0]+1])/2.0)

 plt.plot(mld)
 plt.savefig('./plot/'+label+'/'+file1+'_MLD.eps',bbox_inches='tight')
 plt.close()

 rl = np.linspace(0.9955,0.9965,15)

# TOTAL flux

 fig = plt.figure(figsize=(6, 8)) 
 gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5]) 

 plt.subplot(gs[0])
 plt.plot(flux_t, flux, 'r--')
 plt.plot(time%24,fflux(time%24),'ok')
 plt.autoscale(enable=True, axis='x', tight=True)

 plt.subplot(gs[1])
 v = np.linspace(-0.005, 0.005, 20, endpoint=True)
 vl = np.linspace(-0.005, 0.005, 5, endpoint=True)
 plt.contourf(Xl,depths,w*rho,v,extend='both',cmap=plt.cm.PiYG)
 plt.contourf(Xl,depths,w*rho,v,extend='both',cmap=plt.cm.PiYG)
 plt.contourf(Xl,depths,w*rho,v,extend='both',cmap=plt.cm.PiYG)
 plt.colorbar(ticks=vl,orientation='horizontal')
 plt.contour(Xl,depths,rho,rl,colors='k',linewidths=1)
 #plt.autumn()
 plt.xlabel('X (m)')
 plt.ylabel('Z (m)')
 # plt.xticks(range(lati,lonf,1000),(range(0,15,1)))
 # plt.yticks(range(depthi,depthf,10),(range(0,15,1)))
 plt.title('$rho*w$')

 plt.savefig('./plot/'+label+'/'+file1+'.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/'+file1+'.eps\n'

 # Vertical velocity

 fig = plt.figure(figsize=(6, 8))
 gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5])

 plt.subplot(gs[0])
 plt.plot(flux_t, flux, 'r--')
 plt.plot(time%24,fflux(time%24),'ok')
 plt.autoscale(enable=True, axis='x', tight=True)

 plt.subplot(gs[1])
 v = np.linspace(-0.0001, 0.0001, 20, endpoint=True)
 vl = np.linspace(-0.0001, 0.0001, 5, endpoint=True)
 plt.contourf(Xl,depths,w,v,extend='both',cmap=plt.cm.PiYG)
 plt.contourf(Xl,depths,w,v,extend='both',cmap=plt.cm.PiYG)
 plt.contourf(Xl,depths,w,v,extend='both',cmap=plt.cm.PiYG)
 plt.colorbar(ticks=vl,orientation='horizontal')
 plt.contour(Xl,depths,rho,rl,colors='k',linewidths=1)
 plt.xlabel('X (m)')
 plt.ylabel('Z (m)')
 # plt.xticks(range(lati,lonf,1000),(range(0,15,1)))
 # plt.yticks(range(depthi,depthf,10),(range(0,15,1)))
 plt.title('$w$')

 plt.savefig('./plot/'+label+'/'+file1+'_w.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/'+file1+'_w.eps\n'

 
 os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'_p.jpg ./plot/'+label+'/'+file1+'_p.eps')
 os.system('mogrify ./plot/'+label+'/'+file1+'_p.jpg -trim ./plot/'+label+'/'+file1+'_n.jpg')
 os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'_n.jpg ./plot/'+label+'/'+file1+'_n.eps')
 os.system('mogrify ./plot/'+label+'/'+file1+'_n.jpg -trim ./plot/'+label+'/'+file1+'_n.jpg')
 os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'.jpg ./plot/'+label+'/'+file1+'.eps')
 os.system('mogrify ./plot/'+label+'/'+file1+'.jpg -trim ./plot/'+label+'/'+file1+'.jpg')
 os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'_w.jpg ./plot/'+label+'/'+file1+'_w.eps')
 os.system('mogrify ./plot/'+label+'/'+file1+'_w.jpg -trim ./plot/'+label+'/'+file1+'_w.jpg')

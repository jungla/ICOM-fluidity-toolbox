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

Tmin = 17.00
Tmax = 21.00

lati = 0
loni = 0
depthi = 0

latf = 1000
lonf = 1000
depthf = -200

# dimensions archives

xstep = 5
ystep = 100
zstep = -4

Xlist = np.arange(loni,lonf+xstep,xstep)# x co-ordinates of the desired array shape
Ylist = np.arange(lati,latf+ystep,ystep)# y co-ordinates of the desired array shape
Zlist = np.arange(depthi,depthf+zstep,zstep)# y co-ordinates of the desired array shape
[X,Y,Z] = myfun.meshgrid2(Xlist,Ylist,Zlist)
Y = np.reshape(Y,(np.size(Y),))
X = np.reshape(X,(np.size(X),))
Z = np.reshape(Z,(np.size(Z),))

pts = zip(X,Y,Z)
pts = vtktools.arr(pts)

# define flux

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


#fd = open('./plot/'+label+'/'+label+'_rhow_z.csv','a')

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
 V = data.ProbeData(pts, 'Velocity_CG')
 R = data.ProbeData(pts, 'Density_CG')
 print 'done.'
 w = np.reshape(V[:,2],[len(Zlist),len(Xlist),len(Ylist)])
 rho = np.reshape(R,[len(Zlist),len(Xlist),len(Ylist)])
 #del data
 #
 print 'max: ', (rho).max(), 'min: ', (rho).min()
 #
 #
 # create arrays of velocity and temperature values at the desired points
 #
 mld = np.zeros([len(Xlist),len(Ylist)])
 #
 for x in range(len(Xlist)):
  for y in range(len(Ylist)):
   ml = rho[:,x,y]
   mls = np.cumsum(ml)/range(1,len(ml)+1)
   mlst, = np.where(mls>=ml)
   mld[x,y] = ((Zlist[mlst[len(mlst)-1]]))

#plt.plot(np.mean(mld,1))
#plt.savefig('./plot/'+label+'/'+file1+'_MLD.eps',bbox_inches='tight')
#plt.close()

 rl = np.linspace(1024.5, 1026.5,15)

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
 plt.contourf(Xlist,Zlist,np.mean(w*rho,2),v,extend='both',cmap=plt.cm.PiYG)
 plt.contourf(Xlist,Zlist,np.mean(w*rho,2),v,extend='both',cmap=plt.cm.PiYG)
 plt.contourf(Xlist,Zlist,np.mean(w*rho,2),v,extend='both',cmap=plt.cm.PiYG)
 plt.colorbar(ticks=vl,orientation='horizontal')
 plt.contour(Xlist,Zlist,np.mean(rho,2),rl,colors='k',linewidths=1)
 plt.plot(Xlist,np.mean(mld,1),'r-')
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
 v = np.linspace(1024.5, 1026.5, 20, endpoint=True)
 vl = np.linspace(1024.5, 1026.5, 5, endpoint=True)
 plt.contourf(Xlist,Zlist,np.mean(w,2),v,extend='both',cmap=plt.cm.PiYG)
 plt.contourf(Xlist,Zlist,np.mean(w,2),v,extend='both',cmap=plt.cm.PiYG)
 plt.contourf(Xlist,Zlist,np.mean(w,2),v,extend='both',cmap=plt.cm.PiYG)
 plt.colorbar(ticks=vl,orientation='horizontal')
 plt.contour(Xlist,Zlist,np.mean(rho,2),rl,colors='k',linewidths=1)
 plt.plot(Xlist,np.mean(mld,1),'r-')
 plt.xlabel('X (m)')
 plt.ylabel('Z (m)')
 # plt.xticks(range(lati,lonf,1000),(range(0,15,1)))
 # plt.yticks(range(depthi,depthf,10),(range(0,15,1)))
 plt.title('$w$')

 plt.savefig('./plot/'+label+'/'+file1+'_r.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/'+file1+'_r.eps\n'

 # Density

 fig = plt.figure(figsize=(6, 8))
 gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5])

 plt.subplot(gs[0])
 plt.plot(flux_t, flux, 'r--')
 plt.plot(time%24,fflux(time%24),'ok')
 plt.autoscale(enable=True, axis='x', tight=True)
 
 plt.subplot(gs[1])
 v = np.linspace(-0.0005, 0.0005, 20, endpoint=True)
 vl = np.linspace(-0.0005, 0.0005, 5, endpoint=True)
 plt.contourf(Xlist,Zlist,np.mean(rho,2),v,extend='both',cmap=plt.cm.PiYG)
 plt.contourf(Xlist,Zlist,np.mean(rho,2),v,extend='both',cmap=plt.cm.PiYG)
 plt.contourf(Xlist,Zlist,np.mean(rho,2),v,extend='both',cmap=plt.cm.PiYG)
 plt.colorbar(ticks=vl,orientation='horizontal')
 plt.contour(Xlist,Zlist,np.mean(rho,2),rl,colors='k',linewidths=1)
 plt.plot(Xlist,np.mean(mld,1),'r-')
 plt.xlabel('X (m)')
 plt.ylabel('Z (m)')
 # plt.xticks(range(lati,lonf,1000),(range(0,15,1)))
 # plt.yticks(range(depthi,depthf,10),(range(0,15,1)))
 plt.title('$w$')

 plt.savefig('./plot/'+label+'/'+file1+'_w.eps',bbox_inches='tight')
 plt.close()
 print 'saved '+'./plot/'+label+'/'+file1+'_w.eps\n'


 os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'.jpg ./plot/'+label+'/'+file1+'.eps')
 os.system('mogrify ./plot/'+label+'/'+file1+'.jpg -trim ./plot/'+label+'/'+file1+'.jpg')
 os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'_w.jpg ./plot/'+label+'/'+file1+'_w.eps')
 os.system('mogrify ./plot/'+label+'/'+file1+'_w.jpg -trim ./plot/'+label+'/'+file1+'_w.jpg')
 os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaBits=4 -r300 -sOutputFile=./plot/'+label+'/'+file1+'_r.jpg ./plot/'+label+'/'+file1+'_r.eps')
 os.system('mogrify ./plot/'+label+'/'+file1+'_r.jpg -trim ./plot/'+label+'/'+file1+'_r.jpg')

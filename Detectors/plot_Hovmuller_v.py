#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import numpy as np
import vtktools
import myfun
import os
import spectrum

exp = 'r_3k_B_1F0_s'
filename ='/tamay2/mensa/fluidity/'+exp+'/ring_checkpoint.detectors'
#filename2 = '/tamay2/mensa/fluidity/'+exp+'/ring_30.pvtu'


#data = vtktools.vtu(filename2)
#coords = data.GetLocations()
#depths = sorted(list(set(coords[:,2])))

#Xlist = np.arange(-180000,180000+10000,10000)# x co-ordinates of the desired array shape
#Ylist = np.arange(0,1)*0.0
#Zlist = np.arange(-10,-900,-10)# y co-ordinates of the desired array shape
#[X,Y,Z] = myfun.meshgrid2(Xlist,Ylist,Zlist)
#Y = np.reshape(Y,(np.size(Y),))
#X = np.reshape(X,(np.size(X),))
#Z = np.reshape(Z,(np.size(Z),))

#pts = zip(X,Y,Z)
#pts = vtktools.arr(pts)

#R = data.ProbeData(pts, 'Density_CG')
#rho = np.reshape(R,[len(Zlist),len(Ylist),len(Xlist)])

try: os.stat('./plot/'+exp)
except OSError: os.mkdir('./plot/'+exp)

print 'reading detectors'
det = fluidity_tools.stat_parser(filename)
#keys = det.keys()				 # particles
#print 'done.' 

time = det['ElapsedTime']['value']/86400
tt, = time.shape
pt = 890
step = 1

z = range(-10,-900,-10)
x = np.linspace(0,400,10)**2

par = np.zeros((pt,3,tt))
#time = np.linspace(1800/86400.0,1800*(tt+1)/86400.0,1800/86400.0)

# read particles

pvel = np.zeros((pt,3,tt))
pdns = np.zeros((pt,tt)) 
ptmp = np.zeros((pt,tt))

for d in range(pt):
 temp = det['BoussinesqFluid']['Velocity']['static_detectors_'+myfun.digit(d+1,3)]
 pvel[d,:,:] = temp[:,0:tt]
 temp = det['BoussinesqFluid']['Temperature']['static_detectors_'+myfun.digit(d+1,3)]
 ptmp[d,:] = temp[0:tt]
 temp = det['BoussinesqFluid']['Density']['static_detectors_'+myfun.digit(d+1,3)]
 pdns[d,:] = temp[0:tt]

pwt = pvel[:,2,:]
pu = pvel[:,0,:]
pv = pvel[:,1,:]

pw = np.reshape(pwt,[len(z),len(x),tt])
pt = np.reshape(ptmp,[len(z),len(x),tt])
pke = (np.reshape(pu,[len(z),len(x),tt])**2)*(np.reshape(pu,[len(z),len(x),tt])**2)/2.0

#prb = 7

for prb in range(len(x)):
 v = np.linspace(np.percentile(np.squeeze(pw[:,prb,:]),1),np.percentile(pw[:,prb,:],99), 25, endpoint=True)
 plt.figure()
 plt.contourf(time,z,np.squeeze(pw[:,prb,:]),v,extend='both',cmap='jet')
 #plt.scatter(apoint[:,0],apoint[:,1],marker='.',s=0.1)
 #plt.scatter(par[:,0,999],par[:,2,999])
 plt.colorbar()
 plt.savefig('./plot/'+exp+'/Hovmuller_W_'+exp+'_'+str(prb)+'.eps',bbox_inches='tight')
 plt.close()
 #
 v = np.linspace(np.percentile(np.squeeze(pt[:,prb,:]),1),np.percentile(pt[:,prb,:],99), 25, endpoint=True)
 plt.figure()
 plt.contourf(time,z,np.squeeze(pt[:,prb,:]),v,extend='both',cmap='jet')
 #plt.scatter(apoint[:,0],apoint[:,1],marker='.',s=0.1)
 #plt.scatter(par[:,0,999],par[:,2,999])
 plt.colorbar()
 plt.savefig('./plot/'+exp+'/Hovmuller_T_'+exp+'_'+str(prb)+'.eps',bbox_inches='tight')
 plt.close()
 #
 v = np.linspace(np.percentile(np.squeeze(pke[:,prb,:]),1),np.percentile(pke[:,prb,:],99), 25, endpoint=True)
 plt.figure()
 plt.contourf(time,z,np.squeeze(pke[:,prb,:]),v,extend='both',cmap='jet')
 #plt.scatter(apoint[:,0],apoint[:,1],marker='.',s=0.1)
 #plt.scatter(par[:,0,999],par[:,2,999])
 plt.colorbar()
 plt.savefig('./plot/'+exp+'/Hovmuller_HKE_'+exp+'_'+str(prb)+'.eps',bbox_inches='tight')
 plt.close()
 #




## SPECTRUM
# W at probe n5 - x[5]

# 150m depth - z[14]
pw = np.reshape(pwt,[len(z),len(x),tt])
pw = np.reshape(pwt,[len(z),len(x),tt])
# 500m depth - z[49]

prb = 5
prz = 14

for prz in range(9,49,10):
 pwd = np.squeeze(pw[prz,prb,:-1])
 #
 # plot timeseries
 plt.figure()
 #plt.plot(time,pwd,color=[0.5, 0.5, 0.5],linewidth=1.5)
 plt.plot(pwd,color=[0.5, 0.5, 0.5],linewidth=1.5)
 #plt.xlim([min(time), max(time)])
 plt.savefig('./plot/'+exp+'/Hovmuller_sec_W_'+exp+'_'+str(prb)+'_'+str(z[prz])+'.eps',bbox_inches='tight')
 plt.close()
 #
 # plot spectrum
 #
 pwd = pwd - np.mean(pwd)
 del p
 p = spectrum.Periodogram(pwd, sampling=383)
 p.run()
 window = spectrum.window.create_window(20, 'hamming')
 #
 #from numpy.fft import fft
 #n = len(pwd)
 #I_w = np.abs(fft(pwd))**2 / n
 #w =  np.arange(n) / n
 #w, I_w = w[:int(n/2)+1], I_w[:int(n/2)+1]  # Take only values on [0, pi]
 #
 psdw = np.convolve(window/window.sum(),p.psd,mode='same')
 #  
 plt.figure()
 plt.plot(p.frequencies(),np.log(p.psd))
 plt.plot(p.frequencies(),np.log(psdw))
 #p = spectrum.pyule(pwd, 50, NFFT=tt)
 #p.run()
 #
 plt.savefig('./plot/'+exp+'/Hovmuller_PSD_'+exp+'_'+str(prb)+'_'+str(z[prz])+'.eps',bbox_inches='tight')
 plt.close()

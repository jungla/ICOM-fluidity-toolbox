import os, sys
import fio, myfun
import vtktools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv

## READ archive (too many points... somehow)
# args: label

label = sys.argv[1]

path = '/nethome/jmensa/scripts_fluidity/2D/RST/plot/'+label+'/'+label+'_rhow_z.csv'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

depthi = 0
depthf = -200

# dimensions archives

timef = 192

val = np.zeros([102,timef]) 
time = []
t = 0

with open(path, 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        #time.append() # change later
        i = 0
        for item in row:
         if item==' ':
          item = np.nan
         val[i,t] = float(item)
#         print float(item),i
         i = i+1
        t = t+1

timei = 48
timef = 192

time = val[0,timei:timef]
val = val[1:,timei:timef]

rl = np.linspace(0.9955,0.9965,15)

twrho = np.reshape(val,-1)
Ptwrho = np.reshape(val,-1)
Ntwrho = np.reshape(val,-1)
 #
Ptwrho[twrho<0]=np.nan
Ntwrho[twrho>0]=np.nan
Pval = np.log(Ptwrho)
Nval = np.log(abs(Ntwrho))
Pval[Pval<-7]=np.nan
Nval[Nval<-7]=np.nan
Pval = np.reshape(Pval,val.shape)
Nval = np.reshape(Nval,val.shape)


flux = []

for t in time:
 if t > 1:
  tt = t%24/6.0
  if tt >=0 and tt < 1:
   flux.append(-250 + 1000.0*tt)
  elif tt >= 1 and tt < 2:
   flux.append(1750 - 1000.0*tt)
  else:
   flux.append(-250.0)
 else:
  flux.append(0)


# Full fluxes

fig = plt.figure(figsize=(6, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5])

plt.subplot(gs[0])
plt.plot(time, flux, 'r--')
plt.autoscale(enable=True, axis='x', tight=True)

plt.subplot(gs[1])
v = np.linspace(-np.max(abs(np.min(val)),abs(np.max(val)))*0.1, np.max(abs(np.min(val)),abs(np.max(val)))*0.1, 50, endpoint=True)
vl = np.linspace(-np.max(abs(np.min(val)),abs(np.max(val)))*0.1, np.max(abs(np.min(val)),abs(np.max(val)))*0.1, 5, endpoint=True)
plt.contourf(time,range(depthi,depthf-2,-2),val,v,extend='both',cmap=plt.cm.PiYG)
plt.colorbar(ticks=vl,orientation='horizontal')
#plt.contour(time,range(depthi,depthf-2,-2),val,[0.0])

#plt.contourf(range(depthi,depthf-2,-2),time,val)
#plt.contourf(Xlist,Zlist,Nwrho,v,extend='both')
#plt.contour(Xlist,Zlist,rho,rl,colors='k',linewidths=1)
#plt.autumn()
plt.xlabel('time (hr)')
plt.ylabel('Z (m)')
plt.ylim(-50,0)
# plt.xticks(range(lati,lonf,1000),(range(0,15,1)))
# plt.yticks(range)(depthi,depthf,10),(range(0,15,1)))
#plt.title(r'$log(|rho*w_-|)$')

plt.savefig('./plot/'+label+'/'+label+'_rhow_series.eps',bbox_inches='tight')
plt.close()
print 'saved '+'./plot/'+label+'/'+label+'_rhow_series.eps'+'\n'

# Negative fluxes

fig = plt.figure(figsize=(6, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5])

plt.subplot(gs[0])
plt.plot(time, flux, 'r--')
plt.autoscale(enable=True, axis='x', tight=True)

plt.subplot(gs[1])
v = np.linspace(np.nanmin(Nval)*0.8, np.nanmax(Nval)*0.8, 50, endpoint=True)
vl = np.linspace(np.nanmin(Nval), np.nanmax(Nval), 5, endpoint=True)
plt.contourf(time,range(depthi,depthf-2,-2),Nval,v,extend='both')
plt.colorbar(ticks=vl,orientation='horizontal')
#plt.contour(time,range(depthi,depthf-2,-2),val,[0.0])

plt.xlabel('time (hr)')
plt.ylabel('Z (m)')
plt.ylim(-50,0)
plt.savefig('./plot/'+label+'/'+label+'_rhow_series_n.eps',bbox_inches='tight')
plt.close()
print 'saved '+'./plot/'+label+'/'+label+'_rhow_series_n.eps'+'\n'


# Positive fluxes

fig = plt.figure(figsize=(6, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5])

plt.subplot(gs[0])
plt.plot(time, flux, 'r--')
plt.autoscale(enable=True, axis='x', tight=True)

plt.subplot(gs[1])
v = np.linspace(np.nanmin(Pval)*0.8, np.nanmax(Pval)*0.8, 50, endpoint=True)
vl = np.linspace(np.nanmin(Pval), np.nanmax(Pval), 5, endpoint=True)
plt.contourf(time,range(depthi,depthf-2,-2),Pval,v,extend='both')
plt.colorbar(ticks=vl,orientation='horizontal')
#plt.contour(time,range(depthi,depthf-2,-2),val,[0.0])

plt.xlabel('time (hr)')
plt.ylabel('Z (m)')
plt.ylim(-50,0)
plt.savefig('./plot/'+label+'/'+label+'_rhow_series_p.eps',bbox_inches='tight')
plt.close()
print 'saved '+'./plot/'+label+'/'+label+'_rhow_series_p.eps'+'\n'





os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaits=4 -r300 -soutputfile=./plot/'+label+'/'+label+'_rhow_series.jpg ./plot/'+label+'/'+label+'_rhow_series.eps')
os.system('mogrify ./plot/'+label+'/'+label+'_rhow_series.jpg -trim ./plot/'+label+'/'+label+'_rhow_series.jpg')
os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaits=4 -r300 -soutputfile=./plot/'+label+'/'+label+'_rhow_series_n.jpg ./plot/'+label+'/'+label+'_rhow_series_n.eps')
os.system('mogrify ./plot/'+label+'/'+label+'_rhow_series_n.jpg -trim ./plot/'+label+'/'+label+'_rhow_series_n.jpg')
os.system('gs -sDEVICE=jpeg -dJPEGQ=100 -dNOPAUSE -dBATCH -dSAFER -dTextAlphaits=4 -r300 -soutputfile=./plot/'+label+'/'+label+'_rhow_series_p.jpg ./plot/'+label+'/'+label+'_rhow_series_p.eps')
os.system('mogrify ./plot/'+label+'/'+label+'_rhow_series_p.jpg -trim ./plot/'+label+'/'+label+'_rhow_series_p.jpg')

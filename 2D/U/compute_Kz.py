import csv
from scipy import optimize
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt

label = 'm_25_3b'
time = range(1,80,1)

def uEkman(z,kz):
 global V0
 f = 0.0001 
 a = np.sqrt(f/(2*kz))
 return V0*np.exp(-z*a)*np.cos(np.pi/4.-a*z)

def vEkman(z,kz):
 global V0
 f = 0.0001
 a = np.sqrt(f/(2*kz))
 return V0*np.exp(-z*a)*np.sin(np.pi/4.-a*z)

kx = []
ky = []

for t in time:
 file = './Ekman/UVW_Velocity_CG_'+label+'_'+str(t)+'.csv'
 f = open(file,'r')
 reader = csv.reader(f)
 reader.next()
 depth = []
 u = []
 v = []
 w = []
 for row in reader:
  depth.append(float(row[2]))
  u.append(float(row[3]))
  v.append(float(row[4]))
  #w.append(float(row[5]))
 f.close()


 d = 36

 depth = -1*np.asarray(depth[:d])
 ur = np.asarray(u[:d])
 vr = np.asarray(v[:d])
 #wr = np.asarray(w[:d])

 # rotate currents
 u = -vr
 v = ur

 V0 = np.sqrt(u[0]**2 + v[0]**2)

 sigma = 1 #np.linspace(0.1,1,36)**(-2)

 outU,cov = optimize.curve_fit(uEkman, depth, u, [0.0001], sigma, maxfev=100000)
 outV,cov = optimize.curve_fit(vEkman, depth, v, [0.0001], sigma, maxfev=100000)
 f = 0.0001
# print outU, outV
 kx.append(outU[0])
 ky.append(outV[0])

 au = np.sqrt(f/(2*outU[0]))
 av = np.sqrt(f/(2*outV[0]))
 ue = V0*np.exp(-depth*au)*np.cos(np.pi/4.-au*depth)
 ve = V0*np.exp(-depth*av)*np.sin(np.pi/4.-av*depth)

 pue, = plt.plot(ue,-depth,'k',linewidth=2)
 pu,= plt.plot(u,-depth,'r',linewidth=2)
 perr,= plt.plot(u-ue,-depth,'k--',linewidth=2)
 plt.legend([pue,pu,perr],['$Ekman$','$BW25_m$','$BW25_m$- $Ekman$'],loc=4)
 plt.xlabel(r'u $[ms^{-1}]$', fontsize=18)
 plt.ylabel(r'depth $[m]$', fontsize=18)
 plt.xticks(fontsize=16)
 plt.yticks(fontsize=16)
 plt.savefig('./plot/'+label+'/Ekman_u_'+label+'_'+str(t)+'.eps')
 print       './plot/'+label+'/Ekman_u_'+label+'_'+str(t)+'.eps'
 plt.close()

 pue, = plt.plot(ve,-depth,'k',linewidth=2)
 pu, = plt.plot(v,-depth,'r',linewidth=2)
 perr, = plt.plot(v-ve,-depth,'k--',linewidth=2)
 plt.legend([pue,pu,perr],['$Ekman$','$BW25_m$','$BW25_m$- $Ekman$'],loc=4)
 plt.xlabel(r'v $[ms^{-1}]$', fontsize=18)
 plt.ylabel(r'depth $[m]$', fontsize=18)
 plt.xticks(fontsize=16)
 plt.yticks(fontsize=16)

 plt.savefig('./plot/'+label+'/Ekman_v_'+label+'_'+str(t)+'.eps')
 print       './plot/'+label+'/Ekman_v_'+label+'_'+str(t)+'.eps'
 plt.close()

plt.subplots(figsize=(8,4))
pkx, = plt.plot(time,kx,'k',linewidth=2)
pky, = plt.plot(time,ky,'r',linewidth=2)
plt.legend([pkx,pky],['$K_z$ from $u$','$K_z$ from $v$'],loc=1)
plt.xlabel(r'Time $[hr]$', fontsize=18)
plt.ylabel(r'$K_z$ $[m^2/s]$', fontsize=18)
plt.xticks(np.linspace(0,72,7),np.linspace(0,72,7).astype(int),fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig('./plot/'+label+'/Ekman_k_'+label+'.eps')
print       './plot/'+label+'/Ekman_k_'+label+'.eps'
plt.close()


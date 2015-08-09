import csv
from scipy import optimize
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt

label_BW = 'm_25_2b_particles'
label_B = 'm_25_1b_particles'
time = range(0,240,1)

timel = np.asarray(time)*1440 + 48*3600

def vEkman(z,kz):
 global V0
 f = 0.0001
 a = np.sqrt(f/(2*kz))
 return V0*np.exp(-z*a)*np.sin(np.pi/4.-a*z)

dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = -1*np.cumsum(dl)

for t in time:
 file = './Velocity_CG/z/Velocity_CG_2_'+label_B+'_'+str(t)+'_z.csv'
 f = open(file,'r')
 reader = csv.reader(f)
 w_B = []
 for row in reader:
  w_B.append(row[:])
 f.close()

 file = './Velocity_CG/z/Velocity_CG_2_'+label_BW+'_'+str(t)+'_z.csv'
 f = open(file,'r')
 reader = csv.reader(f)
 w_BW = []
 for row in reader:
  w_BW.append(row[:])
 f.close()


 w_B = np.asarray(w_B[0]).astype(float)
 w_BW = np.asarray(w_BW[0]).astype(float)

 plt.subplots(figsize=(7,7))
 pBW, = plt.plot(w_B,Zlist,'k',linewidth=2)
 pB, = plt.plot(w_BW,Zlist,'k--',linewidth=2)
 plt.xlabel(r'<w> $[ms^{-1}]$', fontsize=24)
 plt.ylabel(r'depth $[m]$', fontsize=24)
 plt.xticks(np.linspace(-3e-6,3e-6,4),np.linspace(-3e-6,3e-6,4),fontsize=20)
 plt.yticks(fontsize=20)
 plt.tight_layout()
 plt.savefig('./plot/'+label_BW+'/W_'+label_BW+'_'+str(t)+'.eps')
 print       './plot/'+label_BW+'/W_'+label_BW+'_'+str(t)+'.eps'
 plt.close()


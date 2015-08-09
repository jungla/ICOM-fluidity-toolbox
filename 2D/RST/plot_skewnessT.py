import scipy.stats as ss
import os, sys
import scipy
from scipy import interpolate
import lagrangian_stats
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import myfun


label = 'm_50_7'
basename = 'mli'
dayi = 4*3
dayf = 2*4*3
days = 8

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days
#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

path = '/tamay2/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

xn = 200
yn = 200
zn = 51

Xlist = np.linspace(0,10000,xn)# x co-ordinates of the desired array shape
Zlist = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] 
Zlist = np.linspace(0,-50,zn)
Ylist = np.linspace(0,10000,yn)# y co-ordinates of the desired array shape
[X,Y] = np.meshgrid(Xlist,Ylist)

time = range(dayi,dayf,days)
angle = np.linspace(0,2*np.pi,50)
Sk = np.zeros((len(angle),len(time)))

t = 0

for tt in time:
 tlabel = str(tt)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file1 = label+'_' + tlabel
 #
 Temp = lagrangian_stats.read_Scalar('./Temperature_CG/Temperature_CG_'+label+'_'+str(tt),zn,xn,yn,[tt])
 U = lagrangian_stats.read_Scalar('/nethome/jmensa/scripts_fluidity/2D/U/csv/Velocity_CG_0_'+label+'_'+str(tt),zn,xn,yn,[tt])
 V = lagrangian_stats.read_Scalar('/nethome/jmensa/scripts_fluidity/2D/U/csv/Velocity_CG_1_'+label+'_'+str(tt),zn,xn,yn,[tt])
 #
 Xi = np.reshape(X,len(Xlist)*len(Ylist),1)-np.mean(X)
 Yi = np.reshape(Y,len(Xlist)*len(Ylist),1)-np.mean(Y)
 for z in range(5,zn,25):
  # determine speed direction
  Uz = np.mean(np.mean(U[z,:,:,0],0),0)  
  Vz = np.mean(np.mean(V[z,:,:,0],0),0)  
  vangle = np.arctan2(Vz,Uz) + 2*np.pi
  print vangle 
  for a in range(len(angle)):
   print 'angle:', angle[a]
   Mr = [[np.cos(angle[a]), -np.sin(angle[a])],[np.sin(angle[a]), np.cos(angle[a])]]
#
   coords = []
#
   for i in range(len(Xlist)*len(Ylist)):
    coords.append(np.matrix([Xi[i],Yi[i]])*np.matrix(Mr))
#
   coords = np.squeeze(np.asarray(coords))
   data = np.reshape(Temp[z,:,:,0],len(Xlist)*len(Ylist),1)
   Tempi = interpolate.griddata((Xi,Yi),data,(coords[:,0],coords[:,1]),method='linear')
   Tempi = np.reshape(Tempi,(len(Ylist),len(Xlist)))
#
   Tx,Ty = np.gradient(Tempi)
   Ty = Ty[~np.isnan(Ty)]
   Sk[a,t] = scipy.stats.skew(Ty)

#   plt.figure()
#   plt.contourf(Xlist-5000,Ylist-5000,Tempi,10,extend='both',cmap=plt.cm.PiYG)
#   plt.colorbar()
#   plt.ylabel('Y [Km]',fontsize=18)
#   plt.xlabel('X [Km]',fontsize=18)
#   plt.savefig('./plot/'+label+'/T_'+str(z)+'_'+file1+'_'+str(np.round(angle[a]*100))+'.eps')
#   plt.close()
#   print       './plot/'+label+'/T_'+str(z)+'_'+file1+'_'+str(np.round(angle[a]*100))+'.eps'

#
   #
  plt.figure()
  plt.contourf(Xlist-5000,Ylist-5000,np.reshape(data,(len(Ylist),len(Xlist))),10,extend='both',cmap=plt.cm.PiYG)
  plt.colorbar()
  plt.ylabel('Y [Km]',fontsize=18)
  plt.xlabel('X [Km]',fontsize=18)
  plt.savefig('./plot/'+label+'/T_'+str(z)+'_'+file1+'.eps')
  plt.close()
  print       './plot/'+label+'/T_'+str(z)+'_'+file1+'.eps'

  plt.figure()
  plt.plot(angle/np.pi,Sk[:,t],'k-',linewidth=2)

# P.arrow( x, y, dx, dy, **kwargs )
  plt.arrow(vangle/(np.pi)+0.5, 0, 0, -0.5, fc="k", ec="k", head_width=0.05, head_length=0.1 )
  plt.arrow(vangle/(np.pi)-0.5, 0, 0, 0.5, fc="k", ec="k", head_width=0.05, head_length=0.1 )
  plt.plot([0, 2],[0, 0],'k--')
  plt.xlabel(r'$\Theta$ $[\pi]$',fontsize=18)
  plt.ylim(-1.5,1.5)
#  plt.xlim(0,2)
  plt.ylabel('Skewness',fontsize=18)
  plt.savefig('./plot/'+label+'/T_Krt_'+str(z)+'_'+file1+'.eps')
  plt.close()
  print       './plot/'+label+'/T_Krt_'+str(z)+'_'+file1+'.eps'

 t = t + 1

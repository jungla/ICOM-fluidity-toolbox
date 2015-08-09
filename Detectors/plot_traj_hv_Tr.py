#!~/python
import gc
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import myfun
import numpy as np
import os
import vtktools
from matplotlib.patches import Ellipse

label = 'm_50_6b_3D_particles'
basename = 'mli_checkpoint'
dayi = 1
dayf = 100
days = 3

exp2D = 'm_50_6b_2D_particles'
exp3D = 'm_50_6b_3D_particles'

filename = './mli_checkpoint.detectors'

filename3D = '/tamay2/mensa/fluidity/'+exp3D+'/'+filename
filename2D = '/tamay2/mensa/fluidity/'+exp2D+'/'+filename

print 'Reading ', filename2D, filename3D

#try: os.stat('./output/'+exp3D)
#except OSError: os.mkdir('./output/'+exp3D)

path = '/tamay2/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

det3D = fluidity_tools.stat_parser(filename3D)
det2D = fluidity_tools.stat_parser(filename2D)

# pt same for 3D and 2D
pt = int(os.popen('grep position '+filename3D+'| wc -l').read()) # read the number of particles grepping all the positions in the file
time3D = det3D['ElapsedTime']['value']
time2D = det2D['ElapsedTime']['value']

tt = min(len(time2D),len(time3D))

if len(time2D) < len(time3D):
 time = time2D[:tt]
else:
 time = time3D[:tt]

depths = [1, 5, 11, 17, 26]



print 'particles:',pt
print 'timesteps:',tt
print 'depths', depths

par3D = np.zeros((pt,3,tt))
par2D = np.zeros((pt,3,tt))

for d in xrange(pt):
 temp3D = det3D['particles_'+myfun.digit(d+1,len(str(pt)))]['position']
 par3D[d,:,:] = temp3D[:,0:tt]
 temp2D = det2D['particles_'+myfun.digit(d+1,len(str(pt)))]['position']
 par2D[d,:,:] = temp2D[:,0:tt]

xn = 50
yn = 50
zn = 30


Xlist = np.linspace(0,5000,xn)# x co-ordinates of the desired array shape
Zlist = np.linspace(0,-30,zn)# x co-ordinates of the desired array shape
Ylist = np.linspace(0,5000,yn)# y co-ordinates of the desired array shape


[X,Y,Z] = myfun.meshgrid2(Xlist,Ylist,Zlist)
Y = np.reshape(Y,(np.size(Y),))
X = np.reshape(X,(np.size(X),))
Z = np.reshape(Z,(np.size(Z),))

pts = zip(X,Y,Z)
pts = vtktools.arr(pts)

# horizontal

for time in range(dayi,dayf,days):

 t = time * 3.0 # particle time

 tlabel = str(time)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file0 = basename+'_' + str(time) + '.pvtu'
 filepath = path+file0
 file1 = label+'_' + tlabel
 fileout  = path + file1
 #
 print 'opening ', filepath
 #
 data = vtktools.vtu(filepath)


 for z in range(len(depths)):
  print 'depth', depths[z]
  T = data.ProbeData(pts,'Tracer_'+str(z+1)+'_CG')
  #
  Tr = np.squeeze(np.reshape(T,[len(Zlist),len(Xlist),len(Ylist)]))
  Ts = np.sum(Tr,0)
  Ts[Ts < 0.1] = np.nan

  par2Dz = np.reshape(par2D,(20,20,30,3,tt))
  par3Dz = np.reshape(par3D,(20,20,40,3,tt))
 
  par2Dzr = par2Dz[:,:,depths[z],:,:]
  par3Dzr = par3Dz[:,:,depths[z],:,:]
 
  par2Dz = np.reshape(par2Dzr,(400,3,tt))
  par3Dz = np.reshape(par3Dzr,(400,3,tt))

  #
  print 'plotting'
  v = np.linspace(-1e-6, 1e-6, 30, endpoint=True)
  vl = np.linspace(-1e-6, 1e-6, 5, endpoint=True)
  #
  fig = plt.figure(figsize=(8,8))
  ax = fig.add_subplot(111, aspect='equal')
  plt.contour(Xlist/1000,Ylist/1000,Ts/3,[0.1, 0.9],colors=('grey'),linewidths=1.5)
#  plt.contourf(Xlist/1000,Ylist/1000,Ts/3,[0.1, 0.9],colors=('silver'),linewidths=1.5)

  plt.plot((1,1),(3,4),'k')
  plt.plot((1,2),(3,3),'k')
  plt.plot((2,2),(3,4),'k')
  plt.plot((1,2),(4,4),'k')
  #
  s3D = plt.scatter(par3Dz[:,0,t]/1000.0, par3Dz[:,1,t]/1000.0, marker='.', s=35, facecolor='r', lw = 0)
  #
  s2D = plt.scatter(par2Dz[:,0,t]/1000.0, par2Dz[:,1,t]/1000.0, marker='.', s=35, facecolor='b', lw = 0)
  #
  plt.legend((s3D,s2D),('3D','2D'))
  #
  print 'Saving 2D to eps'
  # 
  xt3 = par3Dz[:,0,t] - np.mean(par3Dz[:,0,t])
  yt3 = par3Dz[:,1,t] - np.mean(par3Dz[:,1,t])
  xt2 = par2Dz[:,0,t] - np.mean(par2Dz[:,0,t])
  yt2 = par2Dz[:,1,t] - np.mean(par2Dz[:,1,t])

  cov3 = np.cov(xt3, yt3)
  lambda_3, v = np.linalg.eig(cov3)
  lambda_3 = np.sqrt(lambda_3)
  theta3 = np.rad2deg(0.5*np.arctan2(2*cov3[1,0],(cov3[0,0]-cov3[1,1])))

  cov2 = np.cov(xt2, yt2)
  lambda_2, v = np.linalg.eig(cov2)
  lambda_2 = np.sqrt(lambda_2)
  theta2 = np.rad2deg(0.5*np.arctan2(2*cov2[1,0],(cov2[0,0]-cov2[1,1])))

#  [theta3,maj3,min3] = princax.princax(xt3+yt3*1j)
#  [theta2,maj2,min2] = princax.princax(xt2+yt2*1j)

  e0 = Ellipse(xy=(np.mean(par3Dz[:,0,t])/1000,np.mean(par3Dz[:,1,t])/1000),width=2*lambda_3[0]/1000,height=2*lambda_3[1]/1000,angle=theta3)  
  e1 = Ellipse(xy=(np.mean(par2Dz[:,0,t])/1000,np.mean(par2Dz[:,1,t])/1000),width=2*lambda_2[0]/1000,height=2*lambda_2[1]/1000,angle=theta2)  

  ax.add_artist(e0)
  e0.set_facecolor('none')
  e0.set_edgecolor('k')
  e0.set_linewidth(1.5)

  ax.add_artist(e1)
  e1.set_facecolor('none')
  e1.set_edgecolor('k')
  e1.set_linewidth(1.5)
  e1.set_linestyle('dashed')

  plt.xlim([0, 5])
  plt.ylim([0, 5])
  plt.xlabel('X [km]',fontsize=18)
  plt.ylabel('Y [km]',fontsize=18)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.title(str(time/3)+' hr',fontsize=18)
  plt.savefig('./plot/m_50_6_23D/traj_Tr_'+exp2D+'_z'+str(depths[z])+'_'+str(time)+'_h.eps')
  print './plot/m_50_6_23D/traj_Tr_'+exp2D+'_z'+str(depths[z])+'_'+str(time)+'_h.eps'
  plt.close()
 
  # plot ellipse

 # vertical
  Ts = np.sum(Tr,2)
  Ts[Ts < 0.1] = np.nan

  fig = plt.figure(figsize=(8,8))
  plt.contourf(Xlist,Zlist,Ts,50,extend='both',cmap=plt.cm.PiYG)
  plt.plot((1000,1000),(0,-20),'k')
  plt.plot((2000,2000),(0,-20),'k')
  #
  s3D = plt.scatter(par3Dz[:,0,t-1], par3Dz[:,2,t-1],  marker='o', color='r')
  #
  s2D = plt.scatter(par2Dz[:,0,t-1], par2Dz[:,2,t-1], marker='o', color='b')
  #
  plt.legend((s3D,s2D),('3D','2D'))
  plt.xlim([0, 5000])
  plt.ylim([-20, 0])
  #
  print 'Saving 2D to eps'
  # 

  plt.savefig('./plot/m_50_6_23D/traj_Tr_'+exp2D+'_z'+str(depths[z])+'_'+str(time)+'_v.eps')
  print './plot/m_50_6_23D/traj_Tr_'+exp2D+'_z'+str(depths[z])+'_'+str(time)+'_v.eps'
  plt.close()

 del data
 gc.collect()

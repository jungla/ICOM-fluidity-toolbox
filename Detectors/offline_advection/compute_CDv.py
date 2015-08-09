#!~/python
import numpy as np
import lagrangian_stats
import advect_functions
import csv


def CD_t_v(parz,tt):
 temp = np.zeros((len(parz[:,0,0]),len(parz[0,0,:])))
 temp[:,:] = (parz[:,2,:] - np.mean(parz[:,2,:],0))**2
 return np.mean(temp,0)

# read offline
print 'reading offline'

#label = 'm_25_2_big'
label = 'm_25_2b_particles'
#filename = 'traj_m_25_2_512_0_500_big.csv'
filename = './traj/traj_'+label+'_60_500_3D_big.csv'
tt = 500-60 # IC + 24-48 included

#B
x0 = range(0,5010,10)
y0 = range(0,5010,10)
#y0 = range(0,4010,10)
z0 = [0,5,10,15]

xp = len(x0)
yp = len(y0)
zp = len(z0)
pt = xp*yp*zp

time, par = advect_functions.read_particles_csv(filename,pt,tt)
time = (time)*1440
time = time[:-1]

depths = [5, 10, 15] 
depthid = [1, 2, 3] 

nl = len(depths)

CD = np.zeros((tt,nl))

for z in range(len(depths)):
 print z
 print 'depth', depths[z]
 parz = np.reshape(par,(xp,yp,zp,3,tt))
 parzr = parz[:,:,depthid[z],:,:]
 #
 parz = np.reshape(parzr,(xp*yp,3,tt))
 #
# AD[:,z] = np.sqrt(lagrangian_stats.AD_t(parz,tt))
# AD[:,z] = np.sqrt(lagrangian_stats.AD_t(parz,tt))
# CD[:,z] = lagrangian_stats.CD_t(parz,tt)
# CD[:,z] = lagrangian_stats.CD_t(parz,tt)
# ED[:,z] = lagrangian_stats.ED_t(parz,tt)
# ED[:,z] = lagrangian_stats.ED_t(parz,tt)
 CD[:,z] = CD_t_v(parz,tt)
 
# save to file
f = open('./csv/CDv_'+label+'.csv','w')
print    './csv/CDv_'+label+'.csv'
writer = csv.writer(f)
writer.writerow(('time',depths[0],depths[1],depths[2]))
for i in range(len(time)-1):
 writer.writerow((time[i],CD[i,0],CD[i,1],CD[i,2]))
f.close()


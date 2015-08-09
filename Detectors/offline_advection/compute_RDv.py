#!~/python
import numpy as np
import lagrangian_stats
import advect_functions
import csv


def RD_t_v(parz0,parz1,tt):
 RD_m = [] #np.zeros((px+py,tt))
 RD_m.append(np.mean(np.mean(((parz1[:,:,2,:] - parz0[:,:,2,:])**2),0),0))
 return np.asarray(RD_m)


# read offline
print 'reading offline'

#label = 'm_25_2_big'
label = 'm_25_2b_particles'
#filename = 'traj_m_25_2_512_0_500_big.csv'
filename = './traj/traj_'+label+'_60_500_3Dv.csv'
tt = 500-60 # IC + 24-48 included

#B
x0 = range(0,8000,100)
y0 = range(0,8000,100)
z0 = range(0,52,2)
z0 = [5,5.5,10,10.5,15,15.5]

xp = len(x0)
yp = len(y0)
zp = len(z0)
pt = xp*yp*zp

time, par = advect_functions.read_particles_csv(filename,pt,tt)
time = (time)*1440
time = time[:-1]

depths = [5, 10, 15] 
depthid = [0, 2, 4] 

nl = len(depths)

RD = np.zeros((tt,nl))

for z in range(len(depths)):
 print z
 print 'depth', depths[z]
 parz = np.reshape(par,(xp,yp,zp,3,tt))

 RD[:,z] = RD_t_v(parz[:,:,depthid[z],:,:],parz[:,:,depthid[z]+1,:,:],tt)
 
# save to file
f = open('./csv/RDv_'+label+'.csv','w')
print    './csv/RDv_'+label+'.csv'
writer = csv.writer(f)
writer.writerow(('time',depths[0],depths[1],depths[2]))
for i in range(len(time)-1):
 writer.writerow((time[i],RD[i,0],RD[i,1],RD[i,2]))
f.close()


#!~/python
import numpy as np
import lagrangian_stats
import advect_functions
import csv

def RD_t(par2Dzr,tt,px,py):
 RD_2Dm = [] #np.zeros((px+py,tt))
 for i in range(px):
  RD_2Dm.append(np.mean(((par2Dzr[i+1,:,0,:] - par2Dzr[i,:,0,:])**2 + (par2Dzr[i+1,:,1,:] - par2Dzr[i,:,1,:])**2),0))
 for j in range(py):
  RD_2Dm.append(np.mean(((par2Dzr[:,j+1,0,:] - par2Dzr[:,j,0,:])**2 + (par2Dzr[:,j+1,1,:] - par2Dzr[:,j,1,:])**2),0))
 return np.mean(RD_2Dm,0)

# read offline
print 'reading offline'

#label = 'm_25_2_big'
label = 'm_25_1b_particles'
dim = '2D'
filename2D = './traj/traj_'+label+'_60_500_'+dim+'_big.csv'
tt = 500-60 # IC + 24-48 included

#B
x0 = range(0,5010,10)
y0 = range(0,5010,10)
#y0 = range(0,4010,10)
z0 = [0,5,10,15]

#BW
#x0 = range(0,9010,10)
#y0 = range(0,3010,10)
#z0 = [0,5,10,15]

xp = len(x0)
yp = len(y0)
zp = len(z0)
pt = xp*yp*zp

time2D, par2D = advect_functions.read_particles_csv(filename2D,pt,tt)
par2D = lagrangian_stats.periodicCoords(par2D,8000,8000)

#
time2D = (time2D)*1440
#    
time = time2D[:-1]

depths = [5, 10, 15] 
depthid = [1, 2, 3] 

nl = len(depths)

RD_2D = np.zeros((tt,nl))
CD_2D = np.zeros((tt,nl))
ED_2D = np.zeros((tt,nl))

for z in range(len(depths)):
 print z
 print 'depth', depths[z]
 par2Dz = np.reshape(par2D,(xp,yp,zp,3,tt))
 par2Dzr = par2Dz[:,:,depthid[z],:,:]
 #
 par2Dz = np.reshape(par2Dzr,(xp*yp,3,tt))
 #
# AD_2D[:,z] = np.sqrt(lagrangian_stats.AD_t(par2Dz,tt))
# AD_3D[:,z] = np.sqrt(lagrangian_stats.AD_t(par3Dz,tt))
# CD_2D[:,z] = lagrangian_stats.CD_t(par2Dz,tt)
# CD_3D[:,z] = lagrangian_stats.CD_t(par3Dz,tt)
# ED_2D[:,z] = lagrangian_stats.ED_t(par2Dz,tt)
# ED_3D[:,z] = lagrangian_stats.ED_t(par3Dz,tt)
 RD_2D[:,z] = lagrangian_stats.RD_t(par2Dzr,tt,xp-1,yp-1)
 CD_2D[:,z] = lagrangian_stats.CD_t(par2Dz,tt)
 ED_2D[:,z] = lagrangian_stats.ED_t(par2Dz,tt)
 
# save to file
f = open('./csv/RD_'+dim+'_'+label+'.csv','w')
writer = csv.writer(f)
writer.writerow(('time',depths[0],depths[1],depths[2]))
for i in range(len(time)-1):
 writer.writerow((time[i],RD_2D[i,0],RD_2D[i,1],RD_2D[i,2]))
f.close()

f = open('./csv/CD_'+dim+'_'+label+'.csv','w')
writer = csv.writer(f)
writer.writerow(('time',depths[0],depths[1],depths[2]))
for i in range(len(time)-1):
 writer.writerow((time[i],CD_2D[i,0],CD_2D[i,1],CD_2D[i,2]))
f.close()

f = open('./csv/ED_'+dim+'_'+label+'.csv','w')
writer = csv.writer(f)
writer.writerow(('time',depths[0],depths[1],depths[2]))
for i in range(len(time)-1):
 writer.writerow((time[i],ED_2D[i,0],ED_2D[i,1],ED_2D[i,2]))
f.close()

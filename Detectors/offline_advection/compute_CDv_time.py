#!~/python
import numpy as np
import lagrangian_stats
import advect_functions
import csv

# read offline
print 'reading offline'

#label = 'm_25_2_big'
label = 'm_25_1b_particles'
#filename2D = 'traj_m_25_2_512_0_500_2D_big.csv'

def CD_t_v(parz,tt):
 temp = np.zeros((len(parz[:,0,0]),len(parz[0,0,:])))
 temp[:,:] = (parz[:,2,:] - np.mean(parz[:,2,:],0))**2
 return np.mean(temp,0)

for tday in range(60,240,2):
 dayi  = tday+0 #10*24*1  
 dayf  = tday+15 #10*24*4
 days  = 1

 filename2D = './traj/traj_'+label+'_'+str(dayi)+'_'+str(dayf)+'_3Db.csv'
 print filename2D
 tt = 500-60 # IC + 24-48 included

 x0 = range(0,8000,100)
 y0 = range(0,8000,100)
 z0 = range(0,52,2) #[0,5,10,15]

 xp = len(x0)
 yp = len(y0)
 zp = len(z0)
 pt = xp*yp*zp
 
 time2D, par2D = advect_functions.read_particles_csv(filename2D,pt,tt)
 #par2D = lagrangian_stats.periodicCoords(par2D,8000,8000)
 
 #
 time2D = (time2D)*1440
 #    
 time = time2D[:-1]
 
 depths = z0 
 depthid = range(len(z0)) 
 
 nl = len(depths)
 
 CD_2D = np.zeros((tt,nl))
 
 for z in range(len(depths)):
  print z
  print 'depth', depths[z]
  par2Dz = np.reshape(par2D,(xp,yp,zp,3,tt))
  par2Dzr = par2Dz[:,:,depthid[z],:,:]
  #
  par2Dz = np.reshape(par2Dzr,(xp*yp,3,tt))
  #
  CD_2D[:,z] = CD_t_v(par2Dz,tt)
  
 # save to file
 f = open('./csv/CD_vb_'+label+'_'+str(dayi)+'_'+str(dayf)+'.csv','w')
 print    './csv/CD_vb_'+label+'_'+str(dayi)+'_'+str(dayf)+'.csv'
 writer = csv.writer(f)
 row = []
 row.append('time')
 for z in range(len(depths)):
  row.append(depths[z])
 writer.writerow(row)
 for i in range(len(time)-1):
  row = []
  row.append(time[i])
  for z in range(len(depths)):
   row.append(CD_2D[i,z])
  writer.writerow((row))
 f.close()
 

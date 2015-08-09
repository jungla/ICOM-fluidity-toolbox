#!~/python
import numpy as np
import lagrangian_stats
import advect_functions
import csv

# read offline
print 'reading offline'

#label = 'm_25_2_big'
label = 'm_25_2b_particles'
#filename2D = 'traj_m_25_2_512_0_500_2D_big.csv'

def RD_t_v(parz0,parz1,tt):
 RD_m = [] #np.zeros((px+py,tt))
 RD_m.append(np.mean(np.mean(((parz1[:,:,2,:] - parz0[:,:,2,:])**2),0),0))
 return np.asarray(RD_m)

for tday in range(60,240,2):
 dayi  = tday+0 #10*24*1  
 dayf  = tday+100 #10*24*4
 days  = 1

 filename2D = './traj/traj_'+label+'_'+str(dayi)+'_'+str(dayf)+'_3Dv.csv'
 print filename2D
 tt = dayf-dayi # IC + 24-48 included

 x0 = range(0,8000,100)
 y0 = range(0,8000,100)
# z0 = range(0,52,2) #[0,5,10,15]
 z0 = [5,5.5,10,10.5,15,15.5]

 xp = len(x0)
 yp = len(y0)
 zp = len(z0)
 pt = xp*yp*zp
 
 time2D, par = advect_functions.read_particles_csv(filename2D,pt,tt)
 #par2D = lagrangian_stats.periodicCoords(par2D,8000,8000)
 
 #
 time2D = (time2D)*1440
 #    
 time = time2D[:-1]
 
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
 f = open('./csv/RD_v_'+label+'_'+str(dayi)+'_'+str(dayf)+'.csv','w')
 print    './csv/RD_v_'+label+'_'+str(dayi)+'_'+str(dayf)+'.csv'
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
   row.append(RD[i,z])
  writer.writerow((row))
 f.close()
 

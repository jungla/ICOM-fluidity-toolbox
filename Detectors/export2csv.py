#!~/python
import fluidity_tools
import myfun
import numpy as np
import os
import csv
import lagrangian_stats

exp2D = 'm_50_7_2Db_particles'
exp3D = 'm_50_7_3Db_particles'
filename = './mli_checkpoint.detectors'
filename3D = '/tamay2/mensa/fluidity/'+exp3D+'/'+filename
filename2D = '/tamay2/mensa/fluidity/'+exp2D+'/'+filename
filename2Dr = '/tamay2/mensa/fluidity/'+exp2D+'/'+'./mli_checkpoint_checkpoint.detectors'

print 'Reading ', filename2D, filename3D, filename2Dr

#try: os.stat('./output/'+exp3D)
#except OSError: os.mkdir('./output/'+exp3D)

time3D,par3D = lagrangian_stats.read_particles(filename3D)
time2D,par2D = lagrangian_stats.read_particles(filename2D)
time2Dr,par2Dr = lagrangian_stats.read_particles(filename2Dr)

time2D = np.hstack((time2D[time2D<time2Dr[0]],time2Dr))
par2D = np.concatenate((par2D[:,:,time2D<time2Dr[0]],par2Dr),2)

tt3 = len(time3D)
tt2 = len(time2D)

if len(time2D) < len(time3D):
 time = time2D[:tt]
else:
 time = time3D[:tt]

pt = par2D

print 'particles:',pt
print 'timesteps:',tt

f = open('particles_2D.csv', 'wb')
writer = csv.writer(f)

for t in range(tt3):
 for d in range(len(par2D[:,0,0])):
  writer.writerow((d,par2D[d,0,t],par2D[d,1,t],par2D[d,2,t],t))

f.close()

f = open('particles_3D.csv', 'wb')
writer = csv.writer(f)
  
for t in range(tt3):
 for d in range(len(par3D[:,0,0])):
  writer.writerow((d,par3D[d,0,t],par3D[d,1,t],par3D[d,2,t],t))

f.close()

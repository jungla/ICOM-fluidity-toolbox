import csv
import numpy as np
import fio

dayi = 0
dayf = 241
days = 1

basename = 'Velocity_CG_2_m_25_2b_particles'
#basename = 'drate_m_25_1b_particles'

#Xlist = np.linspace(0,10000,801)
#Ylist = np.linspace(0,4000,321)
Xlist = np.linspace(0,8000,641)
Ylist = np.linspace(0,8000,641)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = 1.*np.cumsum(dl)

xn = yn = 641
zn = 36

timeD = range(dayi,dayf,days)

for t in timeD:
 print 'read drate', t
 # read
 file_W = './Velocity_CG/'+basename+'_'+str(t)+'.csv'
 W = fio.read_Scalar(file_W,xn,yn,zn)
 Wz = np.mean(np.mean(W,0),0)
 print Wz

 f = open('./Velocity_CG/z/'+basename+'_'+str(t)+'_z.csv','w')
 writer = csv.writer(f)
 writer.writerow((list(Wz)))
 f.close()

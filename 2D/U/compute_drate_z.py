import csv
import numpy as np

dayi = 60
dayf = 241
days = 1

basename = 'drate_v_m_25_1b_particles'
#basename = 'drate_m_25_1b_particles'

#Xlist = np.linspace(0,10000,801)
#Ylist = np.linspace(0,4000,321)
Xlist = np.linspace(0,8000,641)
Ylist = np.linspace(0,8000,641)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = 1.*np.cumsum(dl)

timeD = range(dayi,dayf,days)

for t in timeD:
 print 'read drate', t
 drate = np.zeros((len(Xlist),len(Ylist),len(Zlist)))
 # read
 with open('./drate_3+1day/'+basename+'_'+str(t)+'_3D.csv', 'rb') as csvfile:
  spamreader = csv.reader(csvfile)
  j = 0; k = 0
  for row in spamreader:
   j = j + 1
   if j == len(Ylist): k = k + 1; j = 0
   if k == len(Zlist): k = 0
   drate[:,j,k] = row[::-1]

 drateD = np.mean(np.mean(drate,0),0)

 f = open('./drate_3+1day/z/'+basename+'_'+str(t)+'_z.csv','w')
 writer = csv.writer(f)
 writer.writerow((list(drateD)))
 f.close()

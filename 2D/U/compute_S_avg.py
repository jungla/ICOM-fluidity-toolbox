import scipy.stats as ss
import os, sys
import scipy
from scipy import interpolate
import fio 
import numpy as np
import matplotlib  as mpl
#mpl.use('ps')
import matplotlib.pyplot as plt
import myfun


label = 'm_25_1b'
basename = 'mli'
dayi = 0 
dayf = 49
days = 1

path = '/tamay2/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

xn = 641
yn = 641
zn = 51

time = np.asarray(range(dayi,dayf,days))*1440

S_avg = []
S_time = []

for tt in range(len(time)):
 print tt
 tlabel = str(tt)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file1 = label+'_' + tlabel
 #
 U = fio.read_Scalar('./Velocity_CG/Velocity_CG_0_'+label+'_'+str(tt)+'.csv',xn,yn,zn)
 V = fio.read_Scalar('./Velocity_CG/Velocity_CG_1_'+label+'_'+str(tt)+'.csv',xn,yn,zn)
 #
 S_avg.append(np.mean(np.sqrt(U**2+V**2)))
 S_time.append(time[tt])

import csv

f = open('./Savg_'+label+'.csv','w')
writer = csv.writer(f)
for tt in range(len(time)):
 writer.writerow((S_time[tt],S_avg[tt]))

f.close()



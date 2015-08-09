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

W_avg = []
W_time = []

for tt in range(len(time)):
 print tt
 tlabel = str(tt)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file1 = label+'_' + tlabel
 #
 W = fio.read_Scalar('./Velocity_CG/Velocity_CG_2_'+label+'_'+str(tt)+'.csv',xn,yn,zn)
 #
 W_avg.append(np.mean(abs(W)))
 W_time.append(time[tt])

import csv

f = open('./Wavg_'+label+'.csv','w')
writer = csv.writer(f)
for tt in range(len(time)):
 writer.writerow((W_time[tt],W_avg[tt]))

f.close()



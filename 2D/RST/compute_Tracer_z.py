import os, sys
import fio 
import numpy as np
import csv

label = 'm_25_1b_tracer'
basename = 'mli'
dayi = 0 
dayf = 181
days = 1

path = '/tamay2/mensa/fluidity/'+label+'/'

xn = 641
yn = 641
zn = 36

time = range(dayi,dayf,days)

T_avg = []

for tt in time:
 print tt
 #
 T = fio.read_Scalar('./Tracer_CG/Tracer_1_CG_'+label+'_'+str(tt)+'.csv',xn,yn,zn)
 #
 T[np.where(T>1)]=1
 T[np.where(T<0)]=0
 T_avg = np.mean(np.mean(T,0),0)
 print T_avg

 f = open('./Tracer_1_CG_'+label+'_'+str(tt)+'_z_clip.csv','w')
 writer = csv.writer(f)
 writer.writerow((T_avg))
 
 f.close()

import lagrangian_stats
import numpy as np
import myfun
import csv

label = 'm_25_1'
basename = 'mli'
dayi = 0
dayf = 100
days = 1

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days
#label = sys.argv[1]
#basename = sys.argv[2]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

path = '/tamay2/mensa/fluidity/'+label+'/'
basename = 'Temperature_CG'

xn = 81
yn = 81
zn = 51

Xlist = np.linspace(0,2000,xn)# x co-ordinates of the desired array shape
Ylist = np.linspace(0,2000,yn)# y co-ordinates of the desired array shape
[X,Y] = np.meshgrid(Xlist,Ylist)
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = np.cumsum(dl)

time = range(dayi,dayf,days)

t = 0

for tt in time:
 tlabel = str(tt)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file1 = basename+'_'+label+'_' + tlabel + '.csv'
 print file1

 f = open(file1,'wr')
 writer = csv.writer(f)

 #
 Temp = lagrangian_stats.read_Scalar('./RST/'+basename+'/'+basename+'_'+label+'_'+str(tt)+'.csv',zn,xn,yn)
 #
 for k in range(len(Zlist)):
  # determine speed direction
  #
  for i in range(len(Xlist)):
   for j in range(len(Ylist)):
    writer.writerow([Xlist[i],Ylist[j],Zlist[k],Temp[k,i,j]])

 f.close()

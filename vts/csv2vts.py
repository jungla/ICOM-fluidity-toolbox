import fio
import numpy as np
import myfun
from evtk.hl import gridToVTK
import sys

dayi = 100
dayf = 101
days = 1 

## READ archive (too many points... somehow)
# args: name, dayi, dayf, days
# python csv2vts.py m_25_2b Temperature_CG 2D/RST/Temperature_CG/
label = sys.argv[1]
basename = sys.argv[2]
path = sys.argv[3]
#dayi  = int(sys.argv[3])
#dayf  = int(sys.argv[4])
#days  = int(sys.argv[5])

#xn = 801
#yn = 321
xn = 641
yn = 641

Xlist = np.linspace(0,8000,xn)# x co-ordinates of the desired array shape
Ylist = np.linspace(0,8000,yn)# y co-ordinates of the desired array shape
#Xlist = np.linspace(0,10000,xn)# x co-ordinates of the desired array shape
#Ylist = np.linspace(0,4000,yn)# y co-ordinates of the desired array shape
dl = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1]
Zlist = -1*np.cumsum(dl)

zn = len(Zlist)

time = range(dayi,dayf,days)

t = 0

for tt in time:
 tlabel = str(tt)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file1 = './vts/'+basename+'_'+label+'_' + tlabel
 print file1

 var = np.zeros((xn,yn,zn))
 x = np.zeros((xn,yn,zn))
 y = np.zeros((xn,yn,zn))
 z = np.zeros((xn,yn,zn))
 #
 Temp = fio.read_Scalar(path+'/'+basename+'_'+label+'_'+str(tt)+'.csv',xn,yn,zn)
 #
 for k in range(len(Zlist)):
  for i in range(len(Xlist)):
   for j in range(len(Ylist)):
    var[i,j,k] = Temp[i,j,k]
    x[i,j,k] = Xlist[i]
    y[i,j,k] = Ylist[j]
    z[i,j,k] = Zlist[k] 

 gridToVTK(file1, x,y,z, pointData = {basename : var})


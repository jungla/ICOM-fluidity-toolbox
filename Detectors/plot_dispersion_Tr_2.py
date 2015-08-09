#!~/python
import fluidity_tools
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import myfun
import numpy as np
import os
import csv


label = 'm_50_7'
label = label+'_3D_particles'
basename = 'mli_checkpoint'
dayi = 0
dayf = 60
days = 1

time = range(dayi,dayf,days)

path = '/tamay2/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

xn = 200
yn = 20
zn = 1

delta = 3

Xlist = np.linspace(0,10000,xn)# x co-ordinates of the desired array shape
Ylist = np.linspace(0,10000,yn)# y co-ordinates of the desired array shape

depths = [1,5,17]
depths = [17]

def tracer_d2(Xlist,Ylist,Tr,id):
 S00 = 0
 S01 = 0
 S02 = 0
 N = len(Ylist)
 Xlist = Xlist - 1500.0
 for j in range(N):
  A = max(Xlist)-min(Xlist)
  S00 = S00 + np.trapz(Tr[id,j,:], Xlist, 0)/A
  S01 = S01 + np.trapz(Tr[id,j,:]*(X[id,j,:]-1500), Xlist, 0)/A
  S02 = S02 + np.trapz(Tr[id,j,:]*(X[id,j,:]-1500)**2, Xlist, 0)/A
  S00 = S00/N; S01 = S01/N; S02 = S02/N
 return (S02-S01**2)/S00

D2 = np.zeros([len(range(dayi,dayf,days)),len(depths)])

# Tracer second moment

Tr = np.zeros((zn,yn,xn,len(range(dayi,dayf,days))))))))))))

for t in time:
 print t
 for z in range(len(depths)):
  print z
  # points of interest (2D only!)
  print 'extracting points'
  #
  fd = open('./T_'+label+'_'+str(z+4)+'_'+str(t)+'.csv','r')
  reader = csv.reader(fd)
  j = 0
  for row in reader:
   i = 0
   for rec in range(len(Xlist)):
    Tr[z,j,i,t] = row[i]
    i = i + 1
   j = j + 1
  fd.close() 
  D2[t,z] = tracer_d2(Xlist,Ylist,Tr[:,:,:,t],0)

fig = plt.figure()
for t in time:
 plt.plot(Xlist,np.mean(Tr[0,:,:,t],0))

plt.savefig('./plot/'+label+'/D_Tr_'+label+'_t_23D.eps')
plt.close()
print       './plot/'+label+'/D_Tr_'+label+'_t_23D.eps'

# plot dispersion Tr

n time:
 tlabel = str(tt)
 while len(tlabel) < 3: tlabel = '0'+tlabel
 #
 file0 = basename+'_' + str(tt) + '.pvtu'
 filepath = path+file0
 file1 = label+'_' + tlabel
 fileout  = path + file1
 #
 #
 #
 print 'opening ', filepath
 data = vtktools.vtu(filepath)pTr, = plt.plot(timeTr/86400,D_Tr[:,0],color=[0,0,0],linewidth=2)

z = 1
pTr5, = plt.plot(timeTr/86400,D_Tr[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
z = 2
pTr11, = plt.plot(timeTr/86400,D_Tr[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
z = 3
pTr17, = plt.plot(timeTr/86400,D_Tr[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)
z = 4
pTr26, = plt.plot(timeTr/86400,D_Tr[:,z],color=[z/float(nl),z/float(nl),z/float(nl)],linewidth=2)

plt.gca().set_yscale('log')
#plt.gca().set_xscale('log')

plt.xlabel('Time [days]')
plt.ylabel('Dispersion [m^2]')
plt.legend((pTr,pTr5,pTr17),('Tr 1m','Tr 5m','Tr 17m'),loc=4,fontsize=12)

plt.savefig('./plot/'+label+'_23D/D_Tr_'+label+'_23D.eps')
print       './plot/'+label+'_23D/D_Tr_'+label+'_23D.eps' 
plt.close()

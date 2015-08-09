#!python
import os, sys
import csv
import scipy.integrate 
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt

 
for label in sys.argv:
 
 label = sys.argv[1]
 
 f_stat = '/tamay/mensa/fluidity/hycom_winter/'+label+'/mli.stat'
 
 try: os.stat('./plot/'+label)
 except OSError: os.mkdir('./plot/'+label)
 
 # parameters from the header
 bins = 1000 
 header = 80
 Ti = 22
 Tf = Ti + bins
 
 m_file = open(f_stat,'r')
 m_file_csv = csv.reader(m_file, delimiter=' ')
 
 # first field: elapsed time
 # from 22 to 522: Temperature bins where bins are defined as: 
 
 #bins = np.linspace(19.8988, 20.9135, 500, endpoint=True)
 bins = np.linspace(17.0, 21.5, bins, endpoint=True)
 
 
 # physical parameters
 
 alpha = 0.000195
 T_zero = 0
 rho_zero = 1
 g = 9.81
 domainheight = 400
 
 
 i = 0
 
 time = []
 Tr   = []
 T    = []
 
 for row in m_file_csv:
  i = i + 1
  if i > header:
   time.append(row[2])
   for id in range(0,Tf-Ti+1): # integral + bins
    Tr.append(float(row[Ti+12+id*2]))
   T.append(Tr)
   Tr = []
 i = i - header - 1
 T = np.asarray(T)
 
 Tint = T[:,1]
 T    = T[:,1:]
 
 T1 = []
 T2 = []
 T3 = []
 T4 = []
 T5 = []
 
 bpe = []
 reference_state = []
 
 for t in range(len(T)):
 
  Tt   = T[t,:]
  # evolution in time of the quartiles
 
  #  i             s
  #b[0] <= T1 <  b[1]
  #b[1] <= T2 <  b[2] 
  #b[2] <= T3 <  b[3]
  #b[3] <= T4 <= b[4]
  
  v, bounds = np.histogram(bins,4)
  
  b4s = np.asarray(np.where(bins <= bounds[4]))
  b3i = np.asarray(np.where(bins >= bounds[3]))
  b3s = np.asarray(np.where(bins <  bounds[3]))
  b2i = np.asarray(np.where(bins >= bounds[2]))
  b2s = np.asarray(np.where(bins <  bounds[2]))
  b1i = np.asarray(np.where(bins >= bounds[1]))
  b1s = np.asarray(np.where(bins <  bounds[1]))
  b0i = np.asarray(np.where(bins >= bounds[0]))
  
  T1.append(sum(Tt[np.intersect1d(b0i,b1s)])/sum(Tt))
  T2.append(sum(Tt[np.intersect1d(b1i,b2s)])/sum(Tt))
  T3.append(sum(Tt[np.intersect1d(b2i,b3s)])/sum(Tt))
  T4.append(sum(Tt[np.intersect1d(b3i,b4s)])/sum(Tt))
 
  # BPE
  # rearrange bins so have nobins = nobounds -1
  # amounts to including any undershoot or overshoots in lower/upper most bin
  # for discussion of impacts see H. Hiester, PhD thesis (2011), chapter 4.
 
  Tt[1] = Tt[0]+Tt[1]
  Tt[-2] = Tt[-2]+Tt[-1]
  Tt = Tt[1:-1]
 
  # get reference state using method of Tseng and Ferziger 2001
  Abins = sum([Tt[k]*(bins[k+1]-bins[k]) for k in range(len(Tt))])
  pdf = [val/Abins for val in Tt]
  rs = [0]
  for k in range(len(pdf)): rs.append(rs[-1]+(domainheight*pdf[k]*(bins[k+1]-bins[k])))
  reference_state.append(tuple(rs))
 
  # get background potential energy, 
  # noting \rho = \rho_zero(1-\alpha(T-T_zero))
  # and reference state is based on temperature
  # bpe_bckgd = 0.5*(g*rho_zero*(1.0+(alpha*T_zero)))*(domainheight**2)
  # but don't include this as will look at difference over time
  bpe.append(-rho_zero*alpha*g*scipy.integrate.trapz(x=reference_state[-1],y=[bins[j]*reference_state[-1][j] for j in range(len(reference_state[-1]))]))
 
 reference_state = np.array(reference_state)
 bpe_zero = bpe[0]
 #bpe = [np.abs((val - bpe_zero)/bpe_zero) for val in bpe]
 
 ## plot
 
 time = [float(t)/3600 for t in time]
 
 # volume fraction
 
 plt.figure()
 p1, = plt.plot(time,T1)
 p2, = plt.plot(time,T2)
 p3, = plt.plot(time,T3)
 p4, = plt.plot(time,T4)
 label1 = str(bounds[0])+" <= T < "+str(bounds[1])
 label2 = str(bounds[1])+" <= T < "+str(bounds[2])
 label3 = str(bounds[2])+" <= T < "+str(bounds[3])
 label4 = str(bounds[3])+" <= T =< "+str(bounds[4])
 plt.legend([p1, p2, p3, p4], [label1,label2,label3,label4],loc=7)
 plt.xlabel('$t$ (hrs)')
 plt.ylabel('$pdf$ T')
 plt.savefig('./plot/'+label+'/'+label+'_quantile.eps',bbox_inches='tight')
 plt.close()
 
 
 # pdfs in time
 
 plt.figure()
 p1, = plt.plot(bins,T[0,:]/sum(T[0,:]))
 p2, = plt.plot(bins,T[i/2,:]/sum(T[i/2,:]))
 p3, = plt.plot(bins,T[i,:]/sum(T[i,:]))
 label1 = "day "+str(np.trunc(time[0]/24))
 label2 = "day "+str(np.trunc(time[i/2]/24))
 label3 = "day "+str(np.trunc(time[i]/24))
 plt.legend([p1, p2, p3], [label1,label2,label3],loc=2)
 plt.xlabel('$T$ (Celsius degree)')
 plt.ylabel('$pdf$')
 plt.savefig('./plot/'+label+'/'+label+'_pdf.eps',bbox_inches='tight')
 plt.close()
 
 
 # BPE
 
 plt.figure()
 p1, = plt.plot(time,bpe)
 plt.xlabel('$t$ (hrs)')
 plt.ylabel('$\\Delta E_b$')
 plt.savefig('./plot/'+label+'/'+label+'_bpe.eps',bbox_inches='tight')
 plt.close()
 

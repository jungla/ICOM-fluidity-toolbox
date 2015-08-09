#!/share/apps/python/2.7.2/bin/python
import os
import numpy as np
import csv
import fluidity_tools
import myfun

global os, np, csv, fluidity_tools, myfun

################# TRACER

def read_Tracer(filepath,pd,zn,yn,xn,timeTr):
 # pd - list of tracer's name
 # zn,xn,yn - tracer's dims
 # timeTr - tracer's time series

 nl = len(pd)

 Tr = np.zeros((nl,zn,xn,yn,len(timeTr)))

 for t in range(len(timeTr)):
  print t
  for z in range(nl):
   print 'Tracer_',pd[z]
   #
   f = open(filepath+'_Tr'+str(pd[z])+'_'+str(t)+'.csv','r')
   reader = csv.reader(f)
   j = 0
   k = 0
   for row in reader:
    if j == yn: k = k + 1; j = 0
    i = 0
    for item in row[:-1]: # new line character !!
     Tr[z,k,i,j,t] = item
     i = i + 1
    j = j + 1
 #  print np.amax(Tr[z,k,:,:,t])
 #  Tr[:,:,:,t] = Tr[:,:,:,t]/3
   f.close()
 #  TrT = np.reshape(Tr[0,:,:,t],)
 #  TrT = Tr[:,:,:,t]
 return Tr

def read_Scalar(filepath,zn,yn,xn):
 # pd - list of tracer's name
 # zn,xn,yn - tracer's dims
 # timeTr - tracer's time series
 Tr = np.zeros((zn,xn,yn))
 f = open(filepath,'r')
 reader = csv.reader(f)
 j = 0
 k = 0
 for row in reader:
  if j == yn: k = k + 1; j = 0
  i = 0
  for item in row[:-1]: # new line character !!
   Tr[k,i,j] = item
   i = i + 1
  j = j + 1
 #  print np.amax(Tr[z,k,:,:,t])
 #  Tr[:,:,:,t] = Tr[:,:,:,t]/3
 f.close()
 #  TrT = np.reshape(Tr[0,:,:,t],)
 #  TrT = Tr[:,:,:,t]
 return Tr

#

def read_dispersion(filename):
 RD = [] #np.zeros((tt_B,nl))
 time = []

 with open(filename, 'r') as csvfile:
  spamreader = csv.reader(csvfile)
  spamreader.next()
  for row in spamreader:
   time.append(row[0])
   RD.append(row[1:])
 
 time = np.asarray(time).astype(float)
 RD = np.asarray(RD).astype(float)
 return time, RD

#def read_particles(filepath):
# det = fluidity_tools.stat_parser(filepath)
# pt = int(os.popen('grep position '+filepath+'| wc -l').read()) # read the number of particles grepping all the positions in the file
# time = det['ElapsedTime']['value']
# tt = len(time)
# par = np.zeros((pt,3,tt))
# #
# for d in xrange(pt):
#  temp = det['particles_'+myfun.digit(d+1,len(str(pt)))]['position']
#  par[d,:,:] = temp[:,:]
# #
# return time, par 
#
#

def periodicCoords(par,xlim,ylim):

 pt,poop,tt = par.shape

 s = np.zeros(par.shape)
 parP = np.zeros(par.shape)

 xlimD = xlim-xlim/5.0
 ylimD = ylim-ylim/5.0
 
 for p in xrange(pt):
  for t in xrange(tt-1):
   if par[p,0,t] - par[p,0,t+1] > xlimD:
    s[p,0,t+1:] = s[p,0,t+1:] + xlim
   if par[p,0,t] - par[p,0,t+1] < -xlimD:
    s[p,0,t+1:] = s[p,0,t+1:] - xlim
   if par[p,1,t] - par[p,1,t+1] > ylimD:
    s[p,1,t+1:] = s[p,1,t+1:] + ylim
   if par[p,1,t] - par[p,1,t+1] < -ylimD:
    s[p,1,t+1:] = s[p,1,t+1:] - ylim

 for p in xrange(pt):
  for t in xrange(tt-1):
   parP[p,:,t] = par[p,:,t] + s[p,:,t]

 return parP

# WRONG!! 

def tracer_d2(Xlist,Ylist,deltax,Tr):
 S00 = 0
 S01 = 0
 S02 = 0
 N = len(Ylist)
 A = float(max(Xlist)-min(Xlist))
 Xlist = Xlist - deltax
 for j in range(N):
  # strip nans
  Tn = Tr[j,~np.isnan(Tr[j,:])]
  Xln = Xlist[~np.isnan(Tr[j,:])]
  S00 = S00 + np.trapz(Tn, Xln, 0)/A
  S01 = S01 + np.trapz(Tn*(Xln), Xln, 0)/A
  S02 = S02 + np.trapz(Tn*(Xln)**2, Xln, 0)/A
  S00 = S00/float(N); S01 = S01/float(N); S02 = S02/float(N)
 return (S02-S01**2)/S00

def tracer_d2_bis(Xlist,Ylist,deltax,Tr):
 S00 = 0
 S01 = 0
 S02 = 0
 S = 0
 N = len(Ylist)
 A = float(max(Xlist)-min(Xlist))
 Xlist = Xlist - deltax
 for j in range(N):
  # strip nans
  Tn = Tr[~np.isnan(Tr[:,j]),j]
  Xln = Xlist[~np.isnan(Tr[:,j])]
  S00 = np.trapz(Tn, Xln, 0)/A
  S01 = np.trapz(Tn*(Xln), Xln, 0)/A
  S02 = np.trapz(Tn*(Xln)**2, Xln, 0)/A
  S = S + (S02-S01**2)/S00
 S = S/float(N)
 return S

################# PARTICLES

def ED_t(par2Dz,tt):
 D_2D = np.zeros(tt)
 for t in range(tt):
  x2 = par2Dz[:,0,t]
  y2 = par2Dz[:,1,t]
#
#  x2 = x2[~np.isnan(x2)]
#  y2 = y2[~np.isnan(y2)]
#
  if len(x2) > 1 and len(y2) > 1:
   xt2 = x2 - np.mean(x2)
   yt2 = y2 - np.mean(y2)
   cov2 = np.cov(xt2, yt2)
   if ~np.isnan(cov2).any():
    lambda_2, v = np.linalg.eig(cov2)
    lambda_2 = np.sqrt(lambda_2)
    D_2D[t] = 2*lambda_2[0]*lambda_2[1]
   else:
    D_2D[t] = np.nan
  else:
   D_2D[t] = np.nan
 return D_2D

def ED2_t(par2Dz,tt):
 D_2D = np.zeros(tt)
 for t in range(tt):
  x2 = par2Dz[:,0,t]
  y2 = par2Dz[:,1,t]
  #
  #  x2 = x2[~np.isnan(x2)]
  #  y2 = y2[~np.isnan(y2)]
  #
  if len(x2) > 1 and len(y2) > 1:
   xt2 = x2 - np.mean(x2)
   yt2 = y2 - np.mean(y2)
   cov = np.cov(xt2, yt2)
   if ~np.isnan(cov).any():
    term1 = (cov[0,0]+cov[1,1]) 
    term2 = np.sqrt((cov[0,0]-cov[1,1])**2 + 4*cov[1,0]**2)
    lambda_1   = np.sqrt(.5*(term1+term2)) 
    lambda_2   = np.sqrt(.5*(term1-term2)) 
    #    lambda_2, v = np.linalg.eig(cov2)
    #    lambda_2 = np.sqrt(lambda_2)
    D_2D[t] = 2*lambda_1*lambda_2
   else:
    D_2D[t] = np.nan
  else:
   D_2D[t] = np.nan
 return D_2D

def CD_t(par2Dz,tt):
	Pt2D = np.zeros((2,tt))
	Pt2D[:] = np.mean(par2Dz[:,0,:],0),np.mean(par2Dz[:,1,:],0)
	return np.mean(np.sqrt((par2Dz[:,0,:] - Pt2D[0,:])**2 + (par2Dz[:,1,:] - Pt2D[1,:])**2),0)**2

def CDx_t(par2Dz,tt):
 Pt2D = np.zeros((tt))
 Pt2D[:] = np.mean(par2Dz[:,0,:],0)
 # 
 return np.mean((par2Dz[:,0,:] - Pt2D[:])**2,0)

def AD_t(par2Dz,tt):
 temp = np.zeros((len(par2Dz[:,0,0]),len(par2Dz[0,0,:])))
 for p in range(len(par2Dz[:,0,0])):
  temp[p,:] = (par2Dz[p,0,:] - par2Dz[p,0,0])**2 + (par2Dz[p,1,:] - par2Dz[p,1,0])**2
 return np.mean(temp,0)
# return np.mean(((par2Dz[p,0,:] - par2Dz[p,0,0])**2 + (par2Dz[p,1,:] - par2Dz[p,1,0])**2),0)**2

def AD_t_v(par2Dz,tt):
 temp = np.zeros((len(par2Dz[:,0,0]),len(par2Dz[0,0,:])))
 for p in range(len(par2Dz[:,0,0])):
  temp[p,:] = (par2Dz[p,2,:] - par2Dz[p,2,0])**2 
 return np.mean(temp,0)
# return np.mean(((par2Dz[p,0,:] - par2Dz[p,0,0])**2 + (par2Dz[p,1,:] - par2Dz[p,1,0])**2),0)**2


def RD_t(par2Dzr,tt,px,py):
 RD_2Dm = [] #np.zeros((px+py,tt))
 for i in range(px):
  RD_2Dm.append(np.mean(((par2Dzr[i+1,:,0,:] - par2Dzr[i,:,0,:])**2 + (par2Dzr[i+1,:,1,:] - par2Dzr[i,:,1,:])**2),0))
 for j in range(py):
  RD_2Dm.append(np.mean(((par2Dzr[:,j+1,0,:] - par2Dzr[:,j,0,:])**2 + (par2Dzr[:,j+1,1,:] - par2Dzr[:,j,1,:])**2),0))
 return np.mean(RD_2Dm,0)

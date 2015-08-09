import numpy as np
from intergrid import Intergrid
import scipy.ndimage.interpolation
import csv

def interp(x0,y0,z0,lo,hi,maps,U0,ord): 
 query_points = np.zeros((len(x0),3)) # lo + np.random.uniform( size=(xp*yp*zp, 3) ) * (hi - lo)
 query_points[:,0] = x0
 query_points[:,1] = y0
 query_points[:,2] = z0
 if_U = Intergrid(U0, lo=lo, hi=hi, order=ord, maps=maps, verbose=1)
# query_points[:,0] = np.interp(query_points[:,0], Xlist, range(len(Xlist)))
# query_points[:,1] = np.interp(query_points[:,1], Ylist, range(len(Ylist)))
# query_points[:,2] = np.interp(query_points[:,2], Zlist, range(len(Zlist)))
 Ui = if_U.at( query_points )
 return Ui

def interp2D(x0,y0,lo,hi,maps,U0,ord):
 query_points = np.zeros((len(x0),2)) # lo + np.random.uniform( size=(xp*yp*zp, 3) ) * (hi - lo)
 query_points[:,0] = x0
 query_points[:,1] = y0
 if_U = Intergrid(U0, lo=lo, hi=hi, order=ord, maps=maps, verbose=1)
# query_points[:,0] = np.interp(query_points[:,0], Xlist, range(len(Xlist)))
# query_points[:,1] = np.interp(query_points[:,1], Ylist, range(len(Ylist)))
# query_points[:,2] = np.interp(query_points[:,2], Zlist, range(len(Zlist)))
 Ui = if_U.at( query_points )
 return Ui




def interp2(x0,y0,z0,limit,U0):
 #U0 = Ut0
 #xp,yp,zp = x0.shape
 # x = 1.*np.reshape(x0, (np.size(x0),))
 # y = 1.*np.reshape(y0, (np.size(y0),))
 # z = 1.*np.reshape(z0, (np.size(z0),))
 x = (1.*x0-np.min(limit[0]))/np.max(limit[0])*xn
 y = (1.*y0-np.min(limit[1]))/np.max(limit[1])*yn
 z = (1.*z0-np.max(limit[2]))/np.min(limit[2])*zn
 # print min(x), max(x), min(y), max(y), min(z), max(z)
 t = scipy.ndimage.interpolation.map_coordinates(U0, np.array([x,y,z]), output=None, order=3, mode='constant', cval=0.0, prefilter=True)
 return t #np.reshape(t,(xp,yp,zp))


import scipy.interpolate

def interp3(x0,y0,z0,U0):
 points = np.array([np.reshape(X, (np.size(X),)),np.reshape(Y, (np.size(Y),)),np.reshape(Z, (np.size(Z),))])
 t = scipy.interpolate.griddata(np.transpose(points), np.reshape(U0,(np.size(U0),)), (x0,y0,z0), method='linear')
 return t



def RK4(x0,y0,z0,Ut0,Vt0,Wt0,Ut1,Vt1,Wt1,lo,hi,maps,dt,ord):
 h2 = dt/2.0
 h6 = dt/6.0
 # linear time interpolation at t = t0 + 0.5dt
 Ut05 = (Ut0 + Ut1)*0.5
 Vt05 = (Vt0 + Vt1)*0.5
 Wt05 = (Wt0 + Wt1)*0.5
 
 U1 = interp(x0,y0,z0,lo,hi,maps,Ut0,ord)
 V1 = interp(x0,y0,z0,lo,hi,maps,Vt0,ord)
 W1 = interp(x0,y0,z0,lo,hi,maps,Wt0,ord)
 
 x1 = x0 + h2*U1
 y1 = y0 + h2*V1
 z1 = z0 + h2*W1
 
 U2 = interp(x1,y1,z1,lo,hi,maps,Ut05,ord)
 V2 = interp(x1,y1,z1,lo,hi,maps,Vt05,ord)
 W2 = interp(x1,y1,z1,lo,hi,maps,Wt05,ord)
 
 x1 = x0 + h2*U2
 y1 = y0 + h2*V2
 z1 = z0 + h2*W2
 
 U3 = interp(x1,y1,z1,lo,hi,maps,Ut05,ord)
 V3 = interp(x1,y1,z1,lo,hi,maps,Vt05,ord)
 W3 = interp(x1,y1,z1,lo,hi,maps,Wt05,ord)

 x1 = x0 + dt*U3
 y1 = y0 + dt*V3
 z1 = z0 + dt*W3

 U4 = interp(x1,y1,z1,lo,hi,maps,Ut1,ord)
 V4 = interp(x1,y1,z1,lo,hi,maps,Vt1,ord)
 W4 = interp(x1,y1,z1,lo,hi,maps,Wt1,ord)

 x0 = x0 + h6 * (U1 + 2.*U2 + 2.*U3 + U4)
 y0 = y0 + h6 * (V1 + 2.*V2 + 2.*V3 + V4)
 z0 = z0 + h6 * (W1 + 2.*W2 + 2.*W3 + W4)

 return x0,y0,z0

def RK4_zyx(z0,y0,x0,Ut0,Vt0,Wt0,Ut1,Vt1,Wt1,lo,hi,maps,dt,ord):
 h2 = dt/2.0
 h6 = dt/6.0
 # linear time interpolation at t = t0 + 0.5dt
 Ut05 = (Ut0 + Ut1)*0.5
 Vt05 = (Vt0 + Vt1)*0.5
 Wt05 = (Wt0 + Wt1)*0.5

 U1 = interp(z0,y0,x0,lo,hi,maps,Ut0,ord)
 V1 = interp(z0,y0,x0,lo,hi,maps,Vt0,ord)
 W1 = interp(z0,y0,x0,lo,hi,maps,Wt0,ord)

 x1 = x0 + h2*U1
 y1 = y0 + h2*V1
 z1 = z0 + h2*W1

 U2 = interp(z1,y1,x1,lo,hi,maps,Ut05,ord)
 V2 = interp(z1,y1,x1,lo,hi,maps,Vt05,ord)
 W2 = interp(z1,y1,x1,lo,hi,maps,Wt05,ord)

 x1 = x0 + h2*U2
 y1 = y0 + h2*V2
 z1 = z0 + h2*W2

 U3 = interp(z1,y1,x1,lo,hi,maps,Ut05,ord)
 V3 = interp(z1,y1,x1,lo,hi,maps,Vt05,ord)
 W3 = interp(z1,y1,x1,lo,hi,maps,Wt05,ord)

 x1 = x0 + dt*U3
 y1 = y0 + dt*V3
 z1 = z0 + dt*W3

 U4 = interp(z1,y1,x1,lo,hi,maps,Ut1,ord)
 V4 = interp(z1,y1,x1,lo,hi,maps,Vt1,ord)
 W4 = interp(z1,y1,x1,lo,hi,maps,Wt1,ord)

 x0 = x0 + h6 * (U1 + 2.*U2 + 2.*U3 + U4)
 y0 = y0 + h6 * (V1 + 2.*V2 + 2.*V3 + V4)
 z0 = z0 + h6 * (W1 + 2.*W2 + 2.*W3 + W4)

 return x0,y0,z0

def EULER(x0,y0,z0,Ut0,Vt0,Wt0,lo,hi,maps,dt):
 U1 = interp(x0,y0,z0,lo,hi,maps,Ut0)
 V1 = interp(x0,y0,z0,lo,hi,maps,Vt0)
 W1 = interp(x0,y0,z0,lo,hi,maps,Wt0)
 x = x0 + U1*dt
 y = y0 + V1*dt
 z = z0 + W1*dt
 return x,y,z
 
def cBC(x0,y0,z0,lo,hi):
 # applies periodic boundaries
 x0[np.where(x0 > hi[0])] = hi[0]
 x0[np.where(x0 < lo[0])] = lo[0]
 y0[np.where(y0 > hi[1])] = hi[1]
 y0[np.where(y0 < lo[1])] = lo[1]
 z0[np.where(z0 > hi[2])] = hi[2]
 z0[np.where(z0 < lo[2])] = lo[2]
 return x0,y0,z0
 
def pBC(x0,y0,z0,lo,hi):
 # applies periodic boundaries
 x0[np.where(x0 > hi[0])] = x0[np.where(x0 > hi[0])] - hi[0]
 x0[np.where(x0 < lo[0])] = x0[np.where(x0 < lo[0])] + hi[0]
 y0[np.where(y0 > hi[1])] = y0[np.where(y0 > hi[1])] - hi[1]
 y0[np.where(y0 < lo[1])] = y0[np.where(y0 < lo[1])] + hi[1]
 z0[np.where(z0 > hi[2])] = hi[2]
 z0[np.where(z0 < lo[2])] = lo[2]
 return x0,y0,z0


def read_particles_csv(filename,pt,tt):
 time = []  
 par = np.zeros((pt,3,tt))
 f = open(filename,'r')
 reader = csv.reader(f)

 j = 0
 k = 0
 for row in reader:
  if k == 0: time.append(float(row[3]));
  if k == pt: j = j + 1; k = 0; print j; time.append(float(row[3])); 
  i = 0
  for item in row[0:3]: # new line character !!
   par[k,i,j] = float(item)
   i = i + 1
  k = k + 1
 time.append(float(row[3]))
 f.close()

 return np.asarray(time), np.asarray(par)

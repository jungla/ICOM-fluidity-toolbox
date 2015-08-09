import numpy as np

def princax(w):
 #   [theta,maj,min,wr] = princax(w)
 #
 #   Input:  w = complex vector time series (u+i*v)
 #
 #   Output: theta = angle of maximum variance, (east == 0, north=90)
 #           maj   = major axis of principal ellipse
 #           min   = minor axis of principal ellipse
 #           wr    = rotated time series, where real(wr) is aligned
 #                   with the major axis.
 
 cv = np.cov(np.real(w[:]),np.imag(w[:]))
 
 #---------------------------------------------------------------------
 #  Find direction of maximum variance
 #---------------------------------------------------------------------
 
 theta = 0.5*np.arctan2(2*cv[1,0],(cv[0,0]-cv[1,1])) 
 #---------------------------------------------------------------------
 #  Find major and minor axis amplitudes
 #---------------------------------------------------------------------
 
 term1 = (cv[0,0]+cv[1,1]) 
 term2 = np.sqrt((cv[0,0]-cv[1,1])**2 + 4*cv[1,0]**2) 
 
 maj   = np.sqrt(.5*(term1+term2)) 
 min   = np.sqrt(.5*(term1-term2)) 
 
 #---------------------------------------------------------------------
 #  Rotate into principal ellipse orientation
 #---------------------------------------------------------------------
 
 # wr = w*np.exp(-1j*theta) 
 theta   = theta*180./np.pi 

 return theta,maj,min

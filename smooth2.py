def smooth2(R,times):

 ids,jds = R.size()
 
 Rf = np.zeros((ids,jds))
 
 for t in range(times):
 
 # smooth
  for i in range(ids):
   for j in range(jds):
 
   if(np.isnan(R[i,j])):
    Rf[i,j] = R[i,j]
    np = 1
 
    for k in range(4): 
     if k == 0:
      ip = -1
      jp = 0
     if k == 1:
      ip = 0
      jp = -1
     if k == 2:
      ip = 1
      jp = 0
     if k == 3:
      ip = 0
      jp = 1
    
    if (i+ip>0 and j+jp>0 and i+ip<ids and j+jp<jds):
     if (np.isnan(R[i+ip,j+jp])):
      Rf[i,j] = Rf[i,j] + R[i+ip,j+jp]
      np = np + 1;
 
    Rf[i,j] = Rf[i,j]/np;
 
 R = Rf

 return R

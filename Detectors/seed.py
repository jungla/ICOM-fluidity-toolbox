#!python

ret = []
dh = 50
dz = -2

for k in -np.round(np.linspace(2,10,30)):
 for j in range(1,15000,500):
  for i in range(1,15000,500):
   for p in range(4):
    if (p == 0):
     ret.append([i,j,k])
    elif (p == 1):
     ret.append([i+dh,j,k])
    elif (p == 2):
     ret.append([i,j+dh,k])
    else:
     ret.append([i,j,k+dz])

print ret

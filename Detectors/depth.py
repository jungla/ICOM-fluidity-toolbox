#!python

dz = [2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10]

depth = []
depth.append(2)

for d in range(1,len(dz)):
 depth.append(depth[d-1]+dz[d])



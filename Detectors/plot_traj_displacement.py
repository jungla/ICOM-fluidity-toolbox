fig = plt.figure(figsize=(6,8))
>>> 
for d in range(len(Zlist)):
# convert coord to distance from center
pc = p[d::6,:]
coord = np.sqrt(pc[:,0]**2 + pc[:,1]**2)
#
parz = par[d::6,:,:]
distz = parz[:,2,tt] - Zlist[d]
#
#
# draw circles: boundaries and seeding region
# p0 = np.histogram2d(coord,distz_0)
v = np.linspace(1,2,2)
heatmap, xedges, yedges = np.histogram2d(distz,coord,bins=(11,21))
# plt.contourf(yedges[:-1]/1000,xedges[:-1]+depths[d],heatmap,v,extend="max")
# plt.contourf(yedges[:-1]/-1000,xedges[:-1]+depths[d],heatmap,v,extend="max")
#
X, Y = np.meshgrid(xedges, yedges)
#plt.pcolormesh(Y/1000, X+depths[d], heatmap)
#plt.pcolormesh(Y/-1000, X+depths[d], heatmap)
#
heatmap[np.where(heatmap>10)] = 10
#plt.pcolormesh(X,Y/1000 + Zlist[d]/1000.0,np.rot90(heatmap))
#plt.pcolormesh(-X,Y/1000 + Zlist[d]/1000.0,np.rot90(heatmap))
plt.pcolormesh(yedges/1000,xedges+Zlist[d],heatmap,cmap=plt.cm.Blues)
plt.pcolormesh(-yedges/1000,xedges+Zlist[d],heatmap,cmap=plt.cm.Blues)


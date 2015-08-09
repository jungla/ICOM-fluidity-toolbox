cdict1 = {'red':   ((0.0, 0.0, 0.0),
                   (0.05, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.05, 1.0, 1.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 1.0),
                   (0.05, 1.0, 1.0),
                   (1.0, 0.0, 0.0))
        }
blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)
plt.imshow(Z, interpolation='nearest', cmap=blue_red1)
plt.colorbar()
plt.savefig('./plot/'+label+'/PV_'+file1+'.eps',bbox_inches='tight')
plt.close()

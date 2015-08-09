import os, sys
import vtktools
import fluidity_tools
import numpy as np
import matplotlib  as mpl
mpl.use('ps')
import matplotlib.pyplot as plt
import spectrum

label = sys.argv[1]
basename = sys.argv[2]

path = '/tamay2/mensa/fluidity/'+label+'/'

try: os.stat('./plot/'+label)
except OSError: os.mkdir('./plot/'+label)

#
file0 = basename+'.stat'
filepath0 = path+file0
stat0 = fluidity_tools.stat_parser(filepath0)

#file1 = 'mli_checkpoint.stat'
#filepath1 = path+file1
#stat1 = fluidity_tools.stat_parser(filepath1)

time0 = stat0["ElapsedTime"]["value"]/86400.0
#time1 = stat1["ElapsedTime"]["value"]/86400.0

KE0 = 0.5*np.sqrt(stat0["BoussinesqFluid"]["Velocity_CG%3"]["l2norm"])

print len(KE0[1:len(KE0)])
pwd = KE0[500:1524]
print len(pwd)

p = spectrum.Periodogram(pwd, sampling=1024, window='hann', NFFT=None, scale_by_freq=False, detrend=True)
p.run()
window = spectrum.window.create_window(20, 'hamming')
#
#from numpy.fft import fft
#n = len(pwd)
#I_w = np.abs(fft(pwd))**2 / n
#w =  np.arange(n) / n
#w, I_w = w[:int(n/2)+1], I_w[:int(n/2)+1]  # Take only values on [0, pi]
#
psdw = np.convolve(window/window.sum(),p.psd,mode='same')
#  
plt.figure()
#plt.plot(KE0[100:len(KE0)])
#plt.plot(p.frequencies(),np.log(p.psd))
plt.plot(p.frequencies(),np.log(psdw))

# plot KE
#plt.plot(time1, KE1, color='k',linewidth=1.5)
plt.xlabel("frequency $[days]$")
plt.ylabel("Vertical Kinetic Energy $[m^2/s^2]$")

plt.savefig('./plot/'+label+'/KEv_spectrum_'+label+'.eps',bbox_inches='tight')
plt.close()
print 'saved '+'./plot/'+label+'/KEv_spectrum_'+label+'.eps\n'
#

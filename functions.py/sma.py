import numpy as np


def sma(data,ndays):
 datac = np.convolve(data,ndays,'valid')
 return datac


import numpy as np
import matplotlib.pyplot as plt
import  scipy.signal as ssig
import numpy.random as rnd
import numpy.fft as nfft

def ButterWorth2D(xx, yy, par):

    nn = par[0]
    l0 = par[1]
    ButterWx = 1./(1.+(xx/l0)**nn)
    ButterWy = 1./(1.+(yy/l0)**nn)

    return ButterWx*ButterWy

NGrid =512
NData = NGrid*NGrid
ll = 200./256.
nn = 12

data = rnd.uniform(0., 1., NData)

data = np.reshape(data, (NGrid, NGrid))

xy = np.linspace(-1., 1., 512)
xx, yy = np.meshgrid(xy,xy)
par_ = [nn,ll]
BW2D = ButterWorth2D(xx, yy, par_)

dataW = BW2D*data

Kernel_size = 12

kernel = np.exp(-(xx**2+yy**2)/2./Kernel_size**2.)/2./np.pi/Kernel_size**2

dataR  = ssig.fftconvolve(dataW,kernel,  mode='same')/NGrid
dataSR = ssig.fftconvolve(dataW**2,kernel,  mode='same')/NGrid

sigD = dataSR - dataR**2
plt.clf()
plt.ion()
plt.subplot(1,2,1)
plt.imshow(dataW)
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(sigD)
plt.colorbar()
plt.show()


import matplotlib.pyplot as plt
import scipy
import scipy.special
import numpy as np

def rotatedCosineWindow(N):  # N = horizontal size of the targeted kernel, also its vertical size, must be odd.
  return np.fromfunction(lambda y, x: np.maximum(np.cos(np.pi/2*np.sqrt(((x - (N - 1)/2)/((N - 1)/2 + 1))**2 + ((y - (N - 1)/2)/((N - 1)/2 + 1))**2)), 0), [N, N])

def circularLowpassKernelX(omega_c, N):  # omega = cutoff frequency in radians (pi is max), N = horizontal size of the kernel, also its vertical size, must be odd.
  kernel = np.fromfunction(lambda y, x: omega_c**2*(x - (N - 1)/2)*scipy.special.jv(2, omega_c*np.sqrt((x - (N - 1)/2)**2 + (y - (N - 1)/2)**2))/(2*np.pi*((x - (N - 1)/2)**2 + (y - (N - 1)/2)**2)), [N, N])
  kernel[(N - 1)//2, (N - 1)//2] = 0
  return kernel

def circularLowpassKernelY(omega_c, N):  # omega = cutoff frequency in radians (pi is max), N = horizontal size of the kernel, also its vertical size, must be odd.
  kernel = np.fromfunction(lambda y, x: omega_c**2*(y - (N - 1)/2)*scipy.special.jv(2, omega_c*np.sqrt((x - (N - 1)/2)**2 + (y - (N - 1)/2)**2))/(2*np.pi*((x - (N - 1)/2)**2 + (y - (N - 1)/2)**2)), [N, N])
  kernel[(N - 1)//2, (N - 1)//2] = 0
  return kernel

N = 41  # Horizontal size of the kernel, also its vertical size. Must be odd.
window = rotatedCosineWindow(N)

# Optional window function plot
#plt.imshow(window, vmin=-np.max(window), vmax=np.max(window), cmap='bwr')
#plt.colorbar()
#plt.show()

omega_c = np.pi/4  # Cutoff frequency in radians <= pi
kernelX = circularLowpassKernelX(omega_c, N)*window
kernelY = circularLowpassKernelY(omega_c, N)*window

# Optional kernel plot
#plt.imshow(kernelX, vmin=-np.max(kernelX), vmax=np.max(kernelX), cmap='bwr')
#plt.colorbar()
#plt.show()
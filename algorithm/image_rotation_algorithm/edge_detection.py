import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
# %matplotlib inline
# plt.rcParams['figure.figsize'] = (12, 8)
# %load_ext autoreload
# %autoreload 2

#This function performs convolution between an image and a kernel
def convolve(image, kernel): 
    
    # Get the size of the image and the kernel
    sizeK = kernel.shape[0]
    xImg = image.shape[0]
    yImg = image.shape[1]
    sizeP = int((sizeK-1)/2)
    
    # Perform convoluttion
    output = np.zeros(image.shape)
    for i in range(sizeP, xImg - sizeP): # for each row in the excluding boundaries 
        for j in range(sizeP, yImg - sizeP): # for each column in the row excluding boundaries 
            accur = 0 
            for kx in range(0, sizeK): # for each row in the kernel
                for ky in range(0, sizeK): # for each column in the row
                    accur = accur + image[i+kx-sizeP,j+ky-sizeP]*kernel[kx,ky] #multiply elment value corresponding to pixel value
            output[i,j] = accur # value from convolution
#             output[i, j] = np.sum(np.multiply(kernel, image[i-sizeP:i+sizeP, j-sizeP:j+sizeP]))
    return output

#This function perfroms edge detection
def simple_detection(img):
    #Initialize kernel to perform edge detection
    kernel = np.ones((3,3)) * -1 
    kernel[1,1] = 8
    
    #Perfrom convolution and adjust pixel values to be between 0 and 255
    out = convolve(img, kernel)
    out = out + np.min(out)*-1
    out = out * 255/(np.max(out) - np.min(out))
    
    return out

def sobel_detection(img):
    
    #Initialize X and Y kernels
    Xkernel = np.array([[1,2,1],[0,0,0], [-1,-2,-1]])
    Ykernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    
    #Perfrom convolution for horizontal and vertical edges
    Xedge = signal.convolve2d(img, Xkernel, boundary = 'symm', mode = 'same')
    Yedge = signal.convolve2d(img, Ykernel, boundary = 'symm', mode = 'same')
    
    #Calculate magnitude of the gradient
    edgeImg = np.sqrt(np.square(Xedge) + np.square(Yedge))
    edgeImg = edgeImg * 255 / (np.max(edgeImg) - np.min(edgeImg))
    
    #Show image
    # plt.figure()
    # plt.imshow(Xedge, cmap = 'gray')
    # plt.title('Horizontal Edge')
    # plt.show()
    
    # plt.figure()
    # plt.imshow(Yedge, cmap = 'gray')
    # plt.title('Vertical Edge')
    # plt.show()
    
    plt.figure()
    plt.imshow(edgeImg, cmap = 'gray')
    plt.title('Edges before thresholding')
    plt.show()
    
    return edgeImg

def smoothing(img, Ksize):
    smoothKernel = np.ones((Ksize,Ksize))*(1/Ksize**2)
    smImg = signal.convolve2d(img, smoothKernel, boundary = 'symm', mode = 'same')
    return smImg

# This function performs thresholding 
def thresholding(img, threshold):
    output = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] > threshold:
                output[i,j] = 255
            else:
                output[i,j] = 0
    return output

def plotHist(img):
    plt.figure()
    plt.hist(img.reshape(-1,1), bins=range(0,256,1), edgecolor='black')
    plt.show()
    
def convertGray(img):
    gray = (0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2])
    return gray

# read image
img = plt.imread("test_img/002.jpg")
img = convertGray(img)

# plt.imshow(img, cmap = 'gray')
# plotHist(img)

smImg = smoothing(img,5)
plt.imshow(smImg, cmap = 'gray')
plotHist(smImg)

print(img.shape)
print(smImg.shape)

# Sobel Detection
edgeImg = sobel_detection(smImg)

biImg = thresholding(edgeImg, 30)
plt.figure()
plt.imshow(biImg, cmap= 'gray')
plt.savefig('test_edge/002_edge.png')
plt.figure()
plt.title('Edges after thresholding')
plt.hist(biImg.reshape(-1,1).reshape(-1,1))

plt.show()

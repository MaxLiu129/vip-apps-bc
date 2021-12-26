#!/usr/bin/env python
# coding: utf-8

# In[380]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[381]:


#This function performs convolution between an image and a kernel
def convolve(image, kernel): 
    
    # Get the size of the image and the kernel
    xKer = kernel.shape[0]
    yKer = kernel.shape[1]
    xImg = image.shape[0]
    yImg = image.shape[1]
    
    # Perform convoluttion
    output = image.copy()
    for i in range(1, xImg-1): # for each row in the excluding boundaries 
        for j in range(1, yImg-1): # for each column in the row excluding boundaries 
            accur = 0 
            for kx in range(0, xKer): # for each row in the kernel
                for ky in range(0, yKer): # for each column in the row
                    accur = accur + image[i+kx-1,j+ky-1]*kernel[kx,ky] #multiply elment value corresponding to pixel value
            output[i,j] = accur # value from convolution
    return output


# In[382]:


#This function perfroms edge detection
def detect_edges(img, threshold):
    #Initize kernel to perform edge detection
    kernel = np.ones((3,3)) * -1 
    kernel[1,1] = 8
    
    #Perfrom convolution and adjust pixel values to be between 0 and 255
    out = convolve(img, kernel)
    out = out + np.min(out)*-1
    out = out * 255/(np.max(out) - np.min(out))
    
    #Perfrom qunantization to either 0 or 255
    output = out.copy()
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if out[i,j] > threshold:
                output[i,j] = 255
            else:
                output[i,j] = 0
    return output


# In[387]:


# read image
img = plt.imread("bc_test1.jpg")

# Convert to gray scale
img = (0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2])
plt.figure()
plt.imshow(img, cmap = 'gray')

# Plot image histogram
plt.figure()
arr = img.reshape(-1,1)
plt.hist(arr, bins=range(0,256,5))
plt.show()

# Image smoothing
smoothKernel = np.ones((3,3))*(1/9)
out1 = convolve(img, kernel1)

# Edge detection
out2 = detect_edges(out1, 100)
arr2 = out2.reshape(-1,1)
plt.hist(arr2, bins=range(0,256,5))
plt.figure()
plt.imshow(out2, cmap='gray')
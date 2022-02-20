import cv as cv
import cv2
from PIL import Image
from PIL.ImageColor import colormap
from matplotlib.pyplot import figure, subplot, gray, title
from numpy import *
import numpy as np
from numpy.distutils.from_template import conv
from scipy import ndimage
from scipy.ndimage import filters
from scipy.signal import convolve2d
from skimage.io import imread,imsave,imshow
from matplotlib import pyplot as plt

def gauss_1(sigma_d):
    #result=cv2.GaussianBlur(image,(999,999),sigma_d)
    #return result
    kernel_size = 2 * 3 * sigma_d + 1
    kernel = np.zeros((int(kernel_size), int(kernel_size)))
    center = kernel_size // 2
    s = sigma_d ** 2
    #sum_val = 0
    for i in range(int(kernel_size)):
        for j in range(int(kernel_size)):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * s))
            # print(i,j,kernel[i,j])
    kernel = kernel / (2 * np.pi * s)
    return kernel



G = imread("in.bmp")
print(G.shape)
graph = np.array(G)
        #kernel = gauss_1(float(sys.argv[2]))
        #im = ndimage.convolve(graph[:,:,0], kernel)
kernel1=gauss_1(1)
#im = cv2.filter2D(graph[:,:,0], -1, kernel1)
im=ndimage.convolve(graph[:, :, 0], kernel1)
#im2=ndimage.convolve(graph[:, :, 0], kernel2)
dx = np.zeros((G.shape[0], G.shape[1]))
dy = np.zeros((G.shape[0], G.shape[1]))
d = np.zeros((G.shape[0], G.shape[1]))
for i in range (1,G.shape[1]-1):
    for j in range (1,G.shape[1]-1):
        dx[i,j] = im[i,j+1] - im[i,j]
        dy[i,j]= im[i+1, j] -im[i,j]
        d[i,j] = sqrt(dx[i,j]*dx[i,j] + dy[i,j]*dy[i,j])
im=d
#im=abs(im1)+abs(im2)
gmax=np.max(im)
print(gmax)

for i in range(int(G.shape[0])):
    for j in range(int(G.shape[1])):
        if(im[i,j]==gmax):
            im[i,j]=255
        else:
            im[i,j]=im[i,j]*255/gmax
#im=abs(im1)+abs(im2)
#im=cv2.addWeighted(im1,0.5,im2,0.5,0)

print(im)
#IMAGE = np.array(im, dtype='uint8')
imsave("new1.bmp", im)

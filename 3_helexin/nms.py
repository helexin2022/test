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
    kernel_size = 2 * 3 * sigma_d + 1
    kernel = np.zeros((int(kernel_size), int(kernel_size)))
    center = kernel_size // 2
    s = sigma_d ** 2
    #sum_val = 0
    for i in range(int(kernel_size)):
        for j in range(int(kernel_size)):
            x, y = i - center, j - center
            kernel[i, j] = (-x/s)*np.exp(-(x ** 2 + y ** 2) / (2 * s))
            # print(i,j,kernel[i,j])
    kernel = kernel / (2 * np.pi * s)
    return kernel

def gauss_2(sigma_d):
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
            kernel[i, j] = (-y/s)*np.exp(-(x ** 2 + y ** 2) / (2 * s))
            # print(i,j,kernel[i,j])
    kernel = kernel / (2 * np.pi * s)
    return kernel

G = imread("in.bmp")
graph = np.array(G,dtype='float')
sigma=2
kernel1=gauss_1(sigma)
kernel2=gauss_2(sigma)
im1=ndimage.convolve(graph[:, :, 0], kernel1)
im2=ndimage.convolve(graph[:, :, 0], kernel2)
im=sqrt(im1*im1+im2*im2)
gmax=np.max(im)

for i in range(int(G.shape[0])):
    for j in range(int(G.shape[1])):
        if(im[i,j]==gmax):
            im[i,j]=255
        else:
            im[i,j]=im[i,j]*255/gmax


#           非极大值抑制插值处理            %%

jd = zeros((G.shape[0],G.shape[1]))#定义一个非极大值图像
for i in range(1,int(G.shape[0])-1):
    for j in range(1,int(G.shape[1])-1):
        if abs(im2[i,j])>abs(im1[i,j]):
            #weight=im1[i,j]/im2[i,j]
            g2=im[i-1,j]
            g4=im[i+1,j]
            if im2[i,j]*im1[i,j]>0:
                g1=im[i-1,j-1]
                g3=im[i+1,j+1]
            elif im2[i,j]*im1[i,j]<0:
                g1=im[i-1,j+1]
                g3=im[i+1,j-1]
        else:
            #weight=im2[i,j]/im1[i,j]
            g2=im[i,j-1]
            g4=im[i,j+1]
            if im2[i,j]*im1[i,j]>0:
                g1=im[i-1,j-1]
                g3=im[i+1,j+1]
            elif im2[i,j]*im1[i,j]<0:
                g1=im[i+1,j-1]
                g3=im[i-1,j+1]
        #dTemp1 = weight*g1 + (1-weight)*g2
        #dTemp2 = weight*g3 + (1-weight)*g4
        if (im[i,j]>=g1 and im[i,j]>=g2 and im[i,j]>=g3 and im[i,j]>=g4):
            jd[i,j] =im[i,j]
        else:
            jd[i,j]=0


IMAGE = np.array(jd, dtype='uint8')
imsave("new2.bmp", IMAGE)

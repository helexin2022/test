from math import atan2

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


#           非极大值抑制插值处理
a=zeros((G.shape[0],G.shape[1]))
for i in range(1,int(G.shape[0])-1):
    for j in range(1,int(G.shape[1])-1):
            a[i, j] = atan2(im2[i,j],im1[i,j])
            # print(a[i,j])
            if ((a[i, j] >= -pi/8) and (a[i, j] < pi/8) or (a[i, j] <= -7*pi/8) and (a[i, j] >= -pi) or (
                    a[i, j] >= 7*pi/8) and (a[i, j] <= pi)):
                a[i, j] = 0
            elif ((a[i, j] >= pi/8) and (a[i, j] < 3*pi/8) or (a[i, j] <= -5*pi/8) and (a[i, j] > -7*pi/8)):
                a[i, j] = 45
            elif ((a[i, j] >= 3*pi/8) and (a[i, j] < 5*pi/8) or (a[i, j] <= -3*pi/8) and (a[i, j] > - 5*pi/8)):
                a[i, j] = 90
            elif ((a[i, j] >= 5*pi/8) and (a[i, j] < 7*pi/8) or (a[i, j] <= -pi/8) and (a[i, j] > -3*pi/8)):
                a[i, j] = -45
            # print(a[i,j])

jd = zeros((G.shape[0],G.shape[1]))#定义一个非极大值图像
for i in range(1,int(G.shape[0])-1):
    for j in range(1,int(G.shape[1])-1):
        if(a[i,j]==90 and im[i,j]==max(im[i,j], im[i,j+1], im[i,j-1])):
            jd[i,j]=im[i,j]
        elif(a[i, j] == -45 and im[i, j] == max(im[i, j], im[i + 1, j - 1], im[i - 1, j + 1])):
            jd[i, j] = im[i, j]
        elif(a[i, j] == 0 and im[i, j] == max(im[i, j], im[i + 1, j], im[i - 1, j])):
            jd[i, j] = im[i, j]
        elif(a[i, j] == 45 and im[i, j] == max(im[i, j], im[i + 1, j + 1], im[i - 1, j - 1])):
            jd[i, j] = im[i, j]

IMAGE = np.array(jd, dtype='uint8')
imsave("new2.bmp", IMAGE)

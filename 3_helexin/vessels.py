from math import atan2
from queue import Queue
import cv as cv
import cv2
from PIL import Image
from PIL.ImageColor import colormap
from matplotlib.pyplot import figure, subplot, gray, title
from numpy import *
from sympy import *
import numpy as np
from numpy.distutils.from_template import conv
from scipy import ndimage
from scipy.ndimage import filters
from scipy.signal import convolve2d
from skimage.io import imread,imsave,imshow
from matplotlib import pyplot as plt

def gauss_xx(sigma_d):
    kernel_size = 2 * 3 * sigma_d + 1
    kernel = np.zeros((int(kernel_size), int(kernel_size)))
    center = kernel_size // 2
    s = sigma_d ** 2
    #sum_val = 0
    for i in range(int(kernel_size)):
        for j in range(int(kernel_size)):
            x, y = symbols('x,y')
            z=diff((exp(-(x ** 2 + y ** 2) / (2 * s))),x,2)
            k, l = i - center, j - center
            kernel[i, j] = z.subs({x:k,y:l})
            # print(i,j,kernel[i,j])
    kernel = kernel / (2 * np.pi * s)
    return kernel

def gauss_yy(sigma_d):
    #result=cv2.GaussianBlur(image,(999,999),sigma_d)
    #return result
    kernel_size = 2 * 3 * sigma_d + 1
    kernel = np.zeros((int(kernel_size), int(kernel_size)))
    center = kernel_size // 2
    s = sigma_d ** 2
    #sum_val = 0
    for i in range(int(kernel_size)):
        for j in range(int(kernel_size)):
            x, y = symbols('x,y')
            z = diff((exp(-(x ** 2 + y ** 2) / (2 * s))), y, 2)
            k, l = i - center, j - center
            kernel[i, j] = z.subs({x: k, y: l})
            # print(i,j,kernel[i,j])
    kernel = kernel / (2 * np.pi * s)
    return kernel

def gauss_xy(sigma_d):
    #result=cv2.GaussianBlur(image,(999,999),sigma_d)
    #return result
    kernel_size = 2 * 3 * sigma_d + 1
    kernel = np.zeros((int(kernel_size), int(kernel_size)))
    center = kernel_size // 2
    s = sigma_d ** 2
    #sum_val = 0
    for i in range(int(kernel_size)):
        for j in range(int(kernel_size)):
            x, y = symbols('x,y')
            k, l = i - center, j - center
            z = diff(diff(exp(-(x ** 2 + y ** 2) / (2 * s)),y),x)
            kernel[i, j] = z.subs({x: k, y: l})
            # print(i,j,kernel[i,j])
    kernel = kernel / (2 * np.pi * s)
    return kernel

G = imread("in2.bmp")
graph = np.array(G,dtype='float')
sigma=2
kernel1=gauss_xx(sigma)
kernel2=gauss_yy(sigma)
kernel3=gauss_xy(sigma)

#print(kernel1)
#print(kernel2)
#print(kernel3)

im1=ndimage.convolve(graph[:, :, 0], kernel1)
im2=ndimage.convolve(graph[:, :, 0], kernel2)
im3=ndimage.convolve(graph[:, :, 0], kernel3)

#print(im3)
kl=np.vstack((im1,im3))
kr=np.vstack((im3,im2))
k=np.hstack((kl,kr))

A=np.zeros((G.shape[0],G.shape[1]))
B=np.zeros((G.shape[0],G.shape[1],2))
for i in range(0,int(G.shape[0])):
    for j in range(0,int(G.shape[1])):
        X=np.zeros((2,2))
        X[0,0]=im1[i,j]
        X[0,1]=im3[i,j]
        X[1,0]=im3[i,j]
        X[1,1]=im2[i,j]
        a, b = np.linalg.eig(X)#a-Eigenvalues,b-vector
        #print(a)
        #print(abs(a))
        A[i,j]=max(abs(a))
        #print(A[i,j])
        for l in range(len(a)):
            if(a[l]==-1*A[i,j]):
                A[i,j]=0
        for l in range(len(a)):
            if(abs(a[l])==A[i,j]):
                B[i,j,:]=b[:,l]
                #print(b[:,l],B[i,j,:])

gmax=np.max(A)
#print(A)
#print(B)
for i in range(int(G.shape[0])):
    for j in range(int(G.shape[1])):
        if(A[i,j]==gmax):
            A[i,j]=255
        else:
            A[i,j]=A[i,j]*255/gmax


#IMAGE = np.array(A, dtype='uint8')
#imsave("new3.bmp", IMAGE)

#           非极大值抑制处理
a=np.zeros((G.shape[0],G.shape[1]),np.float)
for i in range(1,int(G.shape[0])-1):
    for j in range(1,int(G.shape[1])-1):
            a[i, j] = atan2(B[i,j,1],B[i,j,0])
            print(a[i,j])
            if ((a[i, j] >= -np.pi/8) and (a[i, j] < np.pi/8) or (a[i, j] <= -7*np.pi/8) and (a[i, j] >= -np.pi) or (
                    a[i, j] >= 7*np.pi/8) and (a[i, j] <= np.pi)):
                a[i, j] = 0
            elif ((a[i, j] >= np.pi/8) and (a[i, j] < 3*np.pi/8) or (a[i, j] <= -5*np.pi/8) and (a[i, j] > -7*np.pi/8)):
                a[i, j] = 45
            elif ((a[i, j] >= 3*np.pi/8) and (a[i, j] < 5*np.pi/8) or (a[i, j] <= -3*np.pi/8) and (a[i, j] > - 5*np.pi/8)):
                a[i, j] = 90
            elif ((a[i, j] >= 5*np.pi/8) and (a[i, j] < 7*np.pi/8) or (a[i, j] <= -np.pi/8) and (a[i, j] > -3*np.pi/8)):
                a[i, j] = -45
            #print(a[i,j])

jd = np.zeros((G.shape[0],G.shape[1]))#定义一个非极大值图像
for i in range(1,int(G.shape[0])-1):
    for j in range(1,int(G.shape[1])-1):
        if(a[i,j]==90 and A[i,j]==max(A[i,j], A[i,j+1], A[i,j-1])):
            jd[i,j]=A[i,j]
        elif(a[i, j] == -45 and A[i, j] == max(A[i, j], A[i + 1, j - 1], A[i - 1, j + 1])):
            jd[i, j] = A[i, j]
        elif(a[i, j] == 0 and A[i, j] == max(A[i, j], A[i + 1, j], A[i - 1, j])):
            jd[i, j] = A[i, j]
        elif(a[i, j] == 45 and A[i, j] == max(A[i, j], A[i + 1, j + 1], A[i - 1, j - 1])):
            jd[i, j] = A[i, j]
        else:
            jd[i,j] = 0
print(A)
print(a)

'''
gmax=255
ans=zeros((G.shape[0],G.shape[1]))
for i in range(1,int(G.shape[0])-1):
    for j in range(1,int(G.shape[0])-1):
        if(jd[i,j]>thr_high*gmax):
            jd[i,j]=255
        elif(jd[i,j]<thr_low*gmax):
            jd[i,j]=0

for i in range(1,int(G.shape[0])-1):
    for j in range(1, int(G.shape[0]) - 1):
        if (jd[i , j ] > thr_low * gmax and jd[i , j ] < thr_high * gmax):
            su = [jd[i - 1, j - 1], jd[i - 1, j], jd[i - 1, j + 1],
                  jd[i, j - 1], jd[i, j], jd[i, j + 1],
                  jd[i + 1, j - 1], jd[i + 1, j], jd[i + 1, j + 1]]
            Max = max(su)
            if (Max ==255):
                jd[i, j] = 255
            else:
                jd[i, j] = 0
'''
IMAGE = np.array(jd, dtype='uint8')
imsave("new3.bmp", IMAGE)

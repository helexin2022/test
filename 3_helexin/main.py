from math import atan2
import sys
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

def grad(graph,sigma):
    kernel1 = gauss_1(sigma)
    kernel2 = gauss_2(sigma)
    im1 = ndimage.convolve(graph[:, :, 0], kernel1)
    im2 = ndimage.convolve(graph[:, :, 0], kernel2)
    im = sqrt(im1 * im1 + im2 * im2)
    gmax = np.max(im)
    for i in range(int(G.shape[0])):
        for j in range(int(G.shape[1])):
            if (im[i, j] == gmax):
                im[i, j] = 255
            else:
                im[i, j] = im[i, j] * 255 / gmax
    return im

def nonmax(graph,sigma):
    kernel1 = gauss_1(sigma)
    kernel2 = gauss_2(sigma)
    im1 = ndimage.convolve(graph[:, :, 0], kernel1)
    im2 = ndimage.convolve(graph[:, :, 0], kernel2)
    im = sqrt(im1 * im1 + im2 * im2)
    gmax = np.max(im)

    for i in range(int(G.shape[0])):
        for j in range(int(G.shape[1])):
            if (im[i, j] == gmax):
                im[i, j] = 255
            else:
                im[i, j] = im[i, j] * 255 / gmax

    #           非极大值抑制插值处理
    a = zeros((G.shape[0], G.shape[1]))
    for i in range(1, int(G.shape[0]) - 1):
        for j in range(1, int(G.shape[1]) - 1):
            a[i, j] = atan2(im2[i, j], im1[i, j])
            # print(a[i,j])
            if ((a[i, j] >= -pi / 8) and (a[i, j] < pi / 8) or (a[i, j] <= -7 * pi / 8) and (a[i, j] >= -pi) or (
                    a[i, j] >= 7 * pi / 8) and (a[i, j] <= pi)):
                a[i, j] = 0
            elif ((a[i, j] >= pi / 8) and (a[i, j] < 3 * pi / 8) or (a[i, j] <= -5 * pi / 8) and (
                    a[i, j] > -7 * pi / 8)):
                a[i, j] = 45
            elif ((a[i, j] >= 3 * pi / 8) and (a[i, j] < 5 * pi / 8) or (a[i, j] <= -3 * pi / 8) and (
                    a[i, j] > - 5 * pi / 8)):
                a[i, j] = 90
            elif ((a[i, j] >= 5 * pi / 8) and (a[i, j] < 7 * pi / 8) or (a[i, j] <= -pi / 8) and (
                    a[i, j] > -3 * pi / 8)):
                a[i, j] = -45
            # print(a[i,j])

    jd = zeros((G.shape[0], G.shape[1]))  # 定义一个非极大值图像
    for i in range(1, int(G.shape[0]) - 1):
        for j in range(1, int(G.shape[1]) - 1):
            if (a[i, j] == 90 and im[i, j] == max(im[i, j], im[i, j + 1], im[i, j - 1])):
                jd[i, j] = im[i, j]
            elif (a[i, j] == -45 and im[i, j] == max(im[i, j], im[i + 1, j - 1], im[i - 1, j + 1])):
                jd[i, j] = im[i, j]
            elif (a[i, j] == 0 and im[i, j] == max(im[i, j], im[i + 1, j], im[i - 1, j])):
                jd[i, j] = im[i, j]
            elif (a[i, j] == 45 and im[i, j] == max(im[i, j], im[i + 1, j + 1], im[i - 1, j - 1])):
                jd[i, j] = im[i, j]
    return jd

def canny(G,jd,thr_low,thr_high):
    gmax = 255
    for i in range(int(G.shape[0])):
        for j in range(int(G.shape[0])):
            if (jd[i, j] > thr_high * gmax):
                jd[i, j] = 255
            elif (jd[i, j] < thr_low * gmax):
                jd[i, j] = 0

    for i in range(1, int(G.shape[0]) - 1):
        for j in range(1, int(G.shape[0]) - 1):
            if (jd[i, j] > thr_low * gmax and jd[i, j] < thr_high * gmax):
                su = [jd[i - 1, j - 1], jd[i - 1, j], jd[i - 1, j + 1],
                      jd[i, j - 1], jd[i, j], jd[i, j + 1],
                      jd[i + 1, j - 1], jd[i + 1, j], jd[i + 1, j + 1]]
                Max = max(su)
                if (Max == 255):
                    jd[i, j] = 255
                else:
                    jd[i, j] = 0
    return jd

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

def vessels(graph):
    sigma = 2
    kernel1 = gauss_xx(sigma)
    kernel2 = gauss_yy(sigma)
    kernel3 = gauss_xy(sigma)

    im1 = ndimage.convolve(graph[:, :, 0], kernel1)
    im2 = ndimage.convolve(graph[:, :, 0], kernel2)
    im3 = ndimage.convolve(graph[:, :, 0], kernel3)
    kl = np.vstack((im1, im3))
    kr = np.vstack((im3, im2))
    k = np.hstack((kl, kr))

    A = np.zeros((G.shape[0], G.shape[1]))
    B = np.zeros((G.shape[0], G.shape[1], 2))
    for i in range(0, int(G.shape[0])):
        for j in range(0, int(G.shape[1])):
            X = np.zeros((2, 2))
            X[0, 0] = im1[i, j]
            X[0, 1] = im3[i, j]
            X[1, 0] = im3[i, j]
            X[1, 1] = im2[i, j]
            a, b = np.linalg.eig(X)  # a-Eigenvalues,b-vector
            # print(a)
            # print(abs(a))
            A[i, j] = max(abs(a))
            for l in range(len(a)):
                if (a[l] == -1 * A[i, j]):
                    A[i, j] = 0
            for l in range(len(a)):
                if (abs(a[l]) == A[i, j]):
                    B[i, j, :] = b[:, l]
                    # print(b[:,l],B[i,j,:])

    gmax = np.max(A)
    # print(B)
    for i in range(int(G.shape[0])):
        for j in range(int(G.shape[1])):
            if (A[i, j] == gmax):
                A[i, j] = 255
            else:
                A[i, j] = A[i, j] * 255 / gmax

    # IMAGE = np.array(A, dtype='uint8')
    # imsave("new3.bmp", IMAGE)

    #           非极大值抑制插值处理
    a = np.zeros((G.shape[0], G.shape[1]))
    for i in range(1, int(G.shape[0]) - 1):
        for j in range(1, int(G.shape[1]) - 1):
            a[i, j] = atan2(B[i, j, 1], B[i, j, 0])
            # print(a[i,j])
            if ((a[i, j] >= -np.pi / 8) and (a[i, j] < np.pi / 8) or (a[i, j] <= -7 * np.pi / 8) and (
                    a[i, j] >= -np.pi) or (
                    a[i, j] >= 7 * np.pi / 8) and (a[i, j] <= np.pi)):
                a[i, j] = 0
            elif ((a[i, j] >= np.pi / 8) and (a[i, j] < 3 * np.pi / 8) or (a[i, j] <= -5 * np.pi / 8) and (
                    a[i, j] > -7 * np.pi / 8)):
                a[i, j] = 45
            elif ((a[i, j] >= 3 * np.pi / 8) and (a[i, j] < 5 * np.pi / 8) or (a[i, j] <= -3 * np.pi / 8) and (
                    a[i, j] > - 5 * np.pi / 8)):
                a[i, j] = 90
            elif ((a[i, j] >= 5 * np.pi / 8) and (a[i, j] < 7 * np.pi / 8) or (a[i, j] <= -np.pi / 8) and (
                    a[i, j] > -3 * np.pi / 8)):
                a[i, j] = -45
            # print(a[i,j])

    jd = np.zeros((G.shape[0], G.shape[1]))  # 定义一个非极大值图像
    for i in range(1, int(G.shape[0]) - 1):
        for j in range(1, int(G.shape[1]) - 1):
            if (a[i, j] == 90 and A[i, j] == max(A[i, j], A[i, j + 1], A[i, j - 1])):
                jd[i, j] = A[i, j]
            elif (a[i, j] == -45 and A[i, j] == max(A[i, j], A[i + 1, j - 1], A[i - 1, j + 1])):
                jd[i, j] = A[i, j]
            elif (a[i, j] == 0 and A[i, j] == max(A[i, j], A[i + 1, j], A[i - 1, j])):
                jd[i, j] = A[i, j]
            elif (a[i, j] == 45 and A[i, j] == max(A[i, j], A[i + 1, j + 1], A[i - 1, j - 1])):
                jd[i, j] = A[i, j]
            else:
                jd[i, j] = 0
    return jd

if __name__ == '__main__':
    if(sys.argv[1]=="grad"):
        sigma=(float(sys.argv[2]))
        G = imread(sys.argv[3])
        graph = np.array(G, dtype='float')
        im=grad(graph,sigma)
        IMAGE = np.array(im, dtype='uint8')
        imsave(sys.argv[4], IMAGE)
    elif(sys.argv[1]=="nonmax"):
        sigma = (float(sys.argv[2]))
        G = imread(sys.argv[3])
        graph = np.array(G, dtype='float')
        im = nonmax(graph, sigma)
        IMAGE = np.array(im, dtype='uint8')
        imsave(sys.argv[4], IMAGE)
    elif(sys.argv[1]=="canny"):
        sigma = (float(sys.argv[2]))
        thr_high=float(sys.argv[3])
        thr_low=float(sys.argv[4])
        G = imread(sys.argv[5])
        graph = np.array(G, dtype='float')
        im = nonmax(graph, sigma)
        im=canny(G,im,thr_low,thr_high)
        IMAGE = np.array(im, dtype='uint8')
        imsave(sys.argv[6], IMAGE)
    elif(sys.argv[1]=="vessels"):
        G=imread(sys.argv[2])
        graph = np.array(G, dtype='float')
        im=vessels(graph)
        IMAGE = np.array(im, dtype='uint8')
        imsave(sys.argv[3], IMAGE)


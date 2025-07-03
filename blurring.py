import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#BGR to grayscale
img=cv.imread('Photos/cats.jpg')
cv.imshow('OriginalImage',img)

#averaging
average=cv.blur(img,(3,3))
cv.imshow('Average Blur ',average)

#gaussian blur
gauss=cv.GaussianBlur(img,(3,3),0)
cv.imshow('Gaussian Blur ',gauss)

#median blur
median=cv.medianBlur(img,3)
cv.imshow('Median Blur ',median)

#bilateral blurring
bilateral=cv.bilateralFilter(img,10,35,25)
cv.imshow('Bilateral Blur ',bilateral)

cv.waitKey(0)

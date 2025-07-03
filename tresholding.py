import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img =cv.imread('Photos/cats.jpg')
cv.imshow('Original Image',img)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

#simple Tresholding 
threshold,thresh=cv.threshold(gray,150,255,cv.THRESH_BINARY)
cv.imshow('Simple Thresholded',thresh)

#simple Thresholding inverse
threshold,thresh_inv=cv.threshold(gray,150,255,cv.THRESH_BINARY_INV)
cv.imshow('Simple Thresholded Inverse',thresh_inv)
#Adaptive Thresholded 
adaptive_thresh=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,3)
cv.imshow('Adaptive Thresholded',adaptive_thresh)

cv.waitKey(0)

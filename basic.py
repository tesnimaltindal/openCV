import cv2 as cv
import numpy as np
img=cv.imread('Photos/cat.jpg')
cv.imshow('OriginalImage',img)
#convert to grayscale COLOR_BGR2GRAY
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

#blur
blur =cv.GaussianBlur(img,(7,7),cv.BORDER_DEFAULT)
cv.imshow('Blurred',blur)

#edge cascade
canny =cv.Canny(img,125,175)
cv.imshow('Canny Edge Detection',canny)

#dilating the image genişletmek 
dilated =cv.dilate(canny,(3,3),iterations=1)

cv.imshow('Dilated',dilated)
#eroding the image aşındırma  
eroded =cv.erode(dilated,(3,3),iterations=1)
cv.imshow('Eroded',eroded)
#Resize
resized =cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)
cv.imshow('Resize',resized)
#Cropped
cropped=img[50:200,200:400]
cv.imshow('Cropped',cropped)
cv.waitKey(0)
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#BGR to grayscale
img=cv.imread('Photos/park.jpg')
cv.imshow('OriginalImage',img)
plt.imshow(img)
plt.show()

gray =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

#BGR to HSV
hsv =cv.cvtColor(img,cv.COLOR_BGR2HSV)
cv.imshow('HSV',hsv)

#Bgr to L A B 
lab=cv.cvtColor(img,cv.COLOR_BGR2LAB)
cv.imshow('Lab',lab)
#BGR to RGB
rgb =cv.cvtColor(img,cv.COLOR_BGR2RGB)
cv.imshow('RGB',rgb)

#HSV To BGR
hsv_bgr=cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
cv.imshow('HSV-->BGR',hsv_bgr)

#Lab To BGR
lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
cv.imshow('Lab-->BGR',lab_bgr)


plt.imshow(rgb)
plt.show()


cv.waitKey(0)
#grayscale to bgr not directly transformed
#Gri tonlamalı bir görüntüden renkli bir görüntü 
# oluşturmak için genellikle tüm renk kanallarını gri 
# değeriyle doldurarak 3 kanallı bir görüntü oluşturursunuz
#  (örneğin cv.cvtColor(gray, cv.COLOR_GRAY2BGR)).
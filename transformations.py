import cv2 as cv
import numpy as np
img=cv.imread('Photos/park.jpg')
cv.imshow('OriginalImage',img)
#translation 
def translate(img,x,y):
    transMat=np.float32([[1,0,x],[0,1,y]])
    dimensions=(img.shape[1],img.shape[0])  #shape 1= width shape 0= height
    return cv.warpAffine(img,transMat,dimensions)
#-x=left 
#-y=up
#x=right
#y=right
translated =translate(img,-100,100)
cv.imshow('Translated',translated)
#Rotation 
def rotate(img,angle,rotPoint=None):
    (height,width)=img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2,height//2)
    rotMat =cv.getRotationMatrix2D(rotPoint,angle,1.0)
    dimensions=(width,height)

    return cv.warpAffine(img,rotMat,dimensions)

rotated=rotate(img,-45)
cv.imshow('Rotated',rotated)

rotated_rotated= rotate(img,-90)
cv.imshow('Rotated_Rotated',rotated_rotated)

#Resizing
resized=cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)
cv.imshow('Resized',resized)

#flipping
flip=cv.flip(img,0)
cv.imshow('Flipped',flip)
#cropping
cropped=img[200:400,300:400]
cv.imshow('Cropped',cropped)
cv.waitKey(0)
    
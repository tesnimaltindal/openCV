import cv2 as cv
import numpy as np
blank=np.zeros((500,500,3),dtype='uint8')
cv.imshow('Blank',blank)
#1.paint image the certain color 
#blank[200:300,300:400]=0,0,255
#cv.imshow('Red',blank)
#img = cv.imread('Photos/cat.jpg')
#cv.imshow('Cat',img)
#2.Draw a rectangle
cv.rectangle(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(0,250,0),thickness=-1)
cv.imshow('Rectangle',blank)
#3.draw circcle 
cv.circle(blank,(blank.shape[1]//2,blank.shape[0]//2),40,(0,0,255),thickness=-1)
cv.imshow('circle',blank)
#4.draw line 
cv.line(blank,(100,250),(blank.shape[1]//2,blank.shape[0]//2),(255,255,255),thickness=3)
cv.imshow('circle',blank)
#5.write text
cv.putText(blank,'Hello',(255,255),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),2)
cv.imshow('Text',blank)

cv.waitKey(0)
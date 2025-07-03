import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

blank=np.zeros((400,400),dtype='uint8')
rectangle=cv.rectangle(blank.copy(),(30,30),(370,370),255,-1)
circle=cv.circle(blank.copy(),(200,200),200,255,-1)
cv.imshow('Rectangle',rectangle)
cv.imshow('Circle',circle)
#bitwise AND
bitwise_and=cv.bitwise_and(circle,rectangle)
cv.imshow('Bitwise AND',bitwise_and)

#bitwise OR = intersecting regions and non-intersecting regions
bitwise_or=cv.bitwise_or(circle,rectangle)
cv.imshow('Bitwise OR',bitwise_or)

#bitwise XOR = non-intersecting regions
bitwise_xor=cv.bitwise_xor(circle,rectangle)
cv.imshow('Bitwise XOR',bitwise_xor)

#bitwise NOT = tersi alınır white to black 
bitwise_not=cv.bitwise_not(rectangle)
cv.imshow('Bitwise NOT',bitwise_not)

cv.waitKey(0)
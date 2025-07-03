import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#BGR to grayscale
img=cv.imread('Photos/park.jpg')
cv.imshow('OriginalImage',img)

blank=np.zeros(img.shape[:2],dtype='uint8')


b,g,r=cv.split(img)

#orijinal görüntüyü yeniden birleştiren (merge)  

blue=cv.merge([b,blank,blank])
green=cv.merge([blank,g,blank])
red=cv.merge([blank,blank,r])

cv.imshow('Blue',blue)
cv.imshow('Green',green)
cv.imshow('Red',red)

print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merged=cv.merge([b,g,r])
cv.imshow('Merged Image',merged)
#merged = cv.merge([b, g, r]): 
# Daha önce cv.split() ile ayrılmış olan
#  b, g ve r kanallarını alıp, bunları tekrar 
# bir araya getirerek 3 kanallı (renkli) bir görüntü
#  (merged) oluşturur.

cv.waitKey(0)

#Neden BGR sırası?: 
# OpenCV varsayılan olarak BGR kullandığı için,
#  cv.merge fonksiyonuna verilen sıralama da BGR
#  olmalıdır. Yani, [Mavi_Kanal, Yeşil_Kanal,
#  Kırmızı_Kanal] şeklinde sıralanmalıdır.

#Eğer OpenCV ile bir görüntüyü okuyup (BGR olarak),
#  sonra bunu doğrudan Matplotlib'e verirseniz, 
# renkler yanlış (mavi ve kırmızı yer değiştirmiş)
#  görünecektir. Bu durumu düzeltmek için 
# cv.cvtColor() fonksiyonunu kullanmanız gerekir.

#OpenCV varsayılan olarak BGR kullanır.

#Matplotlib, Pillow (PIL), web (CSS/HTML) genellikle RGB kullanır

#ikisi de renkleri Kırmızı, Yeşil ve Mavi bileşenlerle temsil eder,
#  ancak kanal sırası farklıdır ve bu, kodunuzda doğru dönüşümleri
#  yapmazsanız görsel hatalara yol açar.
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img =cv.imread('Photos/cats.jpg')
cv.imshow('Original Image',img)

blank=np.zeros(img.shape[:2],dtype='uint8')

# gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('Gray',gray)

mask=cv.circle(blank,[img.shape[1]//2,img.shape[0]//2],100,255,-1)

masked=cv.bitwise_and(img,img,mask=mask)
cv.imshow('Masked',masked)
# #Grayscale Histogram 
# gray_hist=cv.calcHist([gray],[0],mask,[256],[0,256])

# plt.figure()
# plt.title('GrayScale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('#of pixels')
# plt.plot(gray_hist)
# plt.xlim([0,256])
# plt.show()

#color histogram 
plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('#of pixels')
colors=('b','g','r')
for i,col in enumerate(colors):
    hist=cv.calcHist([img],[i],mask,[256],[0,256])
    plt.plot(hist,color=col)
    plt.xlim([0,256])


plt.show()

cv.waitkey(0)

#gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256]): Bu satır gri tonlamalı görüntünün histogramını hesaplar. Parametrelerin anlamları şunlardır:

# [gray]: Histogramı hesaplanacak olan görüntüyü (gri tonlamalı görüntü) bir liste içinde belirtiriz.

# [0]: Kanal indeksidir. Gri tonlamalı görüntü tek kanallı olduğu için 0 kullanılır.

# None: Maske parametresidir. Tüm görüntünün histogramını hesaplamak istediğimiz için None kullanılır. Belirli bir bölgenin histogramı hesaplanacaksa buraya bir maske görüntüsü verilir.

# [256]: Bin sayısıdır. Görüntü piksel değerleri 0 ile 255 arasında değiştiği için (toplam 256 değer), her bir olası piksel değeri için bir "bin" (kutucuk) kullanırız.

# [0, 256]: Piksel değerlerinin aralığıdır. 0'dan 255'e kadar olan tüm piksel değerleri dahil edilir.

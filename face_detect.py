import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img =cv.imread('Photos/group 1.jpg')

# Resmin yüklenip yüklenmediğini kontrol et
if img is None:
    print("Hata: 'Photos/group 1.jpg' resmi bulunamadı veya yüklenemedi!")
    exit() # Resim yoksa programı sonlandır

cv.imshow('Original Image (Group of People)',img)

#convert Gray scale
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray Person',gray)

haar_cascade=cv.CascadeClassifier('haar_face.xml')

# Haar Cascade dosyasının yüklenip yüklenmediğini kontrol et
if haar_cascade.empty():
    print("Hata: 'haar_face.xml' dosyası yüklenemedi! Dosyanın doğru yolda olduğundan emin olun.")
    exit() # Dosya yoksa programı sonlandır

faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=1)
print(f'Number of faces found = {len(faces_rect)}')


for(x,y,w,h) in faces_rect:
    # Hata buradaydı: img(x,y) yerine img kullanmalıyız.
    # cv.rectangle fonksiyonu ilk argüman olarak görüntünün kendisini alır.
    cv.rectangle(img, (x,y), (x+w,y+h),(0,255,0),thickness=2)
cv.imshow('Detected Faces',img)

cv.waitKey(0)
cv.destroyAllWindows() # Tüm açık pencereleri kapat

#haar_face.xml dosyası, OpenCV'nin bir yüzün neye benzediğini
#  "öğrenmek" için kullandığı, önceden eğitilmiş bir bilgi bankasıdır. 
# İçindeki karmaşık veri yapısı ve özellik sayısı nedeniyle oldukça büyük
#  bir dosya boyutuna sahiptir.
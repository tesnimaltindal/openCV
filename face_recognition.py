import numpy as np
import cv2 as cv
import os # 'os' modülünü kullanmasanız da, dosya yolları için alışkanlık iyidir.

haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Haar Cascade dosyasının yüklenip yüklenmediğini kontrol et
if haar_cascade.empty():
    print("Hata: 'haar_face.xml' dosyası yüklenemedi! Dosyanın doğru yolda olduğundan emin olun.")
    exit()

# Eğitimde kullanılan aynı kişi listesi ve sırası.
# Bu listenin, modelin eğitildiği 'labels' listesiyle aynı sırada olması KRİTİKTİR.
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']


face_recognizer = cv.face.LBPHFaceRecognizer_create()
# Hata düzeltildi: 'face_trained_yml' yerine 'face_trained.yml'
try:
    face_recognizer.read('face_trained.yml')
except cv.error as e:
    print(f"Hata: 'face_trained.yml' modeli yüklenemedi. Modelin var olduğundan ve bozuk olmadığından emin olun.")
    print(f"OpenCV Hatası: {e}")
    exit()

img = cv.imread(r'C:\Users\Tesnim\Downloads\opencv-course-master\opencv-course-master\Resources\Faces\val\mindy_kaling\2.jpg')

# Resmin yüklenip yüklenmediğini kontrol et
if img is None:
    print("Hata: Test edilecek resim bulunamadı veya yüklenemedi!")
    exit()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray) # Gri tonlamalı yüzü göstermek.


# Yüzleri tespit et
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces_rect:

    faces_roi = gray[y:y+h, x:x+w]

    # Tespit edilen yüzü tanı
    label, confidence = face_recognizer.predict(faces_roi)

    # Güven değeri düşükse (yüksekse daha iyi eşleşme) veya tanıdık bir kişi değilse
    # bu kısmı ayarlayarak eşik belirleyebilirsiniz.
    # Örneğin, confidence > 6000 ise "Bilinmeyen" diyebilirsiniz (LBPH için düşük confidence daha iyi eşleşmedir)
    
    # LBPH'de confidence değeri ne kadar küçükse eşleşme o kadar iyidir.
    # Genellikle bir eşik değeri belirlenir. Örneğin, 100'den küçükse güvenilir kabul edilebilir.
    # Bu eşik, eğitim verilerinize ve istediğiniz hassasiyete göre değişir.
    
    print(f'Etiket: {label} ({people[label]}) - Güven: {confidence:.2f}')

    # Yüzün üzerine adı ve güven değerini yaz
    text_to_display = f'{people[label]} ({confidence:.2f})'
    cv.putText(img, text_to_display, (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), thickness=2)
    
    # Yüzün etrafına dikdörtgen çiz
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)
cv.destroyAllWindows()
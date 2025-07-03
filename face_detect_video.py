import cv2 as cv
import numpy as np

# 1. Video yakalama nesnesini oluşturma
# 0, bilgisayarın varsayılan kamerasını kullanır.
# Bir video dosyasını işlemek isterseniz, dosya yolunu buraya yazın:
# cap = cv.VideoCapture('Videos/my_video.mp4')
cap = cv.VideoCapture(0) # Varsayılan kamera

# Video nesnesinin başarıyla açılıp açılmadığını kontrol et
if not cap.isOpened():
    print("Hata: Video kaynağı açılamadı!")
    exit()

# Haar Cascade sınıflandırıcısını yükleme
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Haar Cascade dosyasının yüklenip yüklenmediğini kontrol et
if haar_cascade.empty():
    print("Hata: 'haar_face.xml' dosyası yüklenemedi! Dosyanın doğru yolda olduğundan emin olun.")
    exit()

print("Video akışı başlıyor. Çıkmak için 'q' tuşuna basın.")

while True:
    # 2. Videodan bir kare (frame) oku
    ret, frame = cap.read() # ret: kare başarıyla okundu mu (True/False), frame: okunan kare

    # Eğer kare okunamazsa (video bittiyse veya hata oluştuysa) döngüyü kır
    if not ret:
        print("Video akışı sonlandı veya kare okunamadı.")
        break

    # 3. Kareyi gri tonlamaya dönüştür
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 4. Yüzleri algıla
    # scaleFactor ve minNeighbors değerlerini videonuzun kalitesine göre ayarlamanız gerekebilir.
    # minNeighbors'ı biraz yükseltmek (örn. 3 veya 4) yanlış pozitifleri azaltabilir.
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    #print(f'Number of faces found in current frame = {len(faces_rect)}') # Her karede yazdırmak çok fazla olabilir

    # 5. Algılanan yüzlerin etrafına dikdörtgen çiz
    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    # 6. İşlenmiş kareyi göster
    cv.imshow('Video - Detected Faces', frame)

    # 7. Çıkış koşulu: 'q' tuşuna basıldığında döngüden çık
    # cv.waitKey(1) her kare için 1 milisaniye bekler.
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Kaynakları serbest bırak ve pencereleri kapat
cap.release() # Video yakalama nesnesini serbest bırak
cv.destroyAllWindows() # Tüm açık OpenCV pencerelerini kapat
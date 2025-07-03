import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img =cv.imread('Photos/park.jpg')
cv.imshow('Original Image',img)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

#laplacian 
lap=cv.Laplacian(gray,cv.CV_64F)
lap=np.uint8(np.absolute(lap))
cv.imshow('Laplacian ',lap)

#sobel
sobelx=cv.Sobel(gray,cv.CV_64F,1,0)
sobely=cv.Sobel(gray,cv.CV_64F,0,1)
combined_sobel= cv.bitwise_or(sobelx,sobely)

cv.imshow('Sobel X',sobelx)
cv.imshow('Sobel Y',sobely)
cv.imshow('Combined',combined_sobel)

#canny edge detection 
#daha temiz 
canny =cv.Canny(gray,150,175)
cv.imshow('Canny',canny)


cv.waitKey(0)

#Kenar Algılama Algoritmaları ve Farkları:
# Laplacian Operatörü (cv.Laplacian)

# Anahtar Noktalar:

# İkinci Dereceden Türev: Görüntünün ikinci dereceden türevini alarak kenarları bulur.
#  Bu, yoğunluktaki ani değişikliklerin (kenarların) sıfır geçiş noktalarını tespit etmesi anlamına gelir.

# Gürültüye Duyarlılık: İkinci dereceden türev kullandığı için gürültüye karşı oldukça hassastır.
#  Gürültülü görüntülerde çok fazla "sahte" kenar üretebilir.

# Kenar Yönü Bilgisi Yok: Kenarın yönü hakkında bilgi vermez (yani, kenar dikey mi, yatay mı, çapraz mı).

# Tek Kanal Çıkışı: Genellikle tek bir çıktı görüntüsü verir ve bu çıktı piksel yoğunluğundaki değişimleri gösterir.

# Sektörde Kullanım Alanları:

# Odaklama Tespiti (Focus Detection): Görüntünün ne kadar net olduğunu değerlendirmek için kullanılır.
#  Daha keskin kenarlar, daha iyi odaklanma anlamına gelir.

# Görüntü İyileştirme/Keskinleştirme: Bir görüntüyü daha keskin hale getirmek için Laplacian filtresiyle
#  elde edilen kenarlar orijinal görüntüye eklenebilir.

# Dokular: Belirli doku özelliklerini vurgulamak için kullanılabilir.

# Sobel Operatörü (cv.Sobel)

# Anahtar Noktalar:

# Birinci Dereceden Türev: Görüntünün birinci dereceden türevini (gradyanını) hesaplayarak kenarları tespit eder.
#  Gradyanın büyüklüğü, kenarın "gücünü", yönü ise kenarın oryantasyonunu gösterir.

# Yön Bilgisi: Kenarın yatay (sobelx) veya dikey (sobely) bileşenlerini ayrı ayrı hesaplayabilir. 
# Bu sayede kenarın yönü hakkında bilgi elde edilebilir.

# Gürültüye Orta Hassasiyet: Laplacian'a göre gürültüye daha az hassastır ancak yine de gürültüden etkilenebilir.

# Daha Kalın Kenarlar: Genellikle Canny'ye göre daha kalın veya çift kenarlar üretebilir.

# Sektörde Kullanım Alanları:

# Yüzey Normal Tahmini: 3D yeniden yapılandırma veya bilgisayar grafikleri gibi alanlarda yüzeylerin normal
#  (dik) vektörlerini tahmin etmek için gradyan bilgisi kullanılabilir.

# Dokuların Analizi: Belirli yönlerdeki dokuları veya desenleri algılamak için faydalıdır.

# Nesne Takibi (Basit Durumlarda): Bir nesnenin kenarlarının hareketini takip etmek için gradyan bilgisi kullanılabilir.

# OCR (Optik Karakter Tanıma): Karakterlerin kenarlarını vurgulamak için ön işleme adımı olarak kullanılabilir.

# Canny Kenar Algılayıcı (cv.Canny)

# Anahtar Noktalar:

# Çok Aşamalı Algoritma: En gelişmiş ve en çok kullanılan kenar algılama algoritmasıdır. Bir dizi adımdan oluşur:

# Gürültü Azaltma (Gaussian Bulanıklığı): Görüntüdeki gürültüyü azaltır.

# Gradyan Hesaplama (Sobel benzeri): Görüntünün gradyanını (büyüklük ve yön) hesaplar.

# Yerel Maksimum Olmayan Baskılama (Non-Maximum Suppression): Kenarları inceltir, sadece gradyan yönünde en güçlü olan pikseli korur.

# Histerezis Eşikleme (Hysteresis Thresholding): İki eşik (alt ve üst) kullanarak güçlü ve zayıf kenarları ayırt eder.
#  Güçlü kenarlar kesin olarak korunur, zayıf kenarlar ise sadece güçlü bir kenara bağlılarsa korunur.

# Temiz ve İnce Kenarlar: Gürültüyü iyi bir şekilde bastırır ve tek piksellik ince kenarlar üretir.

# En İyi Performans: Çoğu genel amaçlı kenar algılama görevi için Laplacian ve Sobel'den daha iyi ve daha doğru sonuçlar verir.

# Parametre Ayarı: Eşik değerleri (threshold1, threshold2) doğru şekilde ayarlanmalıdır.

# Sektörde En Çok Kullanılan Alanlar:

# Nesne Tanıma ve Takip: Nesnelerin sınırlarını belirlemede temel bir adımdır. Robotik, otonom araçlar 
# ve güvenlik sistemlerinde yaygın olarak kullanılır.

# Görüntü Segmentasyonu: Bir görüntüyü anlamlı bölgelere ayırmak için kenar bilgisi kullanılır.

# Kalite Kontrol ve Muayene: Üretim hatlarında ürünlerin kenar kusurlarını veya şekil bozukluklarını
#  tespit etmek için kullanılır.

# Tıbbi Görüntüleme: Organ veya tümör sınırlarını belirlemede yardımcı olabilir.

# Artırılmış Gerçeklik (Augmented Reality): Sanal nesneleri gerçek dünyaya doğru bir şekilde yerleştirmek
#  için çevrenin kenarlarını algılamak.

# Bilgisayarlı Görme Uygulamaları: Drone navigasyonundan, insansız hava araçlarının iniş sistemlerine kadar 
# geniş bir yelpazede kenar bilgisi kritik öneme sahiptir.

# Özetle:

# Laplacian: Gürültüye çok hassas, ikinci dereceden türev, odaklama ve keskinleştirme için.

# Sobel: Kenar yönünü yakalayabilen, birinci dereceden türev, doku analizi ve basit nesne takibi için.

# Canny: En yaygın kullanılan, gürültüye dayanıklı, ince ve temiz kenarlar üreten, nesne tanıma, takip,
# segmentasyon ve kalite kontrol gibi birçok ileri düzey bilgisayar görüşü uygulamasının temelini oluşturan 
# daha temiz ve gelişmiş bir algoritmadır.
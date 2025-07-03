import cv2 as cv
img=cv.imread('data/black_cat.jpg')
cv.imshow('Cat',img) #to show img  okunan görüntüyü ekranda göstermek için kullanılır.
#read videoos 
capture =cv.VideoCapture('video/dog.mp4') #bir video dosyasını okumak için bir video yakalama nesnesi (VideoCapture) oluşturur.
while True:
    isTrue,frame=capture.read()
    cv.imshow('Video',frame)
    if cv.waitkey(20) & 0xFF==ord('d'): #cv.waitKey(20): Klavye tuşuna basılmasını 20 milisaniye bekler. Eğer bu süre içinde bir 
        #tuşa basılırsa, o tuşun ASCII kodunu döndürür; basılmazsa 0 döndürür. Bu 20 ms'lik gecikme, videonun saniyedeki kare sayısını
        #  (FPS) yaklaşık olarak kontrol eder. Daha düşük bir değer daha hızlı oynatma, daha yüksek bir değer daha yavaş oynatma demektir
        #& 0xFF: cv.waitKey()'den dönen değeri sadece son 8 bitini alarak platformlar arası uyumluluğu sağlar.
        #& 0xFF: cv.waitKey()'den dönen değeri sadece son 8 bitini alarak platformlar arası uyumluluğu sağlar.
        break 
capture.release() #, video akışını serbest bırakır ve kullanılan kaynakları (örneğin kamera veya video dosyası) kapatır. Bu, iyi bir programlama pratiğidir.
cv.destroyWindows() #, OpenCV tarafından oluşturulmuş tüm pencereleri kapatır.
cv.waitKey(0) #to wait img 

#daha iyi bir kod 
# ... (önceki kod)

capture = cv.VideoCapture('video/dog.mp4')

while True:
    isTrue, frame = capture.read()

    # Eğer kare başarıyla okunamazsa (video bittiyse veya hata varsa) döngüden çık
    if not isTrue:
        print('Video sona erdi veya okuma hatası oluştu.')
        break # Döngüden çık

    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows() # Tüm pencereleri kapatmak için cv.destroyWindows() yerine bunu kullanmak daha yaygın
cv.waitKey(0) # Görüntü penceresinin açık kalmasını sağlar
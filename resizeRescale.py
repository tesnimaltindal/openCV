
#image ,video,live video
import cv2 as cv
img=cv.imread('Photos/cat.jpg')
cv.imshow('Image',img)
def rescaleframe(frame, scale=0.75):
    # 1. Orjinal çerçevenin genişliğini (sütun sayısı) alır.
    #    frame.shape, (height, width, channels) şeklinde bir tuple döndürür.
    #    frame.shape[1] genişliği temsil eder.
    width = int(frame.shape[1] * scale)

    # 2. Orjinal çerçevenin yüksekliğini (satır sayısı) alır.
    #    frame.shape[0] yüksekliği temsil eder.
    height = int(frame.shape[0] * scale)

    # 3. Yeni genişlik ve yükseklikten oluşan bir demet (tuple) oluşturur.
    #    Bu demet, OpenCV'nin resize fonksiyonu için gerekli olan hedef boyutları temsil eder.
    dimensions = (width, height)

    # 4. cv.resize fonksiyonunu kullanarak çerçeveyi yeniden boyutlandırır.
    #    - frame: Yeniden boyutlandırılacak orijinal çerçeve.
    #    - dimensions: Hedef boyutlar (genişlik, yükseklik).
    #    - interpolation: Yeniden boyutlandırma sırasında kullanılacak enterpolasyon (interpolation) yöntemi.
    #      cv.INTER_AREA, küçültme (shrinking) için genellikle tercih edilen bir yöntemdir
    #      çünkü piksel alanlarını hesaplayarak daha iyi sonuçlar verir ve görsel kalitenin korunmasına yardımcı olur.
    #      Büyütme (enlarging) için cv.INTER_LINEAR veya cv.INTER_CUBIC gibi yöntemler daha uygun olabilir.
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
def changeRes(width,height):#just for live video 
    capture.set(3,width)
    capture.set(4,height)

resized_image=rescaleframe(img)
cv.imshow('ResizdedImage',resized_image)
# Read video
# Use a raw string (r'...') for paths on Windows or forward slashes to avoid issues with backslashes
video_path = 'Videos/dog.mp4' # Or the absolute path: r'C:\Users\YourUser\Desktop\Videos\dog.mp4'

capture = cv.VideoCapture(video_path)

if not capture.isOpened():
    print(f"Hata: Video dosyası açılamadı: {video_path}")
    print("Dosya yolunu kontrol edin veya dosyanın bozuk olmadığından emin olun.")
    exit()

while True:
    isTrue, frame = capture.read()

    # If isTrue is False, it means there are no more frames (end of video) or an error occurred.
    if not isTrue:
        print("Video sona erdi veya kare okunamadı.")
        break

    frame_resized = rescaleframe(frame,scale=.2)

    cv.imshow('Original Video', frame)
    cv.imshow('Resized Video', frame_resized)

    # cv.waitKey(20) waits for 20 milliseconds.
    # '& 0xFF' is used to get the last 8 bits of the key code, ensuring cross-platform compatibility.
    # 'ord('d')' gets the ASCII value of the 'd' key.
    if cv.waitKey(20) & 0xFF == ord('d'):
        print("Kullanıcı 'd' tuşuna basarak çıkış yaptı.")
        break

# Release the video capture object and destroy all windows
capture.release()
cv.destroyAllWindows()

# The last cv.waitKey(0) is not necessary after destroyAllWindows()
# You had two, one is definitely not needed.
cv.waitKey(0)
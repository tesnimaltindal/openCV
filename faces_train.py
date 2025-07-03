import cv2 as cv
import numpy as np
import os


# 1.Veri Toplama: Belirli kişilere ait resimlerin bulunduğu klasörlerden yüzleri tespit eder.

# 2.Özellik Çıkarımı: Tespit edilen yüzlerin gri tonlamalı piksellerini (yani yüz bölgesini) birer "özellik" olarak kaydeder.

# 3.Etiketleme: Her yüzün hangi kişiye ait olduğunu belirten bir "etiket" (sayısal ID) atar.

#4. Eğitim: Toplanan yüz özellikleri ve bunlara karşılık gelen etiketler kullanılarak bir yüz tanıma modeli (LBPHFaceRecognizer) eğitilir.

# 5.Modeli Kaydetme: Eğitilmiş modeli ve eğitim verilerini gelecekte kullanmak üzere kaydeder.


people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling'] # 'Modanna' -> 'Madonna' düzeltildi.

# Dizini raw string olarak belirtmek iyi bir uygulama
DIR = r'C:\Users\Tesnim\Downloads\opencv-course-master\opencv-course-master\Resources\Faces\train'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Haar Cascade dosyasının yüklenip yüklenmediğini kontrol et
if haar_cascade.empty():
    print("Hata: 'haar_face.xml' dosyası yüklenemedi! Dosyanın doğru yolda olduğundan emin olun.")
    exit() # Dosya yoksa programı sonlandır

features = [] # Tespit edilen yüzlerin piksel verilerini (gri tonlamalı ROI'leri) depolayacak boş liste. Bu listeye her yüzün görüntüsü eklenir.
labels = [] #Her bir yüzün hangi kişiye ait olduğunu belirten sayısal etiketleri (ID'leri) depolayacak boş liste. Örneğin, 'Ben Afflek' 0, 'Elton John' 1 vb. gibi etiketlenecek.

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person) 

        # Klasörün varlığını kontrol et
        if not os.path.exists(path):
            print(f"Uyarı: '{path}' dizini bulunamadı. Bu kişiye ait resimler işlenmeyecek.")
            continue # Bir sonraki kişiye geç

        for img_name in os.listdir(path): 
            img_path = os.path.join(path, img_name) 

            img_array = cv.imread(img_path)

            # Resmin başarıyla yüklenip yüklenmediğini kontrol et
            if img_array is None:
                print(f"Uyarı: '{img_path}' resmi yüklenemedi. Atlanıyor.")
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY) 
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            #minNeighbors=4: Bir yüz adayının kaç komşu dikdörtgen tarafından doğrulanması gerektiğini belirtir.
            #  Bu, yanlış pozitifleri (yüz olmayan bir şeyin yüz olarak algılanması) azaltmaya yardımcı olur.
            #scaleFactor=1.1: Her ölçek küçültme adımında görüntünün %10 oranında küçültülmesini sağlar.
            #  Bu, farklı boyutlardaki yüzleri bulmak için önemlidir.
            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w] # Region of Interest (İlgi Bölgesi)
                features.append(faces_roi) # Kesilen yüz ROI'sini features listesine ekler. Bu, modelin "gördüğü" yüz görüntüleridir.
                labels.append(label) #labels.append(label): Kesilen yüzün ait olduğu kişinin etiketini labels listesine ekler.

        #faces_roi = gray[y:y+h, x:x+w]: Algılanan yüzün ilgi bölgesini (Region of Interest - ROI) gri tonlamalı resimden keser.
        #  Bu, sadece yüzün kendisinin piksel verilerini alır.

# Fonksiyonu çağır
create_train()
print('Training Done :)')


# print(f'Length of the features = {len(features)}')
# print(f'Length of the labels = {len(labels)}')

features=np.array(features,dtype='object')
labels=np.array(labels)

# Eğitimi burada yap, örnek olarak bir tanıyıcı tanımla
face_recognizer = cv.face.LBPHFaceRecognizer_create()
#face_recognizer = cv.face.LBPHFaceRecognizer_create(): Local Binary
#  Patterns Histograms (LBPH) Yüz Tanıyıcı nesnesini oluşturur.
#  Bu, OpenCV'de yaygın olarak kullanılan bir yüz tanıma algoritmasıdır. 
#Basit ve etkili bir algoritmadır.



# #train the recognizer on the feature list and labels list 
# face_recognizer.train(features, np.array(labels))
# face_recognizer.save('face_trained.yml') # Modeli kaydet

face_recognizer.train(features,labels)
face_recognizer.save('face_trained.yml')

#face_recognizer.train(features, labels): Oluşturulan face_recognizer modelini, toplanan features (yüz görüntüleri) ve labels (kimlik etiketleri) ile eğitir.
#  Bu adımda, tanıyıcı her bir yüzün özelliklerini öğrenir ve hangi etiketle eşleştiğini öğrenir.

#face_recognizer.save('face_trained.yml'): Eğitilmiş yüz tanıma modelini bir YAML dosyasına (face_trained.yml) kaydeder. Bu sayede modeli her seferinde 
# yeniden eğitmek yerine doğrudan yükleyip kullanabilirsiniz.

np.save('features.npy',features)
np.save('labels.npy',labels)

#Bir LBPHFaceRecognizer modeli oluşturulur.
#Bu model, toplanan yüz özellikleri (features) ve bunların etiketleri (labels) kullanılarak eğitilir. Eğitim sırasında, model her bir kişiye ait yüz desenlerini öğrenir.

#os modülü olmadan, kodunuzun diske erişerek eğitim resimlerini bulması ve işlemesi mümkün olmazdı.

#Bir yüz ROI'sindeki tüm hücrelerden elde edilen LBP histogramları birleştirilerek, o yüze özgü,
#  yüksek boyutlu bir özellik vektörü oluşturulur. Bu vektör, o yüzün "parmak izi" gibidir.


### Eğitim Aşaması (Sizin `create_train()` fonksiyonunuz ve `.train()` metodu):

# 1.  **Veri Toplama ve Ön İşleme:**
#     * Siz her bir kişinin (Ben Afflek, Elton John vb.) birden fazla yüz resmini `train` klasörüne yerleştirdiniz.
#     * Kod, bu resimleri okur, gri tonlamaya çevirir ve Haar Cascade kullanarak sadece yüz bölgelerini (ROI - Region of Interest) kesip çıkarır. Bu kesilen yüzler `features` listesine eklenir.
#     * Aynı zamanda, her kesilen yüz için ait olduğu kişinin bir sayısal etiketi (`people.index(person)`) `labels` listesine eklenir. Örneğin, 'Ben Afflek' 0, 'Elton John' 1 gibi.

# 2.  **LBP (Local Binary Patterns) Özellik Çıkarımı:**
#     * `LBPHFaceRecognizer`'ın temelini oluşturan **LBP** algoritması devreye girer. Bu algoritma, bir pikselin komşularına göre ne kadar parlak veya karanlık olduğunu ikili bir desenle kodlar.
#     * Her bir yüz ROI'si (yani `features` listesindeki her bir yüz görüntüsü), küçük hücrelere (örneğin 3x3 piksellik bölgelere) bölünür.
#     * Her hücredeki pikseller için LBP desenleri çıkarılır. Bu desenler, piksel yoğunluğundaki yerel yapısal değişiklikleri yakalar (köşeler, kenarlar, düz alanlar vb.).
#     * Her hücre için bu LBP desenlerinin bir histogramı oluşturulur. Bu histogram, o hücredeki desenlerin frekansını gösterir.

# 3.  **Histogramların Birleştirilmesi:**
#     * Bir yüz ROI'sindeki tüm hücrelerden elde edilen LBP histogramları birleştirilerek, o yüze özgü, yüksek boyutlu bir özellik vektörü oluşturulur. Bu vektör, o yüzün "parmak izi" gibidir.

# 4.  **Eğitim (Sınıflandırma Modelinin Oluşturulması):**
#     * `face_recognizer.train(features, labels)` metodu çağrıldığında, LBPHFaceRecognizer, her bir kişinin etiketine karşılık gelen bu birleştirilmiş LBP histogram vektörlerini öğrenir.
#     * Model, aynı etiketle (aynı kişiye ait) gelen yüzlerin LBP histogramları arasında ortak desenler ve benzerlikler bulmaya çalışır. Temel olarak, her bir etiket için bu histogram vektörlerini temsil eden bir "referans model" veya "prototip" oluşturur.

# ### Tanıma Aşaması (Model kullanıldığında):

# 1.  **Yeni Yüzün Tespiti ve Özellik Çıkarımı:**
#     * Yeni bir görüntü veya video karesi geldiğinde, yine Haar Cascade ile yüzler tespit edilir ve gri tonlamalı yüz ROI'leri kesilir.
#     * Bu yeni yüz ROI'si için de eğitim aşamasındaki gibi LBP özellik çıkarımı yapılır ve bir LBP histogram vektörü oluşturulur.

# 2.  **Karşılaştırma (Mesafe Hesaplama):**
#     * Yeni yüzün LBP histogram vektörü, eğitim aşamasında oluşturulan referans modellerle (her bir etiket/kişi için) karşılaştırılır.
#     * Bu karşılaştırma genellikle bir **uzaklık metriği** (örneğin Öklid Mesafesi veya Chi-Square Mesafesi) kullanılarak yapılır. Bu mesafe, yeni yüzün histogramı ile eğitilmiş referans modeller arasındaki "farklılığı" ölçer.

# 3.  **En Yakın Eşleşme:**
#     * Model, yeni yüzün LBP histogramına en kısa mesafede olan referans modeli bulur.
#     * En kısa mesafenin olduğu referans modele karşılık gelen etiket (ID) tahmin edilen kişi olur.

# 4.  **Güven Eşiği (Confidence):**
#     * Model ayrıca bir "güven" değeri de döndürebilir (bazı tanıyıcılarda). Bu değer, tahminin ne kadar güvenilir olduğunu gösterir (mesafe ne kadar küçükse güven o kadar yüksek olur). Belirli bir eşiğin üzerindeki tahminler kabul edilirken, altındaki tahminler "bilinmeyen" olarak kabul edilebilir.

# Özetle, LBPHFaceRecognizer doğrudan yüzlerin kendilerini değil, yüzlerin içindeki **yerel doku ve desenleri temsil eden histogramları** öğrenir. Daha sonra yeni bir yüz geldiğinde, onun desenlerini çıkarır ve eğitimde gördüğü desenlerle en çok eşleşeni bulmaya çalışır.
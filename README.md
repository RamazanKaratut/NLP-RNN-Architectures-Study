# NLP & RNN Architectures Study: Sentiment Analysis and Character-Level Language Modeling

Bu depo, Doğal Dil İşleme (NLP) ve Tekrarlayan Sinir Ağları (RNN) mimarilerinin yeteneklerini araştıran iki temel projeyi içermektedir. Projelerde LSTM ve GRU hücrelerinin performansları karşılaştırılmış ve karakter seviyesinde metin üretimi gerçekleştirilmiştir.

## 🛠️ Kullanılan Teknolojiler
* **Python 3.x**
* **TensorFlow / Keras** (Model inşası ve eğitimi)
* **NumPy** (Veri manipülasyonu)
* **Matplotlib** (Performans görselleştirmesi)

---

## 📌 Proje 1: IMDB Sentiment Analysis (Duygu Analizi)

Bu projenin amacı, IMDB film eleştirileri veri setini kullanarak metinlerin olumlu (positive) veya olumsuz (negative) olduğunu tahmin eden ikili sınıflandırma (binary classification) modelleri geliştirmektir. 

10 epoch süren eğitim sonucunda modellerin unseen (görülmemiş) test verisi üzerindeki nihai doğruluk oranları şu şekildedir:
* **Uni-LSTM:** %82.72
* **Uni-GRU:** %84.71
* **Bi-LSTM:** %84.98
* **Bi-GRU:** %85.06 🏆

### 📊 Model Performansları ve Karşılaştırma

Aşağıdaki grafiklerde, 4 farklı modelin eğitim süreci boyunca gösterdiği Doğrulama Başarısı (Validation Accuracy) ve Doğrulama Kaybı (Validation Loss) oranlarını görebilirsiniz.

![Modellerin Doğruluk Karşılaştırması](images/accuracy_plot.png)
*Şekil 1: Modellerin epoch bazlı doğruluk (accuracy) karşılaştırması.*

![Modellerin Kayıp Karşılaştırması](images/loss_plot.png)
*Şekil 2: Modellerin epoch bazlı kayıp (loss) karşılaştırması.*

**Proje Çıktıları ve Analiz:** * Çift yönlü (Bidirectional) modeller bağlamı daha iyi yakalayarak **%85** başarı hedefini aşmış ve en yüksek skoru (Bi-GRU) elde etmiştir.
* GRU hücreleri, bu spesifik veri setinde daha az işlem maliyetiyle LSTM'den daha yüksek doğruluk oranlarına ulaşmıştır.
* Eğitim ve Doğrulama (Validation) eğrileri incelendiğinde, modellerin 3. epoch'tan sonra *overfitting* (ezberleme) eğilimine girdiği gözlemlenmiştir. Bu durum, gelecekteki geliştirmelerde `EarlyStopping` veya daha yüksek `Dropout` oranları kullanılarak optimize edilebilir.

---

## 📌 Proje 2: Character-Level Language Model (Karakter Seviyeli Dil Modeli)

Bu projede Karpathy'nin **Tiny Shakespeare** veri seti kullanılarak karakter seviyesinde bir LSTM dil modeli (Char-RNN) eğitilmiştir. Model kelimeleri değil, harfleri öğrenerek bir sonraki karakteri tahmin etmeye çalışır.



### 🧠 Mimari
* **1x Embedding Katmanı:** Karakterlerin yoğun vektör temsili için.
* **1x LSTM Katmanı:** 512 birim, stateful yapıda. Zemberek (sequence) takibi için.
* **1x Dense Katmanı:** Vocabulary (Kelime/Karakter dağarcığı) boyutu kadar çıktı üretir.

### 🎭 Modelden Örnek Çıktılar

Model eğitildikten sonra, sisteme `"ROMEO: "` başlangıç metni (seed) verilerek aşağıdaki metin üretilmiştir:

> **ROMEO:** I shall be to the part in the bear.
> **JULIET:** What is the word in the sight of my father?
> **KING RICHARD III:** He is the man that is so straight for him,
> And leave the state of thy master.

*(Model, Shakespeare'in tiyatro formatını —büyük harfli isimler, alt satıra geçme ve noktalama işaretleri— başarıyla öğrenmiş ve İngilizce kelime yapılarını taklit etmeye başlamıştır.)*

---

## 🚀 Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için:

1. Depoyu klonlayın:
   ```bash
   git clone [https://github.com/KULLANICI_ADINIZ/NLP-RNN-Architectures-Study.git](https://github.com/KULLANICI_ADINIZ/NLP-RNN-Architectures-Study.git)
   cd NLP-RNN-Architectures-Study
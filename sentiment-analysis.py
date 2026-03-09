import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Bidirectional, Dropout, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# Görsellerin kaydedileceği klasörü oluştur
os.makedirs('images', exist_ok=True)

# ==========================================
# 1. HİPERPARAMETRELER VE VERİ HAZIRLIĞI
# ==========================================
max_features = 10000  
maxlen = 200          
batch_size = 64
epochs = 15 # Early stopping eklediğimiz için yüksek tutabiliriz, nasılsa kendi duracak 
embedding_dim = 128

print("IMDB Veri seti yükleniyor...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print("Diziler aynı uzunluğa getiriliyor (Padding)...")
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# ==========================================
# 2. MODEL OLUŞTURMA FONKSİYONU (GÜNCELLENDİ)
# ==========================================
def build_model(model_type='lstm', bidirectional=False):
    model = Sequential()
    
    # Yeni Keras standardına uygun Input katmanı
    model.add(tf.keras.Input(shape=(maxlen,)))
    
    model.add(Embedding(max_features, embedding_dim))
    # YENİ: Embedding katmanında rastgele özellikleri kapatarak ezberlemeyi önle
    model.add(SpatialDropout1D(0.2)) 
    
    # YENİ: RNN katmanlarına dropout eklendi
    if model_type == 'lstm':
        rnn_layer = LSTM(64, dropout=0.2) 
    else:
        rnn_layer = GRU(64, dropout=0.2)
        
    if bidirectional:
        model.add(Bidirectional(rnn_layer))
    else:
        model.add(rnn_layer)
        
    # YENİ: Çıktıdan hemen önce son bir Dropout katmanı
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model

scenarios = [
    {'type': 'lstm', 'bi': False, 'name': 'Uni-LSTM'},
    {'type': 'gru',  'bi': False, 'name': 'Uni-GRU'},
    {'type': 'lstm', 'bi': True,  'name': 'Bi-LSTM'},
    {'type': 'gru',  'bi': True,  'name': 'Bi-GRU'}
]

histories = {}
test_results = {}

# YENİ: Early Stopping Ayarı
# val_accuracy 3 adım boyunca artmazsa dur ve en iyi modeli geri yükle
early_stop = EarlyStopping(monitor='val_accuracy', 
                           patience=3, 
                           restore_best_weights=True,
                           verbose=1)

# ==========================================
# 3. EĞİTİM VE TEST DÖNGÜSÜ
# ==========================================
for s in scenarios:
    name = s['name']
    print(f"\n{'-'*40}")
    print(f"Eğitilen Model: {name}")
    print(f"{'-'*40}")
    
    model = build_model(model_type=s['type'], bidirectional=s['bi'])
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2, 
                        callbacks=[early_stop], # Callback eklendi
                        verbose=1)
    
    histories[name] = history.history
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    test_results[name] = acc
    print(f"-> {name} Nihai Test Doğruluğu: %{acc*100:.2f}")

# ==========================================
# 4. SONUÇLARIN GÖRSELLEŞTİRİLMESİ VE KAYDEDİLMESİ
# ==========================================
plt.figure(figsize=(14, 6))

# 1. Grafik: Accuracy (Doğruluk)
plt.subplot(1, 2, 1)
for name, hist in histories.items():
    label_name = f"{name} (Test: %{test_results[name]*100:.1f})"
    plt.plot(hist['val_accuracy'], label=label_name, marker='o')
plt.title('Modellerin Doğrulama (Validation) Başarısı')
plt.ylabel('Accuracy (Doğruluk)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 2. Grafik: Loss (Kayıp)
plt.subplot(1, 2, 2)
for name, hist in histories.items():
    plt.plot(hist['val_loss'], label=name, marker='s')
plt.title('Modellerin Doğrulama (Validation) Kaybı')
plt.ylabel('Loss (Kayıp/Hata)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('images/accuracy_loss_combined.png')
print("\nBirleştirilmiş grafik 'images/accuracy_loss_combined.png' olarak kaydedildi.")
plt.show()
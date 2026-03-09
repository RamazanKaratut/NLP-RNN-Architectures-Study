import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Bidirectional
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
epochs = 3            
embedding_dim = 128

print("IMDB Veri seti yükleniyor...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print("Diziler aynı uzunluğa getiriliyor (Padding)...")
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# ==========================================
# 2. MODEL OLUŞTURMA FONKSİYONU
# ==========================================
def build_model(model_type='lstm', bidirectional=False):
    model = Sequential()
    model.add(Embedding(max_features, embedding_dim, input_length=maxlen))
    
    if model_type == 'lstm':
        rnn_layer = LSTM(64)
    else:
        rnn_layer = GRU(64)
        
    if bidirectional:
        model.add(Bidirectional(rnn_layer))
    else:
        model.add(rnn_layer)
        
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
                        verbose=1)
    
    histories[name] = history.history
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    test_results[name] = acc
    print(f"-> {name} Nihai Test Doğruluğu: %{acc*100:.2f}")

# ==========================================
# 4. SONUÇLARIN GÖRSELLEŞTİRİLMESİ VE KAYDEDİLMESİ
# ==========================================
# 1. Grafik: Accuracy (Doğruluk)
plt.figure(figsize=(8, 6))
for name, hist in histories.items():
    label_name = f"{name} (Test: %{test_results[name]*100:.1f})"
    plt.plot(hist['val_accuracy'], label=label_name, marker='o')
plt.title('Modellerin Doğrulama (Validation) Başarısı')
plt.ylabel('Accuracy (Doğruluk)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('images/accuracy_plot.png') # Görseli kaydet
print("Accuracy grafiği 'images/accuracy_plot.png' olarak kaydedildi.")
plt.close()

# 2. Grafik: Loss (Kayıp)
plt.figure(figsize=(8, 6))
for name, hist in histories.items():
    plt.plot(hist['val_loss'], label=name, marker='s')
plt.title('Modellerin Doğrulama (Validation) Kaybı')
plt.ylabel('Loss (Kayıp/Hata)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('images/loss_plot.png') # Görseli kaydet
print("Loss grafiği 'images/loss_plot.png' olarak kaydedildi.")
plt.close()
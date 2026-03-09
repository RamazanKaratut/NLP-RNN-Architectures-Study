import tensorflow as tf
import numpy as np
import os
import requests
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Görsellerin kaydedileceği klasörü oluştur
os.makedirs('images', exist_ok=True)

# ==========================================
# 1. VERİ SETİNİN İNDİRİLMESİ VE HAZIRLIĞI
# ==========================================
print("Shakespeare veri seti indiriliyor...")
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
response = requests.get(url)
text = response.text

vocab = sorted(set(text))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# YENİ: Early Stopping çalışabilsin diye veriyi Train (%90) ve Validation (%10) olarak ikiye bölüyoruz
split_idx = int(len(text) * 0.9)
train_text = text[:split_idx]
val_text = text[split_idx:]

train_as_int = np.array([char2idx[c] for c in train_text])
val_as_int = np.array([char2idx[c] for c in val_text])

# ==========================================
# 2. EĞİTİM VERİSİNİN OLUŞTURULMASI
# ==========================================
seq_length = 100 
BATCH_SIZE = 64
BUFFER_SIZE = 10000

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def create_dataset(data_array):
    dataset = tf.data.Dataset.from_tensor_slices(data_array)
    dataset = dataset.batch(seq_length + 1, drop_remainder=True)
    dataset = dataset.map(split_input_target)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    return dataset

# Hem eğitim hem doğrulama setlerini oluştur
train_dataset = create_dataset(train_as_int)
val_dataset = create_dataset(val_as_int)

# ==========================================
# 3. LSTM MODELİNİN İNŞASI (DROPOUT EKLENDİ)
# ==========================================
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 512

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.Input(batch_shape=[batch_size, None]),
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Dropout(0.2), # Dropout eklendi
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, 
                             recurrent_initializer='glorot_uniform', dropout=0.2), # LSTM içine Dropout
        tf.keras.layers.Dropout(0.2), # Çıktıdan önce Dropout
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(vocab_size=len(vocab), embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=BATCH_SIZE)

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# ==========================================
# 4. EĞİTİM, EARLY STOPPING VE GRAFİK KAYDETME
# ==========================================
checkpoint_dir = './training_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Sadece tek bir "en iyi" ağırlık dosyasını tutacağız
best_checkpoint_path = os.path.join(checkpoint_dir, "best_char_rnn.weights.h5")

# Early Stopping: val_loss 5 epoch boyunca düşmezse eğitimi durdur ve en iyiyi yükle
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True, 
    verbose=1
)

checkpoint_callback = ModelCheckpoint(
    filepath=best_checkpoint_path, 
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)

EPOCHS = 40 # Early stopping olduğu için limiti yüksek tutuyoruz
print("\nEğitim başlıyor...")
history = model.fit(
    train_dataset, 
    validation_data=val_dataset, # Validation verisi eklendi
    epochs=EPOCHS, 
    callbacks=[checkpoint_callback, early_stop]
)

# Eğitim ve Doğrulama Kaybı (Loss) Grafiğini Çizdir ve Kaydet
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], marker='o', color='purple', label='Eğitim Kaybı (Train Loss)')
plt.plot(history.history['val_loss'], marker='s', color='orange', label='Doğrulama Kaybı (Val Loss)')
plt.title('Karakter Seviyeli Dil Modeli Kayıp Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Kayıp (Loss)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('images/char_rnn_loss.png')
print("\nChar-RNN Loss grafiği 'images/char_rnn_loss.png' olarak kaydedildi.")
plt.close()

# ==========================================
# 5. METİN ÜRETİMİ (Keras 3 Uyumlu)
# ==========================================
model_gen = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model_gen.load_weights(best_checkpoint_path) # En iyi epoch'un ağırlıklarını yüklüyoruz

def generate_text(model, start_string, temperature=1.0):
    num_generate = 500 
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    # FIX: model.reset_states() satırı yeni Keras 3 uyumluluğu için tamamen kaldırıldı.

    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        
        # Temperature ayarı: Daha düşük değerler daha güvenli/tahmin edilebilir, yüksek değerler yaratıcı sonuçlar verir
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

print("\n--- Modelin Ürettiği Metin (En İyi Ağırlıklarla) ---")
print(generate_text(model_gen, start_string=u"ROMEO: ", temperature=0.7))
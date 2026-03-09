import tensorflow as tf
import numpy as np
import os
import requests
import matplotlib.pyplot as plt

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
text_as_int = np.array([char2idx[c] for c in text])

# ==========================================
# 2. EĞİTİM VERİSİNİN OLUŞTURULMASI
# ==========================================
seq_length = 100 
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# ==========================================
# 3. LSTM MODELİNİN İNŞASI
# ==========================================
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 512

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(vocab_size=len(vocab), embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=BATCH_SIZE)

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# ==========================================
# 4. EĞİTİM VE GRAFİK KAYDETME
# ==========================================
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

EPOCHS = 10 
print("Eğitim başlıyor...")
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# Eğitim Kaybı (Loss) Grafiğini Çizdir ve Kaydet
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], marker='o', color='purple', label='Eğitim Kaybı (Training Loss)')
plt.title('Karakter Seviyeli Dil Modeli Eğitim Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp (Loss)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('images/char_rnn_loss.png')
print("Char-RNN Loss grafiği 'images/char_rnn_loss.png' olarak kaydedildi.")
plt.close()

# ==========================================
# 5. METİN ÜRETİMİ
# ==========================================
model_gen = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model_gen.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model_gen.build(tf.TensorShape([1, None]))

def generate_text(model, start_string, temperature=1.0):
    num_generate = 500 
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

print("\n--- Modelin Ürettiği Metin ---")
print(generate_text(model_gen, start_string=u"ROMEO: ", temperature=0.7))
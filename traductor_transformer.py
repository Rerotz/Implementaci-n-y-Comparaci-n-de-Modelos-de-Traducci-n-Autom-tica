"""
Sistema de Traducción Automática con Transformer (Español ↔ Inglés)
Implementación completa: Preparación de datos, Entrenamiento y Evaluación
Arquitectura: Transformer con Auto-Atención y Atención Cruzada
"""

import tensorflow as tf
import numpy as np
import sentencepiece as spm
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import time
import json
from sacrebleu.metrics import BLEU
import os
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print(" SISTEMA DE TRADUCCIÓN AUTOMÁTICA CON TRANSFORMER")
print(" Español ↔ Inglés")
print("="*70)

# =====================================================
# FASE 1: SELECCIÓN Y PREPARACIÓN DE DATOS
# =====================================================

print("\n" + "="*70)
print("FASE 1: PREPARACIÓN DE DATOS")
print("="*70)

# 1.1 Carga del corpus Tatoeba
print("\n[1/6] Cargando dataset Tatoeba (español-inglés)...")
dataset = load_dataset("opus_books", "en-es", split="train")

# Muestreo para entrenamiento eficiente (10 épocas rápidas)
MAX_SAMPLES = 50000
data_subset = dataset.shuffle(seed=42).select(range(min(MAX_SAMPLES, len(dataset))))

# Extraer pares de oraciones
spanish_sentences = [item['translation']['es'] for item in data_subset]
english_sentences = [item['translation']['en'] for item in data_subset]

print(f"✓ Dataset cargado: {len(spanish_sentences):,} pares de oraciones")

# 1.2 Limpieza y normalización
print("\n[2/6] Limpieza y normalización...")

def clean_text(text):
    """Limpia y normaliza el texto"""
    import re
    text = text.lower().strip()
    # Normalizar espacios
    text = re.sub(r'\s+', ' ', text)
    # Mantener letras, números y puntuación básica
    text = re.sub(r'[^a-záéíóúüñ0-9.,;:()¿?¡!"\'@#$%&*\-\s]+', '', text)
    return text

spanish_sentences = [clean_text(s) for s in spanish_sentences]
english_sentences = [clean_text(e) for e in english_sentences]

# Filtrar oraciones vacías o muy largas
MAX_LEN = 80
filtered_data = [
    (es, en) for es, en in zip(spanish_sentences, english_sentences)
    if es and en and len(es.split()) <= MAX_LEN and len(en.split()) <= MAX_LEN
]

spanish_sentences, english_sentences = zip(*filtered_data)
print(f"✓ Oraciones después de limpieza: {len(spanish_sentences):,}")

# 1.3 División del corpus
print("\n[3/6] División del corpus (train/val/test: 90%/5%/5%)...")

total = len(spanish_sentences)
train_size = int(0.9 * total)
val_size = int(0.05 * total)

train_es = list(spanish_sentences[:train_size])
train_en = list(english_sentences[:train_size])
val_es = list(spanish_sentences[train_size:train_size + val_size])
val_en = list(english_sentences[train_size:train_size + val_size])
test_es = list(spanish_sentences[train_size + val_size:])
test_en = list(english_sentences[train_size + val_size:])

print(f"✓ Entrenamiento: {len(train_es):,} pares")
print(f"✓ Validación: {len(val_es):,} pares")
print(f"✓ Prueba: {len(test_es):,} pares")

# 1.4 Tokenización con SentencePiece (BPE)
print("\n[4/6] Entrenando tokenizador SentencePiece (BPE)...")

VOCAB_SIZE = 16000
SP_PREFIX = "transformer_spen"

# Guardar datos para entrenar tokenizador
with open('train_corpus.txt', 'w', encoding='utf-8') as f:
    for es, en in zip(train_es, train_en):
        f.write(f"{es}\n{en}\n")

# Entrenar modelo SentencePiece
spm.SentencePieceTrainer.train(
    input='train_corpus.txt',
    model_prefix=SP_PREFIX,
    vocab_size=VOCAB_SIZE,
    character_coverage=0.9995,
    model_type='bpe',
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

# Cargar tokenizador
sp = spm.SentencePieceProcessor(model_file=f'{SP_PREFIX}.model')

# Función auxiliar para convertir ID a texto
def id_to_piece(token_id):
    """Convierte un ID de token a su representación de texto"""
    try:
        return sp.id_to_piece(int(token_id))
    except:
        return f"[{token_id}]"

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

print(f"✓ Tokenizador entrenado: {VOCAB_SIZE:,} tokens")

# 1.5 Codificación y creación de datasets
print("\n[5/6] Codificando secuencias y creando datasets...")

def encode_sentence(sent, max_len=60):
    """Codifica una oración añadiendo BOS y EOS"""
    tokens = [BOS_ID] + sp.encode(sent, out_type=int) + [EOS_ID]
    if len(tokens) > max_len:
        tokens = tokens[:max_len-1] + [EOS_ID]
    return tokens

def create_dataset(src_sentences, tgt_sentences, batch_size=32):
    """Crea un tf.data.Dataset optimizado"""
    src_encoded = [encode_sentence(s) for s in src_sentences]
    tgt_encoded = [encode_sentence(t) for t in tgt_sentences]
    
    def generator():
        for src, tgt in zip(src_encoded, tgt_encoded):
            yield (src, tgt[:-1], tgt[1:])  # input, decoder_input, decoder_output
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )
    
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([None], [None], [None]),
        padding_values=(PAD_ID, PAD_ID, PAD_ID)
    )
    
    return dataset.prefetch(tf.data.AUTOTUNE)

BATCH_SIZE = 32

# Datasets bidireccionales
train_es_en = create_dataset(train_es, train_en, BATCH_SIZE)
val_es_en = create_dataset(val_es, val_en, BATCH_SIZE)
test_es_en = create_dataset(test_es, test_en, BATCH_SIZE)

train_en_es = create_dataset(train_en, train_es, BATCH_SIZE)
val_en_es = create_dataset(val_en, val_es, BATCH_SIZE)
test_en_es = create_dataset(test_en, test_es, BATCH_SIZE)

print("✓ Datasets creados y optimizados")

# 1.6 Estadísticas del corpus
print("\n[6/6] Estadísticas del corpus:")
avg_len_es = np.mean([len(s.split()) for s in train_es])
avg_len_en = np.mean([len(e.split()) for e in train_en])
print(f"  • Longitud media (ES): {avg_len_es:.1f} palabras")
print(f"  • Longitud media (EN): {avg_len_en:.1f} palabras")
print(f"  • Vocabulario: {VOCAB_SIZE:,} subpalabras (BPE)")

# =====================================================
# FASE 2: ARQUITECTURA TRANSFORMER
# =====================================================

print("\n" + "="*70)
print("FASE 2: IMPLEMENTACIÓN DEL TRANSFORMER")
print("="*70)

# Hiperparámetros del Transformer
D_MODEL = 256          # Dimensión del modelo
NUM_HEADS = 8          # Número de cabezas de atención
DFF = 1024             # Dimensión de la FFN
NUM_LAYERS = 4         # Capas del encoder/decoder
DROPOUT_RATE = 0.1
MAX_SEQ_LEN = 60

print(f"\nHiperparámetros:")
print(f"  • D_MODEL: {D_MODEL}")
print(f"  • NUM_HEADS: {NUM_HEADS}")
print(f"  • DFF: {DFF}")
print(f"  • NUM_LAYERS: {NUM_LAYERS}")
print(f"  • DROPOUT: {DROPOUT_RATE}")

# 2.1 Positional Encoding
def get_positional_encoding(seq_len, d_model):
    """Calcula el positional encoding"""
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, v, mask):
    """
    Calcula la atención mediante los pasos descritos en el paper:
    Attention(Q, K, V) = softmax(QK^T / sqrt(dk)) V
    """
    # Q * K^T
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_q, seq_k)

    # Escalado por raíz de la dimensión del key
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Aplicar máscara si existe
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Softmax sobre los logits
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Multiplicar por V
    output = tf.matmul(attention_weights, v)  # (..., seq_q, depth_v)

    return output, attention_weights

# 2.2 Multi-Head Attention
class MultiHeadAttention(tf.keras.layers.Layer):
    def _init_(self, d_model, num_heads):
        super(MultiHeadAttention, self)._init_()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, heads, seq, depth)

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weights

# 2.3 Feed Forward Network
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

def positional_encoding(position, d_model):
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )

    # Apply sin to even indices (2i)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # Apply cos to odd indices (2i+1)
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# 2.4 Encoder Layer
class EncoderLayer(tf.keras.layers.Layer):
    def _init_(self, d_model, num_heads, dff, rate=0.1):
        super()._init_()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training=False, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)   # self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

# 2.5 Decoder Layer (CORREGIDO)
class DecoderLayer(tf.keras.layers.Layer):
    def _init_(self, d_model, num_heads, dff, rate=0.1):
        super()._init_()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        # Self-attention con look-ahead mask
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # Cross-attention con encoder output
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        # Feed forward
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

# 2.6 Encoder
class Encoder(tf.keras.layers.Layer):
    def _init_(self, num_layers, d_model, num_heads, dff,
                 input_vocab_size, maximum_position_encoding, rate=0.1):
        super()._init_()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training=False, mask=None):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, training=training, mask=mask)

        return x

# 2.7 Decoder (CORREGIDO)
class Decoder(tf.keras.layers.Layer):
    def _init_(self, num_layers, d_model, num_heads, dff, vocab_size, 
                 maximum_position_encoding, rate=0.1):
        super()._init_()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                          for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i, dec_layer in enumerate(self.dec_layers):
            x, block1, block2 = dec_layer(
                x,
                enc_output,
                training=training,
                look_ahead_mask=look_ahead_mask,
                padding_mask=padding_mask
            )

            attention_weights[f"decoder_layer{i+1}_block1"] = block1
            attention_weights[f"decoder_layer{i+1}_block2"] = block2

        return x, attention_weights

# 2.8 Transformer completo (CORREGIDO)
class Transformer(tf.keras.Model):
    def _init_(self, num_layers, d_model, num_heads, dff, vocab_size, 
                 pe_input, pe_target, rate=0.1):
        super()._init_()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                               vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(vocab_size)
    
    def call(self, inputs, training=False):
        inp, tar = inputs
        
        enc_padding_mask = self.create_padding_mask(inp)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        enc_output = self.encoder(inp, training=training, mask=enc_padding_mask)

        dec_output, attention_weights = self.decoder(
            tar,
            enc_output,
            training=training,
            look_ahead_mask=combined_mask,
            padding_mask=enc_padding_mask
        )
        
        final_output = self.final_layer(dec_output)
        return final_output
    
    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, PAD_ID), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]
    
    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

# 2.9 Crear modelos (ES→EN y EN→ES)
print("\n[1/2] Creando modelos Transformer...")

transformer_es_en = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    vocab_size=VOCAB_SIZE,
    pe_input=MAX_SEQ_LEN,
    pe_target=MAX_SEQ_LEN,
    rate=DROPOUT_RATE
)

transformer_en_es = Transformer(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF,
    vocab_size=VOCAB_SIZE,
    pe_input=MAX_SEQ_LEN,
    pe_target=MAX_SEQ_LEN,
    rate=DROPOUT_RATE
)

# Construir modelos con una pasada de datos real
print("  Construyendo modelos con datos reales...")
for inp, tar_inp, tar_real in train_es_en.take(1):
    # Construir modelo ES→EN
    _ = transformer_es_en((inp, tar_inp), training=False)
    # Construir modelo EN→ES  
    _ = transformer_en_es((inp, tar_inp), training=False)
    break

print("  ✓ Modelos construidos correctamente")

# Contar parámetros
def count_parameters(model):
    """Cuenta el número de parámetros entrenables"""
    total_params = 0
    for variable in model.trainable_variables:
        total_params += np.prod(variable.shape)
    return total_params

total_params = count_parameters(transformer_es_en)
print(f"✓ Parámetros entrenables: {total_params:,}")

# 2.10 Configurar entrenamiento
print("\n[2/2] Configurando optimizador y función de pérdida...")

# Crear optimizadores separados para cada modelo
learning_rate = 0.001
optimizer_es_en = tf.keras.optimizers.Adam(learning_rate)
optimizer_en_es = tf.keras.optimizers.Adam(learning_rate)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, PAD_ID))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

train_loss_es_en = tf.keras.metrics.Mean(name='train_loss_es_en')
train_accuracy_es_en = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy_es_en')

train_loss_en_es = tf.keras.metrics.Mean(name='train_loss_en_es')
train_accuracy_en_es = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy_en_es')

print("✓ Optimizadores configurados (Adam)")

# 2.11 Funciones de entrenamiento
def train_step_es_en(inp, tar_inp, tar_real):
    with tf.GradientTape() as tape:
        predictions = transformer_es_en((inp, tar_inp), training=True)
        loss = loss_function(tar_real, predictions)
    
    gradients = tape.gradient(loss, transformer_es_en.trainable_variables)
    optimizer_es_en.apply_gradients(zip(gradients, transformer_es_en.trainable_variables))
    
    train_loss_es_en(loss)
    train_accuracy_es_en(tar_real, predictions)

def train_step_en_es(inp, tar_inp, tar_real):
    with tf.GradientTape() as tape:
        predictions = transformer_en_es((inp, tar_inp), training=True)
        loss = loss_function(tar_real, predictions)
    
    gradients = tape.gradient(loss, transformer_en_es.trainable_variables)
    optimizer_en_es.apply_gradients(zip(gradients, transformer_en_es.trainable_variables))
    
    train_loss_en_es(loss)
    train_accuracy_en_es(tar_real, predictions)

# =====================================================
# ENTRENAMIENTO (REDUCIDO PARA PRUEBAS)
# =====================================================

print("\n" + "="*70)
print("INICIANDO ENTRENAMIENTO (2 ÉPOCAS PARA PRUEBA)")
print("="*70)

print("\n" + "="*70)
print("INICIANDO ENTRENAMIENTO (10 ÉPOCAS)")
print("="*70)

EPOCHS = 6  # Aumentamos a 10 épocas
history = {
    'train_loss_es_en': [], 
    'val_loss_es_en': [],
    'train_loss_en_es': [],
    'val_loss_en_es': []
}
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    
    # Entrenamiento ES→EN
    print(f"\n[Época {epoch+1}/{EPOCHS}] Entrenando ES→EN...")
    train_loss_es_en.reset_state()
    train_accuracy_es_en.reset_state()
    
    batch_count = 0
    for batch, (inp, tar_inp, tar_real) in enumerate(train_es_en):
        train_step_es_en(inp, tar_inp, tar_real)
        
        if batch % 100 == 0:
            print(f"  Batch {batch}: Loss={train_loss_es_en.result():.4f}, "
                  f"Acc={train_accuracy_es_en.result():.4f}")
        batch_count += 1
    
    train_loss_es_en_value = train_loss_es_en.result().numpy()
    
    # Validación ES→EN
    val_losses = []
    for (inp, tar_inp, tar_real) in val_es_en:
        predictions = transformer_es_en((inp, tar_inp), training=False)
        loss = loss_function(tar_real, predictions)
        val_losses.append(loss.numpy())
    val_loss_es_en_value = np.mean(val_losses)
    
    # Entrenamiento EN→ES
    print(f"\n[Época {epoch+1}/{EPOCHS}] Entrenando EN→ES...")
    train_loss_en_es.reset_state()
    train_accuracy_en_es.reset_state()
    
    batch_count = 0
    for batch, (inp, tar_inp, tar_real) in enumerate(train_en_es):
        train_step_en_es(inp, tar_inp, tar_real)
        
        if batch % 100 == 0:
            print(f"  Batch {batch}: Loss={train_loss_en_es.result():.4f}, "
                  f"Acc={train_accuracy_en_es.result():.4f}")
        batch_count += 1
    
    train_loss_en_es_value = train_loss_en_es.result().numpy()
    
    # Validación EN→ES
    val_losses = []
    for (inp, tar_inp, tar_real) in val_en_es:
        predictions = transformer_en_es((inp, tar_inp), training=False)
        loss = loss_function(tar_real, predictions)
        val_losses.append(loss.numpy())
    val_loss_en_es_value = np.mean(val_losses)
    
    # Guardar métricas
    history['train_loss_es_en'].append(float(train_loss_es_en_value))
    history['val_loss_es_en'].append(float(val_loss_es_en_value))
    history['train_loss_en_es'].append(float(train_loss_en_es_value))
    history['val_loss_en_es'].append(float(val_loss_en_es_value))
    
    epoch_time = time.time() - epoch_start
    print(f"\n{'='*70}")
    print(f"Época {epoch+1} completada en {epoch_time:.1f}s")
    print(f"  Train Loss ES→EN: {train_loss_es_en_value:.4f} | Val Loss: {val_loss_es_en_value:.4f}")
    print(f"  Train Loss EN→ES: {train_loss_en_es_value:.4f} | Val Loss: {val_loss_en_es_value:.4f}")
    
    # Guardar modelos cada 2 épocas
    if (epoch + 1) % 2 == 0:
        transformer_es_en.save_weights(f'transformer_es_en_epoch_{epoch+1}.weights.h5')
        transformer_en_es.save_weights(f'transformer_en_es_epoch_{epoch+1}.weights.h5')
        print(f"  ✓ Modelos guardados (época {epoch+1})")
    
    print(f"{'='*70}")

total_time = time.time() - start_time
print(f"\n✓ Entrenamiento completado en {total_time/60:.1f} minutos")

# =====================================================
# FASE 3: EVALUACIÓN Y TRADUCCIÓN
# =====================================================

print("\n" + "="*70)
print("FASE 3: EVALUACIÓN Y TRADUCCIÓN")
print("="*70)

# 3.1 Función de traducción MEJORADA
def translate(model, sentence, max_length=MAX_SEQ_LEN, temperature=0.8):
    """Traduce una oración usando el modelo con mejor manejo de generación"""
    # Codificar entrada
    input_tokens = encode_sentence(sentence, max_length)
    encoder_input = tf.expand_dims(input_tokens, 0)
    
    # Inicializar decoder con BOS
    decoder_input = [BOS_ID]
    output = tf.expand_dims(decoder_input, 0)
    
    for i in range(max_length):
        predictions = model((encoder_input, output), training=False)
        predictions = predictions[:, -1:, :]  # Tomar solo el último token
        
        # Aplicar temperatura
        predictions = predictions / temperature
        
        # Usar sampling en lugar de argmax para mayor diversidad
        predictions = tf.squeeze(predictions, axis=1)
        predicted_id = tf.random.categorical(predictions, num_samples=1)
        predicted_id = tf.cast(predicted_id, tf.int32)
        
        # Verificar si es EOS
        if predicted_id == EOS_ID:
            break
            
        # Añadir el token predicho a la salida
        output = tf.concat([output, predicted_id], axis=-1)
    
    # Decodificar resultado
    result = output.numpy()[0]
    # Filtrar tokens especiales y decodificar
    filtered_tokens = [int(x) for x in result if x not in [PAD_ID, BOS_ID, EOS_ID]]
    
    if not filtered_tokens:
        return "[No se pudo traducir]"
    
    try:
        decoded = sp.decode(filtered_tokens)
        # Limpiar espacios extra
        decoded = ' '.join(decoded.split())
        return decoded
    except:
        # Fallback: intentar decodificar manualmente
        try:
            pieces = [sp.id_to_piece(int(x)) for x in filtered_tokens if int(x) < VOCAB_SIZE]
            # Remover pieces vacíos y unir
            pieces = [p for p in pieces if p and p != '']
            decoded = ' '.join(pieces).replace('▁', ' ')
            return decoded
        except:
            return "[Error en decodificación]"

# 3.2 Función de traducción con beam search (OPCIONAL)
def translate_beam_search(model, sentence, beam_width=3, max_length=MAX_SEQ_LEN):
    """Traduce usando beam search para mejores resultados"""
    # Codificar entrada
    input_tokens = encode_sentence(sentence, max_length)
    encoder_input = tf.expand_dims(input_tokens, 0)
    
    # Inicializar beams
    beams = [([BOS_ID], 0.0)]  # (sequence, score)
    
    for i in range(max_length):
        new_beams = []
        
        for seq, score in beams:
            # Si la secuencia ya terminó, mantenerla
            if seq[-1] == EOS_ID:
                new_beams.append((seq, score))
                continue
                
            # Preparar entrada para el decoder
            decoder_input = tf.expand_dims(seq, 0)
            
            # Obtener predicciones
            predictions = model((encoder_input, decoder_input), training=False)
            last_predictions = predictions[:, -1, :]
            
            # Obtener top-k tokens
            top_probs, top_indices = tf.math.top_k(
                tf.nn.softmax(last_predictions[0]), k=beam_width
            )
            
            # Expandir beams
            for j in range(beam_width):
                token_id = top_indices[j].numpy()
                token_prob = top_probs[j].numpy()
                new_seq = seq + [token_id]
                new_score = score - np.log(token_prob)  # Log probability
                new_beams.append((new_seq, new_score))
        
        # Seleccionar top beams
        new_beams.sort(key=lambda x: x[1])
        beams = new_beams[:beam_width]
        
        # Verificar si todos los beams terminaron
        if all(seq[-1] == EOS_ID for seq, score in beams):
            break
    
    # Seleccionar el mejor beam
    best_seq, best_score = beams[0]
    
    # Decodificar
    filtered_tokens = [int(x) for x in best_seq if x not in [PAD_ID, BOS_ID, EOS_ID]]
    
    if not filtered_tokens:
        return "[No se pudo traducir]"
    
    try:
        decoded = sp.decode(filtered_tokens)
        return decoded
    except:
        return "[Error en decodificación]"

# 3.3 Probar traducción con ambos métodos
print("\n[1/3] Probando traducción con diferentes métodos...")

# Oración simple para prueba
test_sentences_es = [
    "hola mundo",
    "me llamo juan",
    "el gato come pescado",
    "buenos días",
    "¿cómo estás?"
]

test_sentences_en = [
    "hello world", 
    "my name is john",
    "the cat eats fish",
    "good morning",
    "how are you?"
]

print("\nES→EN (Sampling):")
print("-" * 50)
for i, sentence in enumerate(test_sentences_es[:3]):
    translation = translate(transformer_es_en, sentence)
    print(f"ES: {sentence}")
    print(f"EN: {translation}")
    print(f"REF: {test_sentences_en[i]}")
    print("-" * 50)

print("\nEN→ES (Sampling):")
print("-" * 50)
for i, sentence in enumerate(test_sentences_en[:3]):
    translation = translate(transformer_en_es, sentence)
    print(f"EN: {sentence}")
    print(f"ES: {translation}")
    print(f"REF: {test_sentences_es[i]}")
    print("-" * 50)

# 3.4 Interfaz mejorada con opciones
print("\n[2/3] Interfaz de traducción interactiva mejorada")

def improved_translation_interface():
    while True:
        print("\n" + "="*60)
        print("TRADUCCIÓN INTERACTIVA MEJORADA")
        print("="*60)
        print("1. Español → Inglés (Sampling)")
        print("2. Inglés → Español (Sampling)")
        print("3. Español → Inglés (Beam Search)")
        print("4. Inglés → Español (Beam Search)")
        print("5. Probar oraciones de ejemplo")
        print("6. Salir")
        
        choice = input("\nSeleccione opción (1-6): ").strip()
        
        if choice == '1':
            text = input("\nIngrese texto en español: ").strip()
            if text:
                print("Traduciendo con sampling...")
                translation = translate(transformer_es_en, text, temperature=0.7)
                print(f"\nES: {text}")
                print(f"EN: {translation}")
            else:
                print("Texto vacío")
                
        elif choice == '2':
            text = input("\nIngrese texto en inglés: ").strip()
            if text:
                print("Traduciendo con sampling...")
                translation = translate(transformer_en_es, text, temperature=0.7)
                print(f"\nEN: {text}")
                print(f"ES: {translation}")
            else:
                print("Texto vacío")
                
        elif choice == '3':
            text = input("\nIngrese texto en español: ").strip()
            if text:
                print("Traduciendo con beam search...")
                translation = translate_beam_search(transformer_es_en, text)
                print(f"\nES: {text}")
                print(f"EN: {translation}")
            else:
                print("Texto vacío")
                
        elif choice == '4':
            text = input("\nIngrese texto en inglés: ").strip()
            if text:
                print("Traduciendo con beam search...")
                translation = translate_beam_search(transformer_en_es, text)
                print(f"\nEN: {text}")
                print(f"ES: {translation}")
            else:
                print("Texto vacío")
                
        elif choice == '5':
            print("\nProbando oraciones de ejemplo:")
            print("-" * 40)
            for i in range(min(3, len(test_sentences_es))):
                print(f"\nEjemplo {i+1}:")
                es_trans = translate(transformer_es_en, test_sentences_es[i])
                en_trans = translate(transformer_en_es, test_sentences_en[i])
                print(f"ES: {test_sentences_es[i]} -> EN: {es_trans}")
                print(f"EN: {test_sentences_en[i]} -> ES: {en_trans}")
                
        elif choice == '6':
            print("\n¡Hasta luego!")
            break
        else:
            print("Opción inválida")

# 3.5 Guardar modelos finales
print("\n[3/3] Guardando modelos entrenados...")
transformer_es_en.save_weights('transformer_es_en_final.weights.h5')
transformer_en_es.save_weights('transformer_en_es_final.weights.h5')
print("✓ Modelos guardados: transformer_es_en_final.weights.h5, transformer_en_es_final.weights.h5")

# Ejecutar interfaz
improved_translation_interface()

print("\n" + "="*70)
print("PROGRAMA COMPLETADO EXITOSAMENTE")
print("="*70)
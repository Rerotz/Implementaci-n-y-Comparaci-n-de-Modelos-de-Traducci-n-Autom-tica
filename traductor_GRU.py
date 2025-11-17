# ============================
# BLOQUE ÚNICO: PREPROC, SP, MODELO GRU+ATN VECTORIZADA, TRAIN, BLEU, INPUT
# ============================
import os, time, random, re
import numpy as np
import sentencepiece as spm
import sacrebleu
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.layers import Embedding, GRU, Dense, Bidirectional
from tensorflow.keras import Model, Input

# ----------------------------
# PARÁMETROS CONFIGURABLES
# ----------------------------
MAX_SAMPLES = 50000
SP_VOCAB = 16000
BATCH_SIZE = 16
EMBED_DIM = 256
ENC_UNITS = 256
DEC_UNITS = 256
EPOCHS = 6
MAX_TGT_LEN = 60
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------
# FORZAR/CONFIG GPU (si hay)
# ----------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        tf.config.optimizer.set_jit(True)  # habilitar XLA (opcional, puede mejorar)
    except Exception as e:
        print("Advertencia configurando GPU:", e)

device = "/gpu:0" if gpus else "/cpu:0"
print("Usando dispositivo:", device)
print("GPUs detectadas:", gpus)

# ----------------------------
# 1) LEER LOS ARCHIVOS SUBIDOS
# ----------------------------
with open("Tatoeba.en-es.es", "r", encoding="utf-8") as f:
    es_lines = [l.strip() for l in f if l.strip()]

with open("Tatoeba.en-es.en", "r", encoding="utf-8") as f:
    en_lines = [l.strip() for l in f if l.strip()]

print("Líneas ES:", len(es_lines), "Líneas EN:", len(en_lines))

N_pairs = min(len(es_lines), len(en_lines), MAX_SAMPLES)
pairs = [(es_lines[i].lower(), en_lines[i].lower()) for i in range(N_pairs)]

def clean_text(s):
    s = s.lower()
    s = re.sub(r"[^a-záéíóúüñ¿?¡!0-9.,;:()\"'\/\-\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

pairs = [(clean_text(s), clean_text(t)) for s,t in pairs]

random.shuffle(pairs)

N = len(pairs)
train_pairs = pairs[:int(N*0.9)]
valid_pairs = pairs[int(N*0.9):int(N*0.95)]
test_pairs  = pairs[int(N*0.95):]

print("Ejemplos: train", len(train_pairs), "valid", len(valid_pairs), "test", len(test_pairs))

# ----------------------------
# 2) ENTRENAR SENTENCEPIECE
# ----------------------------
sp_input = "sp_input.txt"
with open(sp_input, "w", encoding="utf-8") as f:
    for s,t in train_pairs:
        f.write(s + "\n")
        f.write(t + "\n")

sp_model_prefix = "sp"
spm.SentencePieceTrainer.Train(
    input=sp_input,
    model_prefix=sp_model_prefix,
    vocab_size=SP_VOCAB,
    model_type='bpe',
    character_coverage=1.0,
    bos_id=1, eos_id=2, pad_id=0, unk_id=3
)

sp = spm.SentencePieceProcessor()
sp.load(sp_model_prefix + ".model")
print("SentencePiece entrenado. Vocab size:", sp.get_piece_size())

BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()
PAD_ID = sp.pad_id()

# ----------------------------
# 3) TOKENIZAR Y PREPARAR DATA
# ----------------------------
def encode_sentence(s, max_len=None):
    ids = sp.encode_as_ids(s)
    ids = [BOS_ID] + ids + [EOS_ID]
    if max_len:
        ids = ids[:max_len]
    return ids

def prepare_dataset(pairs_list, max_src_len=60, max_tgt_len=MAX_TGT_LEN):
    src_seqs = []
    tgt_in_seqs = []
    tgt_out_seqs = []

    for s,t in pairs_list:
        src_ids = encode_sentence(s, max_len=max_src_len)
        tgt_ids = encode_sentence(t, max_len=max_tgt_len)

        src_seqs.append(src_ids)
        tgt_in_seqs.append(tgt_ids[:-1])
        tgt_out_seqs.append(tgt_ids[1:])

    src_max = min(max(len(x) for x in src_seqs), 120)
    tgt_max = min(max(len(x) for x in tgt_in_seqs), max_tgt_len)

    def pad_list(seqs, max_len):
        arr = np.full((len(seqs), max_len), PAD_ID, dtype=np.int32)
        for i, s in enumerate(seqs):
            L = min(len(s), max_len)
            arr[i, :L] = s[:L]
        return arr

    return pad_list(src_seqs, src_max), pad_list(tgt_in_seqs, tgt_max), pad_list(tgt_out_seqs, tgt_max)

src_train, tgt_in_train, tgt_out_train = prepare_dataset(train_pairs)
src_val,   tgt_in_val,   tgt_out_val   = prepare_dataset(valid_pairs)
src_test,  tgt_in_test,  tgt_out_test  = prepare_dataset(test_pairs)

# ----------------------------
# 4) DATASETS (tf.data)
# ----------------------------
train_dataset = tf.data.Dataset.from_tensor_slices((src_train, tgt_in_train, tgt_out_train))
train_dataset = (
    train_dataset
    .shuffle(20000)
    .batch(BATCH_SIZE, drop_remainder=True)   # <<-- MUY IMPORTANTE
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((src_val, tgt_in_val, tgt_out_val))
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

# ----------------------------
# 5) ATENCIÓN VECTORIZADA + DECODER GRU (REEMPLAZO)
# ----------------------------
class BahdanauAttentionVec(tf.keras.layers.Layer):
    def __init__(self, enc_dim, dec_units):
        super().__init__()
        # dec_units = tamaño interno que usaremos para calcular score
        self.W1 = Dense(dec_units)   # proyecta enc_outputs -> (..., u)
        self.W2 = Dense(dec_units)   # proyecta dec_outputs -> (..., u)
        self.V  = Dense(1)           # reduce a un score escalar

    def call(self, enc_outputs, dec_outputs):
        """
        enc_outputs: (batch, T_enc, enc_dim)
        dec_outputs: (batch, T_dec, dec_units)
        devuelve:
            context: (batch, T_dec, enc_dim)
            attention_weights: (batch, T_dec, T_enc)
        """
        proj_enc = self.W1(enc_outputs)          # (b, T_enc, u)
        proj_dec = self.W2(dec_outputs)          # (b, T_dec, u)

        proj_enc_exp = tf.expand_dims(proj_enc, 1)   # (b, 1, T_enc, u)
        proj_dec_exp = tf.expand_dims(proj_dec, 2)   # (b, T_dec, 1, u)

        score = self.V(tf.nn.tanh(proj_enc_exp + proj_dec_exp))  # (b, T_dec, T_enc, 1)
        score = tf.squeeze(score, -1)                            # (b, T_dec, T_enc)

        attention_weights = tf.nn.softmax(score, axis=-1)        # softmax sobre T_enc

        context = tf.matmul(attention_weights, enc_outputs)      # (b, T_dec, enc_dim)

        return context, attention_weights

class Encoder(Model):
    def __init__(self, vocab_size, embed_dim, enc_units):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.bi_gru = Bidirectional(
            GRU(enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform',
        reset_after=True   )
        )
        self.fc = Dense(enc_units)

    def call(self, x):
        x = self.embedding(x)
        outputs, forward_h, backward_h = self.bi_gru(x)
        state = tf.concat([forward_h, backward_h], axis=-1)
        state = self.fc(state)
        return outputs, state

class DecoderVectorized(Model):
    def __init__(self, vocab_size, embed_dim, dec_units, enc_outputs_dim):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim)

        # GRU que procesa TODOS los timesteps de entrada a la vez
        self.gru = GRU(dec_units, return_sequences=True, return_state=True, reset_after=True)

        # Atención vectorizada
        self.attention = BahdanauAttentionVec(enc_outputs_dim, dec_units)

        # Proyección final a vocab
        self.fc = Dense(vocab_size)

    def call(self, dec_in, enc_outputs, dec_initial_state):
        """
        dec_in: (batch, T_dec)
        enc_outputs: (batch, T_enc, enc_dim)
        dec_initial_state: (batch, dec_units)
        """
        x = self.embedding(dec_in)                       # (b, T_dec, embed_dim)
        dec_outputs, dec_state = self.gru(x, initial_state=dec_initial_state)  # dec_outputs: (b, T_dec, dec_units)

        context, attn = self.attention(enc_outputs, dec_outputs)  # context: (b, T_dec, enc_dim)

        concat = tf.concat([dec_outputs, context], axis=-1)  # (b, T_dec, dec_units+enc_dim)
        logits = self.fc(concat)                             # (b, T_dec, vocab)

        return logits, dec_state

    def inference_step(self, prev_id, enc_outputs, dec_hidden):
        """
        prev_id: (batch, 1)
        enc_outputs: (batch, T_enc, enc_dim)
        dec_hidden: (batch, dec_units)
        devuelve logits (batch, vocab), state (batch, dec_units)
        """
        x = self.embedding(prev_id)[:, 0, :]   # (b, embed_dim)
        x = tf.expand_dims(x, 1)               # (b, 1, embed_dim)

        dec_out, state = self.gru(x, initial_state=dec_hidden)  # dec_out: (b, 1, dec_units)
        dec_out = dec_out[:, 0, :]               # (b, dec_units)

        # Atención: expandir dec_out a (b, 1, dec_units) para usar la función vectorizada
        context, _ = self.attention(enc_outputs, tf.expand_dims(dec_out, 1))  # context: (b, 1, enc_dim)
        context = context[:, 0, :]               # (b, enc_dim)

        concat = tf.concat([dec_out, context], axis=-1)  # (b, dec_units+enc_dim)
        logits = self.fc(concat)                         # (b, vocab)

        return logits, state

# ----------------------------
# INICIALIZAR MODELOS
# ----------------------------
VOCAB_SIZE = sp.get_piece_size()

encoder = Encoder(VOCAB_SIZE, EMBED_DIM, ENC_UNITS)
decoder = DecoderVectorized(VOCAB_SIZE, EMBED_DIM, DEC_UNITS, enc_outputs_dim=ENC_UNITS*2)

# CONTRUIR MODELO EJECUTANDO DUMMY (para inicializar pesos)
dummy_src = tf.zeros((1,10), dtype=tf.int32)
enc_out, enc_st = encoder(dummy_src)
dummy_tgt = tf.zeros((1,10), dtype=tf.int32)

# Ajuste: enc_st (batch, enc_units) debe tener tamaño igual a DEC_UNITS.
# En tu arquitectura, ENC_UNITS == DEC_UNITS por defecto, así que esto funcionará.
_ = decoder(dummy_tgt, enc_out, enc_st)

total_params = encoder.count_params() + decoder.count_params()
print("Params encoder:", encoder.count_params())
print("Params decoder:", decoder.count_params())
print("Total:", total_params)

# ----------------------------
# 6) LOSS, OPTIM
# ----------------------------
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    # real: (b, T), pred: (b, T, vocab)
    mask = tf.cast(tf.not_equal(real, PAD_ID), tf.float32)
    loss_ = loss_object(real, pred)   # (b, T)
    loss_ *= mask
    return tf.reduce_sum(loss_) / (tf.reduce_sum(mask) + 1e-9)

# ----------------------------
# 7) TRAIN LOOP (vectorizado, sin sincronizaciones innecesarias)
# ----------------------------
@tf.function
def train_step(src, ti, to):
    with tf.GradientTape() as tape:
        enc_out, enc_st = encoder(src)
        logits, _ = decoder(ti, enc_out, enc_st)
        loss = loss_function(to, logits)
    vars = encoder.trainable_variables + decoder.trainable_variables
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))
    return loss

@tf.function
def val_step(src, ti, to):
    enc_out, enc_st = encoder(src)
    logits, _ = decoder(ti, enc_out, enc_st)
    return loss_function(to, logits)

train_losses = []
val_losses = []

print("\nEntrenando...\n")
start = time.time()

for epoch in range(1, EPOCHS+1):
    print(f"\nEpoch {epoch}/{EPOCHS}")

    # Acumuladores como tensores para evitar sync en cada batch
    tl = tf.constant(0.0)
    steps = 0
    for src,ti,to in tqdm(train_dataset, desc="train"):
        l = train_step(src,ti,to)    # l es tensor escalar
        tl = tl + l
        steps += 1
    train_loss = (tl / tf.cast(steps, tf.float32)).numpy()

    vl = tf.constant(0.0)
    sv = 0
    for src,ti,to in val_dataset:
        l = val_step(src,ti,to)
        vl = vl + l
        sv += 1
    val_loss = (vl / tf.cast(sv, tf.float32)).numpy() if sv>0 else float('nan')

    train_losses.append(float(train_loss))
    val_losses.append(float(val_loss))

    print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

print("\nTiempo total:", time.time()-start, "s")

# ----------------------------
# 8) INFERENCE + BLEU
# ----------------------------
def greedy_decode_one(src_array, max_len=MAX_TGT_LEN):
    enc_out, enc_st = encoder(src_array)
    cur = np.array([[BOS_ID]],dtype=np.int32)
    state = enc_st
    res = []

    for _ in range(max_len):
        logits, state = decoder.inference_step(cur, enc_out, state)  # logits: (b, vocab)
        next_id = int(tf.argmax(logits,axis=-1).numpy())
        if next_id == EOS_ID:
            break
        res.append(next_id)
        cur = np.array([[next_id]],dtype=np.int32)

    return sp.decode_ids(res)

print("\nEvaluando BLEU...")
refs = []
hyps = []
NE = min(200, len(test_pairs))

for i in range(NE):
    s,t = test_pairs[i]
    ids = encode_sentence(s)
    arr = np.full((1, src_train.shape[1]), PAD_ID, dtype=np.int32)
    L = min(len(ids), arr.shape[1])
    arr[0,:L] = ids[:L]
    pred = greedy_decode_one(arr)
    hyps.append(pred)
    refs.append(t)

bleu = sacrebleu.corpus_bleu(hyps, [refs])
print("BLEU:", bleu.score)

# ----------------------------
# 9) EJEMPLOS + INPUT MANUAL
# ----------------------------
print("\n=== Ejemplos ===")
for i in range(5):
    s,t = test_pairs[i]
    ids = encode_sentence(s)
    arr = np.full((1, src_train.shape[1]), PAD_ID, dtype=np.int32)
    L = min(len(ids), arr.shape[1])
    arr[0,:L] = ids[:L]
    pred = greedy_decode_one(arr)
    print("\nES:", s)
    print("GT:", t)
    print("PR:", pred)

print("\n=== Prueba manual ===")
while True:
    txt = input("ES: ")
    if txt.lower().strip()=="salir":
        break
    c = clean_text(txt)
    ids = encode_sentence(c)
    arr = np.full((1, src_train.shape[1]), PAD_ID, dtype=np.int32)
    L = min(len(ids), arr.shape[1])
    arr[0,:L] = ids[:L]
    print("EN:", greedy_decode_one(arr))

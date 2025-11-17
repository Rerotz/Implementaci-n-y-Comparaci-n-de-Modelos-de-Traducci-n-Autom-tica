from pathlib import Path

data_dir = Path("/content/data")
data_dir.mkdir(parents=True, exist_ok=True)

# EJEMPLO
src_sentences = [
    "hola mundo",
    "buenos días",
    "te quiero",
    "gracias",
    "¿cómo estás?"
]
tgt_sentences = [
    "hello world",
    "good morning",
    "i love you",
    "thank you",
    "how are you?"
]

with open(data_dir/"train.src", "w", encoding="utf-8") as f:
    f.write("\n".join(src_sentences))

with open(data_dir/"train.tgt", "w", encoding="utf-8") as f:
    f.write("\n".join(tgt_sentences))

print("Dataset cargado.")
import sentencepiece as spm

train_src = str(data_dir/"train.src")
train_tgt = str(data_dir/"train.tgt")
VOCAB_SIZE = 99

spm.SentencePieceTrainer.Train(
    f"--input={train_src} --model_prefix=sp_src --vocab_size={VOCAB_SIZE} --character_coverage=1.0 --model_type=bpe"
)
spm.SentencePieceTrainer.Train(
    f"--input={train_tgt} --model_prefix=sp_tgt --vocab_size={VOCAB_SIZE} --character_coverage=1.0 --model_type=bpe"
)

sp_src = spm.SentencePieceProcessor()
sp_src.Load("sp_src.model")

sp_tgt = spm.SentencePieceProcessor()
sp_tgt.Load("sp_tgt.model")

print("Tokenizadores entrenados.")
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

BOS = "<s>"
EOS = "</s>"

def encode_src(texts):
    return [sp_src.EncodeAsIds(t) for t in texts]

def encode_tgt(texts):
    ids = []
    for t in texts:
        combined = f"{BOS} {t} {EOS}"
        ids.append(sp_tgt.EncodeAsIds(combined))
    return ids

src_ids = encode_src(src_sentences)
tgt_ids = encode_tgt(tgt_sentences)

max_len_src = max(len(s) for s in src_ids)
max_len_tgt = max(len(s) for s in tgt_ids)

encoder_input = pad_sequences(src_ids, maxlen=max_len_src, padding="post")
decoder_input = pad_sequences([t[:-1] for t in tgt_ids], maxlen=max_len_tgt-1, padding="post")
decoder_target = pad_sequences([t[1:] for t in tgt_ids], maxlen=max_len_tgt-1, padding="post")

decoder_target = np.expand_dims(decoder_target, -1)

print("Secuencias codificadas y con padding listo.")
from sklearn.model_selection import train_test_split

enc_train, enc_val, dec_in_train, dec_in_val, dec_targ_train, dec_targ_val = train_test_split(
    encoder_input, decoder_input, decoder_target,
    test_size=0.2,
    random_state=42
)

print("Train / Validation listos.")
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense
from tensorflow.keras.models import Model

EMBED_DIM = 256
UNITS = 256
vocab_src = sp_src.GetPieceSize()
vocab_tgt = sp_tgt.GetPieceSize()

encoder_inputs = Input(shape=(None,), name="encoder_inputs")
enc_emb = Embedding(vocab_src, EMBED_DIM, mask_zero=True)(encoder_inputs)
_, encoder_state = SimpleRNN(UNITS, return_state=True)(enc_emb)

decoder_inputs = Input(shape=(None,), name="decoder_inputs")
dec_emb = Embedding(vocab_tgt, EMBED_DIM, mask_zero=True)(decoder_inputs)
decoder_outputs, _ = SimpleRNN(UNITS, return_sequences=True, return_state=True)(
    dec_emb, initial_state=encoder_state
)
decoder_logits = Dense(vocab_tgt, activation="softmax")(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_logits)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("model1_simpleRNN.h5", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

history = model.fit(
    [enc_train, dec_in_train],
    dec_targ_train,
    validation_data=([enc_val, dec_in_val], dec_targ_val),
    epochs=20,
    batch_size=32,
    callbacks=callbacks
)

print("Entrenamiento completado.")
import json

history_dict = {
    "loss": list(map(float, history.history["loss"])),
    "val_loss": list(map(float, history.history["val_loss"])),
    "accuracy": list(map(float, history.history["accuracy"])),
    "val_accuracy": list(map(float, history.history["val_accuracy"])),
}

with open("training_history_model1.json", "w") as f:
    json.dump(history_dict, f)

print("Historial guardado en training_history_model1.json")
# ENCODER
enc_emb = Embedding(vocab_src, EMBED_DIM, mask_zero=True, name="enc_emb")(encoder_inputs)
encoder_rnn = SimpleRNN(UNITS, return_state=True, name="encoder_rnn")
_, encoder_state = encoder_rnn(enc_emb)

# DECODER
dec_emb = Embedding(vocab_tgt, EMBED_DIM, mask_zero=True, name="dec_emb")(decoder_inputs)
decoder_rnn = SimpleRNN(UNITS, return_sequences=True, return_state=True, name="decoder_rnn")
decoder_outputs, _ = decoder_rnn(dec_emb, initial_state=encoder_state)

decoder_dense = Dense(vocab_tgt, activation="softmax", name="decoder_dense")
decoder_outputs = decoder_dense(decoder_outputs)
# =========================================================
# MODELO DE INFERENCIA — basado en los nombres reales del modelo
# =========================================================

# Encoder
encoder_model_inf = Model(encoder_inputs, encoder_state)

# Entradas del decoder en inferencia
decoder_state_input = Input(shape=(UNITS,), name="decoder_state_input")
decoder_single_input = Input(shape=(1,), name="decoder_single_input")

# Recuperar capas REALES
dec_emb_layer = model.get_layer("embedding_1")     # decoder embedding
decoder_rnn_layer = model.get_layer("simple_rnn_1") # decoder SimpleRNN
decoder_dense_layer = model.get_layer("dense")      # proyección softmax

# Embedding
single_emb = dec_emb_layer(decoder_single_input)

# RNN un paso
decoder_single_output, dec_state = decoder_rnn_layer(
    single_emb,
    initial_state=decoder_state_input
)

# Softmax final
decoder_single_logits = decoder_dense_layer(decoder_single_output)

# Modelo final del decoder
decoder_model_inf = Model(
    [decoder_single_input, decoder_state_input],
    [decoder_single_logits, dec_state]
)

print("Modelos de inferencia construidos correctamente.")
import sacrebleu

preds = []
refs = []

for s, t in zip(src_sentences, tgt_sentences):
    preds.append(translate(s))
    refs.append(t)

bleu = sacrebleu.corpus_bleu(preds, [refs])
print("BLEU =", bleu.score)
with open("predicciones_modelo1.txt", "w") as f:
    for s, p in zip(src_sentences, preds):
        f.write(f"SRC: {s}\nPRED: {p}\n\n")

print("Predicciones guardadas en predicciones_modelo1.txt")
def translate(sentence, max_len=50):
    # Tokenizar la oración fuente
    src_ids = sp_src.EncodeAsIds(sentence)
    src = np.array(src_ids)[None, :]   # (1, seq_len)

    # Obtener estado del encoder
    state = encoder_model_inf.predict(src)

    # IDs especiales
    bos_id = sp_tgt.PieceToId("<s>")
    eos_id = sp_tgt.PieceToId("</s>")

    # Iniciar con <s>
    cur = np.array([[bos_id]])
    result = []

    for _ in range(max_len):
        # Predicción paso a paso
        logits, state = decoder_model_inf.predict([cur, state])

        # Id del token más probable
        token = int(np.argmax(logits[0, 0, :]))

        # Si es </s> detenemos
        if token == eos_id:
            break

        # Guardar token
        result.append(token)

        # Alimentar el token al siguiente paso
        cur = np.array([[token]])

    # Decodificar IDs → texto
    return sp_tgt.DecodeIds(result)
print(translate("buenos dias"))
print(translate("como estas"))
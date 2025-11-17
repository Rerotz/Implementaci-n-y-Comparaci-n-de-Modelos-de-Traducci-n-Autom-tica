# Notebook completo: ESPAÑOL → INGLÉS (LSTM Encoder + LSTM Decoder + Bahdanau Attention)
# Este notebook incluye:
# - Fase 1: preprocesamiento (limpieza, dedupe, splits) + SentencePiece (BPE)
# - Fase 2: Modelo LSTM (encoder bidireccional) + LSTM (decoder) con Bahdanau attention
# - AMP (mixed precision) compatible
# - Fase 3: evaluación (train/val loss por epoch, BLEU, tiempo y nº parámetros)
# - Interfaz para probar traducciones
# CELDA 1 - instalar dependencias (ejecutar en Colab)
!pip install -q sentencepiece sacrebleu tqdm pandas
# PyTorch (versión para CUDA en Colab). Si no necesitas CUDA, puedes omitir la línea de instalación de torch.
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CELDA 2 - subir dataset
from google.colab import files
print("Sube tu archivo: 'espanol-ingles.tsv' (formato Tatoeba: id_src \\t src \\t id_tgt \\t tgt ...).")
uploaded = files.upload()
# Tras subirlo, estará en /content/espanol-ingles.tsv
# CELDA 3 - FASE 1: limpieza, dedupe, splits y entrenar SentencePiece
import os, re, random, json
from collections import defaultdict, Counter
import sentencepiece as spm

# ---------- CONFIG Fase1 ----------
WORKDIR = "/content/es_en_tatoeba"
os.makedirs(WORKDIR, exist_ok=True)
INPUT_TSV = "/content/espanol-ingles.tsv"
SRC_LANG = "es"; TGT_LANG = "en"

LOWERCASE = True
KEEP_MULTIPLE = False   # False -> seleccionar 1 traducción por fuente
MAX_WORDS = 100          # longitud máxima en palabras (reduce memoria)
VOCAB_SIZE = 32000
SAMPLE_SIZE = 50000
# SAMPLE_SIZE = None      # None = todo; o int para usar subset rápido

PAD_ID = 0; UNK_ID = 1; BOS_ID = 2; EOS_ID = 3
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ---------- funciones ----------
def normalize_text(s, lowercase=True):
    if not s: return ""
    s = str(s)
    s = s.replace("\u200b", " ").replace("\u00A0", " ")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("«", '"').replace("»", '"').replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower() if lowercase else s

def read_tatoeba_tsv(path):
    pairs = []
    bad = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                bad += 1
                continue
            src = parts[1]; tgt = parts[3]
            pairs.append((src, tgt))
    print(f"Líneas válidas: {len(pairs)} (omitidas: {bad})")
    return pairs

def filter_and_normalize(pairs):
    out = []
    for s,t in pairs:
        s2 = normalize_text(s, LOWERCASE)
        t2 = normalize_text(t, LOWERCASE)
        if not s2 or not t2:
            continue
        if len(s2.split()) > MAX_WORDS or len(t2.split()) > MAX_WORDS:
            continue
        out.append((s2, t2))
    print("Tras normalizar/filtrar:", len(out), "pares")
    return out

def dedupe_select(pairs, keep_multiple=False):
    counter = Counter(pairs)
    print("Pares únicos exactos (s,t):", len(counter))
    if keep_multiple:
        return list(counter.keys())
    grouped = defaultdict(list)
    for (s,t),cnt in counter.items():
        grouped[s].append((t,cnt))
    final = []
    for s, tlist in grouped.items():
        tlist_sorted = sorted(tlist, key=lambda x: -x[1])
        final.append((s, tlist_sorted[0][0]))  # elegimos la traducción más frecuente
    print("Pares tras seleccionar una traducción por fuente:", len(final))
    return final

def write_splits(pairs):
    outdir = os.path.join(WORKDIR, "processed")
    os.makedirs(outdir, exist_ok=True)
    random.shuffle(pairs)
    N = len(pairs)
    n_train = int(N * 0.90)
    n_val = int(N * 0.05)
    train = pairs[:n_train]
    val = pairs[n_train:n_train+n_val]
    test = pairs[n_train+n_val:]
    def write(name, data):
        with open(os.path.join(outdir, f"es_en.{name}.src"), "w", encoding="utf-8") as fs, \
             open(os.path.join(outdir, f"es_en.{name}.tgt"), "w", encoding="utf-8") as ft:
            for s,t in data:
                fs.write(s + "\n"); ft.write(t + "\n")
    write("train", train); write("val", val); write("test", test)
    return {"total": N, "train": len(train), "val": len(val), "test": len(test), "outdir": outdir}

def train_sentencepiece(train_src, train_tgt, vocab_size=VOCAB_SIZE):
    prefix = os.path.join(WORKDIR, f"sp_{SRC_LANG}{TGT_LANG}")
    concat = prefix + "_input.txt"
    with open(concat, "w", encoding="utf-8") as fw:
        for p in [train_src, train_tgt]:
            with open(p, "r", encoding="utf-8") as fr:
                for ln in fr:
                    fw.write(ln)
    spm.SentencePieceTrainer.Train(
        f"--input={concat} --model_prefix={prefix} --vocab_size={vocab_size} "
        f"--model_type=bpe --character_coverage=1.0 --pad_id={PAD_ID} --unk_id={UNK_ID} "
        f"--bos_id={BOS_ID} --eos_id={EOS_ID}"
    )
    print("SentencePiece entrenado:", prefix + ".model")
    return prefix + ".model", prefix + ".vocab"

# ---------- pipeline ----------
print("=== FASE 1: lectura ===")
raw_pairs = read_tatoeba_tsv(INPUT_TSV)
if SAMPLE_SIZE:
    raw_pairs = raw_pairs[:SAMPLE_SIZE]

print("=== FASE 1: normalizar y filtrar ===")
clean_pairs = filter_and_normalize(raw_pairs)

print("=== FASE 1: dedupe y seleccionar traducción por fuente ===")
final_pairs = dedupe_select(clean_pairs, keep_multiple=KEEP_MULTIPLE)

print("=== FASE 1: dividir y escribir splits ===")
stats = write_splits(final_pairs)
print("Splits:", stats)

print("=== FASE 1: entrenar SentencePiece (BPE) ===")
train_src = os.path.join(stats["outdir"], "es_en.train.src")
train_tgt = os.path.join(stats["outdir"], "es_en.train.tgt")
sp_model, sp_vocab = train_sentencepiece(train_src, train_tgt)

# guardar resumen
summary = {
    "input_file": INPUT_TSV,
    "raw_lines": len(raw_pairs),
    "after_clean": len(clean_pairs),
    "final_pairs": len(final_pairs),
    "splits": stats,
    "sp_model": sp_model,
    "sp_vocab": sp_vocab
}
with open(os.path.join(stats["outdir"], "preprocessing_summary.json"), "w", encoding="utf-8") as fj:
    json.dump(summary, fj, ensure_ascii=False, indent=2)

print("FASE 1 COMPLETADA. Archivos en:", stats["outdir"])
# CELDA 4 - listar archivos y mostrar resumen
!ls -R /content/es_en_tatoeba
import json
with open("/content/es_en_tatoeba/processed/preprocessing_summary.json", "r", encoding="utf-8") as f:
    print("\nResumen Fase1:")
    print(json.load(f))
# CELDA 5 - configuración entrenamiento (ajusta aquí si quieres)
import torch, random, os, json, time
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

PROCESSED_DIR = "/content/es_en_tatoeba/processed"
SP_MODEL = "/content/es_en_tatoeba/sp_esen.model"  # generado en Fase1

TRAIN_SRC = os.path.join(PROCESSED_DIR, "es_en.train.src")
TRAIN_TGT = os.path.join(PROCESSED_DIR, "es_en.train.tgt")
VAL_SRC   = os.path.join(PROCESSED_DIR, "es_en.val.src")
VAL_TGT   = os.path.join(PROCESSED_DIR, "es_en.val.tgt")
TEST_SRC  = os.path.join(PROCESSED_DIR, "es_en.test.src")
TEST_TGT  = os.path.join(PROCESSED_DIR, "es_en.test.tgt")

# hiperparámetros óptimos para Colab (evitan OOM)
BATCH_SIZE = 16
EMBED_SIZE = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 1
DROPOUT = 0.1
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5
TEACHER_FORCING_RATIO = 0.5
MAX_DECODING_LEN = 40

PAD_ID = 0; UNK_ID = 1; BOS_ID = 2; EOS_ID = 3

MODEL_SAVE_DIR = "/content/es_en_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

random.seed(42)
torch.manual_seed(42)
# CELDA 6 - utilities, SP y dataset
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# cargar SentencePiece entrenado en Fase1
sp = spm.SentencePieceProcessor()
sp.Load(SP_MODEL)
VOCAB_SIZE = sp.get_piece_size()
print("SentencePiece vocab size:", VOCAB_SIZE)

def encode_sentence(s):
    return sp.encode(s, out_type=int)

def decode_ids_to_text(ids):
    ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
    return sp.decode(ids)

class TranslationDataset(Dataset):
    """Devuelve pares (src_ids_tensor, tgt_ids_tensor)."""
    def __init__(self, src_file, tgt_file, max_len=80):
        with open(src_file, "r", encoding="utf-8") as f: self.src = [l.strip() for l in f if l.strip()]
        with open(tgt_file, "r", encoding="utf-8") as f: self.tgt = [l.strip() for l in f if l.strip()]
        assert len(self.src) == len(self.tgt)
        self.max_len = max_len
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        s = encode_sentence(self.src[idx])[:self.max_len]
        t = encode_sentence(self.tgt[idx])[:self.max_len]
        return torch.tensor(s, dtype=torch.long), torch.tensor(t, dtype=torch.long)

def collate_fn(batch):
    """Padding + preparar dec_input (BOS + tokens) y dec_output (tokens + EOS)."""
    src_seqs, tgt_seqs = zip(*batch)
    src_lens = [len(s) for s in src_seqs]
    tgt_lens = [len(t) for t in tgt_seqs]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens) + 1

    src_padded = torch.full((len(batch), max_src), PAD_ID, dtype=torch.long)
    dec_input = torch.full((len(batch), max_tgt), PAD_ID, dtype=torch.long)
    dec_output = torch.full((len(batch), max_tgt), PAD_ID, dtype=torch.long)

    for i, s in enumerate(src_seqs):
        src_padded[i, :len(s)] = s
    for i, t in enumerate(tgt_seqs):
        dec_input[i,0] = BOS_ID
        dec_input[i,1:1+len(t)] = t
        dec_output[i,:len(t)] = t
        dec_output[i,len(t)] = EOS_ID

    src_lens = torch.tensor(src_lens, dtype=torch.long)
    return src_padded, src_lens, dec_input, dec_output
# CELDA 7 - definición de modelos (FP16-safe: masked_fill usa -1e4)
import torch.nn as nn
import torch

class EncoderLSTM(nn.Module):
    """Encoder: Embedding -> BiLSTM -> proyectar estados a tamaño decoder."""
    def __init__(self, input_dim, emb_dim, hid_dim, num_layers=1, dropout=0.1, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc_hidden = nn.Linear(hid_dim*2, hid_dim)
        self.fc_cell = nn.Linear(hid_dim*2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lens):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        cell_cat = torch.cat([cell[-2], cell[-1]], dim=1)
        hidden = torch.tanh(self.fc_hidden(hidden_cat)).unsqueeze(0)
        cell   = torch.tanh(self.fc_cell(cell_cat)).unsqueeze(0)
        return outputs, hidden, cell

class BahdanauAttention(nn.Module):
    """Atención aditiva - uso de -1e4 para evitar overflow en FP16."""
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        dec_h = decoder_hidden.squeeze(0)
        src_len = encoder_outputs.size(1)
        dec_h_rep = dec_h.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((dec_h_rep, encoder_outputs), dim=2)))
        score = self.v(energy).squeeze(2)
        if mask is not None:
            # usar -1e4 (seguro en FP16)
            score = score.masked_fill(mask == 0, -1e4)
        attn_weights = torch.softmax(score, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class DecoderLSTMWithAttention(nn.Module):
    """Decoder LSTM que incorpora contexto (att) en cada paso."""
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers=1, dropout=0.1, pad_id=0):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_id)
        self.attention = BahdanauAttention(enc_hid_dim=enc_hid_dim, dec_hid_dim=dec_hid_dim)
        self.lstm = nn.LSTM(emb_dim + enc_hid_dim, dec_hid_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc_out = nn.Linear(dec_hid_dim + enc_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_step, decoder_hidden, decoder_cell, encoder_outputs, mask=None):
        input_emb = self.dropout(self.embedding(input_step).unsqueeze(1))
        context, attn_weights = self.attention(decoder_hidden, encoder_outputs, mask=mask)
        lstm_input = torch.cat((input_emb, context.unsqueeze(1)), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (decoder_hidden, decoder_cell))
        output = output.squeeze(1)
        output_comb = torch.cat((output, context, input_emb.squeeze(1)), dim=1)
        pred = self.fc_out(output_comb)
        return pred, hidden, cell, attn_weights

class Seq2Seq(nn.Module):
    """Wrapper: durante training aplica teacher forcing con prob. dada."""
    def __init__(self, encoder, decoder, pad_id=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = pad_id

    def create_mask(self, src):
        return (src != self.pad_id)

    def forward(self, src, src_lens, tgt_input, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        tgt_len = tgt_input.size(1)
        outputs = torch.zeros(batch_size, tgt_len, self.decoder.output_dim).to(src.device)
        encoder_outputs, hidden, cell = self.encoder(src, src_lens)
        mask = self.create_mask(src)
        input_tok = tgt_input[:,0]
        for t in range(1, tgt_len):
            pred, hidden, cell, attn = self.decoder(input_tok, hidden, cell, encoder_outputs, mask=mask)
            outputs[:,t,:] = pred
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = pred.argmax(1)
            input_tok = tgt_input[:,t] if teacher_force else top1
        return outputs
# CELDA 8 - helpers
import torch.optim as optim
from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start, end):
    elapsed = end - start
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    return mins, secs, elapsed

def greedy_decode(model, src_tensor, src_lens, max_len=MAX_DECODING_LEN):
    model.eval()
    with torch.no_grad():
        src = src_tensor.to(DEVICE); src_lens = src_lens.to(DEVICE)
        encoder_outputs, hidden, cell = model.encoder(src, src_lens)
        mask = model.create_mask(src)
        batch_size = src.size(0)
        inputs = torch.full((batch_size,), BOS_ID, dtype=torch.long, device=DEVICE)
        outputs_list = [[] for _ in range(batch_size)]
        finished = [False]*batch_size
        for _ in range(max_len):
            preds, hidden, cell, attn = model.decoder(inputs, hidden, cell, encoder_outputs, mask=mask)
            top1 = preds.argmax(1)
            inputs = top1
            for i in range(batch_size):
                tok = top1[i].item()
                if tok == EOS_ID:
                    finished[i] = True
                else:
                    outputs_list[i].append(tok)
            if all(finished):
                break
    texts = [decode_ids_to_text(seq) for seq in outputs_list]
    return texts
# CELDA 9 - preparar dataloaders y construir modelo
SEQ_MAX_LEN = 80  # longitud máxima usada por el dataset (coincide con Fase1/MAX_WORDS)

train_dataset = TranslationDataset(TRAIN_SRC, TRAIN_TGT, max_len=SEQ_MAX_LEN)
val_dataset = TranslationDataset(VAL_SRC, VAL_TGT, max_len=SEQ_MAX_LEN)
test_dataset = TranslationDataset(TEST_SRC, TEST_TGT, max_len=SEQ_MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

enc = EncoderLSTM(input_dim=VOCAB_SIZE, emb_dim=EMBED_SIZE, hid_dim=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT, pad_id=PAD_ID)
dec = DecoderLSTMWithAttention(output_dim=VOCAB_SIZE, emb_dim=EMBED_SIZE, enc_hid_dim=HIDDEN_SIZE*2, dec_hid_dim=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT, pad_id=PAD_ID)
model = Seq2Seq(enc, dec, pad_id=PAD_ID).to(DEVICE)

print("Vocab size:", VOCAB_SIZE)
print("Número de parámetros (trainable):", count_parameters(model))
# CELDA 10 - Entrenamiento con AMP (torch.amp) - versión corregida
from torch.amp import autocast, GradScaler
import time

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
scaler = GradScaler()

best_valid_loss = float('inf')
history = {"train_loss": [], "val_loss": [], "epoch_time_s": []}
total_start = time.time()

for epoch in range(1, NUM_EPOCHS+1):
    print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
    epoch_start = time.time()

    # --- entrenamiento ---
    model.train()
    train_loss_sum = 0.0
    train_batches = 0
    for src, src_lens, dec_in, dec_out in tqdm(train_loader, desc="Train batches"):
        src = src.to(DEVICE); src_lens = src_lens.to(DEVICE)
        dec_in = dec_in.to(DEVICE); dec_out = dec_out.to(DEVICE)
        optimizer.zero_grad()

        # 🔥 AUTOCAST CORREGIDO
        with autocast("cuda"):
            outputs = model(src, src_lens, dec_in, teacher_forcing_ratio=TEACHER_FORCING_RATIO)
            preds = outputs[:,1:,:].contiguous()
            targets = dec_out[:, :preds.size(1)].contiguous()
            loss = criterion(preds.view(-1, VOCAB_SIZE), targets.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss_sum += loss.item()
        train_batches += 1

    train_loss = train_loss_sum / max(1, train_batches)

    # --- validación ---
    model.eval()
    val_loss_sum = 0.0
    val_batches = 0
    with torch.no_grad():
        for src, src_lens, dec_in, dec_out in tqdm(val_loader, desc="Val batches"):
            src = src.to(DEVICE); src_lens = src_lens.to(DEVICE)
            dec_in = dec_in.to(DEVICE); dec_out = dec_out.to(DEVICE)

            # 🔥 TAMBIÉN CORREGIDO
            with autocast("cuda"):
                outputs = model(src, src_lens, dec_in, teacher_forcing_ratio=0.0)
                preds = outputs[:,1:,:].contiguous()
                targets = dec_out[:, :preds.size(1)].contiguous()
                loss = criterion(preds.view(-1, VOCAB_SIZE), targets.view(-1))

            val_loss_sum += loss.item()
            val_batches += 1

    val_loss = val_loss_sum / max(1, val_batches)
    epoch_time_s = time.time() - epoch_start

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["epoch_time_s"].append(epoch_time_s)

    print(f"Epoch {epoch} -> train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | epoch_time: {epoch_time_s:.1f}s")

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_es_en.pt"))
        print("Mejor modelo guardado.")

# guardar final y history
torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "final_es_en.pt"))
with open(os.path.join(MODEL_SAVE_DIR, "training_history.json"), "w", encoding="utf-8") as fh:
    json.dump(history, fh, ensure_ascii=False, indent=2)

total_end = time.time()
print("Entrenamiento completo. Tiempo total (s):", total_end - total_start, "=> GPU hours approx:", (total_end - total_start)/3600.0)
# CELDA 11 - Evaluación en test: pérdida, BLEU y recursos
import sacrebleu

# cargar mejor modelo
best_path = os.path.join(MODEL_SAVE_DIR, "best_es_en.pt")
if not os.path.exists(best_path):
    best_path = os.path.join(MODEL_SAVE_DIR, "final_es_en.pt")
model.load_state_dict(torch.load(best_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# calcular test loss
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
test_loss_sum = 0.0
test_batches = 0
with torch.no_grad():
    for src, src_lens, dec_in, dec_out in tqdm(test_loader, desc="Test batches"):
        src = src.to(DEVICE); src_lens = src_lens.to(DEVICE)
        dec_in = dec_in.to(DEVICE); dec_out = dec_out.to(DEVICE)
        outputs = model(src, src_lens, dec_in, teacher_forcing_ratio=0.0)
        preds = outputs[:,1:,:].contiguous()
        targets = dec_out[:, :preds.size(1)].contiguous()
        loss = criterion(preds.view(-1, VOCAB_SIZE), targets.view(-1))
        test_loss_sum += loss.item()
        test_batches += 1
test_loss = test_loss_sum / max(1, test_batches)
print("Test loss:", test_loss)

# generar hipótesis y referencias para BLEU
hypotheses = []
references = []
for src, src_lens, dec_in, dec_out in tqdm(test_loader, desc="Decoding test set"):
    hyps = greedy_decode(model, src, src_lens, max_len=MAX_DECODING_LEN)
    hypotheses.extend(hyps)
    for i in range(dec_out.size(0)):
        ref_ids = dec_out[i].cpu().tolist()
        if EOS_ID in ref_ids:
            ref_ids = ref_ids[:ref_ids.index(EOS_ID)]
        references.append(decode_ids_to_text(ref_ids))

bleu = sacrebleu.corpus_bleu(hypotheses, [references])
print("BLEU (sacrebleu):", bleu.score)

# recursos: tiempo de entrenamiento y nº parámetros
with open(os.path.join(MODEL_SAVE_DIR, "training_history.json"), "r", encoding="utf-8") as fh:
    hist = json.load(fh)
total_train_time_s = sum(hist.get("epoch_time_s", []))
gpu_hours = total_train_time_s / 3600.0
num_params = count_parameters(model)

test_results = {
    "test_loss": test_loss,
    "bleu": float(bleu.score),
    "num_test_sentences": len(hypotheses),
    "num_parameters": num_params,
    "total_train_time_s": total_train_time_s,
    "gpu_hours_approx": gpu_hours
}
with open(os.path.join(MODEL_SAVE_DIR, "test_results.json"), "w", encoding="utf-8") as fo:
    json.dump(test_results, fo, ensure_ascii=False, indent=2)
print("Resultados guardados en:", os.path.join(MODEL_SAVE_DIR, "test_results.json"))
print(test_results)
# CELDA 12 - graficar losses y mostrar resumen
import matplotlib.pyplot as plt
with open(os.path.join(MODEL_SAVE_DIR, "training_history.json"), "r", encoding="utf-8") as fh:
    history = json.load(fh)
train_loss = history.get("train_loss", [])
val_loss = history.get("val_loss", [])
plt.figure(figsize=(8,4))
plt.plot(train_loss, label="train_loss")
plt.plot(val_loss, label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss por epoch")
plt.show()

with open(os.path.join(MODEL_SAVE_DIR, "test_results.json"), "r", encoding="utf-8") as fr:
    results = json.load(fr)
print("Resumen final:")
print("BLEU:", results["bleu"])
print("Test loss:", results["test_loss"])
print("Nº parámetros:", results["num_parameters"])
print("GPU hours (aprox):", results["gpu_hours_approx"])
# CELDA 13 - interfaz para probar el traductor
best_path = os.path.join(MODEL_SAVE_DIR, "best_es_en.pt")
if not os.path.exists(best_path):
    best_path = os.path.join(MODEL_SAVE_DIR, "final_es_en.pt")
model.load_state_dict(torch.load(best_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("Interfaz ES -> EN. Escribe 'salir' para terminar.")
while True:
    s = input("ES> ").strip()
    if not s: continue
    if s.lower() in ("salir","exit","quit"): break
    s_norm = normalize_text(s, LOWERCASE)
    src_ids = encode_sentence(s_norm)[:SEQ_MAX_LEN]
    src_tensor = torch.tensor([src_ids], dtype=torch.long)
    src_lens = torch.tensor([len(src_ids)], dtype=torch.long)
    pred = greedy_decode(model, src_tensor, src_lens, max_len=MAX_DECODING_LEN)[0]
    print("EN>", pred)

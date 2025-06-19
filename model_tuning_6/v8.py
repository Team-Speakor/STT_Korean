import os, json, random
from dataclasses import dataclass

import torch
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    TrainingArguments, Trainer, set_seed, TrainerCallback
)
import numpy as np


# ---------- 0. 재현 가능성 ----------
set_seed(42)

# ---------- 1. 경로 ----------
BASE_DIR  = "/home/coop1964/STT_Korean/dataset"
LABEL_DIR = os.path.join(BASE_DIR, "script")
VOICE_DIR = os.path.join(BASE_DIR, "audio")

# ---------- 2. JSON + WAV 로드 ----------
data_list = []
for jf in os.listdir(LABEL_DIR):
    if not jf.endswith(".json"):
        continue
    with open(os.path.join(LABEL_DIR, jf), encoding="utf-8") as f:
        meta = json.load(f)

    wav = os.path.join(VOICE_DIR, meta["fileName"])
    txt = meta.get("transcription", "").strip()

    if os.path.exists(wav) and txt:
        data_list.append({"audio": wav, "text": txt})

random.shuffle(data_list)
data_list = data_list[:38_000]

full_ds = Dataset.from_list(data_list)

# ---------- 3. split 8:1:1 ----------
split1 = full_ds.train_test_split(test_size=0.2, seed=42)
split2 = split1["test"].train_test_split(test_size=0.5, seed=42)
dataset = DatasetDict({
    "train":      split1["train"],
    "validation": split2["train"],
    "test":       split2["test"]
})

# ---------- 4. 오디오 컬럼 → 16 kHz ----------
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

# ---------- 5. Processor & 토크나이저 ----------
MODEL_NAME = "kresnik/wav2vec2-large-xlsr-korean"
processor  = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

new_tokens = list(".,?!")
num_added  = processor.tokenizer.add_tokens(new_tokens)
if num_added:
    print(f"🆕 tokenizer에 문장부호 {new_tokens} 추가됨({num_added})")

# ---------- 6. 전처리 ----------
def prepare_batch(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(
        audio["array"], sampling_rate=16000
    ).input_values[0]

    batch["labels"] = processor(
        text=batch["text"], return_tensors="pt"
    ).input_ids.squeeze(0)
    return batch

dataset = dataset.map(
    prepare_batch,
    remove_columns=["audio", "text"],
    load_from_cache_file=False,
    num_proc=1
)

# ---------- 7. DataCollator ----------
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: bool = True

    def __call__(self, features):
        inputs = [torch.tensor(f["input_values"]) for f in features]
        labels = [torch.tensor(f["labels"])       for f in features]

        batch_inputs = torch.nn.utils.rnn.pad_sequence(
            inputs, batch_first=True
        )
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        return {"input_values": batch_inputs, "labels": batch_labels}

data_collator = DataCollatorCTCWithPadding(processor)

# ---------- 8. 모델 로드 ----------
model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_NAME,
    vocab_size=len(processor.tokenizer),
    pad_token_id=processor.tokenizer.pad_token_id,
    attention_dropout=0.05,
    hidden_dropout=0.05,
    feat_proj_dropout=0.0,
    layerdrop=0.05,
    mask_time_prob=0.04,
    ignore_mismatched_sizes=True
)

# ---------- 9. 하이퍼파라미터 (v8) ----------
training_args = TrainingArguments(
    output_dir="./wav2vec2_finetune_ko_v8",
    eval_steps=500,
    save_steps=500,
    logging_steps=50,
    evaluation_strategy="steps",
    save_total_limit=3,
    num_train_epochs=25,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,      # ★ v8 변경: (2 GPU × 8) = 16 effective
    fp16=True,
    group_by_length=True,
    weight_decay=0.005,
    learning_rate=3e-4,                 # ★ v8 변경
    warmup_ratio=0.10,                  # ★ v8 변경
    dataloader_num_workers=4,
    report_to="none",
    load_best_model_at_end=True,        # ★ v8 추가
    metric_for_best_model="wer",        # ★ v8 추가
    greater_is_better=False             # ★ v8 추가
)

# ---------- 10. (선택) WER 계산 함수 ----------
# * metric_for_best_model을 쓰려면 compute_metrics가 필요합니다.
from jiwer import wer
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids    = np.argmax(pred_logits, axis=-1)
    pred_str    = processor.batch_decode(pred_ids)

    label_ids = pred.label_ids
    # -100 → pad_token_id 로 치환
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    return {"wer": wer(label_str, pred_str)}

# ---------- 11. Trainer ----------
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor,
    compute_metrics=compute_metrics,        # ★ v8 추가
)

# ---------- 12. 학습 ----------
trainer.train()

# ---------- 13. 저장 ----------
CKPT_DIR = "./wav2vec2_ko_v10_best"
model.save_pretrained(CKPT_DIR)
processor.save_pretrained(CKPT_DIR)
print(f"✅ 모델과 프로세서가 {CKPT_DIR} 에 저장되었습니다.")

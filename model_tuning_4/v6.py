"""
목표
1)  문장부호 [UNK] 문제 → 토크나이저에 . , ? ! 추가
2)  총 30 000개 샘플, 8 : 1 : 1 split
3)  하이퍼파라미터 조정 (epoch 18, lr 2e-4, batch size 32-equiv)
4)  재현성·안정성(seeds, fp16, checkpoint, tokenizer resize)
"""

# ============ 라이브러리 ============
import os, json, random
from dataclasses import dataclass

import torch, numpy as np
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    TrainingArguments, Trainer, set_seed,
    GenerationConfig               # <- 사용 X, but keep import
)

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
    meta = json.load(open(os.path.join(LABEL_DIR, jf), encoding="utf-8"))

    wav = os.path.join(VOICE_DIR, meta["fileName"])
    txt = meta.get("transcription", "").strip()

    if os.path.exists(wav) and txt:
        data_list.append({"audio": wav, "text": txt})

# 2-①: 샘플 30 000개만 사용  ★ 변경
random.shuffle(data_list)
data_list = data_list[:30_000]

full_ds = Dataset.from_list(data_list)

# ---------- 3. 8 : 1 : 1 split ----------
split1 = full_ds.train_test_split(test_size=0.2, seed=42)
split2 = split1["test"].train_test_split(test_size=0.5, seed=42)
dataset = DatasetDict(
    train      = split1["train"],
    validation = split2["train"],
    test       = split2["test"]
)

# ---------- 4. 오디오 컬럼 16 kHz ----------
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

# ---------- 5. Processor & 토크나이저 ----------
MODEL_NAME = "kresnik/wav2vec2-large-xlsr-korean"
processor  = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

# 5-①: 문장부호 추가 & tokenizer resize  ★ 변경
added = processor.tokenizer.add_tokens(list(".,?!"))
if added:
    print(f"🆕  tokenizer에 문장부호 {added}개 추가")

# ---------- 6. 전처리 ----------
def prepare_batch(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(
        audio["array"], sampling_rate=16_000
    ).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

dataset = dataset.map(
    prepare_batch,
    remove_columns=["audio", "text"],
    num_proc=4
)

# ---------- 7. DataCollator ----------
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor

    def __call__(self, features):
        inputs  = [torch.tensor(f["input_values"]) for f in features]
        labels  = [torch.tensor(f["labels"])       for f in features]

        batch_inputs = torch.nn.utils.rnn.pad_sequence(
            inputs, batch_first=True
        )
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_mask = (batch_inputs != 0).long()
        return {"input_values": batch_inputs,
                "attention_mask": attention_mask,
                "labels": batch_labels}


data_collator = DataCollatorCTCWithPadding(processor)

# ---------- 8. 모델 ----------
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

if added:  # 새 토큰이 실제로 추가됐을 때만
    old_lm_head = model.lm_head
    in_features = old_lm_head.in_features
    # 새 vocab 크기로 새 Linear 레이어 생성
    model.lm_head = torch.nn.Linear(in_features, len(processor.tokenizer))
    # 가중치 초기값을 이전 것으로 복사하고, 추가된 토큰 부분만 Xavier init
    with torch.no_grad():
        model.lm_head.weight[: old_lm_head.out_features] = old_lm_head.weight
        model.lm_head.bias  [: old_lm_head.out_features] = old_lm_head.bias
        torch.nn.init.xavier_uniform_(model.lm_head.weight[old_lm_head.out_features :])
        torch.nn.init.zeros_(model.lm_head.bias[old_lm_head.out_features :])
    print("✅  lm_head 리사이즈 완료")

# ---------- 9. 하이퍼파라미터 ----------  ★ 전면 수정
training_args = TrainingArguments(
    output_dir          = "./wav2vec2_finetune_ko_v6",
    evaluation_strategy = "epoch",   # epoch 끝마다 loss 측정
    save_strategy       = "epoch",
    save_total_limit    = 3,
    num_train_epochs    = 20,        # 15~20 권장
    per_device_train_batch_size = 4,
    per_device_eval_batch_size  = 4,
    gradient_accumulation_steps = 2, # 4×GPU → 실효 BS 32
    fp16               = True,
    learning_rate      = 2e-4,       # 1e-4~2e-4 안정
    warmup_ratio       = 0.05,
    lr_scheduler_type  = "cosine",
    dataloader_num_workers = 4,
    logging_steps      = 100,
    report_to          = "none",
    group_by_length=True,
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    data_collator   = data_collator,
    train_dataset   = dataset["train"],
    eval_dataset    = dataset["validation"],
    tokenizer       = processor.feature_extractor,
    # compute_metrics 삭제 — eval_loss만 확인
)

# ---------- 10. 학습 ----------
trainer.train()

# ---------- 11. 저장 ----------
CKPT_DIR = "./wav2vec2_korean_v6"
model.save_pretrained(CKPT_DIR)
processor.save_pretrained(CKPT_DIR)
print(f"✅ 모델·프로세서 저장: {CKPT_DIR}")

# ---------- 12. 간단 평가 (loss만) ----------
eval_results = trainer.evaluate(dataset["validation"])
print(f"\n🟢  validation loss: {eval_results['eval_loss']:.4f}")

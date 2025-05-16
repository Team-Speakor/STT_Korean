#!/usr/bin/env python
# coding: utf-8
"""
목표
1) 문장부호가 [UNK] 로 출력되는 문제 해결 → 토크나이저에 문장부호(.,?!) 추가
2) 총 샘플 8000개만 사용, 비율 8:1:1
3) 하이퍼파라미터 & 학습 옵션 튜닝 (한국어-청자 입장에 맞춤)
4) 잠재적 오류 예방 (seed, fp16, resume, logging 등)
"""

import os, json, random
from dataclasses import dataclass

import torch
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    TrainingArguments, Trainer, set_seed
)

import numpy as np
import evaluate

# evaluate function
# Hugging Face metric 라이브러리 로드
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # label_ids = -100으로 마스킹된 부분은 tokenizer pad_token으로 교체
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # 토큰 ID → 텍스트로 디코딩
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {
        "wer": wer,
        "cer": cer,
    }


# ====== 0. 재현 가능성 ======  # ★ 수정
set_seed(42)

# ====== 1. 경로 ======
BASE_DIR  = "/home/coop1964/STT_Korean/dataset"
LABEL_DIR = os.path.join(BASE_DIR, "script")
VOICE_DIR = os.path.join(BASE_DIR, "audio")

# ====== 2. JSON + WAV 로드 ======
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

# 2-①: 총 8 000개 샘플만 랜덤 선택  # ★ 수정
random.shuffle(data_list)
data_list = data_list[:8_000]

full_ds = Dataset.from_list(data_list)

# ====== 3. split 8:1:1 ======
split1 = full_ds.train_test_split(test_size=0.2, seed=42)
split2 = split1["test"].train_test_split(test_size=0.5, seed=42)
dataset = DatasetDict({
    "train":      split1["train"],
    "validation": split2["train"],
    "test":       split2["test"]
})

# ====== 4. 오디오 컬럼 → 16 kHz ======
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

# ====== 5. Processor & 토크나이저 확장 ======
MODEL_NAME = "kresnik/wav2vec2-large-xlsr-korean"
processor  = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

# 5-①: 문장부호 토큰 추가 → UNK 문제 해결   # ★ 수정
new_tokens = list(".,?!")
num_added  = processor.tokenizer.add_tokens(new_tokens)
if num_added:
    print(f"🆕 tokenizer에 문장부호 {new_tokens} 추가됨({num_added})")

# ====== 6. 전처리 ======
def prepare_batch(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(
        audio["array"], sampling_rate=16_000
    ).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(
            batch["text"], return_tensors="pt"
        ).input_ids.squeeze(0)
    return batch

dataset = dataset.map(
    prepare_batch,
    remove_columns=["audio", "text"],
    num_proc=4
)

# ====== 7. DataCollator ======
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: bool=True

    def __call__(self, features):
        inputs  = [torch.tensor(f["input_values"]) for f in features]
        labels  = [torch.tensor(f["labels"])       for f in features]

        batch_inputs = torch.nn.utils.rnn.pad_sequence(
            inputs, batch_first=True
        )
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True,
            padding_value=-100
        )
        return {
            "input_values": batch_inputs,
            "labels":       batch_labels
        }

data_collator = DataCollatorCTCWithPadding(processor)

# ====== 8. 모델 로드 ======
model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_NAME,
    vocab_size=len(processor.tokenizer),
    pad_token_id=processor.tokenizer.pad_token_id,
    attention_dropout=0.05,      # ★ 약간 완화
    hidden_dropout=0.05,
    feat_proj_dropout=0.0,
    layerdrop=0.05,
    mask_time_prob=0.04,
    ignore_mismatched_sizes=True
)

# 토크나이저가 확장됐다면 lm_head 리사이즈
if num_added:
    # model.resize_token_embeddings(len(processor.tokenizer))
    print("resize_token")

# ====== 9. 하이퍼파라미터 ======   # ★ 수정
training_args = TrainingArguments(
    output_dir="./wav2vec2_finetune_ko_v3",
    evaluation_strategy="no",
    eval_steps=200,
    save_steps=200,
    logging_steps=50,
    save_total_limit=3,
    num_train_epochs=10,                   # 데이터 8 k → 10epoch면 충분
    per_device_train_batch_size=8,         # 4×3090 이므로 여유 ↑
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,         # 실효 BS=16
    fp16=True,                             # AMP 사용
    learning_rate=1e-4,                    # 살짝 낮춤
    warmup_ratio=0.05,
    dataloader_num_workers=4,
    report_to="none",                      # WandB 안 쓸 때
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics  # ✅ 추가
)

# ====== 10. 학습 ======
trainer.train()

# ====== 11. 저장 ======
CKPT_DIR = "./wav2vec2_korean_v3"
model.save_pretrained(CKPT_DIR)
processor.save_pretrained(CKPT_DIR)
print(f"✅ 모델과 프로세서가 {CKPT_DIR} 에 저장되었습니다.")


# 12. evaluate
eval_results = trainer.evaluate(dataset["validation"])

print("\n========== 모델 평가 결과 ==========")
print(f"평가 손실 (eval_loss): {eval_results.get('eval_loss'):.4f}")
print(f"평가 WER (eval_wer): {eval_results.get('wer'):.4f}")
print(f"평가 CER (eval_cer): {eval_results.get('cer'):.4f}")
print(f"평가 런타임 (초): {eval_results.get('eval_runtime'):.2f}")
print(f"초당 평가 샘플 수: {eval_results.get('eval_samples_per_second'):.2f}")
print(f"초당 평가 스텝 수: {eval_results.get('eval_steps_per_second'):.2f}")
print(f"평가 에포크: {eval_results.get('epoch'):.2f}")
print("====================================")

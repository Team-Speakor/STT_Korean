import os, json, random
from dataclasses import dataclass

import torch
from datasets import Dataset, DatasetDict, Audio
from transformers import (
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    TrainingArguments, Trainer, set_seed, TrainerCallback
)

import numpy as np



# Feature Encoder를 3에폭 이후에 언프리즈하기 위한 콜백
class UnfreezeFeatureEncoderCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.epoch == 3:
            model = kwargs.get('model', None)
            if model is not None:
                model.wav2vec2.feature_extractor._freeze_parameters = False
                for param in model.wav2vec2.feature_extractor.parameters():
                    param.requires_grad = True
                print("\n특징 추출기(Feature Encoder)가 언프리즈 되었습니다!")
    
    # 에폭 끝에 GPU 캐시 정리
    def on_epoch_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        print(f"\n에폭 {state.epoch} 완료, GPU 캐시 정리됨")



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

# 2-①: 총 30000개 샘플만 랜덤 선택  # ★ 수정
random.shuffle(data_list)
data_list = data_list[:30_000]

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
        audio["array"], sampling_rate=16000
    ).input_values[0]

    batch["labels"] = processor(
        text=batch["text"], return_tensors="pt"
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


# ====== 9. 하이퍼파라미터 ======   # ★ 수정
training_args = TrainingArguments(
    output_dir="./wav2vec2_finetune_ko_v6",
    eval_steps=500,
    save_steps=500,
    logging_steps=50,
    eval_strategy="steps",
    save_total_limit=3,
    num_train_epochs=20,                   
    per_device_train_batch_size=2,         
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,         
    fp16=True,                             # AMP 사용
    group_by_length=True,                  # 오디오 길이 기반 동적 배치 구성
    weight_decay=0.005,
    learning_rate=2e-4,                    # 살짝 낮춤
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
    tokenizer=processor,
    callbacks=[UnfreezeFeatureEncoderCallback()]
)

# ====== 10. 학습 ======
trainer.train()

# ====== 11. 저장 ======
CKPT_DIR = "./wav2vec2_korean_v6"
model.save_pretrained(CKPT_DIR)
processor.save_pretrained(CKPT_DIR)
print(f"✅ 모델과 프로세서가 {CKPT_DIR} 에 저장되었습니다.")



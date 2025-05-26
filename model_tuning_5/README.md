dataset : 38000개 ( 24000/ 3000/ 3000)

### 🛠️ TrainingArguments 설정표

| 항목             | 변경 전 (v6) | 변경 후 (v7) | 설명                                         |
| ---------------- | ------------ | ------------ | -------------------------------------------- |
| 데이터셋 크기    | 30,000개     | **38,000개** | 학습에 더 많은 다양성과 일반화 능력 확보     |
| num_proc (GPU당) | 4            | 1            | 4 유지하고 데이터셋 늘림 -> 터짐 -> 1로 줄임 |

### code

```python
training_args = TrainingArguments(
    output_dir="./wav2vec2_finetune_ko_v6",
    eval_steps=500,
    save_steps=500,
    logging_steps=50,
    eval_strategy="steps",
    save_total_limit=3,
    num_train_epochs=20,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    fp16=True,                             # AMP 사용
    group_by_length=True,                  # 오디오 길이 기반 동적 배치 구성
    weight_decay=0.005,
    learning_rate=2e-4,                    # 살짝 낮춤
    warmup_ratio=0.05,
    dataloader_num_workers=4,
    report_to="none",                      # WandB 안 쓸 때
)
```

## PLAN
